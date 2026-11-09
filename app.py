import os,sqlite3,json,uuid,time,threading,subprocess
from pathlib import Path
from datetime import datetime,timezone
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
ROOT=Path(os.getenv('STUDIO_ROOT','/srv/ai-video-studio')); DB=ROOT/'studio.sqlite3'; UP=ROOT/'uploads'; OUT=ROOT/'outputs'; VOICES=ROOT/'voices'; MAX=int(os.getenv('MAX_UPLOAD_MB','500'))*1024*1024
for p in (ROOT,UP,OUT,VOICES): p.mkdir(parents=True,exist_ok=True)
def db():
 c=sqlite3.connect(DB,timeout=30);c.row_factory=sqlite3.Row;c.execute('PRAGMA journal_mode=WAL');return c
def stamp():return datetime.now(timezone.utc).isoformat()
def init():
 with db() as c:c.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,status TEXT,stage TEXT,progress INTEGER,payload TEXT,input_path TEXT,output_path TEXT,error TEXT,created_at TEXT,updated_at TEXT)')
def getjob(i):
 with db() as c:r=c.execute('SELECT * FROM jobs WHERE id=?',(i,)).fetchone()
 if not r: return None
 d=dict(r);d['payload']=json.loads(d['payload']);return d
def enqueue(kind,payload,path=None):
 i=str(uuid.uuid4());t=stamp()
 with db() as c:c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)',(i,kind,'queued','Queued',0,json.dumps(payload),path,None,None,t,t))
 return getjob(i)
app=FastAPI(title='AI Video Studio')
class TTS(BaseModel):text:str=Field(min_length=1,max_length=12000);language:str='hi';voice_id:str=''
@app.on_event('startup')
def startup():init()
@app.get('/api/health')
def health():return {'ok':True,'execution':'Azure server','queue':'SQLite','worker':'separate process'}
@app.get('/api/jobs')
def jobs():
 with db() as c:rows=c.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100').fetchall()
 out=[]
 for r in rows:
  d=dict(r);d['payload']=json.loads(d['payload']);out.append(d)
 return out
@app.post('/api/tts')
def tts(x:TTS):
 if x.language not in {'hi','en','fr','es','de'}:raise HTTPException(400,'Unsupported language')
 return enqueue('tts',x.model_dump())
@app.post('/api/video')
async def video(file:UploadFile=File(...),preset:str=Form('reel')):
 if preset not in {'reel','landscape','square'}:raise HTTPException(400,'Invalid preset')
 ext=Path(file.filename or '').suffix.lower()
 if ext not in {'.mp4','.mov','.mkv','.webm'}:raise HTTPException(415,'Unsupported video')
 folder=UP/str(uuid.uuid4());folder.mkdir();target=folder/('source'+ext);size=0
 with target.open('wb') as f:
  while chunk:=await file.read(1024*1024):
   size+=len(chunk)
   if size>MAX:target.unlink(missing_ok=True);raise HTTPException(413,'Upload too large')
   f.write(chunk)
 return enqueue('video_transform',{'preset':preset},str(target))
@app.post('/api/voices')
async def voice(name:str=Form(...),file:UploadFile=File(...)):
 if Path(file.filename or '').suffix.lower()!='.wav':raise HTTPException(415,'WAV reference required')
 safe=''.join(x for x in name if x.isalnum() or x in '-_')[:48]
 if not safe:raise HTTPException(400,'Invalid name')
 dest=VOICES/(safe+'.wav')
 if dest.exists():raise HTTPException(409,'Profile already exists')
 with dest.open('wb') as f:
  while chunk:=await file.read(1024*1024):f.write(chunk)
 return {'voice_id':safe}
@app.get('/api/voices')
def voices():return [{'voice_id':p.stem} for p in VOICES.glob('*.wav')]
@app.get('/api/jobs/{i}/output')
def output(i):
 j=getjob(i)
 if not j or j['status']!='completed' or not j['output_path']:raise HTTPException(404,'Output not ready')
 p=Path(j['output_path']).resolve()
 if OUT.resolve() not in p.parents or not p.is_file():raise HTTPException(404,'Not found')
 return FileResponse(p,filename=p.name)
def update(i,**kw):
 kw['updated_at']=stamp();sql=','.join(k+'=?' for k in kw)
 with db() as c:c.execute('UPDATE jobs SET '+sql+' WHERE id=?',(*kw.values(),i))
def work(j):
 i=j['id'];payload=j['payload'];folder=OUT/i;folder.mkdir(parents=True,exist_ok=True)
 if j['kind']=='video_transform':
  w,h=(1080,1920) if payload['preset']=='reel' else ((1920,1080) if payload['preset']=='landscape' else (1080,1080));dest=folder/'video.mp4';vf=f'scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1'
  update(i,stage='FFmpeg rendering',progress=20);subprocess.run(['ffmpeg','-y','-i',j['input_path'],'-map','0:v:0','-map','0:a?','-vf',vf,'-c:v','libx264','-preset','veryfast','-crf','21','-c:a','aac','-movflags','+faststart',str(dest)],check=True,timeout=7200)
 elif j['kind']=='tts':
  update(i,stage='Loading Chatterbox CPU model',progress=10)
  import torch,torchaudio
  from chatterbox.mtl_tts import ChatterboxMultilingualTTS
  model=ChatterboxMultilingualTTS.from_pretrained(device='cpu');ref=VOICES/(payload['voice_id']+'.wav') if payload.get('voice_id') else None
  kwargs={'language_id':payload['language']}
  if ref:
   if not ref.is_file():raise RuntimeError('Voice profile missing')
   kwargs['audio_prompt_path']=str(ref)
  update(i,stage='Generating speech',progress=35)
  with torch.inference_mode():audio=model.generate(payload['text'],**kwargs)
  dest=folder/'narration.wav';torchaudio.save(str(dest),audio.detach().cpu(),model.sr);del model
 else:raise RuntimeError('Unsupported job type')
 update(i,status='completed',stage='Completed',progress=100,output_path=str(dest))
def worker():
 init()
 while True:
  j=None
  with db() as c:
   c.execute('BEGIN IMMEDIATE');r=c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
   if r:c.execute("UPDATE jobs SET status='running',stage='Starting',updated_at=? WHERE id=?",(stamp(),r['id']))
  if r:j=getjob(r['id'])
  if not j:time.sleep(2);continue
  try:work(j)
  except Exception as e:update(j['id'],status='failed',stage='Failed',error=str(e)[:2000])
if __name__=='__main__':worker()
