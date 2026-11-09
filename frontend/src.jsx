import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Clapperboard, Film, Mic2, FolderOpen, Upload, Play, RefreshCw,
  Download, WandSparkles, LogOut, LockKeyhole, Trash2
} from 'lucide-react';
import './style.css';

function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [checking, setChecking] = useState(true);
  const [password, setPassword] = useState('');
  const [tab, setTab] = useState('create');
  const [jobs, setJobs] = useState([]);
  const [voices, setVoices] = useState([]);
  const [text, setText] = useState('');
  const [lang, setLang] = useState('hi');
  const [voice, setVoice] = useState('');
  const [file, setFile] = useState(null);
  const [sceneFiles, setSceneFiles] = useState([]);
  const [activeScene, setActiveScene] = useState(0);
  const [scenePreviewUrl, setScenePreviewUrl] = useState('');
  const [projectBusy, setProjectBusy] = useState(false);

  const [previewUrl, setPreviewUrl] = useState('');
  const [trimStart, setTrimStart] = useState('0');
  const [trimEnd, setTrimEnd] = useState('0');
  const [musicVolume, setMusicVolume] = useState(0.35);
  const [preset, setPreset] = useState('reel');
  const [framing, setFraming] = useState('fill');
  const [audioMode, setAudioMode] = useState('keep');
  const [musicFile, setMusicFile] = useState(null);
  const [overlayText, setOverlayText] = useState('');
  const [overlayPosition, setOverlayPosition] = useState('bottom');
  const [subtitleFile, setSubtitleFile] = useState(null);
  const [name, setName] = useState('');
  const [wav, setWav] = useState(null);
  const [msg, setMsg] = useState('');
  const [status, setStatus] = useState('Connecting…');
  const [busy, setBusy] = useState(false);

  async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw Error(data.detail || 'Request failed');
    return data;
  }

  async function checkSession() {
    try {
      await api('/api/auth/session');
      setAuthenticated(true);
    } catch {
      setAuthenticated(false);
    } finally {
      setChecking(false);
    }
  }

  async function login(event) {
    event.preventDefault();
    setBusy(true);
    setMsg('');
    try {
      await api('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      });
      setPassword('');
      setAuthenticated(true);
      setMsg('Signed in securely.');
    } catch (error) {
      setMsg(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function logout() {
    try { await api('/api/auth/logout', { method: 'POST' }); } catch {}
    setAuthenticated(false);
    setJobs([]);
    setVoices([]);
    setMsg('You have been signed out.');
  }

  async function refresh() {
    try {
      const [jobList, voiceList] = await Promise.all([
        api('/api/jobs'), api('/api/voices')
      ]);
      setJobs(jobList);
      setVoices(voiceList);
      setStatus('Azure connected');
    } catch (error) {
      setStatus(error.message === 'Authentication required' ? 'Session expired' : 'Server offline');
      if (error.message === 'Authentication required') setAuthenticated(false);
    }
  }

  useEffect(() => { checkSession(); }, []);
  useEffect(() => {
    const selected = sceneFiles[activeScene]?.file;
    if (!selected) {
      setScenePreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(selected);
    setScenePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [sceneFiles, activeScene]);
  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!authenticated) return;
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [authenticated]);

  async function run(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await api('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: lang, voice_id: voice })
      });
      setText('');
      setMsg('Narration queued on Azure');
    } catch (error) { setMsg(error.message); }
    setBusy(false);
    refresh();
  }

  async function upload(event) {
    event.preventDefault();
    const form = new FormData();
    form.append('file', file);
    form.append('preset', preset);
    form.append('framing', framing);
    form.append('audio', audioMode);
    form.append('trim_start', trimStart || '0');
    form.append('trim_end', trimEnd || '0');
    form.append('music_volume', String(musicVolume));
    if (audioMode === 'music' && musicFile) form.append('music_file', musicFile);
    form.append('overlay_text', overlayText);
    form.append('overlay_position', overlayPosition);
    if (subtitleFile) form.append('subtitle_file', subtitleFile);
    setBusy(true);
    try {
      await api('/api/video', { method: 'POST', body: form });
      setMsg('Video queued on Azure');
    } catch (error) { setMsg(error.message); }
    setBusy(false);
    refresh();
  }

  function addScenes(event) {
    const incoming = Array.from(event.target.files || []);
    setSceneFiles(current => [...current, ...incoming].slice(0, 10).map((scene, index) =>
      scene.file ? scene : { id: `${Date.now()}-${index}-${scene.name}`, file: scene, trimStart: '0', trimEnd: '0' }
    ));
    event.target.value = '';
  }

  function updateScene(id, changes) {
    setSceneFiles(current => current.map(scene => scene.id === id ? { ...scene, ...changes } : scene));
  }

  function moveScene(index, direction) {
    setSceneFiles(current => {
      const next = [...current];
      const target = index + direction;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      setActiveScene(target);
      return next;
    });
  }

  function removeScene(id) {
    setSceneFiles(current => current.filter(scene => scene.id !== id));
    setActiveScene(0);
  }

  async function renderTimeline(event) {
    event.preventDefault();
    if (!sceneFiles.length) return;
    const form = new FormData();
    sceneFiles.forEach(scene => form.append('scenes', scene.file));
    form.append('scene_settings', JSON.stringify(sceneFiles.map(scene => ({
      trim_start: Number(scene.trimStart || 0),
      trim_end: Number(scene.trimEnd || 0)
    }))));
    form.append('preset', preset);
    form.append('framing', framing);
    form.append('audio', audioMode);
    form.append('music_volume', String(musicVolume));
    form.append('overlay_text', overlayText);
    form.append('overlay_position', overlayPosition);
    if (audioMode === 'music' && musicFile) form.append('music_file', musicFile);
    if (subtitleFile) form.append('subtitle_file', subtitleFile);
    setProjectBusy(true);
    try {
      await api('/api/project', { method: 'POST', body: form });
      setMsg(`Project with ${sceneFiles.length} scenes queued on Azure.`);
      setSceneFiles([]);
      setActiveScene(0);
      refresh();
    } catch (error) {
      setMsg(error.message);
    } finally {
      setProjectBusy(false);
    }
  }

  async function deleteMedia(url, description) {
    if (!window.confirm(`Permanently delete ${description}? This cannot be undone.`)) return;
    try {
      await api(url, { method: 'DELETE' });
      setMsg(`${description} deleted.`);
      refresh();
    } catch (error) {
      setMsg(error.message);
    }
  }

  async function addVoice(event) {
    event.preventDefault();
    const form = new FormData();
    form.append('name', name);
    form.append('file', wav);
    try {
      await api('/api/voices', { method: 'POST', body: form });
      setMsg('Voice reference saved');
      refresh();
    } catch (error) { setMsg(error.message); }
  }

  const nav = [
    ['create', 'Create video', WandSparkles],
    ['transform', 'Transform', Film],
    ['timeline', 'Timeline', Film],
    ['voice', 'Voice studio', Mic2],
    ['projects', 'Projects', FolderOpen]
  ];

  if (checking) return <div className="auth-shell"><section><p>Checking secure session…</p></section></div>;

  if (!authenticated) return (
    <div className="auth-shell">
      <section className="auth-card">
        <h2><Clapperboard/> FRAME / AI</h2>
        <span>PRIVATE VIDEO WORKSPACE</span>
        <h1><LockKeyhole/> Sign in</h1>
        <p>This studio is private. Enter your administrator password to continue.</p>
        {msg && <p className="msg">{msg}</p>}
        <form onSubmit={login}>
          <label>ADMINISTRATOR PASSWORD
            <input type="password" value={password} onChange={event => setPassword(event.target.value)}
              autoComplete="current-password" required autoFocus />
          </label>
          <button className="primary" disabled={busy || !password}>Sign in securely</button>
        </form>
      </section>
    </div>
  );

  return (
    <div className="shell">
      <aside>
        <h2><Clapperboard/> FRAME / AI</h2>
        <small>YOUR VIDEO WORKSPACE</small>
        {nav.map(([id, label, Icon]) =>
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>
            <Icon size={17}/>{label}
          </button>
        )}
        <button onClick={logout}><LogOut size={17}/>Sign out</button>
        <p className="server">● {status}<br/><small>Server-side processing</small></p>
      </aside>
      <main>
        <header>
          <div><small>AI VIDEO STUDIO</small><h1>{nav.find(item => item[0] === tab)[1]}</h1></div>
          <button onClick={refresh}><RefreshCw/></button>
        </header>
        {msg && <p className="msg">{msg}</p>}
        {tab === 'create' && <section>
          <span>AI VIDEO WORKFLOW</span>
          <h1>Make your next story<br/><i>worth watching.</i></h1>
          <p>Generate Hindi or English narration on Azure. Prompt-to-video scenes will be enabled after CPU image generation is benchmarked.</p>
          <form onSubmit={run}>
            <label>NARRATION TEXT</label>
            <textarea value={text} onChange={event => setText(event.target.value)} rows="7" placeholder="Write your Hindi or English narration…" required/>
            <div className="row">
              <label>LANGUAGE<select value={lang} onChange={event => setLang(event.target.value)}><option value="hi">Hindi</option><option value="en">English</option></select></label>
              <label>VOICE PROFILE<select value={voice} onChange={event => setVoice(event.target.value)}><option value="">Default</option>{voices.map(item => <option key={item.voice_id} value={item.voice_id}>{item.voice_id}</option>)}</select></label>
            </div>
            <button className="primary" disabled={busy || !text.trim()}><Play size={16}/> Generate narration</button>
          </form>
          <div className="note">● Single neural worker　● CPU inference　● Azure jobs</div>
        </section>}
        {tab === 'transform' && <section>
          <span>UPLOAD WORKFLOW</span><h1>Reframe your footage.</h1>
          <p>Original video stays untouched. New export rendered with FFmpeg on Azure.</p>
          <form onSubmit={upload}>
            <label className="drop"><Upload/><b>{file?.name || 'Choose a video file'}</b><small>MP4 / MOV / MKV / WebM</small><input type="file" accept="video/*" onChange={event => setFile(event.target.files[0])}/></label>
            {previewUrl && <div className="video-preview">
              <video src={previewUrl} controls playsInline preload="metadata"/>
              {overlayText && <div className={"preview-title preview-" + overlayPosition}>{overlayText}</div>}
              <small>Local preview · original file is not uploaded until you queue the export</small>
            </div>}
            <div className="row">
              <label>TRIM START (SECONDS)<input type="number" min="0" max="14400" step="0.1" value={trimStart} onChange={event => setTrimStart(event.target.value)}/></label>
              <label>TRIM END (SECONDS · 0 = FULL LENGTH)<input type="number" min="0" max="14400" step="0.1" value={trimEnd} onChange={event => setTrimEnd(event.target.value)}/></label>
            </div>
            <div className="row">
              <label>EXPORT FORMAT<select value={preset} onChange={event => setPreset(event.target.value)}><option value="reel">Reels / Shorts · 9:16</option><option value="landscape">YouTube · 16:9</option><option value="square">Square · 1:1</option></select></label>
              <label>FRAMING<select value={framing} onChange={event => setFraming(event.target.value)}><option value="fill">Fill frame · crop edges</option><option value="fit">Fit whole video · black bars</option></select></label>
            </div>
            <label>AUDIO<select value={audioMode} onChange={event => setAudioMode(event.target.value)}><option value="keep">Keep original audio</option><option value="mute">Remove audio</option><option value="music">Replace with background music</option></select></label>
            {audioMode === 'music' && <>
              <label className="drop"><Upload/><b>{musicFile?.name || 'Choose background music'}</b><small>MP3 / WAV / M4A / AAC / OGG · 30 MB max</small><input type="file" accept=".mp3,.wav,.m4a,.aac,.ogg,audio/*" onChange={event => setMusicFile(event.target.files[0])}/></label>
              <label>MUSIC VOLUME · {Math.round(musicVolume * 100)}%
                <input type="range" min="0" max="1.5" step="0.05" value={musicVolume} onChange={event => setMusicVolume(Number(event.target.value))}/>
              </label>
            </>}
            <label>TEXT OVERLAY (OPTIONAL)
              <input value={overlayText} maxLength={160} onChange={event => setOverlayText(event.target.value)} placeholder="Add a title or short message (160 characters max)"/>
            </label>
            <label>TEXT POSITION<select value={overlayPosition} onChange={event => setOverlayPosition(event.target.value)}><option value="top">Top</option><option value="center">Center</option><option value="bottom">Bottom</option></select></label>
            <label className="drop"><Upload/><b>{subtitleFile?.name || 'Optional SRT subtitles'}</b><small>UTF-8 SRT · Hindi and English · 2 MB max</small><input type="file" accept=".srt,application/x-subrip" onChange={event => setSubtitleFile(event.target.files[0])}/></label>
            <button className="primary" disabled={!file || busy || (audioMode === 'music' && !musicFile) || (trimEnd !== '0' && Number(trimEnd) <= Number(trimStart))}>Queue video transform</button>
          </form>
        </section>}
        {tab === 'timeline' && <section className="timeline-workspace">
          <span>MULTI-SCENE EDITOR</span><h1>Build your story.</h1>
          <p>Add up to 10 clips, arrange their order, set an individual trim range and render them as one video. Each scene is normalised before joining.</p>
          <label className="drop"><Upload/><b>Add video scenes</b><small>MP4 / MOV / MKV / WebM · up to 10 scenes · 500 MB total</small><input type="file" accept="video/*" multiple onChange={addScenes}/></label>
          {sceneFiles.length > 0 && <>
            <div className="timeline-summary">{sceneFiles.length} / 10 scenes · drag order with the arrows</div>
            {scenePreviewUrl && <div className="video-preview"><video src={scenePreviewUrl} controls playsInline preload="metadata"/><small>Selected scene preview · local browser playback</small></div>}
            <div className="scene-list">
              {sceneFiles.map((scene, index) => <article className={"scene-card " + (activeScene === index ? "selected" : "")} key={scene.id} onClick={() => setActiveScene(index)}>
                <div className="scene-heading">
                  <button type="button" className="scene-select"><b>{String(index + 1).padStart(2, '0')} · {scene.file.name}</b></button>
                  <div className="scene-actions">
                    <button type="button" title="Move scene up" disabled={index === 0} onClick={event => {event.stopPropagation();moveScene(index,-1);}}>↑</button>
                    <button type="button" title="Move scene down" disabled={index === sceneFiles.length - 1} onClick={event => {event.stopPropagation();moveScene(index,1);}}>↓</button>
                    <button type="button" className="danger" onClick={event => {event.stopPropagation();removeScene(scene.id);}}> <Trash2 size={14}/> Remove</button>
                  </div>
                </div>
                <div className="row scene-trim">
                  <label>START (SECONDS)<input type="number" min="0" max="14400" step="0.1" value={scene.trimStart} onChange={event => updateScene(scene.id,{trimStart:event.target.value})}/></label>
                  <label>END (0 = FULL CLIP)<input type="number" min="0" max="14400" step="0.1" value={scene.trimEnd} onChange={event => updateScene(scene.id,{trimEnd:event.target.value})}/></label>
                </div>
                {Number(scene.trimEnd) > 0 && Number(scene.trimEnd) <= Number(scene.trimStart) && <small className="validation-error">End must be greater than start.</small>}
              </article>)}
            </div>
            <form onSubmit={renderTimeline}>
              <div className="row">
                <label>EXPORT FORMAT<select value={preset} onChange={event => setPreset(event.target.value)}><option value="reel">Reels / Shorts · 9:16</option><option value="landscape">YouTube · 16:9</option><option value="square">Square · 1:1</option></select></label>
                <label>FRAMING<select value={framing} onChange={event => setFraming(event.target.value)}><option value="fill">Fill · crop edges</option><option value="fit">Fit · black bars</option></select></label>
              </div>
              <label>AUDIO<select value={audioMode} onChange={event => setAudioMode(event.target.value)}><option value="keep">Keep scene audio</option><option value="mute">Mute all audio</option><option value="music">Replace with background music</option></select></label>
              {audioMode === 'music' && <>
                <label className="drop"><Upload/><b>{musicFile?.name || 'Choose background music'}</b><small>MP3 / WAV / M4A / AAC / OGG · 30 MB max</small><input type="file" accept=".mp3,.wav,.m4a,.aac,.ogg,audio/*" onChange={event => setMusicFile(event.target.files[0])}/></label>
                <label>MUSIC VOLUME · {Math.round(musicVolume * 100)}%<input type="range" min="0" max="1.5" step="0.05" value={musicVolume} onChange={event => setMusicVolume(Number(event.target.value))}/></label>
              </>}
              <label>PROJECT TITLE (OPTIONAL)<input value={overlayText} maxLength={160} onChange={event => setOverlayText(event.target.value)} placeholder="Add a title to the finished video"/></label>
              <label>TITLE POSITION<select value={overlayPosition} onChange={event => setOverlayPosition(event.target.value)}><option value="top">Top</option><option value="center">Center</option><option value="bottom">Bottom</option></select></label>
              <label className="drop"><Upload/><b>{subtitleFile?.name || 'Optional SRT subtitles'}</b><small>UTF-8 · Hindi and English · 2 MB max</small><input type="file" accept=".srt" onChange={event => setSubtitleFile(event.target.files[0])}/></label>
              <button className="primary" disabled={projectBusy || sceneFiles.some(scene => Number(scene.trimEnd) > 0 && Number(scene.trimEnd) <= Number(scene.trimStart)) || (audioMode === 'music' && !musicFile)}>Render {sceneFiles.length} scenes as one video</button>
            </form>
          </>}
        </section>}
        {tab === 'voice' && <section>
          <span>VOICE LIBRARY</span><h1>Your voice profiles.</h1>
          <p>Upload a clean WAV reference for Chatterbox voice conditioning.</p>
          <form onSubmit={addVoice}>
            <label>PROFILE NAME<input value={name} onChange={event => setName(event.target.value)} required/></label>
            <label className="drop"><Mic2/><b>{wav?.name || 'Choose reference WAV'}</b><input type="file" accept=".wav" onChange={event => setWav(event.target.files[0])}/></label>
            <button className="primary" disabled={!wav || !name}>Save voice reference</button>
          </form>
          <h3>Saved voice references</h3>
          {voices.length === 0 ? <p>No saved voice references.</p> : voices.map(item =>
            <article key={item.voice_id} className="media-row">
              <b>🎙 {item.voice_id}</b>
              <button className="danger" onClick={() => deleteMedia('/api/voices/' + encodeURIComponent(item.voice_id), 'Voice profile ' + item.voice_id)}>
                <Trash2 size={15}/> Delete
              </button>
            </article>
          )}
        </section>}
        {tab === 'projects' && <section>
          <span>JOB QUEUE</span>
          {jobs.length === 0 ? <p>No jobs yet.</p> : jobs.map(job =>
            <article key={job.id}>
              <b>{job.kind.replaceAll('_', ' ')}</b><small>{job.stage} · {job.status}</small>
              <progress value={job.progress} max="100"/>
              {job.status === 'completed' && <a href={'/api/jobs/' + job.id + '/output'}><Download size={16}/> Download</a>}
              {job.error && <p>{job.error}</p>}
              <button className="danger" disabled={job.status === 'running'}
                onClick={() => deleteMedia('/api/jobs/' + job.id, 'job and its media')}>
                <Trash2 size={15}/> {job.status === 'running' ? 'Running — cannot delete yet' : 'Delete job and media'}
              </button>
            </article>
          )}
        </section>}
        <footer>AZURE SERVER EXECUTION · ONE NEURAL JOB AT A TIME</footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App/>);
