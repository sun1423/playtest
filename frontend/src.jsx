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
  const [preset, setPreset] = useState('reel');
  const [framing, setFraming] = useState('fill');
  const [audioMode, setAudioMode] = useState('keep');
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
            <div className="row">
              <label>EXPORT FORMAT<select value={preset} onChange={event => setPreset(event.target.value)}><option value="reel">Reels / Shorts · 9:16</option><option value="landscape">YouTube · 16:9</option><option value="square">Square · 1:1</option></select></label>
              <label>FRAMING<select value={framing} onChange={event => setFraming(event.target.value)}><option value="fill">Fill frame · crop edges</option><option value="fit">Fit whole video · black bars</option></select></label>
            </div>
            <label>AUDIO<select value={audioMode} onChange={event => setAudioMode(event.target.value)}><option value="keep">Keep original audio</option><option value="mute">Remove audio</option></select></label>
            <label>TEXT OVERLAY (OPTIONAL)
              <input value={overlayText} maxLength={160} onChange={event => setOverlayText(event.target.value)} placeholder="Add a title or short message (160 characters max)"/>
            </label>
            <label>TEXT POSITION<select value={overlayPosition} onChange={event => setOverlayPosition(event.target.value)}><option value="top">Top</option><option value="center">Center</option><option value="bottom">Bottom</option></select></label>
            <label className="drop"><Upload/><b>{subtitleFile?.name || 'Optional SRT subtitles'}</b><small>UTF-8 SRT · Hindi and English · 2 MB max</small><input type="file" accept=".srt,application/x-subrip" onChange={event => setSubtitleFile(event.target.files[0])}/></label>
            <button className="primary" disabled={!file || busy}>Queue video transform</button>
          </form>
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
