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
  const [storyScript, setStoryScript] = useState('');
  const [storyLanguage, setStoryLanguage] = useState('en');
  const [storySceneLimit, setStorySceneLimit] = useState('8');
  const [storyScenes, setStoryScenes] = useState([]);
  const [storyBusy, setStoryBusy] = useState(false);
  const [storyImageUrls, setStoryImageUrls] = useState({});
  const [storyRenderBusy, setStoryRenderBusy] = useState(false);
  const [voice, setVoice] = useState('');
  const [file, setFile] = useState(null);
  const [sceneFiles, setSceneFiles] = useState([]);
  const [activeScene, setActiveScene] = useState(0);
  const [scenePreviewUrl, setScenePreviewUrl] = useState('');
  const [projectBusy, setProjectBusy] = useState(false);
  const [savedProjects, setSavedProjects] = useState([]);
  const [savedProjectName, setSavedProjectName] = useState('');
  const [savingProject, setSavingProject] = useState(false);

  const [previewUrl, setPreviewUrl] = useState('');
  const [trimStart, setTrimStart] = useState('0');
  const [trimEnd, setTrimEnd] = useState('0');
  const [musicVolume, setMusicVolume] = useState(0.35);
  const [transitionDuration, setTransitionDuration] = useState('0.5');
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
      const [jobList, voiceList, savedList] = await Promise.all([
        api('/api/jobs'), api('/api/voices'), api('/api/saved-projects')
      ]);
      setJobs(jobList);
      setVoices(voiceList);
      setSavedProjects(savedList);
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
    const urls = {};
    storyScenes.forEach((scene, index) => {
      if (scene.imageFile) urls[index] = URL.createObjectURL(scene.imageFile);
    });
    setStoryImageUrls(urls);
    return () => Object.values(urls).forEach(URL.revokeObjectURL);
  }, [storyScenes]);
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
      scene.file ? scene : { id: `${Date.now()}-${index}-${scene.name}`, file: scene, trimStart: '0', trimEnd: '0', transition: 'cut' }
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
      trim_end: Number(scene.trimEnd || 0),
      transition: scene.transition || 'cut'
    }))));
    form.append('transition_duration', transitionDuration);
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

  async function generateStoryboard(event) {
    event.preventDefault();
    setStoryBusy(true);
    try {
      const result = await api('/api/storyboard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: storyScript, language: storyLanguage, max_scenes: Number(storySceneLimit) })
      });
      setStoryScenes(result.scenes.map(scene => ({ ...scene, duration: '4', transition: 'cut', imageFile: null })));
      setMsg(`Storyboard drafted: ${result.scene_count} scenes, ${result.total_words} words.`);
    } catch (error) {
      setMsg(error.message);
    } finally {
      setStoryBusy(false);
    }
  }

  function updateStoryboardScene(index, changes) {
    setStoryScenes(current => current.map((scene, i) => i === index ? { ...scene, ...changes } : scene));
  }

  async function renderStoryboardVideo(event) {
    event.preventDefault();
    if (storyScenes.length < 2 || storyScenes.some(scene => !scene.imageFile)) {
      setMsg('Add one image to every storyboard scene before rendering.');
      return;
    }
    const form = new FormData();
    storyScenes.forEach(scene => form.append('images', scene.imageFile));
    form.append('scene_data', JSON.stringify(storyScenes.map(scene => ({
      duration: Number(scene.duration || 4), transition: scene.transition || 'cut'
    }))));
    form.append('preset', preset);
    form.append('framing', framing);
    form.append('transition_duration', transitionDuration);
    form.append('music_volume', String(musicVolume));
    form.append('overlay_text', overlayText);
    form.append('overlay_position', overlayPosition);
    if (musicFile) form.append('music_file', musicFile);
    setStoryRenderBusy(true);
    try {
      await api('/api/storyboard/render', { method: 'POST', body: form });
      setMsg(`Storyboard video with ${storyScenes.length} still-image scenes queued on Azure.`);
      refresh();
    } catch (error) {
      setMsg(error.message);
    } finally {
      setStoryRenderBusy(false);
    }
  }


  async function saveStoryboardProject() {
    if (storyScenes.length < 2) return;
    const form = new FormData();
    const scenes = storyScenes.map(scene => {
      const item = { ...scene };
      delete item.imageFile;
      if (scene.imageFile) {
        item.image_asset_index = Number(form.getAll('assets').length);
        form.append('assets', scene.imageFile, scene.imageFile.name);
      } else {
        item.image_asset_index = null;
      }
      return item;
    });
    let musicAssetIndex = null;
    if (musicFile) {
      musicAssetIndex = form.getAll('assets').length;
      form.append('assets', musicFile, musicFile.name);
    }
    form.append('name', savedProjectName.trim() || storyScenes[0]?.title || 'Untitled storyboard');
    form.append('kind', 'storyboard');
    form.append('project_data', JSON.stringify({
      script: storyScript, language: storyLanguage, scene_limit: storySceneLimit,
      scenes, music_asset_index: musicAssetIndex,
      preset, framing, transitionDuration, musicVolume, overlayText, overlayPosition
    }));
    setSavingProject(true);
    try {
      await api('/api/saved-projects', { method: 'POST', body: form });
      setMsg('Storyboard saved. You can reopen it from Projects.');
      setSavedProjectName('');
      refresh();
    } catch (error) {
      setMsg(error.message);
    } finally {
      setSavingProject(false);
    }
  }

  async function openSavedProject(projectId) {
    try {
      const result = await api('/api/saved-projects/' + encodeURIComponent(projectId));
      if (result.kind !== 'storyboard') throw Error('Unsupported saved project type');
      const data = result.data;
      const assets = data.assets || [];
      const loadAsset = async index => {
        if (index == null || !assets[index]) return null;
        const response = await fetch(assets[index].url);
        if (!response.ok) throw Error('Could not load a saved media asset');
        const blob = await response.blob();
        return new File([blob], assets[index].original_name || assets[index].filename, { type: blob.type });
      };
      const scenes = await Promise.all((data.scenes || []).map(async scene => ({
        ...scene, imageFile: await loadAsset(scene.image_asset_index)
      })));
      setStoryScenes(scenes);
      setStoryScript(data.script || '');
      setStoryLanguage(data.language || 'en');
      setStorySceneLimit(String(data.scene_limit || scenes.length));
      setMusicFile(await loadAsset(data.music_asset_index));
      setPreset(data.preset || 'reel');
      setFraming(data.framing || 'fill');
      setTransitionDuration(String(data.transitionDuration || '0.5'));
      setMusicVolume(Number(data.musicVolume ?? 0.35));
      setOverlayText(data.overlayText || '');
      setOverlayPosition(data.overlayPosition || 'bottom');
      setTab('storyboard');
      setMsg('Saved storyboard reopened: ' + result.name);
    } catch (error) {
      setMsg(error.message);
    }
  }

  function renumberStoryboard(scenes) {
    return scenes.map((scene, index) => ({ ...scene, scene_number: index + 1 }));
  }

  function moveStoryboardScene(index, direction) {
    setStoryScenes(current => {
      const next = [...current];
      const target = index + direction;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      return renumberStoryboard(next);
    });
  }

  function removeStoryboardScene(index) {
    setStoryScenes(current => renumberStoryboard(current.filter((_, i) => i !== index)));
  }

  function duplicateStoryboardScene(index) {
    setStoryScenes(current => {
      if (current.length >= 10) {
        setMsg('A storyboard can contain at most 10 scenes.');
        return current;
      }
      const next = [...current];
      const copy = { ...current[index], title: current[index].title + ' (copy)' };
      next.splice(index + 1, 0, copy);
      return renumberStoryboard(next);
    });
  }

  function addStoryboardScene() {
    setStoryScenes(current => {
      if (current.length >= 10) {
        setMsg('A storyboard can contain at most 10 scenes.');
        return current;
      }
      return renumberStoryboard([...current, {
        scene_number: current.length + 1,
        title: 'New scene',
        narration: '',
        visual_direction: 'Medium cinematic shot illustrating the main action',
        image_prompt: 'Cinematic realistic visual. No text, no captions, coherent visual style.',
        shot_type: 'cinematic',
        duration: '4',
        transition: 'cut',
        imageFile: null
      }]);
    });
  }

  async function copyStoryboard() {
    const content = storyScenes.map(scene =>
      `SCENE ${scene.scene_number}: ${scene.title}\nNARRATION: ${scene.narration}\nVISUAL: ${scene.visual_direction}\nIMAGE PROMPT: ${scene.image_prompt}`
    ).join('\n\n');
    try {
      await navigator.clipboard.writeText(content);
      setMsg('Storyboard copied to clipboard.');
    } catch {
      setMsg('Clipboard access unavailable. Select and copy the storyboard text manually.');
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
    ['storyboard', 'Storyboard', Film],
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
        {tab === 'storyboard' && <section className="storyboard-workspace">
          <span>SCRIPT TO STORYBOARD</span><h1>Plan every scene.</h1>
          <p>Paste a Hindi or English script. FRAME / AI will split it into editable scenes with narration and suggested visual directions. This first version uses local rules, not a cloud AI model.</p>
          <form onSubmit={generateStoryboard}>
            <label>YOUR STORY / SCRIPT
              <textarea rows="9" maxLength="12000" value={storyScript} onChange={event => setStoryScript(event.target.value)} placeholder="Paste your story here. Use blank lines to indicate scene changes…" required/>
            </label>
            <div className="row">
              <label>LANGUAGE<select value={storyLanguage} onChange={event => setStoryLanguage(event.target.value)}><option value="en">English</option><option value="hi">Hindi</option></select></label>
              <label>MAX SCENES<select value={storySceneLimit} onChange={event => setStorySceneLimit(event.target.value)}>{[2,3,4,5,6,7,8,9,10].map(n => <option key={n} value={n}>{n} scenes</option>)}</select></label>
            </div>
            <button className="primary" disabled={storyBusy || storyScript.trim().length < 20}>{storyBusy ? 'Building storyboard…' : 'Create storyboard draft'}</button>
          </form>
          {storyScenes.length > 0 && <>
            <div className="storyboard-toolbar"><b>{storyScenes.length} / 10 editable scenes</b><div className="storyboard-toolbar-actions"><button type="button" disabled={storyScenes.length >= 10} onClick={addStoryboardScene}>+ Add scene</button><button type="button" onClick={copyStoryboard}>Copy storyboard</button></div></div>
            <div className="saved-project-form"><input aria-label="Saved project name" value={savedProjectName} onChange={event => setSavedProjectName(event.target.value)} placeholder="Project name (optional)" maxLength={100}/><button type="button" className="primary" disabled={savingProject} onClick={saveStoryboardProject}>{savingProject ? 'Saving…' : 'Save project'}</button></div>
            <div className="storyboard-scenes">
              {storyScenes.map((scene,index) => <article className="story-scene" key={scene.scene_number}>
                <div className="story-scene-head">
                  <b>SCENE {String(scene.scene_number).padStart(2,'0')}</b>
                  <small>{scene.narration.trim().split(/\s+/).filter(Boolean).length} narration words</small>
                </div>
                <div className="story-scene-actions">
                  <button type="button" disabled={index === 0} onClick={() => moveStoryboardScene(index,-1)} aria-label="Move scene up">↑ Move up</button>
                  <button type="button" disabled={index === storyScenes.length - 1} onClick={() => moveStoryboardScene(index,1)} aria-label="Move scene down">↓ Move down</button>
                  <button type="button" disabled={storyScenes.length >= 10} onClick={() => duplicateStoryboardScene(index)}>Duplicate</button>
                  <button type="button" className="danger" disabled={storyScenes.length <= 2} onClick={() => removeStoryboardScene(index)}> <Trash2 size={13}/> Remove</button>
                </div>
                <label>SCENE TITLE<input value={scene.title} onChange={event => updateStoryboardScene(index,{title:event.target.value})}/></label>
                <label>NARRATION<textarea rows="3" value={scene.narration} onChange={event => updateStoryboardScene(index,{narration:event.target.value})}/></label>
                <label>VISUAL DIRECTION<textarea rows="2" value={scene.visual_direction} onChange={event => updateStoryboardScene(index,{visual_direction:event.target.value})}/></label>
                <label>IMAGE PROMPT<textarea rows="3" value={scene.image_prompt} onChange={event => updateStoryboardScene(index,{image_prompt:event.target.value})}/></label>
                <label className="drop story-image-upload">
                  <Upload/><b>{scene.imageFile?.name || 'Attach scene image'}</b><small>JPG / PNG / WebP · max 20 MB</small>
                  <input type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" onChange={event => updateStoryboardScene(index,{imageFile:event.target.files?.[0] || null})}/>
                </label>
                {storyImageUrls[index] && <img className="story-image-preview" src={storyImageUrls[index]} alt={scene.title}/>}
                <div className="row">
                  <label>SCENE DURATION<select value={scene.duration} onChange={event => updateStoryboardScene(index,{duration:event.target.value})}>{[2,3,4,5,6,8,10,12,15,20].map(n=><option key={n} value={n}>{n} seconds</option>)}</select></label>
                  <label>TRANSITION<select value={scene.transition} onChange={event => updateStoryboardScene(index,{transition:event.target.value})}><option value="cut">Cut</option><option value="fade">Fade through black</option></select></label>
                </div>
              </article>)}
            </div>
            <form onSubmit={renderStoryboardVideo}>
              <div className="row">
                <label>EXPORT FORMAT<select value={preset} onChange={event => setPreset(event.target.value)}><option value="reel">Reels / Shorts · 9:16</option><option value="landscape">YouTube · 16:9</option><option value="square">Square · 1:1</option></select></label>
                <label>FRAMING<select value={framing} onChange={event => setFraming(event.target.value)}><option value="fill">Fill · crop edges</option><option value="fit">Fit · black bars</option></select></label>
              </div>
              <label>FADE DURATION<select value={transitionDuration} onChange={event => setTransitionDuration(event.target.value)}><option value="0.5">0.5 seconds</option><option value="1">1 second</option></select></label>
              <label>BACKGROUND MUSIC (OPTIONAL)<small> {musicFile?.name || 'No music selected; export will be silent'}</small><input type="file" accept=".mp3,.wav,.m4a,.aac,.ogg,audio/*" onChange={event => setMusicFile(event.target.files?.[0] || null)}/></label>
              {musicFile && <label>MUSIC VOLUME · {Math.round(musicVolume*100)}%<input type="range" min="0" max="1.5" step="0.05" value={musicVolume} onChange={event => setMusicVolume(Number(event.target.value))}/></label>}
              <label>PROJECT TITLE (OPTIONAL)<input value={overlayText} maxLength={160} onChange={event => setOverlayText(event.target.value)}/></label>
              <label>TITLE POSITION<select value={overlayPosition} onChange={event => setOverlayPosition(event.target.value)}><option value="top">Top</option><option value="center">Center</option><option value="bottom">Bottom</option></select></label>
              <button className="primary" disabled={storyRenderBusy || storyScenes.some(scene => !scene.imageFile)}>{storyRenderBusy ? 'Queueing storyboard video…' : 'Render storyboard video'}</button>
            </form>
            <p className="storyboard-note">Images are supplied by you. FRAME / AI adds gentle camera motion, scene transitions and assembles the video. AI image generation is not enabled yet.</p>
          </>}
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
                <label>TRANSITION BEFORE THIS SCENE<select value={scene.transition} onChange={event => updateScene(scene.id,{transition:event.target.value})}><option value="cut">Cut</option><option value="fade">Fade through black</option></select></label>
                {Number(scene.trimEnd) > 0 && Number(scene.trimEnd) <= Number(scene.trimStart) && <small className="validation-error">End must be greater than start.</small>}
              </article>)}
            </div>
            <form onSubmit={renderTimeline}>
              <label>FADE DURATION<select value={transitionDuration} onChange={event => setTransitionDuration(event.target.value)}><option value="0.5">0.5 seconds</option><option value="1">1 second</option></select></label>
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
          <span>SAVED PROJECTS</span>
          <h2>Reopen your work</h2>
          {savedProjects.length === 0 ? <p>No saved projects yet. Save a storyboard to keep its scenes and uploaded images.</p> : savedProjects.map(project => <article className="saved-project-row" key={project.id}>
            <div><b>{project.name}</b><small>{project.kind} · Updated {new Date(project.updated_at).toLocaleString()}</small></div>
            <div className="saved-project-actions"><button type="button" onClick={() => openSavedProject(project.id)}>Open</button><button type="button" className="danger" onClick={() => deleteMedia('/api/saved-projects/' + project.id, 'saved project ' + project.name)}><Trash2 size={14}/> Delete</button></div>
          </article>)}
          <h2>Render queue</h2>
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
