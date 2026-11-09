import os
import sqlite3
import json
import uuid
import time
import hmac
import hashlib
import secrets
import subprocess
import threading
import importlib.util
import shutil
import re
import urllib.request
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(os.getenv("STUDIO_ROOT", "/srv/ai-video-studio"))
DB = ROOT / "studio.sqlite3"
UP = ROOT / "uploads"
OUT = ROOT / "outputs"
VOICES = ROOT / "voices"
PROJECTS = ROOT / "projects"
LLM_LOGS = ROOT / "logs"
LLM_CONFIG = ROOT / "llm_config.json"
LLM_PULL_STATE = ROOT / "llm_pull.json"
OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_INFERENCE_LOCK = threading.Lock()
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
MAX_VOICE = int(os.getenv("MAX_VOICE_MB", "30")) * 1024 * 1024
ENABLE_TTS = os.getenv("ENABLE_TTS", "0") == "1"
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"
SESSION_SECONDS = 12 * 60 * 60
LOGIN_ATTEMPTS = {}
LOGIN_WINDOW = 15 * 60
MAX_LOGIN_ATTEMPTS = 8

for path in (ROOT, UP, OUT, VOICES, PROJECTS, LLM_LOGS):
    path.mkdir(parents=True, exist_ok=True)


@contextmanager
def db():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def init():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, kind TEXT, status TEXT, stage TEXT,
                progress INTEGER, payload TEXT, input_path TEXT,
                output_path TEXT, error TEXT, created_at TEXT, updated_at TEXT
            )"""
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS saved_projects(
            id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
            data TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")


def password_matches(password):
    try:
        salt, expected = ADMIN_PASSWORD_HASH.split("$", 1)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), 310000
        ).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def make_session():
    expires = str(int(time.time()) + SESSION_SECONDS)
    signature = hmac.new(
        SESSION_SECRET.encode(), expires.encode(), hashlib.sha256
    ).hexdigest()
    return expires + "." + signature


def valid_session(token):
    try:
        expires, signature = token.split(".", 1)
        if int(expires) < int(time.time()):
            return False
        expected = hmac.new(
            SESSION_SECRET.encode(), expires.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)
    except (ValueError, TypeError):
        return False


def valid_model_name(model):
    return bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,100}(?::[A-Za-z0-9._-]{1,64})?", model)
        and ".." not in model and "//" not in model and not model.endswith("/")
    )


def ollama_call(path, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        OLLAMA_API + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "Local Ollama runtime is unavailable or did not respond.") from exc


def stop_other_ollama_models(selected=""):
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=8, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        raise HTTPException(503, "Could not inspect local model memory usage.") from exc
    for line in result.stdout.splitlines()[1:]:
        name = line.split()[0] if line.split() else ""
        if name and name != selected:
            stopped = subprocess.run(["ollama", "stop", name], capture_output=True, text=True, timeout=20)
            if stopped.returncode:
                raise HTTPException(503, f"Could not unload inactive model {name}.")


def ollama_chat(model, payload, timeout=180):
    with OLLAMA_INFERENCE_LOCK:
        stop_other_ollama_models(model)
        return ollama_call("/api/chat", payload, timeout=timeout)


def installed_models():
    try:
        return ollama_call("/api/tags", timeout=5).get("models", [])
    except HTTPException:
        return []


def active_model():
    try:
        return json.loads(LLM_CONFIG.read_text()).get("active_model", "")
    except (OSError, ValueError, AttributeError):
        return ""


def set_active_model(model):
    LLM_CONFIG.write_text(json.dumps({"active_model": model}, indent=2))


def pull_state():
    try:
        state = json.loads(LLM_PULL_STATE.read_text())
        pid = int(state.get("pid", 0))
        running = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                stat = Path(f"/proc/{pid}/stat").read_text().split()
                running = len(stat) > 2 and stat[2] != "Z"
            except (OSError, ValueError):
                pass
        state["running"] = running
        state["log"] = Path(state.get("log_path", "")).read_text(errors="replace")[-2500:] if state.get("log_path") and Path(state["log_path"]).is_file() else ""
        state["status"] = "downloading" if running else (
            "downloaded" if "success" in state.get("log", "").lower() else "failed"
        )
        return state
    except (OSError, ValueError, TypeError):
        return None


def tts_runtime_ready():
    if not ENABLE_TTS:
        return False
    try:
        return all(importlib.util.find_spec(name) for name in ("torch", "torchaudio", "chatterbox"))
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def getjob(job_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["payload"] = json.loads(result["payload"])
    return result


def enqueue(kind, payload, path=None):
    job_id = str(uuid.uuid4())
    now = stamp()
    with db() as conn:
        conn.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, kind, "queued", "Queued", 0, json.dumps(payload), path, None, None, now, now),
        )
    return getjob(job_id)


def update(job_id, **values):
    values["updated_at"] = stamp()
    assignments = ",".join(f"{key}=?" for key in values)
    with db() as conn:
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id=?",
            (*values.values(), job_id),
        )


def cleanup_upload(input_path):
    if not input_path:
        return
    source = Path(input_path).resolve()
    upload_root = UP.resolve()
    if upload_root not in source.parents:
        return
    shutil.rmtree(source.parent, ignore_errors=True)


def cleanup_failed_output(job_id):
    folder = (OUT / job_id).resolve()
    output_root = OUT.resolve()
    if output_root not in folder.parents:
        return
    shutil.rmtree(folder, ignore_errors=True)


app = FastAPI(title="AI Video Studio")


@app.middleware("http")
async def require_login(request: Request, call_next):
    if request.url.path in {"/api/auth/login", "/api/auth/session", "/api/health"}:
        return await call_next(request)
    token = request.cookies.get("studio_session", "")
    if not SESSION_SECRET or not ADMIN_PASSWORD_HASH or not valid_session(token):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class TTS(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    language: str = "hi"
    voice_id: str = ""


class ModelNameRequest(BaseModel):
    model: str = Field(min_length=1, max_length=128)


class ModelTestRequest(BaseModel):
    model: str = Field(min_length=1, max_length=128)


class StoryboardRequest(BaseModel):
    script: str = Field(min_length=20, max_length=12000)
    language: str = Field(default="en", pattern="^(hi|en)$")
    max_scenes: int = Field(default=4, ge=2, le=10)


@app.on_event("startup")
def startup():
    if not SESSION_SECRET or not ADMIN_PASSWORD_HASH:
        raise RuntimeError("Authentication secrets are not configured.")
    init()


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "execution": "Azure server",
        "queue": "SQLite",
        "worker": "separate process",
        "tts_ready": tts_runtime_ready(),
        "video_transform_ready": True,
        "authentication": True,
    }


@app.get("/api/llm/health")
def llm_health():
    runtime = ollama_runtime_ready()
    selected = active_model()
    models = installed_models() if runtime else []
    installed = any(item.get("name") == selected for item in models) if selected else False
    pull = pull_state()
    downloading = bool(pull and pull.get("running"))
    ready = runtime and installed and not downloading
    reason = (
        "Model download in progress" if downloading else
        "Ollama runtime unavailable" if not runtime else
        "No active model selected" if not selected else
        "Active model is not downloaded" if not installed else
        "Ready"
    )
    return {"ready": ready, "runtime_ready": runtime, "active_model": selected if installed else "",
            "status": "downloading" if downloading else ("ready" if ready else "not_ready"),
            "reason": reason}


@app.get("/api/llm")
def llm_status():
    models = installed_models()
    names = [item.get("name", "") for item in models]
    selected = active_model()
    if selected not in names:
        selected = ""
    return {
        "runtime_ready": bool(shutil.which("ollama")) and bool(models or ollama_runtime_ready()),
        "active_model": selected,
        "models": models,
        "pull": pull_state(),
        "catalog": [
            {"name": "Qwen3.5 4B · Q4_K_M", "model": "qwen3.5:4b-q4_K_M", "size": "about 3.4 GB", "recommended": True},
            {"name": "Qwen3.5 2B · Q4_K_M", "model": "qwen3.5:2b-q4_K_M", "size": "about 1.9 GB", "recommended": False},
        ],
    }


def ollama_runtime_ready():
    try:
        ollama_call("/api/version", timeout=2)
        return True
    except HTTPException:
        return False


@app.post("/api/llm/pull")
def llm_pull(payload: ModelNameRequest):
    model = payload.model.strip()
    if not valid_model_name(model):
        raise HTTPException(400, "Invalid model name. Use an Ollama library model and tag.")
    state = pull_state()
    if state and state.get("running"):
        raise HTTPException(409, f"Another model download is already running: {state.get('model')}")
    if not shutil.which("ollama") or not ollama_runtime_ready():
        raise HTTPException(503, "Local Ollama runtime is not installed or running.")
    if model in [item.get("name") for item in installed_models()]:
        raise HTTPException(409, "This model is already downloaded.")
    log_path = LLM_LOGS / f"ollama-pull-{int(time.time())}.log"
    log_file = log_path.open("w")
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model], stdout=log_file, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    finally:
        log_file.close()
    LLM_PULL_STATE.write_text(json.dumps({
        "pid": process.pid, "model": model, "log_path": str(log_path),
        "started_at": stamp(), "return_code": None,
    }))
    return {"started": True, "model": model, "pid": process.pid}


@app.post("/api/llm/active")
def llm_set_active(payload: ModelNameRequest):
    model = payload.model.strip()
    if model not in [item.get("name") for item in installed_models()]:
        raise HTTPException(404, "Download this model before activating it.")
    with OLLAMA_INFERENCE_LOCK:
        stop_other_ollama_models()
        set_active_model(model)
    return {"active_model": model}


@app.delete("/api/llm/models")
def llm_delete_model(payload: ModelNameRequest):
    model = payload.model.strip()
    if not valid_model_name(model):
        raise HTTPException(400, "Invalid model name.")
    if model not in [item.get("name") for item in installed_models()]:
        raise HTTPException(404, "Model not found.")
    if active_model() == model:
        set_active_model("")
    try:
        with OLLAMA_INFERENCE_LOCK:
            stop_other_ollama_models()
            result = subprocess.run(
                ["ollama", "rm", model], capture_output=True, text=True, timeout=180, check=False
            )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "Model removal timed out.") from exc
    if result.returncode:
        raise HTTPException(502, result.stderr[-500:] or "Model removal failed.")
    return {"deleted": model}


@app.post("/api/llm/test")
def llm_test(payload: ModelTestRequest):
    model = payload.model.strip()
    if model not in [item.get("name") for item in installed_models()]:
        raise HTTPException(404, "Download this model before testing it.")
    started = time.time()
    result = ollama_chat(model, {
        "model": model,
        "messages": [{"role": "user", "content": "Write exactly one natural Hindi sentence and one natural English sentence about sunrise. Do not explain."}],
        "stream": False, "think": False,
        "keep_alive": "2m",
        "options": {"num_ctx": 2048, "num_predict": 80, "num_thread": 2, "temperature": 0.2},
    }, timeout=180)
    return {
        "model": model,
        "response": result.get("message", {}).get("content", ""),
        "seconds": round(time.time() - started, 2),
        "tokens_per_second": result.get("eval_count", 0) / max(result.get("eval_duration", 1) / 1e9, 0.001),
    }


@app.get("/api/auth/session")
def auth_session(request: Request):
    if valid_session(request.cookies.get("studio_session", "")):
        return {"authenticated": True}
    return JSONResponse({"authenticated": False}, status_code=401)


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request):
    now = time.time()
    client = request.client.host if request.client else "unknown"
    attempts, started = LOGIN_ATTEMPTS.get(client, (0, now))
    if now - started > LOGIN_WINDOW:
        attempts, started = 0, now
    if attempts >= MAX_LOGIN_ATTEMPTS:
        raise HTTPException(429, "Too many login attempts. Try again in 15 minutes.")
    if not password_matches(payload.password):
        LOGIN_ATTEMPTS[client] = (attempts + 1, started)
        raise HTTPException(401, "Incorrect password.")
    LOGIN_ATTEMPTS.pop(client, None)
    response = JSONResponse({"authenticated": True})
    response.set_cookie(
        "studio_session",
        make_session(),
        max_age=SESSION_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="strict",
        path="/api",
    )
    return response


@app.post("/api/auth/logout")
def logout():
    response = JSONResponse({"authenticated": False})
    response.delete_cookie("studio_session", path="/api", httponly=True, samesite="strict")
    return response


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    # Serialize against the worker's queued -> running claim. A running job
    # cannot be removed while FFmpeg or TTS may still be writing its output.
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found")
        if row["status"] == "running":
            raise HTTPException(409, "This job is running. Delete it after it finishes.")
        job = dict(row)
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    cleanup_upload(job.get("input_path"))
    cleanup_failed_output(job_id)
    output_path = job.get("output_path")
    if output_path:
        output_file = Path(output_path).resolve()
        if OUT.resolve() in output_file.parents:
            output_file.unlink(missing_ok=True)
    return {"deleted": True, "job_id": job_id}


@app.get("/api/jobs")
def jobs():
    with db() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        result.append(item)
    return result


@app.get("/api/saved-projects")
def list_saved_projects():
    with db() as conn:
        rows = conn.execute(
            "SELECT id,name,kind,created_at,updated_at FROM saved_projects ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/saved-projects")
async def save_saved_project(
    name: str = Form(..., min_length=1, max_length=100),
    kind: str = Form(...),
    project_data: str = Form(...),
    assets: list[UploadFile] = File(default=[]),
):
    if kind not in {"storyboard", "timeline"}:
        raise HTTPException(400, "Unsupported saved project type")
    try:
        data = json.loads(project_data)
    except (TypeError, ValueError):
        raise HTTPException(400, "Invalid project data")
    scenes = data.get("scenes") if isinstance(data, dict) else None
    minimum = 2 if kind == "storyboard" else 1
    if not isinstance(scenes, list) or not minimum <= len(scenes) <= 10:
        raise HTTPException(400, "A saved project must contain 1 to 10 scenes")
    if len(assets) > 12:
        raise HTTPException(400, "Too many project assets")
    max_total = (120 if kind == "storyboard" else 500) * 1024 * 1024
    project_id = str(uuid.uuid4())
    folder = PROJECTS / project_id
    folder.mkdir()
    saved_assets = []
    total_size = 0
    try:
        for index, upload in enumerate(assets):
            extension = Path(upload.filename or "").suffix.lower()
            allowed = {".jpg", ".jpeg", ".png", ".webp", ".mp3", ".wav", ".m4a", ".aac", ".ogg"}
            if kind == "timeline":
                allowed |= {".mp4", ".mov", ".mkv", ".webm", ".srt"}
            if extension not in allowed:
                raise HTTPException(415, "Unsupported saved project asset")
            target = folder / f"asset_{index + 1:02d}{extension}"
            with target.open("xb") as output_file:
                while chunk := await upload.read(256 * 1024):
                    total_size += len(chunk)
                    if total_size > max_total:
                        raise HTTPException(413, f"Saved project assets exceed {max_total // (1024 * 1024)} MB")
                    output_file.write(chunk)
            saved_assets.append({"filename": target.name, "original_name": Path(upload.filename or target.name).name})
            await upload.close()
        data["assets"] = saved_assets
        now = stamp()
        with db() as conn:
            conn.execute(
                "INSERT INTO saved_projects(id,name,kind,data,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (project_id, name.strip(), kind, json.dumps(data), now, now),
            )
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        for upload in assets:
            await upload.close()
        raise
    return {"id": project_id, "name": name.strip(), "kind": kind, "created_at": now}


@app.get("/api/saved-projects/{project_id}")
def get_saved_project(project_id: str):
    with db() as conn:
        row = conn.execute("SELECT * FROM saved_projects WHERE id=?", (project_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Saved project not found")
    data = json.loads(row["data"])
    for index, asset in enumerate(data.get("assets", [])):
        asset["url"] = f"/api/saved-projects/{project_id}/assets/{asset['filename']}"
    return {"id": row["id"], "name": row["name"], "kind": row["kind"], "data": data}


@app.get("/api/saved-projects/{project_id}/assets/{filename}")
def get_saved_project_asset(project_id: str, filename: str):
    if not re.fullmatch(r"asset_\d{2}\.[a-z0-9]+", filename):
        raise HTTPException(400, "Invalid asset name")
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM saved_projects WHERE id=?", (project_id,)).fetchone()
    path = (PROJECTS / project_id / filename).resolve()
    if not exists or PROJECTS.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Project asset not found")
    return FileResponse(path, filename=filename)


@app.delete("/api/saved-projects/{project_id}")
def delete_saved_project(project_id: str):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT id FROM saved_projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Saved project not found")
        conn.execute("DELETE FROM saved_projects WHERE id=?", (project_id,))
    shutil.rmtree((PROJECTS / project_id).resolve(), ignore_errors=True)
    return {"deleted": True, "project_id": project_id}


@app.post("/api/storyboard")
def create_storyboard(request: StoryboardRequest):
    script = " ".join(request.script.split())
    paragraphs = [script] if script else []
    sentences = []
    for paragraph in paragraphs:
        parts = re.split(r"(?<=[.!?।])\s+", paragraph)
        sentences.extend(part.strip() for part in parts if part.strip())
    if len(sentences) < 2:
        raise HTTPException(400, "Add at least two sentences or separate scene ideas with blank lines.")

    count = min(request.max_scenes, len(sentences))
    groups = [[] for _ in range(count)]
    for index, sentence in enumerate(sentences):
        groups[min(count - 1, index * count // len(sentences))].append(sentence)

    def shot_for(text):
        value = text.lower()
        if any(word in value for word in ("यात्रा", "सफर", "travel", "road", "journey", "चल", "दौड़")):
            return "Wide moving shot showing the journey or movement"
        if any(word in value for word in ("चेहरा", "आँख", "भावना", "डर", "खुशी", "face", "emotion", "fear", "smile")):
            return "Close-up focused on the character's expression"
        if any(word in value for word in ("शहर", "गाँव", "जंगल", "समुद्र", "पहाड़", "city", "village", "forest", "ocean", "mountain")):
            return "Establishing wide shot of the location"
        if any(word in value for word in ("समय", "सुबह", "रात", "दिन", "आसमान", "sunrise", "night", "morning", "sky")):
            return "Atmospheric cinematic establishing shot"
        return "Medium cinematic shot illustrating the main action"

    # The source narration is partitioned deterministically. The LLM is only
    # responsible for creative metadata, never rewriting or shortening the script.
    scenes = []
    for index, group in enumerate(groups, 1):
        narration = " ".join(group)
        scenes.append({
            "scene_number": index,
            "title": f"Scene {index}",
            "narration": narration,
            "visual_direction": shot_for(narration),
            "image_prompt": f"Cinematic realistic visual for this scene: {narration}. No text, no captions, coherent visual style.",
            "shot_type": "cinematic",
        })

    selected_model = active_model()
    if selected_model:
        scene_briefs = [
            {"scene_number": index, "story": " ".join(group)}
            for index, group in enumerate(groups, 1)
        ]
        output_schema = {
            "type": "object",
            "properties": {
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "visual_direction": {"type": "string"},
                            "image_prompt": {"type": "string"},
                            "shot_type": {"type": "string"},
                        },
                        "required": ["title", "visual_direction", "image_prompt", "shot_type"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scenes"],
            "additionalProperties": False,
        }
        metadata_language = "Hindi" if request.language == "hi" else "English"
        prompt = (
            "Create concise cinematic metadata for these pre-split storyboard scenes. Return exactly "
            "one scene object per input, in the same order. Never return narration; source narration is "
            "preserved by the application. Do not invent facts. Keep character appearance, clothing, props "
            "and locations consistent across scenes. Titles and visual_direction MUST be in "
            f"{metadata_language}. image_prompt MUST be in English for image-generator compatibility. "
            "shot_type must be a short English camera-shot label. Titles <=5 words, visual_direction "
            "<=12 words, image_prompt <=20 words. Use specific, filmable descriptions; no text inside images.\n"
            f"SCENE INPUTS:\n{json.dumps(scene_briefs, ensure_ascii=False)}"
        )
        result = ollama_chat(selected_model, {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "format": output_schema, "stream": False, "think": False, "keep_alive": "2m",
            "options": {
                "num_ctx": 3072, "num_predict": min(2000, max(700, count * 200)),
                "num_thread": 2, "temperature": 0.15,
            },
        }, timeout=180)
        try:
            generated = json.loads(result.get("message", {}).get("content", "{}"))
            generated_scenes = generated.get("scenes", [])
            if not isinstance(generated_scenes, list) or len(generated_scenes) != count:
                raise ValueError("Invalid scene count")
            clean_scenes = []
            required_fields = ("title", "visual_direction", "image_prompt", "shot_type")
            for index, (item, base_scene) in enumerate(zip(generated_scenes, scenes), 1):
                if not isinstance(item, dict) or any(
                    not isinstance(item.get(field), str) or not item[field].strip()
                    for field in required_fields
                ):
                    raise ValueError("A scene is missing required storyboard metadata")
                clean_scenes.append({
                    "scene_number": index,
                    "title": item["title"].strip()[:120],
                    "narration": base_scene["narration"],
                    "visual_direction": item["visual_direction"].strip()[:1000],
                    "image_prompt": item["image_prompt"].strip()[:2000],
                    "shot_type": item["shot_type"].strip()[:80],
                })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(502, "The selected local model returned invalid scene metadata. Try another model or the rule-based fallback.") from exc
        return {
            "language": request.language, "scene_count": len(clean_scenes),
            "total_words": len(script.split()), "scenes": clean_scenes,
            "notice": f"Scene metadata generated locally using {selected_model}; original narration preserved exactly."
        }
    return {
        "language": request.language,
        "scene_count": len(scenes),
        "total_words": len(script.split()),
        "scenes": scenes,
        "notice": "Rule-based storyboard draft. Select and activate a downloaded local model to enable AI scene metadata."
    }


@app.post("/api/tts")
def tts(payload: TTS):
    if not tts_runtime_ready():
        raise HTTPException(
            503,
            "Narration is temporarily disabled while the Hindi/English CPU voice engine is being validated.",
        )
    if payload.language not in {"hi", "en", "fr", "es", "de"}:
        raise HTTPException(400, "Unsupported language")
    return enqueue("tts", payload.model_dump())


@app.post("/api/video")
async def video(
    file: UploadFile = File(...),
    preset: str = Form("reel"),
    framing: str = Form("fill"),
    audio: str = Form("keep"),
    overlay_text: str = Form("", max_length=160),
    overlay_position: str = Form("bottom"),
    subtitle_file: UploadFile | None = File(None),
    music_file: UploadFile | None = File(None),
    trim_start: float = Form(0),
    trim_end: float = Form(0),
    music_volume: float = Form(0.35),
):
    if preset not in {"reel", "landscape", "square"}:
        raise HTTPException(400, "Invalid preset")
    if framing not in {"fill", "fit"}:
        raise HTTPException(400, "Invalid framing mode")
    if not 0 <= trim_start <= 14400 or not 0 <= trim_end <= 14400:
        raise HTTPException(400, "Trim values must be between 0 and 4 hours")
    if trim_end and trim_end <= trim_start:
        raise HTTPException(400, "Trim end must be greater than trim start")
    if not 0 <= music_volume <= 1.5:
        raise HTTPException(400, "Music volume must be between 0 and 1.5")
    if audio not in {"keep", "mute", "music"}:
        raise HTTPException(400, "Invalid audio mode")
    if audio == "music" and not music_file:
        raise HTTPException(400, "Choose a background music file")
    if music_file and Path(music_file.filename or "").suffix.lower() not in {
        ".mp3", ".wav", ".m4a", ".aac", ".ogg"
    }:
        raise HTTPException(415, "Unsupported background music format")
    if overlay_position not in {"top", "center", "bottom"}:
        raise HTTPException(400, "Invalid overlay position")
    overlay_text = overlay_text.strip()
    if subtitle_file and Path(subtitle_file.filename or "").suffix.lower() != ".srt":
        raise HTTPException(415, "Subtitles must be an .srt file")
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(415, "Unsupported video")
    folder = UP / str(uuid.uuid4())
    folder.mkdir()
    target = folder / ("source" + extension)
    size = 0
    try:
        with target.open("wb") as output_file:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Upload too large")
                output_file.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        folder.rmdir()
        raise
    finally:
        await file.close()
    subtitle_path = None
    if subtitle_file:
        subtitle_path = folder / "captions.srt"
        subtitle_size = 0
        try:
            with subtitle_path.open("xb") as subtitle_out:
                while chunk := await subtitle_file.read(64 * 1024):
                    subtitle_size += len(chunk)
                    if subtitle_size > 2 * 1024 * 1024:
                        raise HTTPException(413, "Subtitle file exceeds 2 MB")
                    subtitle_out.write(chunk)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await subtitle_file.close()
    music_path = None
    if music_file:
        music_path = folder / ("background" + Path(music_file.filename).suffix.lower())
        music_size = 0
        try:
            with music_path.open("xb") as music_out:
                while chunk := await music_file.read(64 * 1024):
                    music_size += len(chunk)
                    if music_size > 30 * 1024 * 1024:
                        raise HTTPException(413, "Background music exceeds 30 MB")
                    music_out.write(chunk)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await music_file.close()
    return enqueue(
        "video_transform",
        {
            "preset": preset,
            "framing": framing,
            "audio": audio,
            "overlay_text": overlay_text,
            "overlay_position": overlay_position,
            "subtitle_path": str(subtitle_path) if subtitle_path else "",
            "music_path": str(music_path) if music_path else "",
            "trim_start": trim_start,
            "trim_end": trim_end,
            "music_volume": music_volume,
        },
        str(target),
    )


@app.post("/api/project")
async def create_project(
    scenes: list[UploadFile] = File(...),
    scene_settings: str = Form("[]"),
    preset: str = Form("reel"),
    framing: str = Form("fill"),
    audio: str = Form("keep"),
    music_volume: float = Form(0.35),
    transition_duration: float = Form(0.5),
    overlay_text: str = Form("", max_length=160),
    overlay_position: str = Form("bottom"),
    subtitle_file: UploadFile | None = File(None),
    music_file: UploadFile | None = File(None),
):
    if not 1 <= len(scenes) <= 10:
        raise HTTPException(400, "A project must contain 1 to 10 scenes")
    if preset not in {"reel", "landscape", "square"} or framing not in {"fill", "fit"}:
        raise HTTPException(400, "Invalid export settings")
    if audio not in {"keep", "mute", "music"} or not 0 <= music_volume <= 1.5:
        raise HTTPException(400, "Invalid audio settings")
    if audio == "music" and not music_file:
        raise HTTPException(400, "Choose background music")
    if overlay_position not in {"top", "center", "bottom"}:
        raise HTTPException(400, "Invalid title position")
    try:
        settings = json.loads(scene_settings)
    except (TypeError, ValueError):
        raise HTTPException(400, "Invalid scene settings")
    if not isinstance(settings, list) or len(settings) != len(scenes):
        raise HTTPException(400, "Scene settings do not match uploaded clips")
    if transition_duration not in {0.5, 1.0}:
        raise HTTPException(400, "Transition duration must be 0.5 or 1 second")
    for index, item in enumerate(settings):
        if not isinstance(item, dict):
            raise HTTPException(400, "Invalid scene settings")
        start, end = float(item.get("trim_start", 0)), float(item.get("trim_end", 0))
        if not 0 <= start <= 14400 or not 0 <= end <= 14400 or (end and end <= start):
            raise HTTPException(400, "Invalid scene trim range")
        if item.get("transition", "cut") not in {"cut", "fade"}:
            raise HTTPException(400, "Invalid scene transition")

    folder = UP / str(uuid.uuid4())
    folder.mkdir()
    saved_scenes = []
    total_size = 0
    try:
        for index, upload in enumerate(scenes):
            extension = Path(upload.filename or "").suffix.lower()
            if extension not in {".mp4", ".mov", ".mkv", ".webm"}:
                raise HTTPException(415, "Unsupported scene video format")
            target = folder / f"scene_{index + 1:02d}{extension}"
            size = 0
            with target.open("xb") as output_file:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    total_size += len(chunk)
                    if size > MAX_UPLOAD or total_size > MAX_UPLOAD:
                        raise HTTPException(413, "Project uploads exceed the 500 MB total limit")
                    output_file.write(chunk)
            saved_scenes.append({"path": str(target), **settings[index]})
            await upload.close()

        subtitle_path = ""
        if subtitle_file:
            if Path(subtitle_file.filename or "").suffix.lower() != ".srt":
                raise HTTPException(415, "Subtitles must be an .srt file")
            subtitle_path = folder / "captions.srt"
            size = 0
            with subtitle_path.open("xb") as output_file:
                while chunk := await subtitle_file.read(64 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024 * 1024:
                        raise HTTPException(413, "Subtitle file exceeds 2 MB")
                    output_file.write(chunk)
            await subtitle_file.close()

        music_path = ""
        if music_file:
            if Path(music_file.filename or "").suffix.lower() not in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
                raise HTTPException(415, "Unsupported background music format")
            music_path = folder / ("background" + Path(music_file.filename).suffix.lower())
            size = 0
            with music_path.open("xb") as output_file:
                while chunk := await music_file.read(64 * 1024):
                    size += len(chunk)
                    if size > 30 * 1024 * 1024:
                        raise HTTPException(413, "Background music exceeds 30 MB")
                    output_file.write(chunk)
            await music_file.close()
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        for upload in scenes:
            await upload.close()
        if subtitle_file:
            await subtitle_file.close()
        if music_file:
            await music_file.close()
        raise

    payload = {
        "scenes": saved_scenes,
        "preset": preset,
        "framing": framing,
        "audio": audio,
        "transition_duration": transition_duration,
        "music_volume": music_volume,
        "music_path": str(music_path) if music_path else "",
        "subtitle_path": str(subtitle_path) if subtitle_path else "",
        "overlay_text": overlay_text.strip(),
        "overlay_position": overlay_position,
    }
    return enqueue("video_transform", payload, saved_scenes[0]["path"])


@app.post("/api/storyboard/render")
async def render_storyboard(
    images: list[UploadFile] = File(...),
    scene_data: str = Form("[]"),
    preset: str = Form("reel"),
    framing: str = Form("fill"),
    transition_duration: float = Form(0.5),
    music_file: UploadFile | None = File(None),
    music_volume: float = Form(0.35),
    overlay_text: str = Form("", max_length=160),
    overlay_position: str = Form("bottom"),
):
    if not 2 <= len(images) <= 10:
        raise HTTPException(400, "A visual storyboard needs 2 to 10 scene images")
    if preset not in {"reel", "landscape", "square"} or framing not in {"fill", "fit"}:
        raise HTTPException(400, "Invalid export settings")
    if transition_duration not in {0.5, 1.0} or not 0 <= music_volume <= 1.5:
        raise HTTPException(400, "Invalid transition or music settings")
    if overlay_position not in {"top", "center", "bottom"}:
        raise HTTPException(400, "Invalid title position")
    if music_file and Path(music_file.filename or "").suffix.lower() not in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
        raise HTTPException(415, "Unsupported background music format")
    try:
        settings = json.loads(scene_data)
    except (TypeError, ValueError):
        raise HTTPException(400, "Invalid storyboard scene data")
    if not isinstance(settings, list) or len(settings) != len(images):
        raise HTTPException(400, "Scene settings do not match the images")
    for item in settings:
        if not isinstance(item, dict):
            raise HTTPException(400, "Invalid scene settings")
        if not 2 <= float(item.get("duration", 4)) <= 20:
            raise HTTPException(400, "Each still image must stay on screen for 2 to 20 seconds")
        if item.get("transition", "cut") not in {"cut", "fade"}:
            raise HTTPException(400, "Invalid scene transition")

    folder = UP / str(uuid.uuid4())
    folder.mkdir()
    saved = []
    total_size = 0
    try:
        for index, upload in enumerate(images):
            extension = Path(upload.filename or "").suffix.lower()
            if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise HTTPException(415, "Scene images must be JPG, PNG or WebP")
            target = folder / f"scene_{index + 1:02d}{extension}"
            size = 0
            with target.open("xb") as output_file:
                while chunk := await upload.read(256 * 1024):
                    size += len(chunk)
                    total_size += len(chunk)
                    if size > 20 * 1024 * 1024 or total_size > 100 * 1024 * 1024:
                        raise HTTPException(413, "Storyboard images exceed the 20 MB per-image or 100 MB total limit")
                    output_file.write(chunk)
            header = target.read_bytes()[:16]
            valid_header = (
                header.startswith(b"\xff\xd8\xff")
                or header.startswith(b"\x89PNG\r\n\x1a\n")
                or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
            )
            if not valid_header:
                raise HTTPException(415, "An uploaded scene is not a valid JPG, PNG or WebP image")
            saved.append({"image_path": str(target), "duration": float(settings[index].get("duration", 4)),
                          "transition": settings[index].get("transition", "cut")})
            await upload.close()

        music_path = ""
        if music_file:
            music_path = folder / ("background" + Path(music_file.filename).suffix.lower())
            size = 0
            with music_path.open("xb") as output_file:
                while chunk := await music_file.read(64 * 1024):
                    size += len(chunk)
                    if size > 30 * 1024 * 1024:
                        raise HTTPException(413, "Background music exceeds 30 MB")
                    output_file.write(chunk)
            await music_file.close()
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        for upload in images:
            await upload.close()
        if music_file:
            await music_file.close()
        raise

    payload = {
        "scenes": saved, "preset": preset, "framing": framing,
        "audio": "music" if music_path else "mute",
        "transition_duration": transition_duration,
        "music_volume": music_volume, "music_path": str(music_path) if music_path else "",
        "subtitle_path": "", "overlay_text": overlay_text.strip(),
        "overlay_position": overlay_position,
    }
    return enqueue("video_transform", payload, saved[0]["image_path"])


@app.post("/api/voices")
async def voice(name: str = Form(...), file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() != ".wav":
        raise HTTPException(415, "WAV reference required")
    safe_name = "".join(char for char in name if char.isalnum() or char in "-_")[:48]
    if not safe_name:
        raise HTTPException(400, "Invalid name")
    destination = VOICES / (safe_name + ".wav")
    if destination.exists():
        raise HTTPException(409, "Profile already exists")

    size = 0
    try:
        with destination.open("xb") as output_file:
            header = await file.read(12)
            size += len(header)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise HTTPException(415, "Invalid WAV file")
            output_file.write(header)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VOICE:
                    raise HTTPException(413, "Voice reference exceeds 30 MB")
                output_file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"voice_id": safe_name}


@app.get("/api/voices")
def voices():
    return [{"voice_id": path.stem} for path in VOICES.glob("*.wav")]


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    if not voice_id or len(voice_id) > 48 or any(
        not (char.isalnum() or char in "-_") for char in voice_id
    ):
        raise HTTPException(400, "Invalid voice profile ID")
    path = (VOICES / (voice_id + ".wav")).resolve()
    if VOICES.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Voice profile not found")
    path.unlink()
    return {"deleted": True, "voice_id": voice_id}


@app.get("/api/jobs/{job_id}/output")
def output(job_id):
    job = getjob(job_id)
    if not job or job["status"] != "completed" or not job["output_path"]:
        raise HTTPException(404, "Output not ready")
    path = Path(job["output_path"]).resolve()
    if OUT.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, filename=path.name)


def work(job):
    job_id = job["id"]
    payload = job["payload"]
    folder = OUT / job_id
    # A recovered job may have a partial export from before the worker stopped.
    cleanup_failed_output(job_id)
    folder.mkdir(parents=True, exist_ok=True)

    if job["kind"] == "video_transform":
        source_for_render = job["input_path"]
        scene_items = payload.get("scenes", [])
        if scene_items:
            width, height = (
                (1080, 1920) if payload["preset"] == "reel"
                else (1920, 1080) if payload["preset"] == "landscape"
                else (1080, 1080)
            )
            segment_dir = folder / "segments"
            segment_dir.mkdir()
            segment_paths = []
            for index, scene in enumerate(scene_items):
                image_source = bool(scene.get("image_path"))
                source = Path(scene.get("image_path") if image_source else scene["path"]).resolve()
                if UP.resolve() not in source.parents or not source.is_file():
                    raise RuntimeError("A scene source is missing or outside the upload directory.")
                if image_source:
                    segment_duration = float(scene.get("duration", 4))
                    if not 2 <= segment_duration <= 20:
                        raise RuntimeError(f"Scene {index + 1} has an invalid still-image duration.")
                    trim_start = 0
                else:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)],
                        check=True, capture_output=True, text=True, timeout=30,
                    )
                    metadata = json.loads(probe.stdout)
                    video_stream = next((item for item in metadata["streams"] if item["codec_type"] == "video"), None)
                    if not video_stream:
                        raise RuntimeError(f"Scene {index + 1} has no video stream.")
                    duration = float(metadata["format"].get("duration", 0))
                    trim_start = float(scene.get("trim_start", 0))
                    trim_end = float(scene.get("trim_end", 0))
                    effective_end = trim_end if trim_end else duration
                    if duration <= 0 or trim_start >= duration or effective_end <= trim_start or effective_end > duration + 0.05:
                        raise RuntimeError(f"Scene {index + 1} has an invalid trim range.")
                    segment_duration = effective_end - trim_start
                if payload.get("framing", "fill") == "fit":
                    scene_filter = (
                        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
                    )
                else:
                    scene_filter = (
                        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                        f"crop={width}:{height}"
                    )
                scene_filter += ",fps=30,setsar=1,format=yuv420p"
                transition_seconds = float(payload.get("transition_duration", 0.5))
                fade_in = index > 0 and scene.get("transition", "cut") == "fade"
                fade_out = index < len(scene_items) - 1 and scene_items[index + 1].get("transition", "cut") == "fade"
                if fade_in:
                    scene_filter += f",fade=t=in:st=0:d={transition_seconds}"
                if fade_out:
                    scene_filter += f",fade=t=out:st={max(0, segment_duration-transition_seconds)}:d={transition_seconds}"
                segment = segment_dir / f"scene_{index + 1:02d}.mp4"
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
                if image_source:
                    command += ["-loop", "1", "-framerate", "30", "-i", str(source)]
                    image_filter = scene_filter + (
                        f",zoompan=z='min(zoom+0.0005,1.08)':"
                        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                        f"d=1:s={width}x{height}:fps=30"
                    )
                    command += ["-f", "lavfi", "-t", str(segment_duration), "-i",
                                "anullsrc=channel_layout=stereo:sample_rate=48000",
                                "-t", str(segment_duration), "-map", "0:v:0", "-map", "1:a:0",
                                "-vf", image_filter, "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "21", "-threads", "2", "-pix_fmt", "yuv420p",
                                "-c:a", "aac", "-ar", "48000", "-ac", "2",
                                "-movflags", "+faststart", str(segment)]
                    subprocess.run(command, check=True, timeout=7200)
                    segment_paths.append(segment)
                    continue
                if trim_start > 0:
                    command += ["-ss", str(trim_start)]
                command += ["-i", str(source)]
                has_audio = any(item["codec_type"] == "audio" for item in metadata["streams"])
                if not has_audio:
                    command += ["-f", "lavfi", "-t", str(segment_duration), "-i",
                                "anullsrc=channel_layout=stereo:sample_rate=48000"]
                audio_filter = []
                if has_audio and fade_in:
                    audio_filter += [f"afade=t=in:st=0:d={transition_seconds}"]
                if has_audio and fade_out:
                    audio_filter += [f"afade=t=out:st={max(0, segment_duration-transition_seconds)}:d={transition_seconds}"]
                command += ["-t", str(segment_duration), "-map", "0:v:0",
                            "-map", "0:a:0" if has_audio else "1:a:0",
                            "-vf", scene_filter]
                if audio_filter:
                    command += ["-af", ",".join(audio_filter)]
                command += ["-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "21", "-threads", "2", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-ar", "48000", "-ac", "2",
                            "-movflags", "+faststart", str(segment)]
                subprocess.run(command, check=True, timeout=7200)
                segment_paths.append(segment)

            concat_list = segment_dir / "concat.txt"
            concat_list.write_text(
                "".join("file '" + str(path).replace("'", "'\\''") + "\n" for path in segment_paths),
                encoding="utf-8",
            )
            joined = folder / "joined.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c", "copy", "-movflags", "+faststart", str(joined)],
                check=True, timeout=7200,
            )
            source_for_render = str(joined)

        width, height = (
            (1080, 1920) if payload["preset"] == "reel"
            else (1920, 1080) if payload["preset"] == "landscape"
            else (1080, 1080)
        )
        destination = folder / "video.mp4"
        if payload.get("framing", "fill") == "fit":
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
            )
        else:
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1"
            )
        overlay_text = payload.get("overlay_text", "").strip()
        if overlay_text:
            text_file = folder / "overlay.txt"
            text_file.write_text(overlay_text, encoding="utf-8")
            position = {
                "top": "h*0.08",
                "center": "(h-text_h)/2",
                "bottom": "h-text_h-h*0.08",
            }[payload.get("overlay_position", "bottom")]
            drawtext = (
                "drawtext=fontfile=/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
                f":textfile={text_file}:fontcolor=white:fontsize=48"
                ":borderw=2:bordercolor=black:box=1:boxcolor=black@0.55"
                f":boxborderw=16:x=(w-text_w)/2:y={position}"
            )
            video_filter += "," + drawtext
        subtitle_path = payload.get("subtitle_path", "")
        if subtitle_path:
            subtitle_file = Path(subtitle_path).resolve()
            if UP.resolve() not in subtitle_file.parents or not subtitle_file.is_file():
                raise RuntimeError("Subtitle file is missing or outside the upload directory.")
            video_filter += f",subtitles=filename={subtitle_file}:fontsdir=/usr/share/fonts/truetype/noto"
        music_path = payload.get("music_path", "")
        trim_start = 0 if scene_items else float(payload.get("trim_start", 0))
        trim_end = 0 if scene_items else float(payload.get("trim_end", 0))
        input_args = []
        if trim_start > 0:
            input_args += ["-ss", str(trim_start)]
        if music_path:
            music_file = Path(music_path).resolve()
            if UP.resolve() not in music_file.parents or not music_file.is_file():
                raise RuntimeError("Background music is missing or outside the upload directory.")
            input_args += ["-i", source_for_render, "-stream_loop", "-1", "-i", str(music_file)]
            map_args = ["-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:a", "aac",
                        "-af", f"volume={float(payload.get('music_volume', 0.35))}"]
        else:
            input_args += ["-i", source_for_render]
            map_args = ["-map", "0:v:0", "-map", "0:a?"]
            map_args += ["-an"] if payload.get("audio", "keep") == "mute" else ["-c:a", "aac"]
        duration_args = ["-t", str(trim_end - trim_start)] if trim_end > 0 else []
        update(job_id, stage="FFmpeg rendering", progress=20)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-y", *input_args, *map_args, *duration_args,
                "-vf", video_filter, "-filter_threads", "1",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                "-threads", "2", "-movflags", "+faststart",
                str(destination),
            ],
            check=True,
            timeout=7200,
        )
    elif job["kind"] == "tts":
        if not tts_runtime_ready():
            raise RuntimeError("Narration engine is not enabled or installed.")
        update(job_id, stage="Loading Chatterbox Multilingual V3", progress=10)
        import torch
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        model = ChatterboxMultilingualTTS.from_pretrained(device="cpu", t3_model="v3")
        reference = VOICES / (payload["voice_id"] + ".wav") if payload.get("voice_id") else None
        kwargs = {"language_id": payload["language"]}
        if reference:
            if not reference.is_file():
                raise RuntimeError("Voice profile missing")
            kwargs["audio_prompt_path"] = str(reference)
        update(job_id, stage="Generating speech", progress=35)
        with torch.inference_mode():
            audio = model.generate(payload["text"], **kwargs)
        destination = folder / "narration.wav"
        torchaudio.save(str(destination), audio.detach().cpu(), model.sr)
        del model
    else:
        raise RuntimeError("Unsupported job type")

    (folder / "overlay.txt").unlink(missing_ok=True)
    update(job_id, status="completed", stage="Completed", progress=100, output_path=str(destination))


def worker():
    init()
    with db() as conn:
        conn.execute(
            "UPDATE jobs SET status='queued',stage='Recovered after worker restart',updated_at=? WHERE status='running'",
            (stamp(),),
        )
    while True:
        with db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE jobs SET status='running',stage='Starting',updated_at=? WHERE id=?",
                    (stamp(), row["id"]),
                )
        if not row:
            time.sleep(2)
            continue
        job = getjob(row["id"])
        try:
            work(job)
        except Exception as error:
            cleanup_failed_output(job["id"])
            update(job["id"], status="failed", stage="Failed", error=str(error)[:2000])
        finally:
            # Keep the source only while queued/running so a crashed worker can recover it.
            cleanup_upload(job.get("input_path"))


if __name__ == "__main__":
    worker()
