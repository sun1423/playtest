import os
import sqlite3
import json
import uuid
import time
import subprocess
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(os.getenv("STUDIO_ROOT", "/srv/ai-video-studio"))
DB = ROOT / "studio.sqlite3"
UP = ROOT / "uploads"
OUT = ROOT / "outputs"
VOICES = ROOT / "voices"
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
MAX_VOICE = int(os.getenv("MAX_VOICE_MB", "30")) * 1024 * 1024
ENABLE_TTS = os.getenv("ENABLE_TTS", "0") == "1"

for path in (ROOT, UP, OUT, VOICES):
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


app = FastAPI(title="AI Video Studio")


class TTS(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    language: str = "hi"
    voice_id: str = ""


@app.on_event("startup")
def startup():
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
    }


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
async def video(file: UploadFile = File(...), preset: str = Form("reel")):
    if preset not in {"reel", "landscape", "square"}:
        raise HTTPException(400, "Invalid preset")
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
    return enqueue("video_transform", {"preset": preset}, str(target))


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
    folder.mkdir(parents=True, exist_ok=True)

    if job["kind"] == "video_transform":
        width, height = (
            (1080, 1920) if payload["preset"] == "reel"
            else (1920, 1080) if payload["preset"] == "landscape"
            else (1080, 1080)
        )
        destination = folder / "video.mp4"
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )
        update(job_id, stage="FFmpeg rendering", progress=20)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", job["input_path"],
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", video_filter, "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "21",
                "-c:a", "aac", "-movflags", "+faststart",
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
            update(job["id"], status="failed", stage="Failed", error=str(error)[:2000])


if __name__ == "__main__":
    worker()
