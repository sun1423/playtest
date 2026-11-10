#!/usr/bin/env python3
"""Remote GPU worker for AI Video Studio VACE + custom-voice jobs.

Run this on a temporary GPU session (for example, Kaggle) with the
VACE repository/model and Chatterbox Multilingual V3 installed.
Only the runner token is allowed to access the protected runner endpoints.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

API_BASE = os.getenv("STUDIO_API_URL", "").rstrip("/")
RUNNER_TOKEN = os.getenv("STUDIO_VACE_RUNNER_TOKEN", "")
VACE_ROOT = Path(os.getenv("VACE_ROOT", "/kaggle/working/VACE")).resolve()
VACE_CHECKPOINT_DIR = Path(
    os.getenv("VACE_CHECKPOINT_DIR", str(VACE_ROOT / "models" / "Wan2.1-VACE-1.3B"))
).resolve()
WORK_ROOT = Path(os.getenv("VACE_WORK_ROOT", "/kaggle/working/studio-vace-jobs"))
POLL_SECONDS = int(os.getenv("VACE_POLL_SECONDS", "15"))
CLIP_SECONDS = 5.0625
FRAME_COUNT = 81


def load_kaggle_secrets() -> None:
    global API_BASE, RUNNER_TOKEN
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        API_BASE = API_BASE or secrets.get_secret("STUDIO_API_URL").rstrip("/")
        RUNNER_TOKEN = RUNNER_TOKEN or secrets.get_secret("STUDIO_VACE_RUNNER_TOKEN")
    except Exception:
        pass


def session() -> requests.Session:
    client = requests.Session()
    client.headers.update({"Authorization": "Bearer " + RUNNER_TOKEN})
    return client


def api_url(path: str) -> str:
    return urljoin(API_BASE + "/", path.lstrip("/"))


def prepare_vace_runtime() -> None:
    """Use FP16 + PyTorch SDPA on free T4-class GPUs without BF16/FlashAttention."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Enable a Kaggle GPU accelerator.")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 7:
        raise RuntimeError(f"Unsupported GPU compute capability {capability}; use T4 or newer.")
    print(f"GPU: {torch.cuda.get_device_name(0)} · compute capability {capability}")

    if torch.cuda.is_bf16_supported():
        print("GPU supports BF16; using the official VACE/Wan precision defaults.")
        return

    if importlib.util.find_spec("flash_attn") or importlib.util.find_spec("flash_attn_interface"):
        raise RuntimeError("This GPU requires FP16 PyTorch SDPA. Remove FlashAttention packages and follow KAGGLE_VACE_SETUP.md.")

    config_path = VACE_ROOT / "vace/models/wan/configs/shared_config.py"
    config_text = config_path.read_text(encoding="utf-8")
    config_old = "wan_shared_cfg.param_dtype = torch.bfloat16"
    config_new = "wan_shared_cfg.param_dtype = torch.float16"
    if config_old in config_text:
        config_path.write_text(config_text.replace(config_old, config_new, 1), encoding="utf-8")
    elif config_new not in config_text:
        raise RuntimeError("Could not safely configure VACE transformer precision for this GPU.")

    model_path = VACE_ROOT / "vace/models/wan/wan_vace.py"
    model_text = model_path.read_text(encoding="utf-8")
    model_old = "            self.model.to(self.device)"
    model_new = "            self.model.to(device=self.device, dtype=self.param_dtype)"
    if model_old in model_text:
        model_path.write_text(model_text.replace(model_old, model_new, 1), encoding="utf-8")
    elif model_new not in model_text:
        raise RuntimeError("Could not safely cast the VACE transformer weights to FP16.")

    wan_spec = importlib.util.find_spec("wan")
    if not wan_spec or not wan_spec.submodule_search_locations:
        raise RuntimeError("Wan2.1 Python package is missing; install it using the setup guide.")
    attention_path = Path(next(iter(wan_spec.submodule_search_locations))) / "modules/attention.py"
    attention_text = attention_path.read_text(encoding="utf-8")
    attention_old = "    dtype=torch.bfloat16,\n    version=None,"
    attention_new = "    dtype=torch.float16,\n    version=None,"
    if attention_old in attention_text:
        attention_path.write_text(attention_text.replace(attention_old, attention_new, 1), encoding="utf-8")
    elif attention_new not in attention_text:
        raise RuntimeError("Could not safely configure Wan attention precision for this GPU.")
    print("Configured FP16 transformer + FP16 SDPA; T5 remains CPU-offloaded in BF16.")


def post_progress(client: requests.Session, job_id: str, stage: str, progress: int) -> None:
    response = client.post(
        api_url(f"/api/vace/runner/jobs/{job_id}/progress"),
        json={"stage": stage, "progress": progress},
        timeout=30,
    )
    response.raise_for_status()


def download_file(client: requests.Session, path: str, destination: Path) -> None:
    with client.get(api_url(path), stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def run_command(command: list[str], cwd: Path, log_path: Path, client: requests.Session,
                job_id: str, stage: str, start_progress: int, end_progress: int) -> None:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
            text=True, env=os.environ.copy(),
        )
        last_ping = 0.0
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - last_ping >= 20:
                    elapsed = now - started
                    progress = min(end_progress, start_progress + int(elapsed // 30))
                    post_progress(client, job_id, stage, progress)
                    last_ping = now
                time.sleep(5)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"{stage} failed (exit {process.returncode}):\n{tail}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(result.stdout.strip())


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return bool(result.stdout.strip())


def generate_custom_voice(text: str, language: str, reference_wav: Path, destination: Path) -> None:
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    with torch.inference_mode():
        audio = model.generate(
            text,
            language_id=language,
            audio_prompt_path=str(reference_wav),
        )
    torchaudio.save(str(destination), audio.detach().cpu(), model.sr)
    del model, audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mux_narration(vace_video: Path, narration_wav: Path, source_video: Path,
                  keep_source_audio: bool, destination: Path) -> None:
    audio_duration = probe_duration(narration_wav)
    video_duration = probe_duration(vace_video)
    pad_duration = max(0.0, audio_duration - video_duration) + 0.25
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-i", str(vace_video), "-i", str(narration_wav)]
    mix_filter = "[1:a:0]volume=1.0[voice]"
    audio_map = "[voice]"
    if keep_source_audio and has_audio(source_video):
        command += ["-i", str(source_video)]
        mix_filter += ";[2:a:0]volume=0.16[original];[voice][original]amix=inputs=2:duration=first:dropout_transition=2[mix]"
        audio_map = "[mix]"
    command += [
        "-filter_complex", mix_filter,
        "-map", "0:v:0", "-map", audio_map,
        "-vf", "tpad=stop_mode=clone:stop_duration=" + f"{pad_duration:.3f}",
        "-t", f"{audio_duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-threads", "2",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True, timeout=1800, capture_output=True, text=True)


def process_job(client: requests.Session, job: dict) -> None:
    job_id = job["id"]
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / "source_video"
    reference = job_dir / "voice_reference.wav"
    normalized = job_dir / "normalized.mp4"
    output_dir = job_dir / "vace_output"
    output_dir.mkdir(exist_ok=True)
    narration_wav = job_dir / "custom_narration.wav"
    final_video = job_dir / "final_video.mp4"

    try:
        post_progress(client, job_id, "Downloading source and custom voice", 8)
        download_file(client, job["input_url"], source)
        download_file(client, job["voice_url"], reference)

        post_progress(client, job_id, "Preparing 480p VACE input", 12)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", str(source),
             "-vf", "fps=16,tpad=stop_mode=clone:stop_duration=5.1,trim=end_frame=81,"
                    "scale=832:480:force_original_aspect_ratio=decrease,"
                    "pad=832:480:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p",
             "-frames:v", "81",
             "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
             "-threads", "2", str(normalized)],
            check=True, timeout=300, capture_output=True, text=True,
        )

        post_progress(client, job_id, "Running Wan2.1 VACE video editing", 20)
        command = [
            os.getenv("VACE_PYTHON", "python"), "vace/vace_pipeline.py",
            "--base", "wan", "--task", job.get("motion_mode", "flow"),
            "--video", str(normalized), "--prompt", job["prompt"],
            "--ckpt_dir", str(VACE_CHECKPOINT_DIR),
            "--size", "480p", "--frame_num", str(FRAME_COUNT),
            "--save_dir", str(output_dir), "--save_file", str(output_dir / "out_video.mp4"),
            "--offload_model", "True", "--t5_cpu",
        ]
        run_command(
            command, VACE_ROOT, job_dir / "vace.log", client, job_id,
            "VACE video generation", 22, 70,
        )
        generated_video = output_dir / "out_video.mp4"
        if not generated_video.is_file() or generated_video.stat().st_size < 1024:
            raise RuntimeError("VACE finished without creating the expected out_video.mp4.")

        post_progress(client, job_id, "Generating your custom voice narration", 76)
        generate_custom_voice(
            job["narration"], job["language"], reference, narration_wav
        )

        post_progress(client, job_id, "Combining video and custom voice", 88)
        mux_narration(
            generated_video, narration_wav, source,
            bool(job.get("keep_source_audio")), final_video,
        )

        post_progress(client, job_id, "Uploading finished video to Studio", 94)
        with final_video.open("rb") as video_file:
            response = client.post(
                api_url(f"/api/vace/runner/jobs/{job_id}/complete"),
                files={"file": ("final_video.mp4", video_file, "video/mp4")},
                timeout=(30, 900),
            )
        response.raise_for_status()
        print(f"Completed VACE job {job_id}")
    except Exception as exc:
        try:
            client.post(
                api_url(f"/api/vace/runner/jobs/{job_id}/fail"),
                json={"error": str(exc)[:2000]},
                timeout=30,
            ).raise_for_status()
        except Exception as report_error:
            print(f"Could not report VACE failure: {report_error}")
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Video Studio remote VACE GPU runner")
    parser.add_argument("--once", action="store_true", help="Check for at most one job, then exit.")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args()

    load_kaggle_secrets()
    if not API_BASE.startswith("https://"):
        raise SystemExit("STUDIO_API_URL must use HTTPS. Do not send the runner token over plain HTTP.")
    if not RUNNER_TOKEN:
        raise SystemExit("Set STUDIO_VACE_RUNNER_TOKEN as an environment variable or Kaggle secret.")
    if not (VACE_ROOT / "vace" / "vace_pipeline.py").is_file():
        raise SystemExit(f"VACE source not found at {VACE_ROOT}. Follow KAGGLE_VACE_SETUP.md.")
    if not VACE_CHECKPOINT_DIR.is_dir():
        raise SystemExit(f"VACE checkpoint directory not found: {VACE_CHECKPOINT_DIR}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("FFmpeg and ffprobe must be installed in the GPU runtime.")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_vace_runtime()
    client = session()
    print("VACE remote runner connected. Model: Wan2.1-VACE-1.3B; custom voice: Chatterbox V3.")
    while True:
        response = client.post(api_url("/api/vace/runner/next"), timeout=30)
        response.raise_for_status()
        job = response.json().get("job")
        if job:
            try:
                process_job(client, job)
            except Exception as exc:
                print(f"Job {job.get('id')} failed: {exc}")
        elif args.once:
            print("No queued VACE jobs.")
            return
        if args.once:
            return
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
