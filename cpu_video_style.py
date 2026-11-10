"""Streaming ONNX full-frame AI video styles for CPU-only rendering."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

STYLE_MODELS = {
    "anime_512": {"file": "animeganv2_512.onnx", "size": 512, "family": "animegan"},
    "candy": {"file": "candy.onnx", "size": 224, "family": "fast_style"},
    "mosaic": {"file": "mosaic.onnx", "size": 224, "family": "fast_style"},
    "rain_princess": {"file": "rain-princess.onnx", "size": 224, "family": "fast_style"},
    "udnie": {"file": "udnie.onnx", "size": 224, "family": "fast_style"},
}
STYLE_LABELS = {
    "anime_512": "AnimeGANv2 · 512px quality",
    "candy": "Candy painting · fast",
    "mosaic": "Mosaic painting · fast",
    "rain_princess": "Rain Princess painting · fast",
    "udnie": "Udnie painting · fast",
}


def engine_ready(models_root: Path, style: str | None = None) -> bool:
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("cv2") is None:
        return False
    root = Path(models_root).resolve()
    names = [style] if style and style != "none" else list(STYLE_MODELS)
    return all(
        name in STYLE_MODELS
        and (root / STYLE_MODELS[name]["file"]).is_file()
        and (root / STYLE_MODELS[name]["file"]).stat().st_size > 100_000
        for name in names
    )


def style_catalog(models_root: Path) -> list[dict]:
    root = Path(models_root).resolve()
    dependencies_ready = all(
        importlib.util.find_spec(name) is not None for name in ("onnxruntime", "cv2")
    )
    return [{
        "id": key,
        "label": STYLE_LABELS[key],
        "ready": dependencies_ready
            and (root / value["file"]).is_file()
            and (root / value["file"]).stat().st_size > 100_000,
        "input_size": value["size"],
        "family": value["family"],
    } for key, value in STYLE_MODELS.items()]


def _square_frame(cv2, frame, size: int):
    """Letterbox to the fixed model input without stretching the source subject."""
    height, width = frame.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("Video frame is too small for style transfer.")
    scale = min(size / width, size / height)
    rw = max(1, min(size, int(round(width * scale))))
    rh = max(1, min(size, int(round(height * scale))))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
    left, top = (size - rw) // 2, (size - rh) // 2
    right, bottom = size - rw - left, size - rh - top
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
    )
    return canvas, (left, top, rw, rh, width, height)


def _infer_frame(cv2, session, style: str, frame, strength: float):
    import numpy as np

    model = STYLE_MODELS[style]
    canvas, geometry = _square_frame(cv2, frame, model["size"])
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None, :, :, :]
    if model["family"] == "animegan":
        tensor = tensor / 127.5 - 1.0
        output = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: tensor},
        )[0][0]
        styled_rgb = np.clip((output.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    else:
        output = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: tensor},
        )[0][0]
        styled_rgb = np.clip(output.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    styled_bgr = cv2.cvtColor(styled_rgb, cv2.COLOR_RGB2BGR)
    left, top, rw, rh, width, height = geometry
    crop = styled_bgr[top:top + rh, left:left + rw]
    restored = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    amount = max(0.0, min(1.0, float(strength)))
    return restored if amount >= 0.999 else cv2.addWeighted(frame, 1.0 - amount, restored, amount, 0)


def _run(command: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Video style operation exceeded {timeout} seconds.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError((exc.stderr or exc.stdout or "FFmpeg failed")[-1600:]) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable is missing: {command[0]}") from exc


def _video_duration(video: dict, container: dict) -> float:
    """Use video timestamps/frame count before falling back to container duration."""
    for candidate in (video.get("duration"),):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    try:
        value = float(video.get("duration_ts")) * float(Fraction(str(video.get("time_base"))))
        if math.isfinite(value) and value > 0:
            return value
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        frames = float(video.get("nb_frames"))
        rate = float(Fraction(str(video.get("avg_frame_rate") or video.get("r_frame_rate"))))
        value = frames / rate
        if math.isfinite(value) and value > 0:
            return value
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        value = float(container.get("duration"))
    except (TypeError, ValueError):
        value = 0.0
    if math.isfinite(value) and value > 0:
        return value
    raise ValueError("The style input has invalid video timing.")


def _probe(path: Path) -> dict:
    result = _run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ])
    data = json.loads(result.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError("The style input contains no video stream.")
    duration = _video_duration(video, data.get("format") or {})
    try:
        fps = float(Fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "25/1"))
    except (ValueError, ZeroDivisionError):
        fps = 25.0
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("The style input has invalid video timing.")
    return {
        "duration": duration, "width": int(video["width"]), "height": int(video["height"]),
        "fps": fps, "has_audio": audio is not None,
    }


def run_video_style(
    source_video: Path,
    destination: Path,
    models_root: Path,
    style: str = "anime_512",
    strength: float = 0.85,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    """Stylise every video frame on CPU while copying the generated speech audio."""
    if style not in STYLE_MODELS:
        raise ValueError(f"Unsupported video style: {style}")
    if not math.isfinite(float(strength)) or not 0.25 <= float(strength) <= 1.0:
        raise ValueError("Style strength must be between 0.25 and 1.0.")
    if not engine_ready(models_root, style):
        raise RuntimeError("The selected CPU video-style model or ONNX Runtime is not installed.")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and FFprobe must be installed.")

    import cv2
    import onnxruntime as ort

    source_video = Path(source_video).resolve()
    destination = Path(destination).resolve()
    models_root = Path(models_root).resolve()
    if not source_video.is_file():
        raise ValueError("The lip-synced source video is missing.")

    started = time.monotonic()
    media = _probe(source_video)
    width, height, fps = media["width"], media["height"], media["fps"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    working = destination.parent / "style_work"
    shutil.rmtree(working, ignore_errors=True)
    working.mkdir(parents=True)
    log_path = working / "ffmpeg.log"

    if progress:
        progress(4, "Loading AI style model on CPU")
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(2, os.cpu_count() or 2))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(models_root / STYLE_MODELS[style]["file"]),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if "CPUExecutionProvider" not in session.get_providers():
        raise RuntimeError("The video-style model did not initialise on CPU.")
    model_size = STYLE_MODELS[style]["size"]
    input_shape = session.get_inputs()[0].shape
    if (isinstance(input_shape[2], int) and input_shape[2] != model_size) or (
        isinstance(input_shape[3], int) and input_shape[3] != model_size
    ):
        raise RuntimeError(
            f"Model input shape {input_shape} does not match the selected {model_size}px preset. "
            "Reinstall the model weights with scripts/install_video_style_models.py."
        )

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError("Could not open the lip-synced MP4 for style transfer.")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError("Could not read the source video frame count.")

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8f}", "-i", "pipe:0",
        "-i", str(source_video),
        "-map", "0:v:0", "-map", "1:a?",
        "-frames:v", str(frame_count),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-threads", "2", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(destination),
    ]
    processed = 0
    with log_path.open("w", encoding="utf-8") as log:
        encoder = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log
        )
        try:
            if progress:
                progress(8, "Applying the AI style to the complete video")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                styled = _infer_frame(cv2, session, style, frame, float(strength))
                encoder.stdin.write(styled.tobytes())
                processed += 1
                if progress and (processed % 5 == 0 or processed == frame_count):
                    percent = 8 + int(86 * processed / frame_count)
                    progress(min(94, percent), f"AI-stylising frame {processed:,} of {frame_count:,}")
            if processed != frame_count:
                raise RuntimeError(f"Video decode ended at frame {processed} of {frame_count}; refusing partial output.")
            encoder.stdin.close()
            return_code = encoder.wait(timeout=3600)
            if return_code:
                raise RuntimeError("FFmpeg could not encode the styled MP4: " + log_path.read_text(errors="replace")[-1200:])
        except Exception:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait(timeout=30)
            destination.unlink(missing_ok=True)
            raise
        finally:
            capture.release()

    if not destination.is_file() or destination.stat().st_size < 1024:
        raise RuntimeError("The AI style output MP4 was not created.")
    result = _probe(destination)
    if media["has_audio"] and not result["has_audio"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("The styled MP4 lost the generated voice audio track.")
    if abs(result["duration"] - media["duration"]) > max(0.5, 2.0 / fps):
        destination.unlink(missing_ok=True)
        raise RuntimeError("The styled MP4 duration differs from the lip-synced source.")
    shutil.rmtree(working, ignore_errors=True)
    if progress:
        progress(97, "Validating the styled video and voice audio")
    return {
        "style": style, "style_label": STYLE_LABELS[style], "style_strength": round(float(strength), 2),
        "duration_seconds": round(result["duration"], 3), "frames_processed": processed,
        "width": result["width"], "height": result["height"], "fps": round(fps, 3),
        "engine": "ONNX Runtime · CPUExecutionProvider",
        "processing_seconds": round(time.monotonic() - started, 2),
    }
