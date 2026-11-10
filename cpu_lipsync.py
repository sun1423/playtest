"""CPU-only ONNX lip-sync for video dubbing.

Wav2Lip changes the mouth region only. The original video supplies identity,
eyes, facial expression, head motion, body movement and background.
"""
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

FPS = 25.0
MEL_STEP_SIZE = 16
MEL_FPS = 80.0
FRAME_BATCH = 8
MAX_DIMENSION = 1280
MAX_SLOWDOWN = 1.20
IMAGE_SIZE = 96

MODEL_FILES = {
    "generator": "wav2lip_gan.onnx",
    "face_detector": "yunet.onnx",
}


def engine_ready(models_root: Path) -> bool:
    """Return true only when both pinned model files and runtime deps exist."""
    root = Path(models_root).resolve()
    if not all((root / name).is_file() and (root / name).stat().st_size > 100_000
               for name in MODEL_FILES.values()):
        return False
    return all(importlib.util.find_spec(name) is not None
               for name in ("onnxruntime", "cv2", "numpy", "librosa", "scipy"))


def plan_video_timing(video_duration: float, speech_duration: float,
                      max_slowdown: float = MAX_SLOWDOWN) -> dict:
    """Keep the full video; only slow it modestly if the new speech is longer."""
    video_duration = float(video_duration)
    speech_duration = float(speech_duration)
    if not math.isfinite(video_duration) or not math.isfinite(speech_duration):
        raise ValueError("Video and narration durations must be finite.")
    if video_duration <= 0 or speech_duration <= 0:
        raise ValueError("Video and narration must both have a positive duration.")
    ratio = max(1.0, speech_duration / video_duration)
    if ratio > max_slowdown + 1e-6:
        raise ValueError(
            "The narration is more than 20% longer than the video. "
            "Shorten the script or choose a longer source video; the Studio "
            "will not cut off speech or freeze the picture for a long time."
        )
    target = max(video_duration, speech_duration)
    frames = max(1, int(math.ceil(target * FPS)))
    return {
        "video_duration": video_duration,
        "speech_duration": speech_duration,
        "target_duration": frames / FPS,
        "video_speed_factor": ratio,
        "frame_count": frames,
        "fps": FPS,
    }


def build_mel_chunks(mel, frame_count: int, fps: float = FPS):
    """Build exactly one Wav2Lip 80x16 mel window per output frame."""
    import numpy as np

    if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] == 0:
        raise ValueError("Wav2Lip mel input must have shape (80, time) and be non-empty.")
    if frame_count < 1 or fps <= 0:
        raise ValueError("Lip-sync frame count and FPS must be positive.")
    chunks = []
    for index in range(frame_count):
        start = int(index * MEL_FPS / fps)
        if start + MEL_STEP_SIZE <= mel.shape[1]:
            chunk = mel[:, start:start + MEL_STEP_SIZE]
        else:
            chunk = mel[:, max(0, mel.shape[1] - MEL_STEP_SIZE):]
            if chunk.shape[1] < MEL_STEP_SIZE:
                chunk = np.pad(chunk, ((0, 0), (MEL_STEP_SIZE - chunk.shape[1], 0)))
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def _run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Video preparation exceeded {timeout} seconds.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-1600:]
        raise RuntimeError(detail or "A video-processing command failed.") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable is missing: {command[0]}") from exc


def media_stream_duration(data: dict, stream: dict | None) -> float:
    """Prefer the requested stream's duration over the container's longest stream."""
    candidates = ((stream or {}).get("duration"),)
    for candidate in candidates:
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    # Some containers omit stream.duration while still exposing exact stream
    # timestamps or a frame count. Prefer those over the longer container
    # duration (which may belong to a separate audio track).
    try:
        duration_ts = float((stream or {}).get("duration_ts"))
        time_base = float(Fraction(str((stream or {}).get("time_base"))))
        derived = duration_ts * time_base
        if math.isfinite(derived) and derived > 0:
            return derived
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        frames = float((stream or {}).get("nb_frames"))
        rate_text = (stream or {}).get("avg_frame_rate") or (stream or {}).get("r_frame_rate")
        rate = float(Fraction(str(rate_text)))
        derived = frames / rate
        if math.isfinite(derived) and derived > 0:
            return derived
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        fallback = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        fallback = 0.0
    if math.isfinite(fallback) and fallback > 0:
        return fallback
    raise ValueError("The media stream has an invalid duration.")


def probe_media(path: Path, require_video: bool = True) -> dict:
    """Probe either a video (default) or an audio-only file such as the TTS WAV."""
    result = _run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(Path(path).resolve()),
    ], timeout=30)
    data = json.loads(result.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if require_video and video is None:
        raise ValueError("The source file contains no video stream.")
    if not require_video and audio is None:
        raise ValueError("The source file contains no audio stream.")
    duration = media_stream_duration(data, video if require_video else audio)
    return {
        "duration_seconds": duration,
        "width": int(video.get("width") or 0) if video else 0,
        "height": int(video.get("height") or 0) if video else 0,
        "has_audio": audio is not None,
        "fps": (video.get("avg_frame_rate") or video.get("r_frame_rate") or "unknown") if video else "unknown",
    }


def _mouth_mask(cv2, width: int, height: int, cache: dict):
    """Soft elliptical mask over mouth/lower lip area; eyes and upper face stay original."""
    import numpy as np

    key = (width, height)
    mask = cache.get(key)
    if mask is not None:
        return mask
    mask = np.zeros((height, width), dtype=np.float32)
    center = (width // 2, int(height * 0.73))
    axes = (max(8, int(width * 0.43)), max(8, int(height * 0.22)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    sigma = max(2.0, min(width, height) * 0.035)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)[:, :, None]
    cache[key] = mask
    return mask


def wav2lip_melspectrogram(waveform):
    """Reproduce the Wav2Lip reference mel pipeline (amplitude, not power mels)."""
    import librosa
    import numpy as np
    from scipy import signal

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size < 800 or not np.all(np.isfinite(waveform)):
        raise ValueError("Wav2Lip audio must contain at least 800 finite samples.")
    emphasized = signal.lfilter([1.0, -0.97], [1.0], waveform)
    stft = librosa.stft(
        y=emphasized, n_fft=800, hop_length=200, win_length=800
    )
    basis = librosa.filters.mel(
        sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600
    )
    magnitude = np.dot(basis, np.abs(stft))
    mel_db = 20.0 * np.log10(np.maximum(1e-5, magnitude)) - 20.0
    # Original Wav2Lip hparams: min_level_db=-100, max_abs_value=4,
    # symmetric_mels=True. This exact scaling matters for lip timing.
    return np.clip(8.0 * ((mel_db + 100.0) / 100.0) - 4.0, -4.0, 4.0).astype(np.float32)


def _choose_face(faces, previous, frame_width: int, frame_height: int):
    if faces is None or len(faces) == 0:
        return None
    candidates = []
    for face in faces:
        x, y, width, height, *landmarks_and_score = face.tolist()
        score = float(landmarks_and_score[-1]) if landmarks_and_score else 0.0
        cx, cy = x + width / 2, y + height / 2
        area_ratio = (width * height) / max(1, frame_width * frame_height)
        if previous is None:
            rank = score + min(0.35, area_ratio * 2.0)
        else:
            px, py, pw, ph = previous
            pcx, pcy = px + pw / 2, py + ph / 2
            distance = math.hypot((cx - pcx) / max(1, frame_width),
                                  (cy - pcy) / max(1, frame_height))
            size_change = abs(math.log(max(1, width * height) / max(1, pw * ph)))
            rank = score - distance * 2.0 - size_change * 0.08
        candidates.append((rank, (int(x), int(y), int(width), int(height))))
    return max(candidates, key=lambda item: item[0])[1]


def _padded_box(box, frame_width: int, frame_height: int):
    x, y, width, height = box
    # Wav2Lip expects a close face crop. Small chin padding avoids shifting
    # the generated mouth onto the neck; do not crop away the forehead/eyes.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + width)
    y2 = min(frame_height, y + int(height * 1.10))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def run_cpu_lipsync(
    source_video: Path,
    narration_wav: Path,
    destination: Path,
    models_root: Path,
    keep_source_ambience: bool = False,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    """Create a full-length, CPU-only lip-synced MP4 using ONNX Runtime.

    Video frames are decoded and processed in batches of at most eight; the
    entire decoded video is never kept in memory.
    """
    if not engine_ready(models_root):
        raise RuntimeError(
            "CPU lip-sync models or Python dependencies are missing. "
            "Run scripts/install_cpu_lipsync.py and install requirements.txt first."
        )
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and FFprobe must be installed.")
    source_video = Path(source_video).resolve()
    narration_wav = Path(narration_wav).resolve()
    destination = Path(destination).resolve()
    models_root = Path(models_root).resolve()
    if not source_video.is_file() or not narration_wav.is_file():
        raise ValueError("The source video or generated narration WAV is missing.")

    import cv2
    import librosa
    import numpy as np
    import onnxruntime as ort

    started = time.monotonic()
    source_info = probe_media(source_video)
    audio_info = probe_media(narration_wav, require_video=False)
    timing = plan_video_timing(source_info["duration_seconds"], audio_info["duration_seconds"])
    frame_count = timing["frame_count"]
    fps = timing["fps"]
    target_duration = timing["target_duration"]
    speed_factor = timing["video_speed_factor"]
    speech_frame_count = min(frame_count, int(math.ceil(audio_info["duration_seconds"] * fps)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    working = destination.parent / "lipsync_work"
    shutil.rmtree(working, ignore_errors=True)
    working.mkdir(parents=True)
    normalized_video = working / "normalized.mp4"
    padded_audio = working / "narration_16k.wav"
    ffmpeg_log = working / "ffmpeg.log"

    if progress:
        progress(8, "Preparing the complete source video")
    video_filter = []
    if speed_factor > 1.00001:
        video_filter.append(f"setpts={speed_factor:.8f}*PTS")
    video_filter.extend([
        f"fps={fps:g}",
        f"scale='min({MAX_DIMENSION},iw)':'min({MAX_DIMENSION},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
        "setsar=1",
        "format=yuv420p",
        "tpad=stop_mode=clone:stop_duration=2",
        f"trim=end_frame={frame_count}",
        f"setpts=N/({fps:g}*TB)",
    ])
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source_video), "-an", "-vf", ",".join(video_filter),
        "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p",
        str(normalized_video),
    ], timeout=max(300, int(source_info["duration_seconds"] * 30)))

    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(narration_wav), "-af", "apad",
        "-t", f"{target_duration:.6f}", "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", "-threads", "1", str(padded_audio),
    ], timeout=180)

    video_probe = probe_media(normalized_video)
    width, height = video_probe["width"], video_probe["height"]
    if width < 96 or height < 96:
        raise RuntimeError("The source video resolution is too small for reliable lip-sync.")

    if progress:
        progress(14, "Loading CPU ONNX lip-sync model")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, min(2, os.cpu_count() or 2))
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(models_root / MODEL_FILES["generator"]),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    if "CPUExecutionProvider" not in session.get_providers():
        raise RuntimeError("Lip-sync model did not initialise on the CPU provider.")
    mel_input, face_input = [item.name for item in session.get_inputs()[:2]]
    output_name = session.get_outputs()[0].name

    if progress:
        progress(18, "Preparing Hindi speech timing")
    waveform, _ = librosa.load(str(padded_audio), sr=16000, mono=True)
    mel = wav2lip_melspectrogram(waveform)
    mel_chunks = build_mel_chunks(mel, frame_count, fps)

    detector = cv2.FaceDetectorYN.create(
        str(models_root / MODEL_FILES["face_detector"]),
        "",
        (320, 320),
        score_threshold=0.55,
        nms_threshold=0.3,
        top_k=5000,
    )
    capture = cv2.VideoCapture(str(normalized_video))
    if not capture.isOpened():
        raise RuntimeError("Could not open the normalised source video.")

    if progress:
        progress(20, "Synchronising mouth movement to your voice")
    include_ambience = bool(keep_source_ambience and source_info["has_audio"])
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:g}", "-i", "pipe:0",
        "-i", str(padded_audio),
    ]
    if include_ambience:
        command += ["-i", str(source_video)]

    if include_ambience:
        ambience_speed = 1.0 / speed_factor
        filter_complex = (
            "[1:a:0]volume=1.0[voice];"
            f"[2:a:0]atempo={ambience_speed:.6f},volume=0.12,"
            f"apad,atrim=duration={target_duration:.6f}[ambience];"
            "[voice][ambience]amix=inputs=2:duration=first:dropout_transition=0:"
            "normalize=0,alimiter=limit=0.95:level=0[aout]"
        )
    else:
        filter_complex = "[1:a:0]alimiter=limit=0.95:level=0[aout]"
    command += [
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
        "-frames:v", str(frame_count), "-t", f"{target_duration:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-threads", "2", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", str(destination),
    ]
    with ffmpeg_log.open("w", encoding="utf-8") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        previous_box = None
        feather_cache = {}
        processed = 0
        detected = 0
        total_batches = int(math.ceil(frame_count / FRAME_BATCH))
        try:
            batch_frames = []
            batch_mels = []
            batch_meta = []

            def flush_batch():
                nonlocal batch_frames, batch_mels, batch_meta, processed, detected
                if not batch_frames:
                    return
                face_inputs = []
                selected = []
                for local_index, (frame, box) in enumerate(batch_meta):
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    face = cv2.resize(frame[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE))
                    masked = face.copy()
                    masked[IMAGE_SIZE // 2:, :] = 0
                    image = np.concatenate((masked, face), axis=2).astype(np.float32) / 255.0
                    face_inputs.append(image)
                    selected.append(local_index)
                predictions = None
                if face_inputs:
                    image_batch = np.asarray(face_inputs, dtype=np.float32).transpose(0, 3, 1, 2)
                    selected_mels = np.asarray([batch_mels[i] for i in selected], dtype=np.float32)[:, None, :, :]
                    predictions = session.run(
                        [output_name],
                        {mel_input: selected_mels, face_input: image_batch},
                    )[0]
                    predictions = predictions.transpose(0, 2, 3, 1) * 255.0
                generated_index = 0
                for local_index, frame in enumerate(batch_frames):
                    box = batch_meta[local_index][1]
                    if box is not None:
                        detected += 1
                        x1, y1, x2, y2 = box
                        bw, bh = x2 - x1, y2 - y1
                        generated = np.clip(predictions[generated_index], 0, 255).astype(np.uint8)
                        generated_index += 1
                        generated = cv2.resize(generated, (bw, bh), interpolation=cv2.INTER_LINEAR)
                        mask = _mouth_mask(cv2, bw, bh, feather_cache)
                        original_roi = frame[y1:y2, x1:x2].astype(np.float32)
                        frame[y1:y2, x1:x2] = np.clip(
                            generated.astype(np.float32) * mask
                            + original_roi * (1.0 - mask), 0, 255
                        ).astype(np.uint8)
                    encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                    processed += 1
                batch_frames = []
                batch_mels = []
                batch_meta = []
                if progress:
                    current = 20 + int(73 * min(1.0, processed / frame_count))
                    progress(min(93, current), f"Lip-syncing frame {processed:,} of {frame_count:,}")

            for index in range(frame_count):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"The source video ended at frame {index + 1} of {frame_count}; "
                        "the full video could not be synchronised."
                    )
                if index < speech_frame_count:
                    detector.setInputSize((width, height))
                    _, faces = detector.detect(frame)
                    detected_box = _choose_face(faces, previous_box, width, height)
                    if detected_box is not None:
                        previous_box = detected_box
                        box = _padded_box(detected_box, width, height)
                    else:
                        box = None
                else:
                    # The new narration has ended. Preserve the source performance
                    # rather than repeating the final synthetic mouth pose.
                    box = None
                batch_frames.append(frame)
                batch_mels.append(mel_chunks[index])
                batch_meta.append((frame, box))
                if len(batch_frames) >= FRAME_BATCH:
                    flush_batch()
            flush_batch()
            encoder.stdin.close()
            return_code = encoder.wait(timeout=3600)
            if return_code != 0:
                raise RuntimeError("FFmpeg could not encode the lip-synced MP4: " +
                                   ffmpeg_log.read_text(errors="replace")[-1200:])
        except Exception:
            capture.release()
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait(timeout=30)
            raise
        finally:
            capture.release()

    # Only count frames during active narration; the silent tail deliberately
    # preserves the original performance and must not lower face-detection quality.
    detection_ratio = detected / max(1, speech_frame_count)
    if detection_ratio < 0.80:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "A clear face was detected in less than 80% of speech frames. Use footage with "
            "one clearly visible main speaker; long occlusions and frequent off-camera shots "
            "cannot be lip-synchronised reliably by this CPU model."
        )
    if not destination.is_file() or destination.stat().st_size < 1024:
        raise RuntimeError("The lip-sync output MP4 was not created.")
    output_info = probe_media(destination)
    if not output_info["has_audio"]:
        raise RuntimeError("The generated MP4 is missing the speech audio track.")
    if abs(output_info["duration_seconds"] - target_duration) > max(0.5, 2.0 / fps):
        raise RuntimeError("The generated MP4 duration does not match the lip-sync timeline.")

    if progress:
        progress(97, "Validating the complete MP4")
    shutil.rmtree(working, ignore_errors=True)
    return {
        "duration_seconds": round(output_info["duration_seconds"], 3),
        "frames_processed": processed,
        "face_detected_ratio": round(detection_ratio, 3),
        "width": output_info["width"],
        "height": output_info["height"],
        "fps": fps,
        "video_speed_factor": round(speed_factor, 4),
        "engine": "Wav2Lip ONNX · CPUExecutionProvider",
        "processing_seconds": round(time.monotonic() - started, 2),
        "expressions": "Source face, eyes, head movement and body performance preserved; mouth region synchronised to the new speech.",
    }
