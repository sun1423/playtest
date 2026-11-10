"""WAV inspection, 20-second reference preparation and local Hindi transcription."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import unicodedata
from pathlib import Path

HINDI_ASR_PROMPT = (
    "यह हिंदी भाषा का ऑडियो है। बोले गए शब्दों को हिंदी की देवनागरी लिपि में लिखें। "
    "उर्दू या अरबी लिपि का उपयोग न करें।"
)
HINDI_ASR_RETRY_PROMPT = (
    "यह हिंदी भाषण है, उर्दू नहीं। शब्दों को केवल देवनागरी में लिखें, "
    "जैसे: नमस्ते, आप कैसे हैं? कृपया अरबी/उर्दू अक्षर न लिखें।"
)


def _contains_urdu_script(text: str) -> bool:
    """Detect Arabic/Urdu letters in ASR output without flagging punctuation."""
    return any(
        0x0600 <= ord(char) <= 0x08FF
        and unicodedata.category(char)[0] in {"L", "M"}
        for char in text
    )


def _transcribe_hindi_once(model, audio_path: str, prompt: str):
    segments, info = model.transcribe(
        audio_path,
        language="hi",
        task="transcribe",
        initial_prompt=prompt,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False,
        temperature=0.0,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info

_ASR_LOCK = threading.Lock()
_ASR_MODEL = None
_ASR_MODEL_NAME = ""


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio processing exceeded its time limit.") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required audio tool is not installed: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()[-1200:]
        raise RuntimeError(detail or "Audio processing failed.") from exc


def probe_audio(path: Path) -> dict:
    """Read actual audio properties; do not trust the filename or browser MIME type."""
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size < 44:
        raise ValueError("The uploaded WAV is empty or incomplete.")
    with path.open("rb") as source:
        header = source.read(12)
    if header[8:12] != b"WAVE" or header[:4] not in {b"RIFF", b"RF64"}:
        raise ValueError("The upload is not a valid WAV file.")

    result = _run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    import json

    metadata = json.loads(result.stdout)
    stream = next(
        (item for item in metadata.get("streams", []) if item.get("codec_type") == "audio"),
        None,
    )
    if not stream:
        raise ValueError("The WAV contains no audio stream.")
    duration = float(metadata.get("format", {}).get("duration") or 0)
    if not 0.25 <= duration <= 3600:
        raise ValueError("The WAV duration must be between 0.25 seconds and one hour.")
    return {
        "duration_seconds": round(duration, 3),
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "bits_per_sample": int(stream.get("bits_per_sample") or stream.get("bits_per_raw_sample") or 0),
        "format": metadata.get("format", {}).get("format_name", "wav"),
        "file_size_bytes": path.stat().st_size,
    }


def _active_intervals_from_silence(log_text: str, duration: float) -> list[tuple[float, float]]:
    events = [
        (kind, float(value))
        for kind, value in re.findall(r"silence_(start|end):\s*([0-9]+(?:\.[0-9]+)?)", log_text)
    ]
    active: list[tuple[float, float]] = []
    cursor = 0.0
    silence_start = None
    for kind, value in events:
        value = min(max(value, 0.0), duration)
        if kind == "start" and silence_start is None:
            if value > cursor:
                active.append((cursor, value))
            silence_start = value
        elif kind == "end" and silence_start is not None:
            cursor = max(cursor, value)
            silence_start = None
    if silence_start is None and cursor < duration:
        active.append((cursor, duration))
    return [(start, end) for start, end in active if end > start]


def choose_active_audio_window(
    duration: float,
    active_intervals: list[tuple[float, float]],
    window_seconds: float = 20.0,
) -> tuple[float, float, float]:
    """Choose the fixed-length window with the most non-silent audio."""
    length = min(float(window_seconds), float(duration))
    if length <= 0:
        raise ValueError("No usable audio duration was found.")
    if duration <= window_seconds:
        active = sum(
            max(0.0, min(end, duration) - max(start, 0.0))
            for start, end in active_intervals
        )
        return 0.0, duration, min(1.0, active / duration if duration else 0.0)

    max_start = duration - length
    candidates = {0.0, max_start}
    for start, end in active_intervals:
        for point in (start, end, start - length, end - length):
            candidates.add(min(max(point, 0.0), max_start))

    def coverage(start_at: float) -> float:
        end_at = start_at + length
        return sum(
            max(0.0, min(end, end_at) - max(start, start_at))
            for start, end in active_intervals
        )

    best_start = max(sorted(candidates), key=coverage)
    speech_seconds = coverage(best_start)
    return best_start, length, min(1.0, speech_seconds / length)


def _detect_active_intervals(path: Path, duration: float) -> list[tuple[float, float]]:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-vn", "-af", "silencedetect=noise=-48dB:d=0.30",
                "-f", "null", "-",
            ],
            check=False, capture_output=True, text=True, timeout=180,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise RuntimeError("FFmpeg could not analyse the uploaded WAV.") from exc
    if result.returncode != 0:
        raise RuntimeError("Could not analyse audio activity in the uploaded WAV.")
    return _active_intervals_from_silence(result.stderr or "", duration)


def _measure_volume(path: Path) -> tuple[float, float]:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-vn", "-af", "volumedetect", "-f", "null", "-",
            ],
            check=False, capture_output=True, text=True, timeout=90,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise RuntimeError("FFmpeg could not measure WAV loudness.") from exc
    if result.returncode != 0:
        raise RuntimeError("FFmpeg could not measure WAV loudness.")
    output = result.stderr or ""
    mean_match = re.search(r"mean_volume:\s*(-?(?:[0-9]+(?:\.[0-9]+)?|inf))\s*dB", output)
    peak_match = re.search(r"max_volume:\s*(-?(?:[0-9]+(?:\.[0-9]+)?|inf))\s*dB", output)
    if not mean_match or not peak_match:
        raise RuntimeError("Could not measure WAV loudness.")
    mean = float(mean_match.group(1)) if mean_match.group(1) != "-inf" else -90.0
    peak = float(peak_match.group(1)) if peak_match.group(1) != "-inf" else -90.0
    return mean, peak


def gain_for_levels(mean_db: float, peak_db: float) -> float:
    """Return gain only for clearly quiet or hot audio; avoid unnecessary processing."""
    if peak_db > -1.0:
        return round(-3.0 - peak_db, 2)
    if mean_db < -28.0:
        requested_gain = min(12.0, -22.0 - mean_db)
        peak_headroom = max(0.0, -3.0 - peak_db)
        return round(min(requested_gain, peak_headroom), 2)
    return 0.0


def prepare_voice_audio(source: Path, destination: Path) -> dict:
    """Create a mono 24 kHz PCM16 reference clip without altering the source WAV."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    original = probe_audio(source)
    active = _detect_active_intervals(source, original["duration_seconds"])
    start, duration, active_audio_ratio = choose_active_audio_window(
        original["duration_seconds"], active, 20.0
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_clip = destination.with_name(f".{destination.stem}.raw.wav")
    try:
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", str(raw_clip),
        ], timeout=180)
        mean_db, peak_db = _measure_volume(raw_clip)
        gain_db = gain_for_levels(mean_db, peak_db)
        if gain_db:
            _run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(raw_clip), "-af",
                f"volume={gain_db:.2f}dB,alimiter=limit=0.95:level=0",
                "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(destination),
            ], timeout=120)
            adjustment = f"Applied {gain_db:+.2f} dB gain and peak limiting."
        else:
            shutil.copyfile(raw_clip, destination)
            adjustment = "No loudness processing needed; original clip level retained."
        prepared = probe_audio(destination)
        if (
            prepared["codec"] != "pcm_s16le"
            or prepared["sample_rate"] != 24000
            or prepared["channels"] != 1
        ):
            raise RuntimeError("Prepared reference did not normalize to mono 24 kHz PCM16.")
        warnings = []
        if active_audio_ratio < 0.5:
            warnings.append("The selected clip contains substantial silence or low-level audio; listen before saving.")
        if mean_db < -28.0 and gain_db == 0.0:
            warnings.append("Average level is low but peak headroom is limited; gain was not increased to avoid clipping.")
        if peak_db >= -0.1:
            warnings.append("The source contains near-clipped or clipped peaks; lost distortion cannot be restored.")
        return {
            "original": original,
            "prepared": prepared,
            "clip_start_seconds": round(start, 3),
            "clip_duration_seconds": round(duration, 3),
            "non_silent_audio_ratio": round(active_audio_ratio, 3),
            "mean_volume_db": round(mean_db, 2),
            "peak_volume_db": round(peak_db, 2),
            "gain_db": gain_db,
            "processing_note": adjustment + " Noise reduction was not applied because it can damage speaker characteristics.",
            "warnings": warnings,
        }
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        raw_clip.unlink(missing_ok=True)


def transcribe_hindi(path: Path, model_cache: Path) -> dict:
    """Run one local CPU/int8 Hindi transcription; model weights are cached outside Git."""
    global _ASR_MODEL, _ASR_MODEL_NAME
    model_name = os.getenv("VOICE_ASR_MODEL", "base").strip() or "base"
    with _ASR_LOCK:
        if _ASR_MODEL is None or _ASR_MODEL_NAME != model_name:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Local Hindi transcription is not installed.") from exc
            model_cache = Path(model_cache).resolve()
            model_cache.mkdir(parents=True, exist_ok=True)
            _ASR_MODEL = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(1, min(os.cpu_count() or 2, 2)),
                num_workers=1,
                download_root=str(model_cache),
            )
            _ASR_MODEL_NAME = model_name
        audio_path = str(Path(path).resolve())
        transcript, info = _transcribe_hindi_once(
            _ASR_MODEL, audio_path, HINDI_ASR_PROMPT
        )
        attempts = 1
        if _contains_urdu_script(transcript):
            retry_text, retry_info = _transcribe_hindi_once(
                _ASR_MODEL, audio_path, HINDI_ASR_RETRY_PROMPT
            )
            attempts = 2
            if retry_text:
                transcript, info = retry_text, retry_info
    script_warning = (
        "The recognizer still returned Urdu/Arabic characters. Review and correct "
        "the transcript in Devanagari before saving this Hindi voice profile."
        if _contains_urdu_script(transcript) else ""
    )
    return {
        "text": transcript,
        "language": getattr(info, "language", "hi"),
        "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
        "model": model_name,
        "transcription_attempts": attempts,
        "script_warning": script_warning,
    }
