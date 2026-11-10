"""Inspect and prepare WAV voice references without modifying their original files."""

from __future__ import annotations

import json
import math
import statistics
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any


class VoiceAudioError(ValueError):
    """Raised when an uploaded voice recording cannot be inspected or prepared."""


HIGH_PASS_OPTIONS = {0, 60, 80, 100}
LOW_PASS_OPTIONS = {0, 8000, 10000, 12000}


def validate_settings(
    trim_silence: bool,
    normalize_loudness: bool,
    noise_reduction: bool,
    highpass_hz: int,
    lowpass_hz: int,
) -> dict[str, Any]:
    if highpass_hz not in HIGH_PASS_OPTIONS:
        raise VoiceAudioError("Unsupported high-pass setting.")
    if lowpass_hz not in LOW_PASS_OPTIONS:
        raise VoiceAudioError("Unsupported low-pass setting.")
    if highpass_hz and lowpass_hz and highpass_hz >= lowpass_hz:
        raise VoiceAudioError("High-pass must be lower than low-pass.")
    return {
        "trim_silence": bool(trim_silence),
        "normalize_loudness": bool(normalize_loudness),
        "noise_reduction": bool(noise_reduction),
        "highpass_hz": highpass_hz,
        "lowpass_hz": lowpass_hz,
        "output_format": "WAV PCM16, mono, 24000 Hz",
    }


def probe_wav(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_format", "-show_streams",
                "-of", "json", str(path),
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise VoiceAudioError("Could not inspect the WAV file with FFprobe.") from exc

    streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        raise VoiceAudioError("The WAV file contains no readable audio stream.")
    stream = streams[0]
    try:
        duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise VoiceAudioError("The WAV file has an invalid duration.")
    if duration > 120:
        raise VoiceAudioError("Voice references are limited to 120 seconds for safe CPU processing.")

    format_info = data.get("format", {})
    tags = format_info.get("tags", {})
    safe_tags = {
        str(key)[:80]: str(value)[:300]
        for key, value in tags.items()
        if isinstance(key, str) and isinstance(value, (str, int, float))
    }
    return {
        "container": format_info.get("format_name", "unknown"),
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate_hz": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "sample_format": stream.get("sample_fmt", "unknown"),
        "bits_per_sample": int(stream.get("bits_per_sample") or stream.get("bits_per_raw_sample") or 0),
        "duration_seconds": round(duration, 3),
        "tags": safe_tags,
    }


def decode_pcm16(path: Path) -> tuple[bytes, int]:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-ac", "1", "-ar", "24000", "-f", "s16le", "pipe:1",
            ],
            check=True, capture_output=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VoiceAudioError("Could not decode the WAV audio.") from exc
    if not result.stdout or len(result.stdout) % 2:
        raise VoiceAudioError("The WAV did not decode into valid audio samples.")
    return result.stdout, 24000


def analyze_pcm16(pcm: bytes, sample_rate: int = 24000) -> dict[str, Any]:
    if not pcm or len(pcm) % 2:
        raise VoiceAudioError("No valid PCM16 samples were available for analysis.")
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    peak = max(abs(value) for value in samples)
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    duration = len(samples) / sample_rate
    window_size = max(1, sample_rate // 10)
    active_levels = []
    silent_samples = 0
    for offset in range(0, len(samples), window_size):
        segment = samples[offset:offset + window_size]
        if not segment:
            continue
        segment_rms = math.sqrt(sum(value * value for value in segment) / len(segment))
        dbfs = 20 * math.log10(max(segment_rms, 1) / 32767)
        if dbfs < -45:
            silent_samples += len(segment)
        else:
            active_levels.append(dbfs)

    def dbfs(value: float) -> float:
        return round(20 * math.log10(max(value, 1) / 32767), 2)

    silence_percent = round(100 * silent_samples / len(samples), 2)
    warnings = []
    if duration < 5:
        warnings.append("Short reference: 5–10 seconds of clear speech is preferred.")
    elif duration > 15:
        warnings.append("Long reference: trim to a clean 5–10 second speech sample when possible.")
    if silence_percent > 10:
        warnings.append("Long or frequent near-silent sections were detected.")
    if peak >= 32700:
        warnings.append("Audio is very close to digital clipping; check the original recording level.")
    if sum(abs(value) >= 32760 for value in samples) > 0:
        warnings.append("Clipped samples were detected.")
    if dbfs(rms) < -35:
        warnings.append("Recording level is very low.")
    if len(active_levels) >= 5:
        spread = statistics.quantiles(active_levels, n=10)[8] - statistics.quantiles(active_levels, n=10)[0]
        if spread > 18:
            warnings.append("Speech loudness varies significantly between sections.")

    return {
        "duration_seconds": round(duration, 3),
        "peak_dbfs": dbfs(peak),
        "rms_dbfs": dbfs(rms),
        "near_silence_percent": silence_percent,
        "clipped_sample_count": sum(abs(value) >= 32760 for value in samples),
        "warnings": warnings,
    }


def prepare_wav(
    source: Path,
    destination: Path,
    trim_silence: bool = False,
    normalize_loudness: bool = False,
    noise_reduction: bool = False,
    highpass_hz: int = 0,
    lowpass_hz: int = 0,
) -> dict[str, Any]:
    settings = validate_settings(
        trim_silence, normalize_loudness, noise_reduction, highpass_hz, lowpass_hz
    )
    source_info = probe_wav(source)
    filters = []
    if trim_silence:
        filters.append(
            "silenceremove=start_periods=1:start_duration=0.15:"
            "start_threshold=-45dB:stop_periods=-1:stop_duration=0.25:"
            "stop_threshold=-45dB"
        )
    if highpass_hz:
        filters.append(f"highpass=f={highpass_hz}")
    if lowpass_hz:
        filters.append(f"lowpass=f={lowpass_hz}")
    if noise_reduction:
        filters.append("afftdn=nr=8:nf=-45")
    if normalize_loudness:
        filters.append("loudnorm=I=-23:LRA=7:TP=-2.0")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn",
    ]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend([
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
        "-map_metadata", "-1", str(destination),
    ])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        destination.unlink(missing_ok=True)
        raise VoiceAudioError("FFmpeg could not prepare the voice reference.") from exc

    try:
        prepared_info = probe_wav(destination)
        if prepared_info["codec"] != "pcm_s16le" or prepared_info["sample_rate_hz"] != 24000 or prepared_info["channels"] != 1:
            raise VoiceAudioError("Prepared audio was not converted to mono 24 kHz PCM16.")
        pcm, _ = decode_pcm16(destination)
        analysis = analyze_pcm16(pcm)
        if analysis["duration_seconds"] < 2:
            raise VoiceAudioError("The prepared reference is too short; keep at least 2 seconds of speech.")
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return {
        "source": source_info,
        "prepared": {
            "container": "wav",
            "codec": "pcm_s16le",
            "sample_rate_hz": 24000,
            "channels": 1,
            "bits_per_sample": 16,
            "duration_seconds": prepared_info["duration_seconds"],
        },
        "source_analysis": analyze_pcm16(decode_pcm16(source)[0]),
        "prepared_analysis": analysis,
        "processing": settings,
        "source_preserved": True,
        "conditioning_status": "not_tested",
    }
