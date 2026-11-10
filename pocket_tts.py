"""Safe CPU-only integration for Saryps Labs Pocket TTS Hindi (ONNX)."""

from __future__ import annotations

import json
import os
import unicodedata
import subprocess
import uuid
from pathlib import Path


def valid_voice_id(voice_id: object) -> bool:
    """Accept Unicode letters, numbers and combining marks without path syntax."""
    if not isinstance(voice_id, str) or not 1 <= len(voice_id) <= 48:
        return False
    if not voice_id[0].isalnum():
        return False
    return all(
        char.isalnum() or unicodedata.category(char).startswith("M") or char in "-_"
        for char in voice_id
    )


def normalize_voice_id(value: str) -> str:
    """Keep legacy punctuation-stripping while preserving Unicode combining marks."""
    if not isinstance(value, str):
        return ""
    return "".join(
        char for char in value
        if char.isalnum() or unicodedata.category(char).startswith("M") or char in "-_"
    )[:48]


DEFAULT_SEED = 1
DEFAULT_EOS_THRESHOLD = -2.0
DEFAULT_GENERATION_SETTINGS = {
    "temperature": 0.3,
    "flow_steps": 1,
    "eos_threshold": -2.0,
    "seed": 1,
}


def normalize_generation_settings(settings: dict | None = None) -> dict:
    """Validate the small set of Pocket TTS controls exposed in Voice Studio."""
    import math

    supplied = settings or {}
    if not isinstance(supplied, dict):
        raise ValueError("Voice generation settings must be an object.")
    result = dict(DEFAULT_GENERATION_SETTINGS)
    for key in result:
        if key in supplied:
            result[key] = supplied[key]
    for key in ("temperature", "eos_threshold"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {key} generation setting.")
        result[key] = float(value)
    if not 0.05 <= result["temperature"] <= 1.0:
        raise ValueError("Temperature must be between 0.05 and 1.0.")
    if not -6.0 <= result["eos_threshold"] <= 0.0:
        raise ValueError("EOS threshold must be between -6.0 and 0.0.")
    steps = result["flow_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 8:
        raise ValueError("Flow steps must be between 1 and 8.")
    seed = result["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be between 0 and 2147483647.")
    return result

REQUIRED_MODEL_FILES = (
    "onnx/flow_lm_flow_int8.onnx",
    "onnx/flow_lm_main_int8.onnx",
    "onnx/mimi_decoder_int8.onnx",
    "onnx/mimi_encoder.onnx",
    "onnx/text_conditioner.onnx",
    "onnx/tokenizer.model",
    "onnx/bos_before_voice.npy",
)


def engine_ready(enabled: bool, model_root: Path, binary_path: Path | None = None) -> bool:
    """Return True only when explicitly enabled and every required runtime file exists."""
    if not enabled:
        return False
    root = Path(model_root).resolve()
    binary = Path(binary_path or root / "onnx/runtime/pocket-tts").resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return False
    if not (binary.parent / "libonnxruntime.so").is_file():
        return False
    return all((root / relative).is_file() for relative in REQUIRED_MODEL_FILES)


def resolve_voice_reference(voices_dir: Path, voice_id: str, max_bytes: int) -> Path:
    """Resolve an uploaded WAV reference without allowing path traversal or symlinks out."""
    if not isinstance(voice_id, str) or not valid_voice_id(voice_id):
        raise ValueError("Choose a valid saved WAV voice profile.")
    root = Path(voices_dir).resolve()
    reference = (root / f"{voice_id}.wav").resolve()
    if root not in reference.parents or not reference.is_file():
        raise ValueError("The selected WAV voice profile was not found.")
    if reference.suffix.lower() != ".wav":
        raise ValueError("The selected voice profile must be a WAV file.")
    if reference.stat().st_size > max_bytes:
        raise ValueError("The selected voice profile exceeds the 30 MB limit.")
    with reference.open("rb") as source:
        header = source.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("The selected voice profile is not a valid RIFF/WAVE file.")
    return reference


def clear_voice_cache(voices_dir: Path, voice_id: str) -> None:
    """Remove this profile's embedding/KV cache when a WAV is replaced or deleted."""
    if not isinstance(voice_id, str) or not valid_voice_id(voice_id):
        raise ValueError("Invalid voice profile ID.")
    root = Path(voices_dir).resolve()
    cache_dir = (root / ".cache").resolve()
    if root not in cache_dir.parents or not cache_dir.is_dir():
        return
    for extension in ("emb", "kv"):
        cached = (cache_dir / f"{voice_id}.{extension}").resolve()
        if cache_dir in cached.parents and cached.is_file():
            cached.unlink()


def build_command(
    text: str,
    voice_name: str,
    model_root: Path,
    voices_dir: Path,
    binary_path: Path | None = None,
    settings: dict | None = None,
) -> list[str]:
    """Build the tested two-thread, INT8 CLI command; keep voice caches enabled."""
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        raise ValueError("Hindi narration must contain 1–12,000 characters.")
    if Path(voice_name).name != voice_name or not voice_name.lower().endswith(".wav"):
        raise ValueError("Invalid voice reference filename.")
    root = Path(model_root).resolve()
    binary = Path(binary_path or root / "onnx/runtime/pocket-tts").resolve()
    generation = normalize_generation_settings(settings)
    return [
        str(binary),
        "--precision", "int8",
        "--models-dir", str(root / "onnx"),
        "--tokenizer", str(root / "onnx/tokenizer.model"),
        "--voices-dir", str(Path(voices_dir).resolve()),
        "--threads", "2",
        "--temperature", str(generation["temperature"]),
        "--lsd-steps", str(generation["flow_steps"]),
        "--eos-threshold", str(generation["eos_threshold"]),
        "--seed", str(generation["seed"]),
        "--no-append-terminal-punctuation",
        text.strip(),
        voice_name,
    ]


def _probe_pcm16_wav(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    if not streams:
        raise RuntimeError("Hindi narration output contains no audio stream.")
    stream = streams[0]
    if (
        stream.get("codec_name") != "pcm_s16le"
        or str(stream.get("sample_rate")) != "24000"
        or int(stream.get("channels", 0)) != 1
    ):
        raise RuntimeError("Hindi narration output is not normalized 24 kHz mono PCM16.")
    duration = float(metadata.get("format", {}).get("duration", 0))
    if duration <= 0:
        raise RuntimeError("Hindi narration output has an invalid duration.")
    return duration


def synthesize(
    text: str,
    voice_id: str,
    voices_dir: Path,
    model_root: Path,
    destination: Path,
    max_voice_bytes: int,
    binary_path: Path | None = None,
    settings: dict | None = None,
) -> float:
    """Generate and safely normalize a mono 24 kHz PCM16 WAV using CPU-only ONNX."""
    reference = resolve_voice_reference(voices_dir, voice_id, max_voice_bytes)
    root = Path(model_root).resolve()
    binary = Path(binary_path or root / "onnx/runtime/pocket-tts").resolve()
    if not engine_ready(True, root, binary):
        raise RuntimeError("Pocket TTS Hindi model or native CPU runtime is incomplete.")
    command = build_command(text, reference.name, root, voices_dir, binary, settings)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    float_output = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.float.wav"
    )
    try:
        try:
            result = subprocess.run(
                [*command, str(float_output)],
                cwd=str(root),
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Hindi narration exceeded the 15-minute safety limit.") from exc
        if result.returncode != 0 or not float_output.is_file() or float_output.stat().st_size < 44:
            raise RuntimeError("Pocket TTS Hindi failed to generate a valid audio file.")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(float_output),
                    "-af", "alimiter=limit=0.89:attack=5:release=50:level=0",
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", "-threads", "1",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Hindi narration audio normalization timed out.") from exc
        if not destination.is_file() or destination.stat().st_size < 44:
            raise RuntimeError("Hindi narration audio normalization failed.")
        return _probe_pcm16_wav(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        float_output.unlink(missing_ok=True)
