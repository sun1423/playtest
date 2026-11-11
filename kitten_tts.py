"""KittenTTS 2 CPU/C++ backend; kept separate from Pocket TTS."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path

from pocket_tts import resolve_voice_reference, valid_voice_id


MAX_TEXT_CHARS = 100_000
MAX_TEXT_ARG_BYTES = 100_000


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"KittenTTS configuration is unreadable: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"KittenTTS configuration is invalid: {Path(path).name}")
    return value


def _safe_asset(root: Path, descriptor: dict) -> Path:
    relative = Path(str(descriptor.get("file", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("KittenTTS manifest contains an unsafe asset path.")
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if base not in path.parents:
        raise RuntimeError("KittenTTS asset path escapes the configured model directory.")
    if not path.is_file():
        raise RuntimeError(f"KittenTTS asset is missing: {relative.as_posix()}")
    expected_size = descriptor.get("size")
    if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
        raise RuntimeError(f"KittenTTS asset size mismatch: {relative.as_posix()}")
    return path


def _model_configuration(model_root: Path, decoder: str) -> tuple[dict, dict]:
    config = _read_json(Path(model_root) / "config.json")
    if config.get("type") != "KITTEN2":
        raise RuntimeError("The configured model directory is not a KittenTTS 2 checkpoint.")
    cpp = config.get("cpp")
    if not isinstance(cpp, dict) or cpp.get("version") != 1:
        raise RuntimeError("This KittenTTS model does not contain the supported C++ CPU asset manifest.")
    decoders = cpp.get("decoders")
    if not isinstance(decoders, dict) or decoder not in decoders:
        raise RuntimeError(f"KittenTTS decoder is unavailable: {decoder}")
    return config, decoders[decoder]


def engine_ready(
    enabled: bool,
    binary_path: Path,
    model_root: Path,
    prep_python: Path,
    prep_script: Path,
    python_package: Path,
    decoder: str = "default",
    torch_lib_dir: Path | None = None,
) -> bool:
    """Check local installation only. Never downloads or initializes model weights."""
    if not enabled:
        return False
    try:
        binary = Path(binary_path).resolve()
        root = Path(model_root).resolve()
        python = Path(prep_python).resolve()
        script = Path(prep_script).resolve()
        package = Path(python_package).resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            return False
        if not python.is_file() or not os.access(python, os.X_OK) or not script.is_file():
            return False
        if not (package / "kittenml" / "kittentts2" / "speaker.py").is_file():
            return False
        if not (package / "kittenml" / "kittentts2" / "vocoder.py").is_file():
            return False
        config, decoder_assets = _model_configuration(root, decoder)
        _safe_asset(root, config["cpp"]["gguf"])
        _safe_asset(root, decoder_assets["torchscript"])
        _safe_asset(root, decoder_assets["voices"])
        for required in (
            root / str(config["speaker_embedding"]),
            root / str(config["lm_dir"]) / "config.json",
            root / str(config["lm_packed"]),
        ):
            if not required.is_file():
                return False
        if torch_lib_dir is not None:
            library_dir = Path(torch_lib_dir).resolve()
            if not library_dir.is_dir() or not any(library_dir.glob("libtorch*.so")):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False


def validate_text(text: str) -> str:
    """Keep recognised emotion, vocal-event and emphasis markup intact."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter text for KittenTTS 2 to speak.")
    value = text.strip()
    if len(value) > MAX_TEXT_CHARS:
        raise ValueError("KittenTTS text exceeds the 100,000-character limit.")
    if len(value.encode("utf-8")) > MAX_TEXT_ARG_BYTES:
        raise ValueError(
            "This script is too large for one KittenTTS CPU request. "
            "Use shorter sections so expression tags stay with the intended lines."
        )
    return value


@contextmanager
def _engine_lock(profile_root: Path):
    profile_root = Path(profile_root).resolve()
    profile_root.mkdir(parents=True, exist_ok=True)
    lock_path = profile_root / ".registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_runtime_assets(model_root: Path, profile_root: Path, decoder: str) -> tuple[dict, dict, Path]:
    model_root = Path(model_root).resolve()
    profile_root = Path(profile_root).resolve()
    config, decoder_assets = _model_configuration(model_root, decoder)
    cpp = config["cpp"]
    registry_descriptor = decoder_assets["voices"]
    base_registry = _safe_asset(model_root, registry_descriptor)
    runtime_registry = profile_root / registry_descriptor["file"]
    runtime_registry.parent.mkdir(parents=True, exist_ok=True)

    for descriptor in (cpp["gguf"], decoder_assets["torchscript"]):
        source = _safe_asset(model_root, descriptor)
        destination = profile_root / descriptor["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source:
                destination.unlink()
                destination.symlink_to(source)
        elif not destination.exists():
            destination.symlink_to(source)

    config_hash = hashlib.sha256((model_root / "config.json").read_bytes()).hexdigest()
    marker = profile_root / ".base-config-sha256"
    existing_hash = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if existing_hash != config_hash or not runtime_registry.is_file():
        temporary = runtime_registry.with_name(f".{runtime_registry.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(base_registry, temporary)
            os.replace(temporary, runtime_registry)
        finally:
            temporary.unlink(missing_ok=True)
        marker.write_text(config_hash, encoding="utf-8")

    runtime_config = dict(config)
    runtime_config["cpp"] = dict(cpp)
    runtime_config["cpp"]["decoders"] = dict(cpp["decoders"])
    selected = dict(decoder_assets)
    selected["voices"] = dict(registry_descriptor)
    selected["voices"]["size"] = runtime_registry.stat().st_size
    runtime_config["cpp"]["decoders"][decoder] = selected
    runtime_config["default_decoder"] = decoder
    _atomic_json(profile_root / "config.json", runtime_config)
    return config, decoder_assets, runtime_registry


def _prepare_voice(
    voice_id: str,
    reference: Path,
    transcript: str,
    model_root: Path,
    profile_root: Path,
    python: Path,
    prepare_script: Path,
    python_package: Path,
    decoder: str,
    revision: str,
    threads: int,
) -> None:
    if not valid_voice_id(voice_id):
        raise ValueError("Choose a valid saved voice profile.")
    transcript = str(transcript or "").strip()
    if not transcript:
        raise ValueError(
            "KittenTTS 2 needs the exact transcript of the saved voice recording. "
            "Edit the profile transcript in Voice Studio and try again."
        )
    reference = Path(reference).resolve()
    with wave.open(str(reference), "rb") as wav:
        duration = wav.getnframes() / float(wav.getframerate())
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise ValueError("KittenTTS voice reference must be mono 24 kHz PCM16 WAV.")
        if not 5.0 <= duration <= 30.0:
            raise ValueError("KittenTTS voice cloning needs a 5–30 second reference clip.")

    config, _, registry = _ensure_runtime_assets(model_root, profile_root, decoder)
    config_hash = hashlib.sha256((Path(model_root) / "config.json").read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(
        (_sha256_file(reference) + "\0" + transcript + "\0" + revision + "\0" + decoder + "\0" + config_hash).encode("utf-8")
    ).hexdigest()
    metadata_dir = Path(profile_root) / ".profiles"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{voice_id}.json"
    if metadata_path.is_file():
        try:
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == fingerprint and voice_id in _read_json(registry):
                return
        except (OSError, ValueError, TypeError):
            pass

    with tempfile.TemporaryDirectory(prefix="kitten-profile-", dir=str(profile_root)) as temp_name:
        stage = Path(temp_name)
        stage_config = dict(config)
        stage_config["cpp_decoder"] = decoder
        (stage / "config.json").write_text(
            json.dumps(stage_config, ensure_ascii=False), encoding="utf-8"
        )
        existing_voices = _read_json(registry)
        if voice_id in existing_voices and not metadata_path.is_file():
            raise ValueError(
                "This voice profile name conflicts with a built-in KittenTTS voice. "
                "Rename the saved Studio voice profile and try again."
            )
        existing_voices.pop(voice_id, None)
        (stage / "voices.json").write_text(
            json.dumps(existing_voices, ensure_ascii=False), encoding="utf-8"
        )
        command = [
            str(python), str(prepare_script),
            "--repo", str(Path(model_root).resolve()),
            "--assets", str(stage),
            "--python-package", str(Path(python_package).resolve()),
            "--reference", str(reference),
            "--transcript", transcript,
            "--name", voice_id,
            "--threads", str(max(1, min(int(threads), 2))),
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(max(1, min(int(threads), 2)))
        env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
        env["OPENBLAS_NUM_THREADS"] = "1"
        hf_home = os.getenv("KITTEN_TTS_HF_HOME")
        if hf_home:
            env["HF_HOME"] = str(Path(hf_home).resolve())
        result = subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=1800, env=env,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Reference preparation failed.").strip()
            raise RuntimeError(f"KittenTTS voice preparation failed: {detail[-1600:]}")
        prepared_voices = _read_json(stage / "voices.json")
        if voice_id not in prepared_voices:
            raise RuntimeError("KittenTTS did not create the expected cloned voice.")

        temporary_registry = registry.with_name(f".{registry.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_registry.write_text(
                json.dumps(prepared_voices, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary_registry, registry)
        finally:
            temporary_registry.unlink(missing_ok=True)

    _ensure_runtime_assets(model_root, profile_root, decoder)
    _atomic_json(metadata_path, {
        "voice_id": voice_id,
        "fingerprint": fingerprint,
        "decoder": decoder,
        "model_revision": revision,
        "prepared_at": int(__import__("time").time()),
    })


def synthesize(
    text: str,
    voice_id: str,
    voices_dir: Path,
    model_root: Path,
    destination: Path,
    max_voice_bytes: int,
    metadata: dict,
    binary_path: Path,
    prep_python: Path,
    prep_script: Path,
    python_package: Path,
    profile_root: Path,
    torch_lib_dir: Path | None = None,
    decoder: str = "default",
    revision: str = "main",
    threads: int = 2,
) -> float:
    text = validate_text(text)
    reference = resolve_voice_reference(voices_dir, voice_id, max_voice_bytes)
    transcript = str((metadata or {}).get("transcript") or "").strip()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    float_output = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.float.wav")
    report_path = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.json")
    thread_count = max(1, min(int(threads), 2))

    with _engine_lock(profile_root):
        try:
            _prepare_voice(
                voice_id, reference, transcript, model_root, profile_root,
                prep_python, prep_script, python_package, decoder, revision, thread_count,
            )
            command = [
                str(Path(binary_path).resolve()),
                "--assets", str(Path(profile_root).resolve()),
                "--decoder", decoder,
                "--text", text,
                "--voice", voice_id,
                "--threads", str(thread_count),
                "--decoder-threads", str(thread_count),
                "--preset", "expressive",
                "--no-normalize",
                "--no-repack",
                "--output", str(float_output),
                "--report", str(report_path),
            ]
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(thread_count)
            env["MKL_NUM_THREADS"] = str(thread_count)
            env["OPENBLAS_NUM_THREADS"] = "1"
            hf_home = os.getenv("KITTEN_TTS_HF_HOME")
            if hf_home:
                env["HF_HOME"] = str(Path(hf_home).resolve())
            if torch_lib_dir:
                library_dir = str(Path(torch_lib_dir).resolve())
                env["LD_LIBRARY_PATH"] = library_dir + (
                    ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
                )
            result = subprocess.run(
                command, check=False, capture_output=True, text=True,
                timeout=900, env=env,
            )
            if result.returncode != 0 or not float_output.is_file() or float_output.stat().st_size < 44:
                detail = (result.stderr or result.stdout or "KittenTTS inference failed.").strip()
                raise RuntimeError(f"KittenTTS 2 synthesis failed: {detail[-1600:]}")
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(float_output),
                    "-af", "alimiter=limit=0.89:attack=5:release=50:level=0",
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", "-threads", "1",
                    str(destination),
                ],
                check=True, capture_output=True, text=True, timeout=120,
            )
            with wave.open(str(destination), "rb") as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
                    raise RuntimeError("KittenTTS output did not normalize to mono 24 kHz PCM16.")
                duration = wav.getnframes() / float(wav.getframerate())
            if duration <= 0:
                raise RuntimeError("KittenTTS generated an empty audio file.")
            return duration
        except subprocess.TimeoutExpired as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError("KittenTTS CPU generation exceeded its safety timeout.") from exc
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            float_output.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)


def remove_voice_profile(
    voice_id: str, profile_root: Path, model_root: Path, decoder: str = "default"
) -> None:
    """Remove only this Studio clone from the shared Kitten registry."""
    if not valid_voice_id(voice_id):
        return
    profile_root = Path(profile_root).resolve()
    metadata_path = profile_root / ".profiles" / f"{voice_id}.json"
    registry = profile_root / "cpp" / decoder / "voices.json"
    try:
        with _engine_lock(profile_root):
            if metadata_path.is_file() and registry.is_file():
                voices = _read_json(registry)
                voices.pop(voice_id, None)
                _atomic_json(registry, voices)
    except (OSError, ValueError, RuntimeError):
        pass
    finally:
        metadata_path.unlink(missing_ok=True)
