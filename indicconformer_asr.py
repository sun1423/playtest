"""CPU Hindi ASR using the AI4Bharat IndicConformer 120M CTC ONNX export."""

from __future__ import annotations

import json
import os
import threading
import time
import unicodedata
from pathlib import Path

_SESSION_LOCK = threading.Lock()
_SESSION = None
_SESSION_KEY = None
_VOCAB = None


def contains_non_devanagari_indic_script(text: str) -> bool:
    """Detect Urdu/Arabic output so Hindi transcripts are never silently accepted."""
    for char in text:
        code = ord(char)
        if 0x0600 <= code <= 0x08FF and unicodedata.category(char)[0] in {"L", "M"}:
            return True
    return False


def ctc_greedy_decode(logits, vocab: list[str]) -> str:
    """Collapse CTC repeats and blanks, then restore SentencePiece word boundaries."""
    import numpy as np

    values = np.asarray(logits)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError("Hindi ASR expects one audio clip per inference.")
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected IndicConformer output shape: {values.shape}")
    blank_id = len(vocab)
    token_ids = np.argmax(values, axis=-1)
    pieces: list[str] = []
    previous = -1
    for raw_id in token_ids:
        token_id = int(raw_id)
        if token_id != previous and token_id != blank_id:
            if 0 <= token_id < len(vocab):
                pieces.append(vocab[token_id])
        previous = token_id
    return "".join(pieces).replace("\u2581", " ").strip()


def _paths(model_cache: Path) -> tuple[Path, Path]:
    configured = os.getenv("INDICCONFORMER_HI_ROOT", "").strip()
    root = Path(configured).expanduser() if configured else Path(model_cache).resolve().parent / "indicconformer-hi"
    return root / "model.onnx", root / "vocab.json"


def _load_model(model_cache: Path):
    global _SESSION, _SESSION_KEY, _VOCAB
    import onnxruntime as ort

    model_path, vocab_path = _paths(model_cache)
    if not model_path.is_file() or not vocab_path.is_file():
        raise RuntimeError(
            "IndicConformer Hindi is not installed. Run "
            "python scripts/install_indicconformer_asr.py on the Studio server."
        )
    key = (str(model_path), model_path.stat().st_mtime_ns, model_path.stat().st_size)
    with _SESSION_LOCK:
        if _SESSION is None or _SESSION_KEY != key:
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(os.cpu_count() or 2, 2))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            inputs = {item.name for item in session.get_inputs()}
            if not {"audio_signal", "length"}.issubset(inputs):
                raise RuntimeError(
                    "The installed IndicConformer model has unexpected inputs: "
                    + ", ".join(sorted(inputs))
                )
            with vocab_path.open("r", encoding="utf-8") as source:
                vocab = json.load(source)
            if not isinstance(vocab, list) or len(vocab) < 1000 or not all(
                isinstance(token, str) for token in vocab
            ):
                raise RuntimeError("IndicConformer vocabulary is invalid.")
            _SESSION = session
            _SESSION_KEY = key
            _VOCAB = vocab
    return _SESSION, _VOCAB


def transcribe_indicconformer_hindi(audio_path: Path, model_cache: Path) -> dict:
    """Transcribe a prepared WAV with the Hindi-specific ONNX model on CPU."""
    import numpy as np
    import librosa

    session, vocab = _load_model(model_cache)
    audio, sample_rate = librosa.load(
        str(Path(audio_path).resolve()), sr=16000, mono=True, dtype=np.float32
    )
    if audio.size < 1600:
        raise ValueError("The prepared recording is too short for Hindi transcription.")

    # Match the ONNX export's NeMo preprocessing recipe.
    preemphasized = np.empty_like(audio, dtype=np.float32)
    preemphasized[0] = audio[0]
    preemphasized[1:] = audio[1:] - 0.97 * audio[:-1]
    mel = librosa.feature.melspectrogram(
        y=preemphasized,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hann",
        center=True,
        pad_mode="constant",
        power=2.0,
        n_mels=80,
        fmin=0.0,
        fmax=8000.0,
        htk=False,
        norm="slaney",
    )
    log_mel = np.log(mel + (2.0 ** -24)).astype(np.float32)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, ddof=1, keepdims=True) + 1e-5
    features = ((log_mel - mean) / std).astype(np.float32)
    feature_batch = features[np.newaxis, :, :]
    lengths = np.asarray([features.shape[1]], dtype=np.int64)

    started = time.perf_counter()
    with _SESSION_LOCK:
        logits = session.run(
            None, {"audio_signal": feature_batch, "length": lengths}
        )[0]
    elapsed = round(time.perf_counter() - started, 3)
    transcript = ctc_greedy_decode(logits, vocab)
    warning = (
        "Hindi recognizer returned Urdu/Arabic characters. Review the transcript "
        "before saving the voice profile."
        if contains_non_devanagari_indic_script(transcript)
        else ""
    )
    if not transcript:
        warning = warning or "No speech could be recognised. Check the recording and try again."
    return {
        "text": transcript,
        "language": "hi",
        "model": "indicconformer-hi-120m-ctc",
        "engine": "AI4Bharat IndicConformer Hindi · ONNX CPU",
        "transcription_attempts": 1,
        "inference_seconds": elapsed,
        "script_warning": warning,
    }
