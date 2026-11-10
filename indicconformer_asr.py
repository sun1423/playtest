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


def _read_audio_16k(audio_path: Path):
    """Decode any prepared audio file through FFmpeg into mono float32 16 kHz PCM."""
    import numpy as np
    import subprocess

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(Path(audio_path).resolve()), "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le",
        "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, timeout=120
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Hindi ASR audio decoding exceeded its time limit.") from exc
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(detail[-1000:] or "FFmpeg could not decode the Hindi reference WAV.") from exc
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("The Hindi reference WAV did not decode to valid audio.")
    return audio


def _hz_to_mel_slaney(frequencies):
    import numpy as np

    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mel = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    mel[log_region] = min_log_mel + np.log(
        frequencies[log_region] / min_log_hz
    ) / logstep
    return mel


def _mel_to_hz_slaney(mels):
    import numpy as np

    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    frequencies = mels * f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    frequencies[log_region] = min_log_hz * np.exp(
        logstep * (mels[log_region] - min_log_mel)
    )
    return frequencies


def _nemo_mel_filterbank(sample_rate: int = 16000, n_fft: int = 512, n_mels: int = 80):
    """Build the Slaney-normalized triangular bank used by the ONNX model card."""
    import numpy as np

    mel_edges = np.linspace(
        _hz_to_mel_slaney(np.asarray([0.0]))[0],
        _hz_to_mel_slaney(np.asarray([sample_rate / 2.0]))[0],
        n_mels + 2,
    )
    hz_edges = _mel_to_hz_slaney(mel_edges)
    fft_frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    ramps = hz_edges[:, np.newaxis] - fft_frequencies[np.newaxis, :]
    lower = -ramps[:-2] / (hz_edges[1:-1] - hz_edges[:-2])[:, np.newaxis]
    upper = ramps[2:] / (hz_edges[2:] - hz_edges[1:-1])[:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (hz_edges[2:] - hz_edges[:-2]))[:, np.newaxis]
    return weights.astype(np.float32)


def nemo_mel_spectrogram(audio, sample_rate: int = 16000):
    """Return NeMo-compatible log-mel features with shape [80, T]."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != 16000:
        raise ValueError("IndicConformer Hindi expects audio sampled at 16 kHz.")
    if audio.size == 0:
        raise ValueError("The Hindi audio recording is empty.")

    preemphasized = np.empty_like(audio)
    preemphasized[0] = audio[0]
    preemphasized[1:] = audio[1:] - 0.97 * audio[:-1]

    n_fft, hop_length, win_length = 512, 160, 400
    padded = np.pad(preemphasized, (n_fft // 2, n_fft // 2), mode="constant")
    frame_count = 1 + (padded.size - n_fft) // hop_length
    shape = (frame_count, n_fft)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides, writeable=False
    )

    # librosa/NeMo use a periodic Hann window, centered in the 512-sample FFT.
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(win_length, dtype=np.float64) / win_length
    )
    fft_window = np.zeros(n_fft, dtype=np.float32)
    offset = (n_fft - win_length) // 2
    fft_window[offset:offset + win_length] = window.astype(np.float32)

    spectrum = np.fft.rfft(
        frames.astype(np.float32) * fft_window[np.newaxis, :], axis=1
    )
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    mel = _nemo_mel_filterbank(sample_rate, n_fft) @ power.T
    log_mel = np.log(mel + (2.0 ** -24)).astype(np.float32)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, ddof=1, keepdims=True) + 1e-5
    return ((log_mel - mean) / std).astype(np.float32)


def transcribe_indicconformer_hindi(audio_path: Path, model_cache: Path) -> dict:
    """Transcribe a prepared WAV with the Hindi-specific ONNX model on CPU."""
    import numpy as np

    session, vocab = _load_model(model_cache)
    audio = _read_audio_16k(audio_path)
    if audio.size < 1600:
        raise ValueError("The prepared recording is too short for Hindi transcription.")
    features = nemo_mel_spectrogram(audio)
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
