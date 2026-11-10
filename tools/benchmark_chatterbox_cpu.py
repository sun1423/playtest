#!/usr/bin/env python3
"""Measure Chatterbox Multilingual V3 CPU inference without touching Studio data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_TEXT = {
    "hi": (
        "आज हम एक महत्वपूर्ण विषय को सरल भाषा में समझेंगे। "
        "छोटे-छोटे फैसले कई बार हमारे भविष्य की दिशा बदल देते हैं। "
        "इसलिए किसी भी निर्णय से पहले तथ्यों को समझना जरूरी है।"
    ),
    "en": (
        "Today we will explain an important subject in simple language. "
        "Small decisions can change the direction of our future. "
        "That is why it is important to understand the facts before deciding."
    ),
}


def read_proc_values() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, raw, *_ = line.split()
                values[key.rstrip(":")] = int(raw) / 1024
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                values["MemAvailable"] = int(line.split()[1]) / 1024
                break
    except (OSError, ValueError):
        pass
    return values


def peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-wav", required=True, type=Path,
                        help="A WAV voice reference the user is authorised to use.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/chatterbox-cpu-benchmark"))
    parser.add_argument("--threads", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--model", choices=("v3", "v2"), default="v3")
    parser.add_argument("--hi-text", default=DEFAULT_TEXT["hi"])
    parser.add_argument("--en-text", default=DEFAULT_TEXT["en"])
    parser.add_argument("--skip-hi", action="store_true")
    parser.add_argument("--skip-en", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 2:
        parser.error("--threads must be 1 or 2 for this 2-vCPU host")
    if args.skip_hi and args.skip_en:
        parser.error("At least one language must be benchmarked")
    return args


def main() -> int:
    args = parse_args()
    reference = args.reference_wav.expanduser().resolve()
    if not reference.is_file() or reference.suffix.lower() != ".wav":
        raise SystemExit("Reference must be an existing .wav file.")
    if reference.stat().st_size > 30 * 1024 * 1024:
        raise SystemExit("Reference WAV exceeds the Studio 30 MB voice-profile limit.")

    try:
        import soundfile as sf
        info = sf.info(str(reference))
    except Exception as exc:
        raise SystemExit(f"Could not read reference WAV: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0:
        raise SystemExit("Reference WAV contains no readable audio.")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = []
    if not args.skip_hi:
        languages.append(("hi", args.hi_text))
    if not args.skip_en:
        languages.append(("en", args.en_text))
    destinations = [args.output_dir / f"chatterbox_{lang}.wav" for lang, _ in languages]
    if not args.overwrite:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise SystemExit("Output already exists; use --overwrite: " + ", ".join(existing))

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Benchmark: Chatterbox Multilingual TTS on CPU only", flush=True)
    print(f"Host CPUs visible: {os.cpu_count() or 'unknown'}; Torch threads: {args.threads}", flush=True)
    print(f"Reference: {reference.name}; {info.duration:.1f}s; {info.samplerate}Hz; {info.channels} channel(s)", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)

    mem_before = read_proc_values()
    process_cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    started_load = time.perf_counter()

    import torch
    import soundfile as sf

    if torch.cuda.is_available():
        raise SystemExit("CUDA is available in this runtime; refusing to run a non-CPU benchmark.")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    print(f"PyTorch: {torch.__version__}; loading model {args.model} on CPU…", flush=True)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu", t3_model=args.model)
    load_seconds = time.perf_counter() - started_load
    mem_loaded = read_proc_values()
    print(f"Model loaded in {load_seconds:.1f}s; process peak RSS {peak_rss_mib():.0f} MiB", flush=True)

    results: list[dict[str, Any]] = []
    for index, (language, text) in enumerate(languages):
        destination = destinations[index]
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        mem_at_start = read_proc_values()
        wall_started = time.perf_counter()
        with torch.inference_mode():
            audio = model.generate(
                text,
                language_id=language,
                audio_prompt_path=str(reference),
            )
        generation_seconds = time.perf_counter() - wall_started
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = (
            (cpu_after.ru_utime - cpu_before.ru_utime)
            + (cpu_after.ru_stime - cpu_before.ru_stime)
        )

        waveform = audio.detach().float().cpu().numpy().squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise RuntimeError(f"{language} generation returned an invalid waveform shape: {waveform.shape}")
        if not bool(torch.isfinite(torch.from_numpy(waveform)).all()):
            raise RuntimeError(f"{language} generation returned NaN or infinite samples.")
        rms = float((waveform.astype("float64") ** 2).mean() ** 0.5)
        if rms < 1e-5:
            raise RuntimeError(f"{language} generation appears silent (RMS={rms:.8f}).")
        audio_seconds = waveform.size / float(model.sr)
        sf.write(str(destination), waveform, int(model.sr), subtype="PCM_16")
        rtf = generation_seconds / audio_seconds if audio_seconds else None
        result = {
            "language": language,
            "text_characters": len(text),
            "generation_seconds": round(generation_seconds, 3),
            "audio_seconds": round(audio_seconds, 3),
            "real_time_factor": round(rtf, 3) if rtf is not None else None,
            "cpu_seconds": round(cpu_seconds, 3),
            "cpu_utilization_percent": round(100 * cpu_seconds / generation_seconds, 1)
                if generation_seconds else None,
            "audio_rms": round(rms, 6),
            "sample_rate": int(model.sr),
            "wav_bytes": destination.stat().st_size,
            "output": str(destination),
            "memory_at_generation_start_mib": mem_at_start.get("VmRSS"),
            "available_host_memory_after_generation_mib": read_proc_values().get("MemAvailable"),
        }
        results.append(result)
        print(
            f"{language}: generated {audio_seconds:.1f}s audio in {generation_seconds:.1f}s "
            f"(RTF {rtf:.2f}x); WAV {destination.name}",
            flush=True,
        )

    process_cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    total_cpu_seconds = (
        (process_cpu_after.ru_utime - process_cpu_before.ru_utime)
        + (process_cpu_after.ru_stime - process_cpu_before.ru_stime)
    )
    report = {
        "benchmark": "chatterbox-multilingual-cpu",
        "model": args.model,
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "visible_cpu_count": os.cpu_count(),
        "torch_threads": args.threads,
        "reference": {
            "filename": reference.name,
            "duration_seconds": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "size_bytes": reference.stat().st_size,
        },
        "model_load_seconds": round(load_seconds, 3),
        "process_peak_rss_mib": round(peak_rss_mib(), 1),
        "memory_before_model_mib": mem_before,
        "memory_after_model_load_mib": mem_loaded,
        "total_process_cpu_seconds_including_load": round(total_cpu_seconds, 3),
        "language_results": results,
        "note": "The first run may include Hugging Face model download time. Repeat after the model cache is populated to separate download from cold-start latency.",
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
