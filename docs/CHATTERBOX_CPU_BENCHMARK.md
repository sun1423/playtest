# Chatterbox CPU benchmark (isolated)

This is a validation tool only. It does not alter the Studio service, database, voice library, Ollama configuration, or production environment. Run it in a separate Python 3.11 virtual environment on the Azure **2-vCPU / 15-GB RAM CPU host**. It explicitly refuses to run if CUDA is available.

## Why benchmark first

The current Studio worker already calls Chatterbox Multilingual V3 on `device="cpu"` for standalone TTS jobs, but TTS is disabled in production. This benchmark measures model load time, Hindi and English voice-conditioned generation, real-time factor (RTF), process peak RSS, host available memory, and CPU time before any integration is enabled.

Chatterbox's public README lists Multilingual V3 as a 500M multilingual model supporting Hindi and English. It also documents a smaller Nano model as English-only, so Nano is not a drop-in replacement for Hindi narration. The project README currently says the repository is retired research and is no longer maintained. Review the license and upstream status before shipping it. Sources: [Chatterbox README](https://github.com/resemble-ai/chatterbox) · [Hindi V3 model card](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-hi).

## Install in a separate environment

Do **not** install these packages into `/opt/ai-video-studio/.venv`. Keep Chatterbox isolated while we establish its memory and runtime cost.

```bash
python3.12 -m venv /opt/chatterbox-cpu-venv
source /opt/chatterbox-cpu-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r /opt/ai-video-studio/requirements-chatterbox-cpu.txt

# Install the archived V3-capable source at the exact upstream revision.
git clone --filter=blob:none https://github.com/resemble-ai/chatterbox.git /opt/chatterbox-source
git -C /opt/chatterbox-source checkout ab55923c27757c90cfcc407baa57dc25a7d25217
python -m pip install --no-deps /opt/chatterbox-source
```

The CPU wheels are selected explicitly so pip does not install a CUDA build. Do not install the PyPI 0.1.7 wheel for this V3 test: that wheel predates V3 checkpoint support. The pinned source revision includes the V3 loader. The package is installed without its full dependency bundle because its Gradio UI is not needed for the benchmark. The benchmark environment is separate from the API and worker virtual environment. Python 3.12 is used because that is the system Python on this Azure host; the pinned Torch 2.6.0 wheels support Python 3.12.

## Run against an authorised reference WAV

Use a voice recording whose speaker has explicitly consented to voice synthesis. The reference WAV must be readable and no larger than the Studio 30-MB voice-profile limit.

```bash
# Built-in reference voice baseline (can run before a personal WAV is provided)
/opt/chatterbox-cpu-venv/bin/python /opt/ai-video-studio/tools/benchmark_chatterbox_cpu.py \
  --output-dir /tmp/chatterbox-cpu-benchmark \
  --threads 2

# Later, test a saved voice profile after confirming the speaker authorised it:
/opt/chatterbox-cpu-venv/bin/python /opt/ai-video-studio/tools/benchmark_chatterbox_cpu.py \
  --reference-wav /path/to/authorised-reference.wav \
  --output-dir /tmp/chatterbox-custom-voice-benchmark \
  --threads 2
```

The output directory contains Hindi and English WAV samples and `benchmark.json`. Without `--reference-wav`, it tests the model's built-in reference conditioning; this is only a CPU baseline, not the custom-voice test. Use `--reference-wav` to test an authorised saved voice profile. Listen to both samples; timing and non-silent audio are not sufficient proof of pronunciation or speaker similarity. If the built-in conditioning is absent in the pinned model snapshot, the baseline run will fail clearly and the custom WAV test will be required. The text can be overridden with `--hi-text` and `--en-text`. Re-run after the first model download to measure warm-cache model loading. Use `--overwrite` to replace an earlier benchmark output.

## Decision gate before integration

Do not enable `ENABLE_TTS` or modify the production service based only on a successful import. Review:

- Full host memory headroom while API, worker and Ollama are in their normal state; no swap storm or OOM.
- Peak RSS and whether a 60–180 second narration completes reliably.
- Hindi pronunciation, English pronunciation, and similarity to the selected authorised voice.
- Cold model load time, subsequent load time, generation RTF and CPU contention with Ollama/FFmpeg.

Only after the host is reachable and these measurements are reviewed should we integrate narration into storyboard rendering. The planned workflow is to use the exact storyboard narration text, synthesize one WAV using the selected voice profile, then combine narration and optional background music with the rendered video through FFmpeg. No GPU or external inference service is part of this plan.
