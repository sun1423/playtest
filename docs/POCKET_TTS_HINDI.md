# Pocket TTS Hindi — CPU voiceover

## Status

Pocket TTS Hindi is the selected CPU-only Hindi voice engine for this Studio branch. It uses the model's native C++/ONNX runtime; it does not use PyTorch, CUDA, a GPU, a remote inference API, or the model's unauthenticated HTTP server.

**It is disabled by default.** The production API and worker must both have `ENABLE_TTS=1` and the model/runtime installed before the health endpoint reports `tts_ready: true`. This branch does not change any production environment setting or install model weights.

This release is **Hindi-only**. The current model does not provide English narration. The project is a research release, not a production-certified speech system; listen to generated speech before publishing.

## Why this engine

On the current Azure host (2 vCPU, AMD EPYC 9V74, 15 GiB RAM, no GPU), the bundled INT8 ONNX runtime was tested with two inference threads:

| Test | Speech produced | Generation time | Speed |
|---|---:|---:|---:|
| Short first-run sentence | 2.00 s | 1.13 s | 1.77× real time |
| 10-second script | 10.00 s | 2.47 s | 4.04× |
| Longer article | 25.52 s | 5.19 s | 4.92× |
| Warm-cache short repeat | 1.92 s | 0.39 s | 4.94× |

Observed cold model load was about 0.42 s. First audio chunk on cold voice conditioning took about 0.8 s; a cached repeat was about 38 ms. Peak RSS for the longer narration process was about 634 MiB. The model's voice embedding and KV conditioning caches reduced repeat work and are intentionally left enabled.

These measurements used the model's published sample WAV as the reference. The production branch now uses a fixed seed and the more concise EOS threshold validated in the bundle's own release record; the random default seed could produce markedly longer output for the same short phrase.

## Azure custom-WAV validation (10 November 2026)

A separate, isolated native CLI benchmark used the existing 20-second mono 24 kHz profile `b1.wav` from the Studio voice library. The source profile was copied into a temporary benchmark folder; the original was not modified. Host: 2 vCPU, 15 GiB RAM, CPU-only. Settings: INT8, two threads, temperature 0.3, one flow step, EOS threshold `-2.0`, deterministic seed `1`.

| Test | Generated audio | Total wall time | End-to-end RTF |
|---|---:|---:|---:|
| Short Hindi phrase | 3.84 s | 1.23 s | 3.1× |
| Longer 180-character Hindi script | 13.28 s | 3.01 s | 4.4× |

Three identical short-phrase runs with the same warmed voice cache produced byte-identical 3.84-second WAV files (matching SHA-256). The native runtime reported roughly 5.0–5.2× RTF for generation alone; direct CLI wall time includes model loading.

The exact branch helper was measured separately across a fresh voice cache and warm repeats:
- First run with a newly created voice cache: 6.85 seconds wall time for 3.84 seconds of audio. Peak child-process RSS was approximately 2.53 GiB; this includes initial speaker conditioning and cache creation.
- Three subsequent warm-cache helper runs: 1.35 seconds average wall time for the same 3.84-second audio. Warm child-process peak RSS was approximately 418 MiB.
- The fresh-cache first output and warm-cache output had different WAV hashes despite identical duration; the three warm repeats were byte-identical. Human listening remains necessary to compare voice quality across cache states.
The exact branch helper (`pocket_tts.synthesize()`) was also exercised with the copied profile. Its first run with a fresh per-profile cache took 6.83 seconds for the 3.84-second short output (cold voice conditioning/cache creation). Three subsequent short repeats took 1.39 seconds on average including CLI invocation, WAV normalization and output probing; all had identical 3.84-second duration and PCM16 format. A longer 180-character script completed through the helper in 3.11 seconds and returned a 13.28-second mono 24 kHz PCM16 WAV.

An isolated FFmpeg storyboard smoke test then mixed that long narration under a 5-second synthetic visual clip, extended the last frame by 8.28 seconds, and added a low-level synthetic music bed. The resulting MP4 was 13.28 seconds with H.264 video and AAC stereo audio at 48 kHz. This validates the audio-mix/extend path, not visual quality or music balancing in a real storyboard.

A seed sweep with the random/default seed produced 3.84–13.68 seconds for the same short text, so production synthesis now pins seed `1` and EOS `-2.0` to avoid that duration instability. This is a deterministic operating point validated on one saved WAV, not a universal claim about all voices or scripts.

The custom-WAV check proves that this WAV is accepted for conditioning and that audio is generated. It does **not** establish speaker similarity or pronunciation quality: a person must listen to the output and compare it with the reference before using it in a published video. The model is research-grade and has known weaker cases including very short isolated utterances, punctuation-heavy text, numbers, and Latin-script content. Its training data had effectively no punctuation coverage, so the Studio removes punctuation and decorative characters before synthesis and splits long text only at word boundaries. Prefer clear, plain Hindi script.

## Model and runtime provenance

- Model repository: [saryps-labs/pocket-tts-hindi](https://huggingface.co/saryps-labs/pocket-tts-hindi)
- Pinned model revision: `6bff9c1e776bb90c68dd05c2467c62d1ecdf1b51`
- Model weights: CC BY 4.0. Attribute as **“Pocket TTS Hindi by Saryps Labs”**, link the CC BY 4.0 licence, and note modifications when redistributing/adapting weights.
- Bundled PocketTTS.cpp runtime: MIT; retain its runtime `LICENSE`.
- The model repository's `NOTICE.md` contains component and dataset attribution; retain it beside the downloaded model files.
- The ONNX files are an INT8 CPU-oriented conversion. The bundled runtime includes compatibility code for the Hindi model's BOS-before-voice embedding.

Do not commit model files, downloaded datasets, voice profiles, cache files, or generated audio to Git.

## Build and install (manual, administrator-only)

The model is approximately 230 MiB for the ONNX bundle. The first native build downloads ONNX Runtime 1.23.2 and pinned C++ dependencies. Keep all files in the shared Studio data/model location, not in the Git checkout.

### 1. Prepare the separate build tools environment

Use a separate venv so no build dependencies enter the production API/worker Python environment:

```bash
sudo mkdir -p /srv/ai-video-studio/models/pocket-tts-hindi
sudo chown -R aivideo:aivideo /srv/ai-video-studio/models/pocket-tts-hindi
python3.12 -m venv /opt/pocket-tts-build-venv
/opt/pocket-tts-build-venv/bin/python -m pip install --upgrade pip
/opt/pocket-tts-build-venv/bin/python -m pip install cmake==3.31.6 huggingface_hub
```

### 2. Download the pinned Hindi ONNX bundle

```bash
/opt/pocket-tts-build-venv/bin/hf download saryps-labs/pocket-tts-hindi \
  --revision 6bff9c1e776bb90c68dd05c2467c62d1ecdf1b51 \
  --include 'onnx/*' \
  --include 'MODEL_LICENSE.md' \
  --include 'NOTICE.md' \
  --local-dir /srv/ai-video-studio/models/pocket-tts-hindi
```

Review and retain `MODEL_LICENSE.md` and `NOTICE.md` from the model directory.

### 3. Compile the native runtime with a bounded CPU build

```bash
MODEL=/srv/ai-video-studio/models/pocket-tts-hindi
nice -n 10 /opt/pocket-tts-build-venv/bin/cmake \
  -S "$MODEL/onnx/runtime" \
  -B "$MODEL/onnx/runtime/build" \
  -DCMAKE_BUILD_TYPE=Release
nice -n 10 /opt/pocket-tts-build-venv/bin/cmake \
  --build "$MODEL/onnx/runtime/build" --config Release -j2
sudo chown -R aivideo:aivideo "$MODEL"
sudo -u aivideo "$MODEL/onnx/runtime/pocket-tts" --help
```

The native binary and `libonnxruntime.so` are expected under `onnx/runtime/`. Keep the ONNX model and binary together. Do not run the upstream `--server` mode: its HTTP listener binds to all interfaces and has no authentication. Studio launches the CLI through its authenticated job worker instead.

### 4. Explicitly enable only after verifying the installation

Set these values on **both** `ai-video-studio-api.service` and `ai-video-studio-worker.service` using systemd overrides. Do not copy them into the Git repository:

```ini
[Service]
Environment="ENABLE_TTS=1"
Environment="POCKET_TTS_HINDI_ROOT=/srv/ai-video-studio/models/pocket-tts-hindi"
Environment="POCKET_TTS_HINDI_BINARY=/srv/ai-video-studio/models/pocket-tts-hindi/onnx/runtime/pocket-tts"
```

After an operator intentionally enables it, reload systemd and restart the API and worker services in a maintenance window. The API health endpoint must report `tts_ready: true` and `tts_languages: ["hi"]`. If a required ONNX file, shared library, executable permission, or model file is missing, readiness remains false.

The worker user must be able to write to `/srv/ai-video-studio/voices/.cache`; the runtime stores voice embedding and KV conditioning caches there. It does not edit the original WAV reference.

## Voice profiles and workflow

Voice Studio now uses a single preparation and profile workflow:

1. Upload your original WAV (maximum 30 MB). The original is kept untouched in a temporary staging folder.
2. The Studio inspects the real WAV codec, sample rate, channels, bit depth, duration and file size. If the recording is longer than 20 seconds, it selects a 20-second window with the highest non-silent audio coverage. A shorter recording is kept at its original length; it is never padded.
3. The working reference is converted to mono, 24 kHz, PCM16. Loudness is changed only when the clip is clearly too quiet or too hot. Noise reduction is intentionally not applied by default because it can damage speaker identity. The UI shows audio properties and warnings and lets you listen before saving.
4. The prepared clip is transcribed locally in Hindi by faster-whisper (CPU, INT8, base model by default). Its model downloads to `/srv/ai-video-studio/models/whisper` on first use, outside Git. The transcript is editable; when transcription is unavailable, type the spoken text manually.
5. Save the prepared WAV, editable transcript and generation parameters under the voice profile. The original WAV is preserved separately in the private originals directory; the prepared 20-second reference is stored as the profile WAV and metadata/settings are kept in the profiles directory. Temporary staging copies are removed after save. Abandoned staging uploads are removed after 24 hours.
6. Enter any Hindi text, generate one sample and play it in Voice Studio. The first generation creates Pocket TTS speaker-conditioning caches; subsequent generations reuse them. This is speaker conditioning, not per-speaker model training.

**Selection limitation:** the initial 20-second window is chosen with FFmpeg silence/activity detection. It maximises non-silent audio time; it does not semantically distinguish speech from music or background noise. Listen to the result before saving.

### Profile settings

Each profile stores temperature, flow steps, EOS threshold and seed in a sidecar JSON file beside its prepared WAV. Defaults remain temperature `0.3`, one flow step, EOS `-2.0`, and seed `1`. Preview and saved project/standalone narration use the selected profile's stored generation settings. Settings are validated before being passed to the native runtime.

- Standalone narration: Create Video → Hindi text → saved WAV profile → Generate Hindi narration.
- Storyboard video: create/edit scenes and attach images, then enable **Add Hindi voiceover from the scene narration** and select a saved profile. The exact editable scene narration is synthesized using that profile's settings.
- Optional background music is mixed below the narration. The output duration extends by cloning the final video frame if narration runs longer than the storyboard. Shorter narration is padded with silence to the video duration.
- The generated float WAV is measured with FFmpeg; quiet speech receives conservative static gain toward a -20 dBFS mean target, capped by -3 dBFS peak headroom and +9 dB maximum gain. This preserves dynamics without compression or auto-makeup. A peak limiter remains as a safety ceiling, then audio is normalized to mono 24 kHz PCM16 before muxing.

Use only voice recordings you have permission to use. The WAV stays on your Studio server; the transcription model is downloaded once to that server and inference is local.

## Safety and operational limits

- `ENABLE_TTS` defaults to `0`; no TTS inference occurs until an administrator explicitly enables the engine.
- No GPU, external GPU, cloud inference API, or unauthenticated model server.
- Maximum narration text is 12,000 characters; individual saved WAV references are subject to the existing 30 MB upload limit.
- CLI runs with a two-thread budget and a 15-minute subprocess timeout. One worker serializes TTS and rendering jobs.
- English voice generation remains unsupported by this Pocket TTS Hindi model.
- The model does not guarantee transcript-perfect pronunciation or speaker similarity. Review every output.

