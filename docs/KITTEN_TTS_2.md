# KittenTTS 2 · expressive Hindi CPU voice engine

KittenTTS 2 is an optional second speech engine. Pocket TTS Hindi remains installed, unchanged and available as the default/fallback. The application uses the selected engine on each job; queued jobs retain their engine choice.

## What it adds

- Hindi speech from the saved WAV voice profile.
- Recognised emotion tags: \`[angry]\`, \`[contemplative]\`, \`[excited]\`, \`[joyful]\`, \`[mundane]\`, \`[nervous]\`, \`[sad]\`, \`[stern]\`, \`[surprised]\`, and \`[tender]\`.
- Inline vocal events: \`<gasp>\`, \`<giggle>\`, \`<growl>\`, \`<gulp>\`, \`<laugh>\`, \`<pause>\`, \`<scoff>\`, \`<sigh>\`, \`<sob>\`, and \`<um>\`.
- Emphasis: \`(((word or short phrase)))\`.
- Existing speaker profile WAV plus its editable transcript. No voice profile is overwritten; Kitten's derived speaker conditioning is saved separately.

Put one leading emotion tag at the start of each intended utterance, for example:

\`\`\`text
[contemplative] आज की घटना ने मुझे बहुत कुछ सोचने पर मजबूर कर दिया।
[pause]
[angry] लेकिन अब यह बिल्कुल बर्दाश्त नहीं होगा!
[tender] तुम मेरे लिए (((बहुत खास))) हो।
\`\`\`

Emotion control is beta: tags steer delivery but do not guarantee a specific emotion. Hindi quality and recognition can vary by voice and sentence. The app passes Hindi text with the upstream-required \`--no-normalize\` setting so its English-focused text normalizer does not rewrite Hindi numbers, dates or abbreviations.

## Hindi quality defaults

When the submitted text contains Devanagari, the backend selects a Hindi-focused path:

- Stable sampling: temperature 0.7, top-k 50 and top-p 0.8.
- Stronger loop control: repetition penalty 1.12 and codec-token run penalty 1.8 after 5 repeats.
- Sentence-aware chunks target 240 characters, with a 90-character minimum.
- Timbre-only cloning: use the saved profile's speaker embedding but omit its reference audio/transcript tokens from the generation prompt. This avoids carrying errors from a long or imperfect Hindi reference transcript into the spoken sentence.
- A modest +5 dB output gain followed by the existing peak limiter. Non-Devanagari requests retain their previous audio level and generation settings.

The mode is selected automatically from Devanagari text. It does not modify the saved WAV, transcript or prepared voice profile.

## CPU architecture and isolation

- Native KittenML C++ runner / GGML CPU model; CUDA and Metal builds are disabled.
- Two CPU threads maximum for synthesis and speaker preparation.
- A separate preparation virtualenv isolates PyTorch and Kitten dependencies from the production API virtualenv.
- Model files and reference-conditioning assets live outside Git in \`/srv/ai-video-studio/models\`.
- No automatic download occurs on app startup or when the engine is selected. Admin runs the installer explicitly.
- The engine stays disabled until \`ENABLE_KITTEN_TTS=1\` is added to both API and worker systemd services.

The installer downloads the selected native GGUF, decoder, voice registry, speaker encoder and language-model checkpoint required to derive cloned voices. It also prefetches the S3 decoder weights. Default decoder is the upstream quality-focused \`default\` decoder, not the smaller student decoder.

## Install on Azure

Run from the deployed Studio checkout as root (or through sudo):

\`\`\`bash
cd /opt/ai-video-studio
sudo .venv/bin/python scripts/install_kitten_tts.py
\`\`\`

The installer:
1. Creates a dedicated CPU PyTorch 2.6 preparation environment.
2. Checks out pinned KittenML Python and C++ source revisions.
3. Builds the official CPU C++ runner with CUDA, Metal, server mode and test/example targets disabled.
4. Downloads the model's current immutable Hugging Face revision and writes it to \`models/kitten-tts-2/.download_revision\`.
5. Pre-downloads the reference S3 model weights and reports the exact service environment values.
6. Leaves Kitten disabled and does not restart services.

The script expects the systemd service account to be \`aivideo\`; set \`AI_VIDEO_STUDIO_USER\` to the real service user if it differs. It creates a persistent, writable profile registry and HF cache for that user. The model, source checkout and dedicated venv remain separate from the application's code and Pocket TTS files.

## Enable after installation

Review the model's Stellon Labs Community License and the separate decoder/model notices first. Then add every \`KITTEN_TTS_*\` value printed by the installer plus:

\`\`\`ini
[Service]
Environment="ENABLE_KITTEN_TTS=1"
\`\`\`

to **both**:
- \`ai-video-studio-api.service\`
- \`ai-video-studio-worker.service\`

Keep the existing \`ENABLE_TTS=1\` and Pocket model paths untouched. Reload systemd and restart the API and worker in a maintenance window. Check \`GET /api/health\`: \`tts_engines.kitten.ready\` must be true. If it is false, do not select Kitten for production jobs.

The Voice Studio engine selector applies to previews, script MP3, storyboard narration, Transform voice-over and video lip-sync dubbing. The selected engine is saved with each job and is restored when editing a voice-over.

## First use of a saved voice

Kitten requires an exact transcript with the saved mono 24 kHz PCM16 reference. It prepares speaker conditioning on first use or after that transcript/WAV changes. Existing profiles with an empty or inaccurate transcript must be corrected in Voice Studio before using Kitten. The original WAV, prepared WAV and transcript remain preserved; Kitten keeps its derived assets in \`models/kitten-tts-2-profiles\`.

A new preparation can take time and use extra CPU/RAM. The engine lock serializes Kitten requests so the shared voice registry is not changed during a concurrent synthesis. Long Script-to-Audio exports run through one native process, allowing Kitten's sentence-aware chunking to reuse the loaded speech model rather than reloading it for every part. No audio-generation smoke test is run by the installer.

## Validation and rollback

Static tests cover engine opt-in, model-manifest integrity, Hindi text handling and expression-markup preservation. The frontend build and service health should be checked after deployment. Do not mark voice quality or inference speed as validated until an actual Hindi sample has been generated and listened to on the 2-vCPU server.

To disable Kitten, set \`ENABLE_KITTEN_TTS=0\` or remove that environment setting from both services, then restart them. Pocket TTS remains available and does not share Kitten's runtime, model cache, profile registry or generation parameters.
