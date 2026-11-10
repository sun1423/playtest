# Voice Studio Hindi transcription

Voice Studio uses AI4Bharat IndicConformer Hindi (120M, CTC) through ONNX Runtime CPU for the prepared reference WAV. This is a Hindi-specific recognizer, avoiding multilingual language-selection between Hindi and Urdu. It takes mono audio, resamples to 16 kHz, computes the NeMo-compatible 80-bin log-mel frontend and decodes the CTC output into Hindi text.

## Install on Azure

\`\`\`bash
cd /opt/ai-video-studio
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/install_indicconformer_asr.py
\`\`\`

The model and vocabulary are downloaded to /srv/ai-video-studio/models/indicconformer-hi by default. The installer pins a model repository revision, exact file sizes and SHA-256 digests. The ONNX weights are approximately 493 MB; no PyTorch or GPU is needed at inference time.

## Runtime behavior

- VOICE_ASR_BACKEND=indicconformer selects the Hindi-specific ONNX recognizer. This is the default.
- VOICE_ASR_BACKEND=whisper restores the existing Faster-Whisper CPU/int8 implementation.
- VOICE_ASR_FALLBACK_TO_WHISPER=1 (default) falls back to Faster-Whisper if model loading or ONNX inference fails.
- The ONNX session is loaded lazily and serialized; it uses the CPU execution provider and at most two intra-op threads.
- The response metadata includes the recognizer name and inference duration. Urdu/Arabic-script output is explicitly flagged for manual correction rather than silently accepted.

## Licence and attribution

The original IndicConformer Hindi model is published by AI4Bharat. The ONNX CTC export used here is downloaded from sulabhkatiyar/indicconformer-120m-onnx, whose model card declares CC-BY-4.0. Preserve attribution to AI4Bharat and the ONNX exporter when redistributing the model or derivative artifacts. Review the model repository's current licence and terms before commercial redistribution.

## Validation

Unit tests cover greedy CTC collapse, Devanagari/Urdu-script checks, backend selection and Whisper fallback. They do not establish transcription quality on an individual recording. Before considering Voice Studio fixed, compare the exact same representative Hindi WAV against the existing recognizer and manually check names, punctuation, Sanskritised terms, Urdu loanwords and the full transcript.
