# Voice Studio Hindi transcription

## Locked production choice

AI4Bharat IndicConformer Hindi (120M, CTC) through ONNX Runtime CPU is the selected and default Hindi ASR backend. The exact same uploaded 30-second female reference was tested against the current production Whisper base recognizer: IndicConformer produced a coherent Devanagari transcript in about 3.46 seconds of model inference, while Whisper base produced repetitive and largely incorrect text. IndicConformer still needs human review for names and ambiguous words; a Hindi language-confidence score is not a transcript-accuracy score.

The model is Hindi-specific, avoiding multilingual language-selection between Hindi and Urdu. It takes mono audio, resamples to 16 kHz, computes the NeMo-compatible 80-bin log-mel frontend and decodes the CTC output into Hindi text.

## Install on Azure

```bash
cd /opt/ai-video-studio
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/install_indicconformer_asr.py
```

The model and vocabulary are downloaded to `/srv/ai-video-studio/models/indicconformer-hi` by default. The installer pins a model repository revision, exact file sizes and SHA-256 digests. The ONNX weights are approximately 493 MB; no PyTorch or GPU is needed at inference time.

## Runtime selection and rollback

- `VOICE_ASR_BACKEND=indicconformer` selects the locked Hindi-specific ONNX recognizer. This is the default.
- `VOICE_ASR_BACKEND=whisper` explicitly selects the existing Faster-Whisper CPU/int8 implementation for rollback or comparison.
- Automatic fallback is **disabled by default**. If IndicConformer fails to load or run, Voice Studio reports the error rather than silently substituting Whisper's lower-quality transcript.
- To explicitly allow Whisper as a fallback, set `VOICE_ASR_FALLBACK_BACKEND=whisper`. The result records both the requested backend and the backend actually used, plus the fallback reason.
- The old `VOICE_ASR_FALLBACK_TO_WHISPER=1` setting remains supported as a compatibility switch.
- The ONNX session is loaded lazily and serialized; it uses the CPU execution provider and at most two intra-op threads.
- Urdu/Arabic-script output is explicitly flagged for manual correction rather than silently accepted.

## Adding another ASR model later

ASR engines use a small adapter contract: a callable receiving `(audio_path: Path, model_cache: Path)` and returning a result dictionary containing at least `text`. A new model can be integrated without rewriting the voice-profile workflow:

1. Add a separate adapter module that installs/loads the new model and implements that callable.
2. Register it with `register_asr_backend("backend-name", transcribe_function)`.
3. For an optional adapter module, set `VOICE_ASR_BACKEND_MODULES=package.module` (multiple module names can be comma-separated). The module registers its adapter when imported.
4. Select it with `VOICE_ASR_BACKEND=backend-name`. Keep IndicConformer as the default until the replacement has been benchmarked on the same held-out Hindi recordings.

Each adapter should own its model-specific installer, pinned model revision, checksums, licence notes, CPU/RAM limits, and tests. New models must not replace the locked default automatically.

## Licence and attribution

The original IndicConformer Hindi model is published by AI4Bharat. The ONNX CTC export used here is downloaded from `sulabhkatiyar/indicconformer-120m-onnx`, whose model card declares CC-BY-4.0. Preserve attribution to AI4Bharat and the ONNX exporter when redistributing the model or derivative artifacts. Review the model repository's current licence and terms before commercial redistribution.

## Validation

Unit tests cover greedy CTC collapse, Devanagari/Urdu-script checks, backend registration/selection and explicit fallback behavior. The real reference comparison demonstrates much better output than Whisper base on this recording; it does not prove accuracy on every speaker, dialect or recording condition. Before changing the default model, compare the exact same representative Hindi WAV corpus and manually check names, punctuation, Sanskritised terms, Urdu loanwords and full transcripts.
