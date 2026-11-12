# AI Video Studio

Azure-hosted social video studio. Browser UI is a control panel; uploads, inference and rendering run on Azure. React/Vite + FastAPI + SQLite queue + one worker. Hindi voice narration is an optional Pocket TTS ONNX feature that remains disabled unless explicitly enabled after the model is installed. The current local voice engine supports Hindi only; English narration is not provided by this engine. FFmpeg transforms uploaded video while preserving or mixing audio. Prompt-to-video diffusion is intentionally disabled until a CPU image pipeline is benchmarked; ComfyUI is optional, not the main UI.

Voice Studio prepares an up-to-20-second WAV reference, transcribes it locally in Hindi for editing, and saves reusable voice profiles with generation settings and in-app sample playback. See [Hindi ASR engine setup and licence notes](docs/VOICE_STUDIO_HINDI_ASR.md). Keep model weights, voice references and media outside Git. See [Pocket TTS Hindi setup and benchmark](docs/POCKET_TTS_HINDI.md) and docs/deploy.md for deployment notes.


CPU Hindi video dubbing and lip-sync is available on the feature branch. It uses the saved Pocket TTS voice profile and ONNX Wav2Lip CPU inference; setup, hardware benchmark, limitations and weight-license notes are in [docs/CPU_VIDEO_DUB.md](docs/CPU_VIDEO_DUB.md). This is mouth re-animation of the real source footage, not full facial-expression generation.


The CPU video-dub workflow uses saved Pocket TTS Hindi profiles, Wav2Lip ONNX lip-sync, and optional full-frame ONNX styles (AnimeGANv2 512px or faster 224px painterly presets). See [docs/CPU_VIDEO_DUB.md](docs/CPU_VIDEO_DUB.md) for setup and measured CPU performance.
