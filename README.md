# AI Video Studio

Azure-hosted social video studio. Browser UI is a control panel; uploads, inference and rendering run on Azure. React/Vite + FastAPI + SQLite queue + one worker. Hindi voice narration is an optional Pocket TTS ONNX feature that remains disabled unless explicitly enabled after the model is installed. The current local voice engine supports Hindi only; English narration is not provided by this engine. FFmpeg transforms uploaded video while preserving or mixing audio. Prompt-to-video diffusion is intentionally disabled until a CPU image pipeline is benchmarked; ComfyUI is optional, not the main UI.

Voice Studio prepares an up-to-20-second WAV reference, transcribes it locally in Hindi for editing, and saves reusable voice profiles with generation settings and in-app sample playback. Keep model weights, voice references and media outside Git. See [Pocket TTS Hindi setup and benchmark](docs/POCKET_TTS_HINDI.md) and docs/deploy.md for deployment notes.
