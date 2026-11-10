# AI Video Studio
Azure-hosted social video studio. Browser UI is a control panel; uploads, inference and rendering run on Azure. React/Vite + FastAPI + SQLite queue + one worker. Chatterbox Multilingual Hindi/English TTS runs on CPU; FFmpeg transforms uploaded video while preserving source audio. Prompt-to-video diffusion is intentionally disabled until a CPU image pipeline is benchmarked; ComfyUI is optional, not the main UI. Keep model weights, voice references and media outside Git. See docs/deploy.md for deployment notes.

## Optional AI video-to-video runner

The `AI Video` workflow uses open-weight Wan2.1 VACE 1.3B for video editing and the existing WAV voice profiles with Chatterbox Multilingual V3 for custom narration. Diffusion inference runs on a separately started GPU runner (Kaggle setup: `KAGGLE_VACE_SETUP.md`); the Azure CPU host only queues jobs and stores results. It is intentionally not enabled until the remote runner token and GPU session are configured. Initial output is about 5 seconds at 480p.
