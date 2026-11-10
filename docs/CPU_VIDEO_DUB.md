# CPU Hindi video dubbing and lip-sync

## What this feature does

The Studio accepts a real source video, a saved Pocket TTS Hindi voice profile, and replacement Hindi dialogue. The worker synthesises the dialogue with the selected speaker profile, then runs Wav2Lip through ONNX Runtime on the CPU. It processes the complete supported clip, not a fixed first-five-second segment.

The original performance supplies the face identity, eyes, expressions, head movement, body motion, camera motion, and background. The lip-sync model regenerates the lower mouth area only. It does **not** invent new emotions, eyebrow movement, eye expression, or new head gestures. The source actor must already perform those things in the source footage.

## Azure hardware and measured baseline

Validated host: Azure x86_64, 2 vCPU, about 16 GiB RAM, no NVIDIA device. The smoke test processed 156 frames (6.27 seconds) through the ONNX CPU provider and produced an MP4 with video and audio. End-to-end time, including model startup and encoding, was 56.4 seconds on this host. This is a slow offline render: do not advertise it as real time. Longer clips take substantially longer; processing time varies with resolution, face detection, video encoding, and the source.

The pipeline caps inference at 2 ONNX threads and eight decoded frames per batch. It does not load an entire uncompressed video into memory. Frames are normalised to a maximum dimension of 1280 before CPU inference.

## One-time installation

From the deployed checkout, install the pinned CPU dependencies into the Studio virtual environment:

\`\`\`bash
cd /opt/ai-video-studio
.venv/bin/pip install -r requirements.txt
\`\`\`

Download the pinned ONNX model files. They are stored under private persistent model storage, not in Git:

\`\`\`bash
.venv/bin/python scripts/install_cpu_lipsync.py
\`\`\`

The installer verifies SHA-256 hashes before publishing either model. Default destination:

\`/srv/ai-video-studio/models/wav2lip-cpu\`

The API and worker must use the same model path. Override with \`LIPSYNC_MODELS_ROOT\` only when deliberately moving the model directory. The model is not available until both runtime dependencies and both verified model files are present. Check \`GET /api/lipsync/status\` after installation.

## In-app workflow

1. Open **AI Video · Lip Sync**.
2. Upload MP4, MOV, MKV, or WebM footage (1–180 seconds, within the existing 500 MB upload cap).
3. Select a saved Hindi WAV voice profile from Voice Studio.
4. Enter the replacement Hindi dialogue. Punctuation and paragraph breaks help the TTS engine produce pauses.
5. Queue the job and monitor it in **Projects → Render queue**.
6. Download the completed \`video_with_lipsync.mp4\`.

The full speech audio replaces the original dialogue by default. The optional ambience switch mixes the source audio quietly underneath; it can also retain the original dialogue, so leave it off for clean dubbing.

## Timing policy

- When narration is shorter than the source, the video keeps its full duration and the new speech ends naturally.
- When narration is longer, the source video may be slowed by up to 20% to fit. More than 20% is rejected instead of silently cutting speech or holding a frozen frame for a long time.
- The speech audio is padded to the output frame duration so the final MP4 has a stable audio/video timeline.

For best results, use one clearly visible, mostly front-facing speaker whose mouth remains visible in at least 80% of frames. Multi-person scenes, fast cuts, profile-only views, hands over the mouth, masks, or heavy occlusion can fail the face-tracking quality gate.

## Known quality and licensing limits

Wav2Lip is a mouth re-animation baseline, not full video-to-video diffusion. It preserves source emotion; it cannot create new facial expressions. A 96×96 face generator can look softer than a modern diffusion lip-sync system. The refined blend keeps the upper face/eyes from the source frame, but the mouth can still show seams or blur. Check the actual footage and generated output before relying on it.

The Wav2Lip pretrained generator weights distributed by the selected model source carry personal/research/non-commercial restrictions. Do not use these weights for a commercial product or monetised production without first obtaining written rights or replacing them with commercially cleared weights. Open-source code does not automatically make pretrained weights commercially licensed.

## Troubleshooting

- **Feature reports not ready:** install \`requirements.txt\` and run the model download script; then restart both API and worker services.
- **No visible mouth change:** check that the input contains one large, front-facing face and the face-detection ratio passes the 80% gate.
- **Job rejects speech duration:** shorten the Hindi narration or provide a longer source clip; the worker will not truncate dialogue.
- **Output mouth is soft or seams are visible:** this is a Wav2Lip quality limitation. Do not assume the source voice profile is the cause.
