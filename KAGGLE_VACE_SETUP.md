# Free GPU VACE runner (Kaggle)

This keeps the Studio API, SQLite database, uploads and projects on Azure. A Kaggle GPU notebook acts as an optional remote worker. It pulls queued jobs, runs Wan2.1 VACE, generates narration with Chatterbox Multilingual V3 using the WAV voice profile already stored in Voice Studio, mixes the audio, and uploads the finished MP4.

**No paid inference API is used.** Kaggle free GPU availability and quotas are controlled by Kaggle and are not guaranteed. VACE and Chatterbox model weights are downloaded into the temporary Kaggle session; they are not installed on the 2-vCPU Azure host.

## 1. Configure the Studio API runner token

After this branch is merged and deployed, generate a long random token on the Azure host:

```bash
openssl rand -hex 32
```

Keep the output private. Add it to the API service environment using a systemd override:

```bash
sudo systemctl edit ai-video-studio-api.service
```

Enter this override, replacing the placeholder with the generated value:

```ini
[Service]
Environment="VACE_RUNNER_TOKEN=REPLACE_WITH_RANDOM_TOKEN"
```

Then reload and restart only the API service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ai-video-studio-api.service
```

Do not commit the token or put it in a notebook cell. The API does not return the token from any status endpoint. The runner endpoints accept only the matching Bearer token.

## 2. Create a Kaggle notebook

1. Create a new Kaggle Notebook.
2. In Notebook options, enable **Internet** and select a **GPU T4** accelerator when Kaggle makes one available.
3. Add two Kaggle secrets (Notebook add-ons / Secrets):
   - `STUDIO_API_URL`: the public base URL of the deployed Studio, without a trailing slash.
   - `STUDIO_VACE_RUNNER_TOKEN`: the same random token configured on the API service.
4. Do not paste the token into a notebook cell or save it in the Git repository.

Run these setup cells:

```bash
!nvidia-smi
!python --version
!git clone --branch feature/vace-video-to-video-custom-voice https://github.com/sun1423/playtest.git /kaggle/working/playtest
!git clone https://github.com/ali-vilab/VACE.git /kaggle/working/VACE
```

The VACE authors report testing with Python 3.10.13, PyTorch 2.5.1 and CUDA 12.4. Kaggle's preinstalled runtime can differ; do not force-reinstall CUDA/PyTorch if that would break the selected Kaggle image. Use a compatible runtime or adjust to the official VACE installation instructions.

Install the official VACE requirements and the Wan adapter:

```bash
%cd /kaggle/working/VACE
!pip install -r requirements.txt
!pip install "wan @ git+https://github.com/Wan-Video/Wan2.1"
```

Install Chatterbox Multilingual TTS into the same Python environment:

```bash
!pip install chatterbox-tts
```

Download the 1.3B VACE checkpoint and preprocessing annotators:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.1-VACE-1.3B",
    local_dir="/kaggle/working/VACE/models/Wan2.1-VACE-1.3B",
)
snapshot_download(
    repo_id="ali-vilab/VACE-Annotators",
    local_dir="/kaggle/working/VACE/models/VACE-Annotators",
)
```

If the annotator/model download or dependency install fails, stop and inspect the Kaggle cell output before starting the runner. Do not switch to the 14B model on a T4.

## 3. Start the remote runner

```bash
%cd /kaggle/working/playtest
!pip install -r requirements-vace-runner.txt
%env VACE_ROOT=/kaggle/working/VACE
%env VACE_CHECKPOINT_DIR=/kaggle/working/VACE/models/Wan2.1-VACE-1.3B
!python vace_runner.py --poll-seconds 15
```

Leave the final cell running while you queue an AI Video job in Studio. The runner checks for jobs, downloads the source video and chosen saved WAV voice profile, runs VACE flow/depth editing, synthesizes the narration in that custom voice, and uploads the MP4. Stop the notebook when finished to release the GPU session.

## 4. Use it in Studio

1. Upload a clean 5–10 second WAV voice reference in **Voice Studio**. Avoid music, reverb and background conversation.
2. Open **AI Video**, upload a source video and enter an English or Hindi visual prompt.
3. Enter narration text, choose the matching language and select the saved voice profile.
4. Choose **Flow** to prioritize motion continuity or **Depth** to prioritize scene structure.
5. Optionally mix the original video's audio quietly under the narration, then queue the job.

## Current prototype limits

- Wan2.1 VACE 1.3B generates about 5.06 seconds at 480p per job. The runner takes the first 5.06 seconds of the uploaded source in this first version.
- If the generated voice lasts longer than the generated visuals, the last video frame is held until narration finishes.
- T4 availability is not guaranteed. If VACE runs out of VRAM, stop the job and use a GPU runtime with more memory; the Azure production host has no GPU and is deliberately not used for diffusion inference.
- The custom voice is generated as narration with Chatterbox. VACE itself does not clone or synthesize voices.
- Kaggle temporarily receives your uploaded video and reference WAV for processing. The runner deletes its local job folder after completion/failure; Kaggle session storage is temporary.
