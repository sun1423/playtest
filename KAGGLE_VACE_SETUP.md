# Free GPU VACE runner (Kaggle)

This keeps the Studio API, SQLite database, uploads and projects on Azure. A Kaggle GPU notebook acts as an optional remote worker. It pulls queued jobs, runs open-weight Wan2.1 VACE 1.3B, generates custom Hindi/English narration with Chatterbox Multilingual V3 using a WAV profile already stored in Voice Studio, mixes the audio and uploads the finished MP4.

**No paid inference API is used.** Kaggle's free GPU availability and weekly quota are controlled by Kaggle and can change. Its September 2026 notice retired the free P100 and says T4 x2 remains available. The runner currently uses the first T4 GPU; it does not distribute a single VACE job across both cards. Verify the actual accelerator with `nvidia-smi` before downloading models. The Azure host remains CPU-only and does not load diffusion weights.

## 1. Configure the Studio API runner token

After this feature branch is merged and deployed, generate a random token on the Azure host:

```bash
openssl rand -hex 32
```

Keep the output private. Add it to the API service environment using a systemd override:

```bash
sudo systemctl edit ai-video-studio-api.service
```

Enter this override, replacing the placeholder with the generated token:

```ini
[Service]
Environment="VACE_RUNNER_TOKEN=REPLACE_WITH_RANDOM_TOKEN"
```

Then reload and restart only the API service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ai-video-studio-api.service
```

Do not commit the token or paste it into a notebook cell. The runner API accepts only the matching Bearer token. The runner refuses to start unless `STUDIO_API_URL` is HTTPS: configure a publicly trusted TLS endpoint for Studio first. Do not expose this token over the current plain-HTTP IP endpoint.

## 2. Create a Kaggle notebook

1. Create a Kaggle Notebook and enable Internet.
2. Select **GPU T4 x2** under Settings → Accelerator when it is available. Kaggle can queue or limit free GPU sessions.
3. Add two Kaggle Secrets:
   - `STUDIO_API_URL`: the HTTPS base URL of the deployed Studio, without a trailing slash.
   - `STUDIO_VACE_RUNNER_TOKEN`: the same random token configured in the API service.
4. Do not paste the token into notebook cells or save it in Git.

Run these setup cells:

```bash
!nvidia-smi
!python --version
!git clone --branch feature/vace-video-to-video-custom-voice https://github.com/sun1423/playtest.git /kaggle/working/playtest
!git clone https://github.com/ali-vilab/VACE.git /kaggle/working/VACE
```

The official VACE instructions report testing with Python 3.10.13, PyTorch 2.5.1 and CUDA 12.4. Kaggle's preinstalled runtime may differ. Do not force-reinstall PyTorch, torchvision or CUDA from pip; the TTS and video pipelines must share a working CUDA-enabled Torch installation.

## 3. Install VACE without incompatible GPU kernels

VACE's stock Wan config defaults to BF16, which is not supported by T4. This runner applies a narrow local runtime adjustment on GPUs without BF16: Wan's transformer autocast and SDPA attention use FP16. The text encoder remains CPU-offloaded. The adjustment only edits the Kaggle checkout and installed Wan attention module; it does not alter the Git repository or Azure host.

FlashAttention is intentionally omitted: this path uses PyTorch SDPA instead. Install the framework requirements while preserving Kaggle's Torch/CUDA packages:

```python
from pathlib import Path

source = Path("/kaggle/working/VACE/requirements/framework.txt")
skip = ("torch", "torchvision", "flash_attn", "onnxruntime-gpu")
filtered = [
    line for line in source.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith(skip)
]
Path("/kaggle/working/vace-kaggle-requirements.txt").write_text("\n".join(filtered) + "\n")
```

```bash
%cd /kaggle/working/VACE
!pip install -r /kaggle/working/vace-kaggle-requirements.txt
!pip install --no-deps "wan @ git+https://github.com/Wan-Video/Wan2.1"
```

Create a separate Chatterbox environment so its pinned Diffusers and Transformers do not replace the VACE versions. It inherits the notebook's CUDA-enabled Torch/TorchAudio from the base environment:

```bash
!python -m venv --system-site-packages /kaggle/working/chatterbox-env
!/kaggle/working/chatterbox-env/bin/pip install --no-deps chatterbox-tts==0.1.7
!/kaggle/working/chatterbox-env/bin/pip install "numpy>=1.24,<2" "librosa==0.11.0" s3tokenizer "transformers==5.2.0" "diffusers==0.29.0" "conformer==0.3.2" "safetensors==0.5.3" spacy-pkuseg "pykakasi==2.3.0" "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master" pyloudnorm omegaconf
```

Do not install Torch or TorchAudio into this second environment. Confirm it inherits the same working CUDA versions:

```python
!/kaggle/working/chatterbox-env/bin/python -c "import torch,torchaudio; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available()); assert torch.__version__.split('+')[0] == torchaudio.__version__.split('+')[0]"
```

Check the selected VACE runtime before continuing:

```python
import torch, torchaudio
print("Torch:", torch.__version__, "TorchAudio:", torchaudio.__version__)
print("CUDA:", torch.cuda.is_available(), "BF16:", torch.cuda.is_bf16_supported())
print("GPU 0:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "Enable a CUDA GPU accelerator"
assert torch.__version__.split("+")[0] == torchaudio.__version__.split("+")[0], "Torch and TorchAudio versions must match"
```

If pip changes Torch or TorchAudio, stop and restore a matching Kaggle CUDA build before proceeding. If a dependency is missing at runtime, resolve it without upgrading/downgrading Torch, TorchAudio, Diffusers or Transformers as a group.

## 4. Download model weights

Download the 1.3B VACE checkpoint and preprocessing annotators into the temporary Kaggle workspace:

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

The runner uses the flow or depth annotator checkpoint from this download. Do not switch to the 14B model on a free T4.

## 5. Start the remote runner

```bash
%cd /kaggle/working/playtest
!pip install -r requirements-vace-runner.txt
%env VACE_ROOT=/kaggle/working/VACE
%env VACE_CHECKPOINT_DIR=/kaggle/working/VACE/models/Wan2.1-VACE-1.3B
%env CHATTERBOX_PYTHON=/kaggle/working/chatterbox-env/bin/python
!python vace_runner.py --poll-seconds 15
```

The runner verifies the GPU and applies the FP16/SDPA compatibility adjustment when BF16 is unavailable. It then polls the API, downloads the source video and selected WAV voice, runs VACE, generates the custom narration, mixes both and uploads the MP4. Leave the cell running while jobs are queued; stop it when finished to release the free GPU session.

## 6. Use it in Studio

1. In **Voice Studio**, upload a clean 5–10 second WAV reference of a voice you have permission to use. Avoid music, reverberation and background conversation.
2. Open **AI Video**, upload a source video and write the visual transformation prompt.
3. Enter the narration text, select Hindi or English, then choose the saved voice profile.
4. Choose **Flow** to preserve movement cues or **Depth** to guide scene structure.
5. Optionally mix the source video's sound quietly below the narration, then queue the job.

The voice selector uses the existing Voice Studio profile; you do not need to upload the WAV again. Chatterbox generates the requested spoken narration from the reference voice. VACE performs only the visual transformation.

## Prototype limits and validation status

- Initial output is 81 frames (about 5.06 seconds) at 480p. The first implementation uses the first 5.06 seconds of the uploaded footage; it does not process the full 120-second upload as one clip.
- A short source clip is padded by repeating its last frame to create the required 81-frame input. The final video holds its last frame if narration is longer than the generated visual segment.
- T4 x2 has 16 GB per GPU; this runner uses GPU 0 only. FP16/SDPA makes the default BF16 configuration compatible in principle, but VACE memory use and generation speed still need a real T4 notebook test. A job can fail with CUDA out-of-memory; the Azure server is not used as a fallback.
- The feature branch has not yet been run on a real Kaggle GPU. Do not treat generation as production-validated until a sample job completes end-to-end.
- Chatterbox is an archived open-source research project; it can still be run locally, but it no longer receives upstream maintenance. Its Multilingual V3 model supports Hindi and English. The TTS worker runs in a separate Python environment to avoid dependency conflicts with VACE, while reusing the notebook's already-installed CUDA Torch/TorchAudio.
- Kaggle temporarily receives the uploaded video and voice WAV. The runner deletes its per-job workspace after completion/failure; Kaggle session storage is temporary.

