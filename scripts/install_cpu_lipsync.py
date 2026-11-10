#!/usr/bin/env python3
"""Download pinned CPU Wav2Lip ONNX and YuNet weights outside Git."""
from __future__ import annotations
import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

WAV2LIP_URL = "https://huggingface.co/wanesoft/faceswap_pack/resolve/2740234ae2d37df437fda5137dc48967f5dbd011/wav2lip_gan.onnx"
WAV2LIP_SHA256 = "087d4b12b203232b293ae4594de73bdf1f2358fe86068da8208faac944d4648e"
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODELS = {
    "wav2lip_gan.onnx": (WAV2LIP_URL, WAV2LIP_SHA256, 100_000_000),
    "yunet.onnx": (YUNET_URL, YUNET_SHA256, 100_000),
}


def fetch(destination: Path, url: str, expected_sha: str, minimum_size: int) -> None:
    if destination.is_file() and destination.stat().st_size >= minimum_size:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected_sha:
            print(f"verified existing {destination.name}")
            return
        raise RuntimeError(f"Existing model hash mismatch: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AI-Video-Studio/CPU-Lipsync"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size < minimum_size or digest.hexdigest() != expected_sha:
            raise RuntimeError(f"Downloaded model failed size/hash verification: {destination.name}")
        temporary.replace(destination)
        print(f"installed {destination.name}: {size:,} bytes · sha256 verified")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        default=os.getenv("LIPSYNC_MODELS_ROOT", "/srv/ai-video-studio/models/wav2lip-cpu"),
    )
    args = parser.parse_args()
    root = Path(args.models_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for filename, (url, digest, minimum) in MODELS.items():
        fetch(root / filename, url, digest, minimum)
    print(f"CPU lip-sync models ready: {root}")
    print("Usage note: verify Wav2Lip pretrained-weight terms before commercial use.")


if __name__ == "__main__":
    main()
