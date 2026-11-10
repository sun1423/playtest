#!/usr/bin/env python3
"""Install the pinned AI4Bharat IndicConformer Hindi ONNX model outside Git."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

REPOSITORY = "sulabhkatiyar/indicconformer-120m-onnx"
REVISION = "16f1697d0631c9bb88e371fca24577efe0aa8617"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/hi"
FILES = {
    "model.onnx": (
        f"{BASE_URL}/model.onnx",
        "fb362cae38fd183546d92a614bab2dcb6098702d53efb8e2f6fe850cda43e0d1",
        493_060_285,
    ),
    "vocab.json": (
        f"{BASE_URL}/vocab.json",
        "69fb197067f139f7961e70fed986274f366c7d1c26f777aada635f6237b75d43",
        68_706,
    ),
}


def install_file(destination: Path, url: str, expected_hash: str, expected_size: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected_hash:
            print(f"verified existing {destination.name}")
            return
        print(f"replacing mismatched {destination.name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "AI-Video-Studio-IndicConformer/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError(
                f"Verification failed for {destination.name}: "
                f"{size} bytes, SHA-256 {digest.hexdigest()}"
            )
        temporary.replace(destination)
        print(f"installed {destination.name}: {size:,} bytes · SHA-256 verified")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        default=os.getenv(
            "INDICCONFORMER_HI_ROOT",
            "/srv/ai-video-studio/models/indicconformer-hi",
        ),
    )
    args = parser.parse_args()
    root = Path(args.models_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for filename, (url, sha256, size) in FILES.items():
        install_file(root / filename, url, sha256, size)
    print(f"IndicConformer Hindi model ready: {root}")
    print("Model source: AI4Bharat IndicConformer; ONNX CTC export by community.")
    print("License: CC-BY-4.0; retain model attribution when redistributing.")


if __name__ == "__main__":
    main()
