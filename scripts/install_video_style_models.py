#!/usr/bin/env python3
"""Download pinned CPU video-style ONNX models into private persistent storage."""
from __future__ import annotations
import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

ANIME_URL = "https://raw.githubusercontent.com/Kazuhito00/AnimeGANv2-ONNX-Sample/1597c3ad45b2db87e1ac4944ee62cfdb30d0670d/model/face_paint_512_v2_0.onnx"
STYLE_MODELS = "https://huggingface.co"
MODELS = {
    "animeganv2_512.onnx": (ANIME_URL, "1381b17ed988e14a1f3cf8954d88073c0884ff4b70128ebe599f0d6021bca63e", 8_000_000),
    "candy.onnx": (f"{STYLE_MODELS}/onnxmodelzoo/candy-9/resolve/f24fcf558bd4564360f41cf3b5321975a33f962f/candy-9.onnx", "9d11a3529d1e547da6ae07201d93484dbab2ec0a3614535752c8f40f0fe2968a", 6_000_000),
    "mosaic.onnx": (f"{STYLE_MODELS}/onnxmodelzoo/mosaic-9/resolve/9a267ba8540fba8fb12bc78f3770e0fd9c923c56/mosaic-9.onnx", "fa646dedade881243f8d5a2ceb7de2b93675b21fc24f7482894ac4851a9a0a47", 6_000_000),
    "rain-princess.onnx": (f"{STYLE_MODELS}/onnxmodelzoo/rain-princess-9/resolve/205625e4c00565b6c0946814de6165116b271e58/rain-princess-9.onnx", "4162912e6f75fedef6f810ae989b9e10d3d5d43308dab34b027c850cf255e152", 6_000_000),
    "udnie.onnx": (f"{STYLE_MODELS}/onnxmodelzoo/udnie-9/resolve/ab74109f694dc8117f5e39a8571ac53bf83618f6/udnie-9.onnx", "8656b6ce7dec8f22ee13c2d557d6b67bd6f550dde88d0f2e7c9972aeb765cc0d", 6_000_000),
}


def fetch(destination: Path, url: str, expected_sha: str, minimum_size: int) -> None:
    if destination.is_file() and destination.stat().st_size >= minimum_size:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected_sha:
            print(f"verified existing {destination.name}")
            return
        print(f"replacing outdated or mismatched model {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AI-Video-Studio-CPU-Style"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size < minimum_size or digest.hexdigest() != expected_sha:
            raise RuntimeError(f"Downloaded style model failed size/hash verification: {destination.name}")
        temporary.replace(destination)
        print(f"installed {destination.name}: {size:,} bytes · SHA-256 verified")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=os.getenv(
        "VIDEO_STYLE_MODELS_ROOT", "/srv/ai-video-studio/models/video-style"
    ))
    args = parser.parse_args()
    root = Path(args.models_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for filename, (url, digest, minimum) in MODELS.items():
        fetch(root / filename, url, digest, minimum)
    print(f"CPU video styles ready: {root}")


if __name__ == "__main__":
    main()
