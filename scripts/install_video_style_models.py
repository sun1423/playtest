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
STYLE_BASE = "https://raw.githubusercontent.com/yakhyo/fast-neural-style-transfer/32f6cb17f1bf63e6c3e4feb2125671266dc083d1/weights/"
MODELS = {
    "animeganv2_512.onnx": (ANIME_URL, "1381b17ed988e14a1f3cf8954d88073c0884ff4b70128ebe599f0d6021bca63e", 8_000_000),
    "candy.onnx": (STYLE_BASE + "candy.onnx", "dd589a07b80f7868e56289fa268e56614846850077e92ec8a9bd80d65c017698", 6_000_000),
    "mosaic.onnx": (STYLE_BASE + "mosaic.onnx", "9b561367c27898fae588442454c74820ac49e738dad141b8e87fb6ee537da45e", 6_000_000),
    "rain-princess.onnx": (STYLE_BASE + "rain-princess.onnx", "fe142dc5feabea07231e5993a52009b6c1187f59eff3b91804f0471585ea818b", 6_000_000),
    "udnie.onnx": (STYLE_BASE + "udnie.onnx", "dc52ca2c6b9426403764341fc09e0587843c78d17f5a48a6652b4f1b80156bb6", 6_000_000),
}


def fetch(destination: Path, url: str, expected_sha: str, minimum_size: int) -> None:
    if destination.is_file() and destination.stat().st_size >= minimum_size:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected_sha:
            print(f"verified existing {destination.name}")
            return
        raise RuntimeError(f"Existing style model SHA-256 mismatch: {destination}")
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
