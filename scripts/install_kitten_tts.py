#!/usr/bin/env python3
"""Install the optional KittenTTS 2 CPU runtime. Run manually as an administrator."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.getenv("STUDIO_ROOT", "/srv/ai-video-studio")).resolve()
MODEL_ROOT = Path(os.getenv("KITTEN_TTS_MODEL_ROOT", str(ROOT / "models" / "kitten-tts-2"))).resolve()
CPP_ROOT = Path(os.getenv("KITTEN_TTS_CPP_ROOT", "/opt/kitten-tts-2-cpp")).resolve()
PY_PACKAGE = Path(os.getenv("KITTEN_TTS_PYTHON_PACKAGE", "/opt/KittenTTS")).resolve()
VENV = Path(os.getenv("KITTEN_TTS_PREP_VENV", "/opt/kitten-tts-prep-venv")).resolve()
HF_HOME = Path(os.getenv("KITTEN_TTS_HF_HOME", str(ROOT / "models" / "kitten-tts-2-hf-cache"))).resolve()
PROFILE_ROOT = Path(os.getenv("KITTEN_TTS_PROFILE_ROOT", str(ROOT / "models" / "kitten-tts-2-profiles"))).resolve()
SERVICE_USER = os.getenv("AI_VIDEO_STUDIO_USER", "aivideo")
CPP_REVISION = "2b9912e72245f5597c664d2be44a6e7ba0a3f293"
PY_REVISION = "b9517e3050adced306219f8bea5f117d87ecfaa7"
MODEL_REPO = "KittenML/kitten-tts-2"
DECODER = "default"


def run(command: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def checkout(url: str, destination: Path, revision: str) -> None:
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise RuntimeError(f"Refusing to overwrite a non-Git directory: {destination}")
        run(["git", "fetch", "--tags", "origin"], cwd=destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", url, str(destination)])
    run(["git", "checkout", "--detach", revision], cwd=destination)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(destination), text=True
    ).strip()
    if actual != revision:
        raise RuntimeError(f"Unexpected Git revision in {destination}: {actual}")


def main() -> None:
    if os.name != "posix":
        raise SystemExit("This installer targets the Linux CPU server.")
    if os.geteuid() != 0:
        raise SystemExit("Run with sudo so the selected /opt and /srv directories can be prepared.")
    for command in ("git", "cmake", "c++"):
        if subprocess.run(["bash", "-lc", f"command -v {command}"], capture_output=True).returncode:
            raise SystemExit(f"Missing build prerequisite: {command}. Install git, CMake and a C++20 compiler first.")

    import venv
    VENV.parent.mkdir(parents=True, exist_ok=True)
    if not (VENV / "bin" / "python").is_file():
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    python = VENV / "bin" / "python"
    pip = [str(python), "-m", "pip"]
    run([*pip, "install", "--upgrade", "pip", "wheel"])
    run([
        *pip, "install", "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.6.0", "torchaudio==2.6.0",
    ])
    run([*pip, "install", "cmake==3.31.6", "ninja", "huggingface_hub", "safetensors"])
    checkout("https://github.com/KittenML/KittenTTS.git", PY_PACKAGE, PY_REVISION)
    checkout("https://github.com/KittenML/kitten-tts-2-cpp.git", CPP_ROOT, CPP_REVISION)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=CPP_ROOT)
    run([*pip, "install", "-e", str(PY_PACKAGE)])

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_HOME)
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_HOME)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    prep_site_packages = VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if not prep_site_packages.is_dir():
        raise RuntimeError(f"Dedicated preparation environment is missing site-packages: {prep_site_packages}")
    sys.path.insert(0, str(prep_site_packages))
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    model_revision = HfApi().model_info(MODEL_REPO).sha
    allow_patterns = [
        "config.json", "README.md", "LICENSE.md", "LICENSE",
        "speaker/model.safetensors", "lm/config.json", "lm/model-ternary.safetensors",
        "cpp/model-tq2_1.gguf", "cpp/default/decoder.pt", "cpp/default/voices.json",
    ]
    print(f"Downloading selected KittenTTS 2 assets at revision {model_revision}", flush=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=model_revision,
        local_dir=str(MODEL_ROOT),
        allow_patterns=allow_patterns,
    )
    (MODEL_ROOT / ".download_revision").write_text(model_revision + "\n", encoding="utf-8")
    print("Prefetching the CPU waveform decoder S3 weights.", flush=True)
    hf_hub_download(repo_id="ResembleAI/chatterbox-turbo", filename="s3gen_meanflow.safetensors")

    torch_prefix = subprocess.check_output(
        [str(python), "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
        text=True,
    ).strip()
    cmake = VENV / "bin" / "cmake"
    build_dir = CPP_ROOT / "build"
    env["CMAKE_PREFIX_PATH"] = torch_prefix
    run([
        str(cmake), "-S", str(CPP_ROOT), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=20",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
        "-DLLAMA_BUILD_KITTEN_TTS=ON",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
    ], env=env)
    run([str(cmake), "--build", str(build_dir), "--target", "kitten-tts", "-j2"], env=env)
    binary = build_dir / "bin" / "kitten-tts"
    torch_lib = prep_site_packages / "torch" / "lib"
    env["LD_LIBRARY_PATH"] = str(torch_lib) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    run([str(binary), "--help"], env=env)
    required = [
        MODEL_ROOT / "cpp/model-tq2_1.gguf",
        MODEL_ROOT / "cpp/default/decoder.pt",
        MODEL_ROOT / "cpp/default/voices.json",
        MODEL_ROOT / "speaker/model.safetensors",
        MODEL_ROOT / "lm/model-ternary.safetensors",
        torch_lib / "libtorch_cpu.so",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Installation is incomplete:\n" + "\n".join(missing))

    try:
        import pwd
        account = pwd.getpwnam(SERVICE_USER)
        for directory in (HF_HOME, PROFILE_ROOT):
            for current, subdirs, filenames in os.walk(directory):
                os.chown(current, account.pw_uid, account.pw_gid)
                for filename in filenames:
                    os.chown(Path(current) / filename, account.pw_uid, account.pw_gid, follow_symlinks=False)
    except KeyError as exc:
        raise RuntimeError(
            f"Service account {SERVICE_USER!r} does not exist. Set AI_VIDEO_STUDIO_USER to the actual systemd service user."
        ) from exc

    print("\nKittenTTS 2 CPU installation is ready but NOT enabled.")
    print("Add these environment values to BOTH API and worker service overrides:")
    settings = {
        "ENABLE_KITTEN_TTS": "1",
        "KITTEN_TTS_MODEL_ROOT": MODEL_ROOT,
        "KITTEN_TTS_BINARY": binary,
        "KITTEN_TTS_PREP_PYTHON": python,
        "KITTEN_TTS_PREP_SCRIPT": CPP_ROOT / "tools/kitten-tts/prepare_reference.py",
        "KITTEN_TTS_PYTHON_PACKAGE": PY_PACKAGE,
        "KITTEN_TTS_PROFILE_ROOT": PROFILE_ROOT,
        "KITTEN_TTS_TORCH_LIB": torch_lib,
        "KITTEN_TTS_HF_HOME": HF_HOME,
        "KITTEN_TTS_DECODER": DECODER,
        "KITTEN_TTS_MODEL_REVISION": model_revision,
        "KITTEN_TTS_THREADS": "2",
    }
    for key, value in settings.items():
        print(f'{key}="{value}"')
    print("Leave ENABLE_TTS=1 unchanged to retain Pocket TTS as the fallback.")
    print("Review model and decoder licences before enabling; no Studio service was restarted.")


if __name__ == "__main__":
    main()
