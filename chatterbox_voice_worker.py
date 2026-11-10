#!/usr/bin/env python3
"""Generate narration with a saved Studio WAV voice profile using Chatterbox V3."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--language", choices=("hi", "en"), required=True)
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--output-wav", required=True)
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    reference = Path(args.reference_wav)
    destination = Path(args.output_wav)
    if not text:
        raise SystemExit("Narration text is empty.")
    if not reference.is_file() or reference.stat().st_size < 44:
        raise SystemExit("The selected custom voice WAV is missing or empty.")

    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(
        device=device,
        t3_model="v3",
    )
    with torch.inference_mode():
        audio = model.generate(
            text,
            language_id=args.language,
            audio_prompt_path=str(reference),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(destination), audio.detach().cpu(), model.sr)
    del model, audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved custom voice narration: {destination}")


if __name__ == "__main__":
    main()
