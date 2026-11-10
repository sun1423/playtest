import os
import tempfile
import unittest
from pathlib import Path

from pocket_tts import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_SEED,
    REQUIRED_MODEL_FILES,
    build_command,
    clear_voice_cache,
    engine_ready,
    normalize_generation_settings,
    normalize_voice_id,
    normalize_dialogue_text,
    split_dialogue_chunks,
    resolve_voice_reference,
    valid_voice_id,
)


class PocketTtsTests(unittest.TestCase):
    def make_engine(self, root: Path) -> Path:
        binary = root / "onnx/runtime/pocket-tts"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        (binary.parent / "libonnxruntime.so").write_bytes(b"runtime")
        for relative in REQUIRED_MODEL_FILES:
            model_file = root / relative
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_bytes(b"model")
        return binary

    def make_voice(self, root: Path, name: str = "my_voice") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.wav"
        path.write_bytes(b"RIFF" + (32).to_bytes(4, "little") + b"WAVE" + b"data")
        return path


    def test_dialogue_normalization_preserves_prosody_and_removes_markup_noise(self):
        text = normalize_dialogue_text(
            "नमस्ते!!!\nआप कैसे हैं???\n\n**मैं ठीक हूँ…** 😊"
        )
        self.assertIn("नमस्ते!", text)
        self.assertIn("आप कैसे हैं?", text)
        self.assertIn("मैं ठीक हूँ।", text)
        self.assertNotIn("😊", text)
        self.assertNotIn("**", text)
        self.assertIn("\n\n", text)

    def test_dialogue_chunking_keeps_chunks_bounded_and_paragraph_pauses(self):
        source = ("यह एक वाक्य है। " * 90) + "\n\n" + ("दूसरा पैराग्राफ है। " * 30)
        chunks = split_dialogue_chunks(source, max_chars=180)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(0 < len(chunk) <= 180 for chunk, _ in chunks))
        self.assertEqual(chunks[-1][1], 0.0)
        self.assertTrue(any(pause >= 0.38 for _, pause in chunks[:-1]))

    def test_dialogue_normalization_rejects_symbols_without_speech(self):
        with self.assertRaises(ValueError):
            normalize_dialogue_text("😊 ★ !!!")

    def test_engine_is_disabled_by_default_even_with_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = self.make_engine(root)
            self.assertFalse(engine_ready(False, root, binary))
            self.assertTrue(engine_ready(True, root, binary))

    def test_engine_requires_all_model_files_and_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = self.make_engine(root)
            (root / REQUIRED_MODEL_FILES[0]).unlink()
            self.assertFalse(engine_ready(True, root, binary))
            (root / REQUIRED_MODEL_FILES[0]).write_bytes(b"model")
            binary.chmod(0o644)
            self.assertFalse(engine_ready(True, root, binary))

    def test_voice_id_rejects_path_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            voices = base / "voices"
            self.make_voice(voices)
            outside = self.make_voice(base / "outside", "external")
            (voices / "escape.wav").symlink_to(outside)
            with self.assertRaises(ValueError):
                resolve_voice_reference(voices, "../outside/external", 1024)
            with self.assertRaises(ValueError):
                resolve_voice_reference(voices, "escape", 1024)
            self.assertEqual(resolve_voice_reference(voices, "my_voice", 1024).name, "my_voice.wav")
            self.make_voice(voices, "आवाज़")
            self.assertEqual(normalize_voice_id("आवाज़"), "आवाज़")
            self.assertTrue(valid_voice_id("आवाज़"))
            self.assertFalse(valid_voice_id("../आवाज़"))
            self.assertEqual(resolve_voice_reference(voices, "आवाज़", 1024).name, "आवाज़.wav")

    def test_voice_size_and_wav_header_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            voices = Path(temp)
            self.make_voice(voices)
            with self.assertRaises(ValueError):
                resolve_voice_reference(voices, "my_voice", 8)
            (voices / "bad.wav").write_bytes(b"not-a-wave-file")
            with self.assertRaises(ValueError):
                resolve_voice_reference(voices, "bad", 1024)

    def test_voice_cache_is_cleared_when_a_profile_is_replaced_or_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            voices = Path(temp)
            cache = voices / ".cache"
            cache.mkdir()
            (cache / "my_voice.emb").write_bytes(b"embedding")
            (cache / "my_voice.kv").write_bytes(b"kv-state")
            (cache / "other_voice.emb").write_bytes(b"keep")
            clear_voice_cache(voices, "my_voice")
            self.assertFalse((cache / "my_voice.emb").exists())
            self.assertFalse((cache / "my_voice.kv").exists())
            self.assertTrue((cache / "other_voice.emb").exists())

    def test_command_uses_two_threads_and_keeps_disk_cache_enabled(self):
        command = build_command(
            "यह हिंदी परीक्षण है",
            "my_voice.wav",
            Path("/models/pocket"),
            Path("/voices"),
        )
        self.assertIn("--threads", command)
        self.assertEqual(command[command.index("--threads") + 1], "2")
        self.assertIn("--precision", command)
        self.assertEqual(command[command.index("--precision") + 1], "int8")
        self.assertEqual(command[command.index("--eos-threshold") + 1], str(DEFAULT_EOS_THRESHOLD))
        self.assertEqual(command[command.index("--seed") + 1], str(DEFAULT_SEED))
        self.assertEqual(DEFAULT_SEED, 1)
        self.assertEqual(DEFAULT_EOS_THRESHOLD, -2.0)
        self.assertNotIn("--no-cache", command)
        self.assertEqual(command[-2], "यह हिंदी परीक्षण है")
        self.assertEqual(command[-1], "my_voice.wav")

    def test_generation_settings_defaults_and_overrides_are_validated(self):
        self.assertEqual(normalize_generation_settings(), {
            "temperature": 0.3, "flow_steps": 1, "eos_threshold": -2.0, "seed": 1
        })
        settings = normalize_generation_settings({
            "temperature": 0.65, "flow_steps": 3, "eos_threshold": -1.2, "seed": 42
        })
        command = build_command(
            "यह हिंदी परीक्षण है", "my_voice.wav", Path("/models"), Path("/voices"),
            settings=settings,
        )
        self.assertEqual(command[command.index("--temperature") + 1], "0.65")
        self.assertEqual(command[command.index("--lsd-steps") + 1], "3")
        self.assertEqual(command[command.index("--eos-threshold") + 1], "-1.2")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        for invalid in (
            {"temperature": 2.0}, {"flow_steps": 0}, {"eos_threshold": -7.0},
            {"seed": -1}, {"seed": True}, {"temperature": float("nan")},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_generation_settings(invalid)

    def test_command_rejects_empty_or_oversized_text(self):
        with self.assertRaises(ValueError):
            build_command(" ", "my_voice.wav", Path("/models"), Path("/voices"))
        with self.assertRaises(ValueError):
            build_command("x" * 12001, "my_voice.wav", Path("/models"), Path("/voices"))


if __name__ == "__main__":
    unittest.main()
