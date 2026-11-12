import json
import tempfile
import unittest
from pathlib import Path

from kitten_tts import engine_ready, is_hindi_text, validate_text


class KittenTtsTests(unittest.TestCase):
    def write_asset(self, root: Path, relative: str, content: bytes = b"model") -> dict:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"file": relative, "size": len(content)}

    def make_install(self, root: Path):
        model = root / "model"
        config = {
            "type": "KITTEN2",
            "lm_dir": "lm",
            "lm_packed": "lm/model-ternary.safetensors",
            "speaker_embedding": "speaker/model.safetensors",
            "cpp": {"version": 1, "gguf": None, "decoders": {"default": {}}},
        }
        config["cpp"]["gguf"] = self.write_asset(model, "cpp/model-tq2_1.gguf")
        decoder = config["cpp"]["decoders"]["default"]
        decoder["torchscript"] = self.write_asset(model, "cpp/default/decoder.pt")
        decoder["voices"] = self.write_asset(model, "cpp/default/voices.json")
        self.write_asset(model, "speaker/model.safetensors")
        self.write_asset(model, "lm/config.json", b"{}")
        self.write_asset(model, "lm/model-ternary.safetensors")
        (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        binary = root / "kitten-tts"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        python = root / "python"
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
        script = root / "prepare_reference.py"
        script.write_text("# reference preparation test fixture\n", encoding="utf-8")
        package = root / "KittenTTS"
        package_dir = package / "kittenml" / "kittentts2"
        package_dir.mkdir(parents=True)
        (package_dir / "speaker.py").write_text("", encoding="utf-8")
        (package_dir / "vocoder.py").write_text("", encoding="utf-8")
        return binary, model, python, script, package

    def test_hindi_detection_is_devanagari_specific(self):
        self.assertTrue(is_hindi_text("नमस्ते, दुनिया"))
        self.assertFalse(is_hindi_text("Namaste, world"))

    def test_expression_markup_and_hindi_are_preserved(self):
        value = "[joyful] नमस्ते <laugh> यह (((बहुत))) अच्छा है"
        self.assertEqual(validate_text(value), value)

    def test_tts_text_normalization_handles_devanagari_canonical_forms(self):
        typed = "पीएम मोदी से सवाल पूछने वाली नार्वे की पत्रकार का भारतीय वीसा उड़ान से कुछ घंटे पहले रद्द"
        normalized = validate_text(typed)
        self.assertIn("उड़ान", normalized)
        # These are real spelling differences and must not be autocorrected.
        self.assertIn("नार्वे", normalized)
        self.assertIn("वीसा", normalized)

    def test_tts_text_normalization_removes_accidental_invisible_controls(self):
        value = validate_text("नम\\u200bस्ते\\u2060 दुनिया\\ufeff")
        self.assertEqual(value, "नमस्ते दुनिया")

    def test_tts_text_normalization_preserves_indic_joiners_and_markup(self):
        value = "[joyful] क्\\u200dष और क्\\u200cष <laugh>"
        self.assertEqual(validate_text(value), value)

    def test_tts_text_normalization_handles_nonbreaking_spaces_and_newlines(self):
        value = validate_text("पहला\\u00a0वाक्य\\r\\nदूसरा\\u202fवाक्य")
        self.assertEqual(value, "पहला वाक्य\\nदूसरा वाक्य")

    def test_rejects_empty_and_oversized_cli_argument(self):
        with self.assertRaises(ValueError):
            validate_text("   ")
        with self.assertRaises(ValueError):
            validate_text("ह" * 40_000)

    def test_engine_readiness_is_opt_in_and_checks_manifest_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            binary, model, python, script, package = self.make_install(Path(temp))
            self.assertFalse(engine_ready(False, binary, model, python, script, package))
            self.assertTrue(engine_ready(True, binary, model, python, script, package))
            (model / "cpp/default/decoder.pt").unlink()
            self.assertFalse(engine_ready(True, binary, model, python, script, package))

    def test_manifest_size_mismatch_disables_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            binary, model, python, script, package = self.make_install(Path(temp))
            (model / "cpp/model-tq2_1.gguf").write_bytes(b"wrong-size")
            self.assertFalse(engine_ready(True, binary, model, python, script, package))


if __name__ == "__main__":
    unittest.main()
