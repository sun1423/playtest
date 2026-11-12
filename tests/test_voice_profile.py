import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voice_profile import (
    HINDI_ASR_PROMPT,
    _active_intervals_from_silence,
    choose_active_audio_window,
    gain_for_levels,
    transcribe_hindi,
    transcribe_language,
)


class VoiceProfileTests(unittest.TestCase):
    def test_hindi_transcription_prompts_devanagari_and_retries_urdu_output(self):
        class FakeASR:
            def __init__(self):
                self.outputs = ["آپ کیسے ہیں؟", "आप कैसे हैं?"]
                self.calls = []

            def transcribe(self, audio_path, **kwargs):
                self.calls.append(kwargs)
                return [SimpleNamespace(text=self.outputs.pop(0))], SimpleNamespace(
                    language="hi", language_probability=0.99
                )

        fake = FakeASR()
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"VOICE_ASR_MODEL": "base", "VOICE_ASR_BACKEND": "whisper"}), \
             patch("voice_profile._ASR_MODEL", fake), \
             patch("voice_profile._ASR_MODEL_NAME", "base"):
            result = transcribe_hindi(
                Path(directory) / "reference.wav", Path(directory) / "models"
            )

        self.assertEqual(result["text"], "आप कैसे हैं?")
        self.assertEqual(result["transcription_attempts"], 2)
        self.assertEqual(result["script_warning"], "")
        self.assertEqual(fake.calls[0]["language"], "hi")
        self.assertIn("देवनागरी", fake.calls[0]["initial_prompt"])
        self.assertIn("देवनागरी", HINDI_ASR_PROMPT)
        self.assertEqual(fake.calls[0]["beam_size"], 5)

    def test_multilingual_whisper_uses_the_selected_language_code(self):
        class FakeASR:
            def __init__(self):
                self.calls = []

            def transcribe(self, audio_path, **kwargs):
                self.calls.append(kwargs)
                return [SimpleNamespace(text="Bonjour, comment allez-vous ?")], SimpleNamespace(
                    language="fr", language_probability=0.98
                )

        fake = FakeASR()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"VOICE_ASR_MODEL": "base"}):
                with patch("voice_profile._ASR_MODEL", fake), patch("voice_profile._ASR_MODEL_NAME", "base"):
                    result = transcribe_language(
                        Path(directory) / "reference.wav",
                        Path(directory) / "models",
                        "fr",
                    )

        self.assertEqual(result["text"], "Bonjour, comment allez-vous ?")
        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["requested_language"], "fr")
        self.assertEqual(fake.calls[0]["language"], "fr")
        self.assertIsNone(fake.calls[0]["initial_prompt"])

    def test_multilingual_dispatch_rejects_unsupported_language(self):
        with self.assertRaisesRegex(ValueError, "supports these languages"):
            transcribe_language(Path("reference.wav"), Path("models"), "xx")

    def test_persistent_urdu_script_is_reported_instead_of_silently_accepted(self):
        class FakeASR:
            def transcribe(self, audio_path, **kwargs):
                return [SimpleNamespace(text="آپ کیسے ہیں؟")], SimpleNamespace(
                    language="hi", language_probability=0.8
                )

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"VOICE_ASR_MODEL": "base", "VOICE_ASR_BACKEND": "whisper"}), \
             patch("voice_profile._ASR_MODEL", FakeASR()), \
             patch("voice_profile._ASR_MODEL_NAME", "base"):
            result = transcribe_hindi(
                Path(directory) / "reference.wav", Path(directory) / "models"
            )

        self.assertEqual(result["transcription_attempts"], 2)
        self.assertTrue(result["script_warning"])
        self.assertIn("Urdu/Arabic", result["script_warning"])


    def test_indicconformer_is_preferred_when_selected_backend_is_default(self):
        expected = {
            "text": "यह एक हिंदी परीक्षण है।",
            "language": "hi",
            "engine": "AI4Bharat IndicConformer Hindi · ONNX CPU",
            "script_warning": "",
        }
        with patch.dict(os.environ, {"VOICE_ASR_BACKEND": "indicconformer"}), \
             patch("indicconformer_asr.transcribe_indicconformer_hindi", return_value=expected) as recognizer:
            result = transcribe_hindi(Path("reference.wav"), Path("models/whisper"))
        for key, value in expected.items():
            self.assertEqual(result.get(key), value)
        self.assertEqual(result["backend"], "indicconformer")
        recognizer.assert_called_once()

    def test_indicconformer_failure_falls_back_only_when_explicitly_configured(self):
        from unittest.mock import Mock
        from voice_profile import _ASR_BACKENDS

        fallback = {"text": "नमस्ते", "engine": "Whisper CPU/int8"}
        whisper = Mock(return_value=fallback)
        original_whisper = _ASR_BACKENDS["whisper"]
        _ASR_BACKENDS["whisper"] = whisper
        try:
            with patch.dict(os.environ, {"VOICE_ASR_BACKEND": "indicconformer", "VOICE_ASR_FALLBACK_BACKEND": "whisper"}), \
                 patch("indicconformer_asr.transcribe_indicconformer_hindi", side_effect=RuntimeError("model not installed")):
                result = transcribe_hindi(Path("reference.wav"), Path("models/whisper"))
        finally:
            _ASR_BACKENDS["whisper"] = original_whisper
        self.assertEqual(result["text"], "नमस्ते")
        self.assertEqual(result["backend"], "whisper")
        self.assertEqual(result["requested_backend"], "indicconformer")
        self.assertIn("model not installed", result["fallback_reason"])
        whisper.assert_called_once()


    def test_indicconformer_is_the_default_backend(self):
        expected = {"text": "यह हिंदी है।", "language": "hi"}
        with patch.dict(os.environ, {}, clear=True), patch(
            "indicconformer_asr.transcribe_indicconformer_hindi", return_value=expected
        ) as recognizer:
            result = transcribe_hindi(Path("reference.wav"), Path("models/whisper"))
        self.assertEqual(result["backend"], "indicconformer")
        self.assertEqual(result["requested_backend"], "indicconformer")
        recognizer.assert_called_once()

    def test_selected_backend_failure_does_not_silently_downgrade(self):
        from unittest.mock import Mock
        from voice_profile import _ASR_BACKENDS

        whisper = Mock()
        original_whisper = _ASR_BACKENDS["whisper"]
        _ASR_BACKENDS["whisper"] = whisper
        try:
            with patch.dict(os.environ, {"VOICE_ASR_BACKEND": "indicconformer"}, clear=True), patch(
                "indicconformer_asr.transcribe_indicconformer_hindi",
                side_effect=RuntimeError("model unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "(?i)selected hindi asr backend.*model unavailable"):
                    transcribe_hindi(Path("reference.wav"), Path("models/whisper"))
        finally:
            _ASR_BACKENDS["whisper"] = original_whisper
        whisper.assert_not_called()

    def test_new_backend_can_be_registered_and_selected_later(self):
        from voice_profile import register_asr_backend, available_asr_backends

        def future_model(path, model_cache):
            return {"text": "नया मॉडल", "language": "hi", "engine": "Future model"}

        with patch("voice_profile._ASR_BACKENDS", {}):
            register_asr_backend("future-model", future_model)
            self.assertIn("future-model", available_asr_backends())
            with patch.dict(os.environ, {"VOICE_ASR_BACKEND": "future-model"}, clear=True):
                result = transcribe_hindi(Path("reference.wav"), Path("models"))
        self.assertEqual(result["text"], "नया मॉडल")
        self.assertEqual(result["backend"], "future-model")

    def test_registered_backend_name_must_be_unique(self):
        from voice_profile import register_asr_backend

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_asr_backend("indicconformer", lambda path, cache: {})

    def test_selects_high_activity_twenty_second_clip(self):
        start, duration, ratio = choose_active_audio_window(
            60.0,
            [(0.0, 8.0), (10.0, 27.0), (35.0, 58.0)],
            20.0,
        )
        self.assertEqual(duration, 20.0)
        self.assertEqual(start, 35.0)
        self.assertEqual(ratio, 1.0)

    def test_short_clip_is_kept_without_padding(self):
        start, duration, ratio = choose_active_audio_window(8.0, [(0.5, 7.5)], 20.0)
        self.assertEqual((start, duration), (0.0, 8.0))
        self.assertEqual(ratio, 0.875)

    def test_silence_detection_is_converted_to_active_intervals(self):
        events = (
            "silence_start: 3.000 | silence_end: 4.000\n"
            "silence_start: 8.000 | silence_end: 9.000\n"
        )
        self.assertEqual(
            _active_intervals_from_silence(events, 10.0),
            [(0.0, 3.0), (4.0, 8.0), (9.0, 10.0)],
        )

    def test_gain_only_changes_audio_when_level_requires_it(self):
        self.assertEqual(gain_for_levels(-22.0, -5.0), 0.0)
        self.assertEqual(gain_for_levels(-35.0, -8.0), 5.0)
        self.assertEqual(gain_for_levels(-35.0, -2.0), 0.0)
        self.assertEqual(gain_for_levels(-18.0, -0.5), -2.5)


if __name__ == "__main__":
    unittest.main()
