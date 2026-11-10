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
             patch.dict(os.environ, {"VOICE_ASR_MODEL": "base"}), \
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

    def test_persistent_urdu_script_is_reported_instead_of_silently_accepted(self):
        class FakeASR:
            def transcribe(self, audio_path, **kwargs):
                return [SimpleNamespace(text="آپ کیسے ہیں؟")], SimpleNamespace(
                    language="hi", language_probability=0.8
                )

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"VOICE_ASR_MODEL": "base"}), \
             patch("voice_profile._ASR_MODEL", FakeASR()), \
             patch("voice_profile._ASR_MODEL_NAME", "base"):
            result = transcribe_hindi(
                Path(directory) / "reference.wav", Path(directory) / "models"
            )

        self.assertEqual(result["transcription_attempts"], 2)
        self.assertTrue(result["script_warning"])
        self.assertIn("Urdu/Arabic", result["script_warning"])

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
