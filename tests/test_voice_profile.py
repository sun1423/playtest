import unittest

from voice_profile import (
    _active_intervals_from_silence,
    choose_active_audio_window,
    gain_for_levels,
)


class VoiceProfileTests(unittest.TestCase):
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
