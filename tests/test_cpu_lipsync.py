import unittest

from cpu_lipsync import build_mel_chunks, plan_video_timing


class CpuLipSyncTests(unittest.TestCase):
    def test_keeps_source_duration_when_speech_is_shorter(self):
        result = plan_video_timing(30.0, 18.0)
        self.assertEqual(result["video_speed_factor"], 1.0)
        self.assertEqual(result["frame_count"], 750)
        self.assertEqual(result["target_duration"], 30.0)

    def test_slowdowns_source_modestly_when_speech_is_longer(self):
        result = plan_video_timing(20.0, 22.0)
        self.assertEqual(result["video_speed_factor"], 1.1)
        self.assertGreaterEqual(result["target_duration"], 22.0)
        self.assertEqual(result["frame_count"], 550)

    def test_rejects_narration_that_would_freeze_or_cut_video(self):
        with self.assertRaisesRegex(ValueError, "more than 20% longer"):
            plan_video_timing(10.0, 13.0)

    def test_rejects_invalid_durations(self):
        for video, speech in ((0, 2), (2, 0), (float("nan"), 2), (2, float("inf"))):
            with self.subTest(video=video, speech=speech), self.assertRaises(ValueError):
                plan_video_timing(video, speech)

    def test_builds_exact_number_of_mel_windows(self):
        import numpy as np
        mel = np.arange(80 * 400, dtype=np.float32).reshape(80, 400)
        chunks = build_mel_chunks(mel, frame_count=151, fps=25)
        self.assertEqual(len(chunks), 151)
        self.assertEqual(chunks[0].shape, (80, 16))
        self.assertEqual(chunks[-1].shape, (80, 16))

    def test_pads_short_mel_without_changing_frame_count(self):
        import numpy as np
        mel = np.ones((80, 3), dtype=np.float32)
        chunks = build_mel_chunks(mel, frame_count=5)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(chunk.shape == (80, 16) for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
