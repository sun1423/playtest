import unittest
import tempfile
import wave
from pathlib import Path


import numpy as np

from cpu_lipsync import (
    _mouth_mask,
    build_mel_chunks,
    plan_video_timing,
    wav2lip_melspectrogram,
)


class CpuLipSyncTests(unittest.TestCase):
    def test_audio_only_probe_accepts_a_valid_wav(self):
        from cpu_lipsync import probe_media
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 16000)
            info = probe_media(path, require_video=False)
            self.assertAlmostEqual(info["duration_seconds"], 1.0, places=2)
            self.assertTrue(info["has_audio"])
            self.assertEqual(info["width"], 0)

    def test_video_probe_rejects_audio_only_wav(self):
        from cpu_lipsync import probe_media
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 16000)
            with self.assertRaisesRegex(ValueError, "no video stream"):
                probe_media(path)

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
        mel = np.arange(80 * 400, dtype=np.float32).reshape(80, 400)
        chunks = build_mel_chunks(mel, frame_count=151, fps=25)
        self.assertEqual(len(chunks), 151)
        self.assertEqual(chunks[0].shape, (80, 16))
        self.assertEqual(chunks[-1].shape, (80, 16))

    def test_pads_short_mel_without_changing_frame_count(self):
        mel = np.ones((80, 3), dtype=np.float32)
        chunks = build_mel_chunks(mel, frame_count=5)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(chunk.shape == (80, 16) for chunk in chunks))

    def test_wav2lip_mel_features_are_finite_and_normalized(self):
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        mel = wav2lip_melspectrogram(waveform)
        self.assertEqual(mel.shape[0], 80)
        self.assertGreater(mel.shape[1], 70)
        self.assertTrue(np.all(np.isfinite(mel)))
        self.assertGreaterEqual(float(mel.min()), -4.0)
        self.assertLessEqual(float(mel.max()), 4.0)

    def test_mouth_mask_preserves_upper_face_and_feathers_edges(self):
        import cv2
        mask = _mouth_mask(cv2, 200, 240, {})
        self.assertEqual(mask.shape, (240, 200, 1))
        self.assertLess(float(mask[:int(240 * 0.38)].max()), 0.01)
        self.assertGreater(float(mask[160, 100, 0]), 0.8)
        self.assertLess(float(mask[160, 4, 0]), 0.05)


if __name__ == "__main__":
    unittest.main()
