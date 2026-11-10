import hashlib
import tempfile
import unittest
import wave
from pathlib import Path

from voice_reference import VoiceAudioError, analyze_pcm16, prepare_wav, validate_settings


class VoiceReferenceTests(unittest.TestCase):
    def test_processing_settings_are_allowlisted(self):
        settings = validate_settings(False, False, False, 80, 10000)
        self.assertEqual(settings["highpass_hz"], 80)
        self.assertEqual(settings["output_format"], "WAV PCM16, mono, 24000 Hz")
        with self.assertRaises(VoiceAudioError):
            validate_settings(False, False, False, 10000, 8000)
        with self.assertRaises(VoiceAudioError):
            validate_settings(False, False, False, 75, 0)

    def test_analysis_reports_silence_and_clipping_without_claiming_clone_quality(self):
        pcm = (b"\x00\x00" * 24000) + (b"\xff\x7f" * 24000)
        result = analyze_pcm16(pcm)
        self.assertGreater(result["near_silence_percent"], 40)
        self.assertGreater(result["clipped_sample_count"], 0)
        self.assertTrue(result["warnings"])

    def test_prepare_converts_copy_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uploaded-source.wav"
            prepared = root / "prepared.wav"
            rate = 48000
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                frames = bytearray()
                for index in range(rate * 7):
                    value = int(7000 * __import__("math").sin(2 * __import__("math").pi * 220 * index / rate))
                    frames.extend(value.to_bytes(2, "little", signed=True))
                    frames.extend(value.to_bytes(2, "little", signed=True))
                handle.writeframes(frames)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            metadata = prepare_wav(source, prepared)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(metadata["source"]["sample_rate_hz"], 48000)
            self.assertEqual(metadata["source"]["channels"], 2)
            self.assertEqual(metadata["prepared"]["sample_rate_hz"], 24000)
            self.assertEqual(metadata["prepared"]["channels"], 1)
            self.assertEqual(metadata["prepared"]["codec"], "pcm_s16le")
            self.assertTrue(metadata["source_preserved"])
            self.assertGreater(metadata["prepared_analysis"]["duration_seconds"], 6)


if __name__ == "__main__":
    unittest.main()
