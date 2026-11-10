import unittest

import numpy as np

from indicconformer_asr import ctc_greedy_decode, contains_non_devanagari_indic_script


class IndicConformerAsrTests(unittest.TestCase):
    def test_ctc_decoder_collapses_repeats_and_blanks(self):
        vocab = ["क", "▁राम", "है"]
        ids = [0, 0, 3, 1, 1, 3, 2, 3]
        logits = np.full((1, len(ids), len(vocab) + 1), -10.0, dtype=np.float32)
        for frame, token in enumerate(ids):
            logits[0, frame, token] = 10.0
        self.assertEqual(ctc_greedy_decode(logits, vocab), "क रामहै")

    def test_hindi_text_is_not_flagged_as_urdu(self):
        self.assertFalse(contains_non_devanagari_indic_script("यह हिंदी वाक्य है।"))

    def test_urdu_script_is_flagged(self):
        self.assertTrue(contains_non_devanagari_indic_script("یہ اردو متن ہے"))


if __name__ == "__main__":
    unittest.main()
