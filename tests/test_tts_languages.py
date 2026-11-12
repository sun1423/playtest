import unittest

from tts_languages import KITTEN_TTS_LANGUAGES, POCKET_TTS_LANGUAGES, language_name, language_options_for_engine, validate_tts_language


class TtsLanguageTests(unittest.TestCase):
    def test_kitten_exposes_its_twenty_supported_languages(self):
        self.assertEqual(len(language_options_for_engine("kitten")), 20)
        self.assertEqual(validate_tts_language("kitten", "fr"), "fr")
        self.assertIn(("he", "Hebrew"), KITTEN_TTS_LANGUAGES)

    def test_pocket_exposes_hindi_only(self):
        self.assertEqual(language_options_for_engine("pocket"), [{"code": "hi", "name": "Hindi"}])
        self.assertEqual(POCKET_TTS_LANGUAGES, (("hi", "Hindi"),))
        with self.assertRaises(ValueError):
            validate_tts_language("pocket", "en")

    def test_language_validation_rejects_unknown_codes(self):
        with self.assertRaises(ValueError):
            validate_tts_language("kitten", "xx")

    def test_language_name(self):
        self.assertEqual(language_name("ja"), "Japanese")


if __name__ == "__main__":
    unittest.main()
