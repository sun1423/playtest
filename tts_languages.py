"""Speech-language capabilities exposed by the installed local TTS engines."""

KITTEN_TTS_LANGUAGES = (
    ("en", "English"), ("ar", "Arabic"), ("zh", "Chinese"), ("fr", "French"),
    ("de", "German"), ("hi", "Hindi"), ("it", "Italian"), ("pt", "Portuguese"),
    ("ru", "Russian"), ("es", "Spanish"), ("ja", "Japanese"), ("ko", "Korean"),
    ("tr", "Turkish"), ("nl", "Dutch"), ("sv", "Swedish"), ("da", "Danish"),
    ("fi", "Finnish"), ("sw", "Swahili"), ("el", "Greek"), ("he", "Hebrew"),
)

POCKET_TTS_LANGUAGES = (("hi", "Hindi"),)


def language_options_for_engine(engine: str) -> list[dict[str, str]]:
    """Return only languages genuinely supported by the selected local TTS engine."""
    if engine == "kitten":
        options = KITTEN_TTS_LANGUAGES
    elif engine == "pocket":
        options = POCKET_TTS_LANGUAGES
    else:
        raise ValueError(f"Unsupported speech engine: {engine}")
    return [{"code": code, "name": name} for code, name in options]


def validate_tts_language(engine: str, language: str) -> str:
    """Validate an ISO language code against an engine's actual capability list."""
    code = str(language or "").strip().lower()
    options = language_options_for_engine(engine)
    if code not in {item["code"] for item in options}:
        names = ", ".join(item["name"] for item in options)
        raise ValueError(f"{engine.title()} supports these languages only: {names}.")
    return code


def language_name(language: str) -> str:
    """Resolve a supported language code to its display name."""
    code = str(language or "").strip().lower()
    for item in language_options_for_engine("kitten"):
        if item["code"] == code:
            return item["name"]
    return code.upper() if code else "Unknown"
