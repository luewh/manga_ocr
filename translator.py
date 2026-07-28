"""
translator.py

Thin wrapper so the translation backend is a one-line swap. Default is
deep-translator's GoogleTranslator (free, no API key). For production
quality on Japanese, DeepL (deep-translator also wraps it) is usually
noticeably better -- see DeepLTranslatorBackend below, just needs an API key.
"""

from typing import List


class GoogleTranslatorBackend:
    def __init__(self, source: str = "auto", target: str = "en"):
        try:
            from deep_translator import GoogleTranslator
        except ImportError as e:
            raise ImportError("Run: pip install deep-translator") from e
        self._cls = GoogleTranslator
        self.source = source
        self.target = target

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            return self._cls(source=self.source, target=self.target).translate(text)
        except Exception as e:
            return f"[translation failed: {e}]"

    def translate_batch(self, texts: List[str]) -> List[str]:
        return [self.translate(t) for t in texts]


class DeepLTranslatorBackend:
    """Requires a DeepL API key (free tier available at deepl.com/pro-api)."""

    def __init__(self, api_key: str, source: str = "auto", target: str = "EN-US"):
        try:
            from deep_translator import DeeplTranslator
        except ImportError as e:
            raise ImportError("Run: pip install deep-translator") from e
        self._translator = DeeplTranslator(api_key=api_key, source=source, target=target)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            return self._translator.translate(text)
        except Exception as e:
            return f"[translation failed: {e}]"

    def translate_batch(self, texts: List[str]) -> List[str]:
        return [self.translate(t) for t in texts]
