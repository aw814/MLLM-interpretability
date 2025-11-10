"""
macro_features_reorg.py
-----------------------
Clean module exposing exactly five functions:
  - check_languages_family(language1, language2) -> bool
  - check_languages_genus(language1, language2) -> bool
  - get_script(language: str) -> str | None
  - get_syllable_count(language: str) -> int | None
  - get_wiki_size(language: str) -> int | str | None  (returns cached count or "FAILED: ..." string)

No third-party dependencies are required.
Script detection uses Unicode name heuristics on short sample texts.
Family/Genus come from a lightweight mapping table tailored to common languages.
Wikipedia sizes default to a cached snapshot; if 'requests' is available, a live fetch is attempted first.
"""

from typing import Optional
import unicodedata

# ------------------------------------------------------------------
# Lightweight language -> (family, genus) mapping (lowercased keys)
# ------------------------------------------------------------------
_LANG2FAMILY = {
    "english": ("Indo-European", "Germanic"),
    "german": ("Indo-European", "Germanic"),
    "dutch": ("Indo-European", "Germanic"),
    "swedish": ("Indo-European", "Germanic"),
    "danish": ("Indo-European", "Germanic"),
    "norwegian": ("Indo-European", "Germanic"),
    "icelandic": ("Indo-European", "Germanic"),

    "french": ("Indo-European", "Romance"),
    "spanish": ("Indo-European", "Romance"),
    "italian": ("Indo-European", "Romance"),
    "portuguese": ("Indo-European", "Romance"),
    "romanian": ("Indo-European", "Romance"),

    "russian": ("Indo-European", "Slavic"),
    "polish": ("Indo-European", "Slavic"),
    "czech": ("Indo-European", "Slavic"),
    "ukrainian": ("Indo-European", "Slavic"),
    "bulgarian": ("Indo-European", "Slavic"),

    "greek": ("Indo-European", "Hellenic"),
    "hindi": ("Indo-European", "Indo-Aryan"),
    "bengali": ("Indo-European", "Indo-Aryan"),
    "punjabi": ("Indo-European", "Indo-Aryan"),
    "urdu": ("Indo-European", "Indo-Aryan"),

    "persian": ("Indo-European", "Iranian"),
    "kurdish": ("Indo-European", "Iranian"),
    "pashto": ("Indo-European", "Iranian"),

    "turkish": ("Turkic", "Turkic"),
    "kazakh": ("Turkic", "Turkic"),
    "uzbek": ("Turkic", "Turkic"),

    "finnish": ("Uralic", "Finnic"),
    "estonian": ("Uralic", "Finnic"),
    "hungarian": ("Uralic", "Ugric"),

    "arabic": ("Afro-Asiatic", "Semitic"),
    "hebrew": ("Afro-Asiatic", "Semitic"),
    "amharic": ("Afro-Asiatic", "Semitic"),

    "swahili": ("Niger-Congo", "Bantu"),
    "zulu": ("Niger-Congo", "Bantu"),
    "yoruba": ("Niger-Congo", "Other"),

    "mandarin chinese": ("Sino-Tibetan", "Sinitic"),
    "cantonese": ("Sino-Tibetan", "Sinitic"),
    "tibetan": ("Sino-Tibetan", "Tibetic"),
    "burmese": ("Sino-Tibetan", "Burmish"),

    "japanese": ("Japonic", "Japonic"),
    "korean": ("Koreanic", "Koreanic"),

    "vietnamese": ("Austroasiatic", "Vietic"),
    "khmer": ("Austroasiatic", "Mon-Khmer"),
    "thai": ("Tai-Kadai", "Kra-Dai"),
    "lao": ("Tai-Kadai", "Kra-Dai"),
    "chinese": ("Sino-Tibetan", "Sinitic"),
    "indonesian": ("Austronesian", "Malayo-Polynesian"),
}

# ------------------------------------------------------------------
# Short sample texts for Unicode-based script detection
# ------------------------------------------------------------------
_SAMPLES = {
    "english": "Hello world",
    "french": "Bonjour tout le monde",
    "german": "Guten Morgen",
    "hebrew": "שלום עולם",
    "hindi": "नमस्ते दुनिया",
    "indonesian": "Selamat pagi",
    "italian": "Ciao mondo",
    "japanese": "こんにちは世界",
    "korean": "안녕하세요 세상",
    "portuguese": "Olá mundo",
    "spanish": "Hola mundo",
    "chinese": "你好世界",
    "cantonese": "你好，世界",
    "greek": "Γειά σου Κόσμε",
    "russian": "Привет мир",
}

def _detect_script(text: str) -> Optional[str]:
    """Return the dominant Unicode script name in text (e.g., 'LATIN', 'CYRILLIC')."""
    scripts = []
    for ch in text:
        if ch.isalpha():
            try:
                scripts.append(unicodedata.name(ch).split()[0])
            except ValueError:
                continue
    if not scripts:
        return None
    top = max(set(scripts), key=scripts.count)
    # Coalesce common East-Asian labels into a single friendly tag
    if top in {'CJK', 'HIRAGANA', 'KATAKANA'}:
        return 'HAN/JAPANESE'
    if top == 'HANGUL':
        return 'HANGUL'
    return top

# ------------------------------------------------------------------
# Cached Wikipedia article counts (snapshot). Live fetch attempted if possible.
# ------------------------------------------------------------------
_LANG_CODES = {
    "english": "en", "french": "fr", "german": "de", "hebrew": "he",
    "hindi": "hi", "indonesian": "id", "italian": "it", "japanese": "ja",
    "korean": "ko", "portuguese": "pt", "spanish": "es", "chinese": "zh",
}

_WIKI_COUNTS_CACHE = {
    'english': 7087876, 'french': 2719280, 'german': 3067614, 'hebrew': 385688,
    'hindi': 167234, 'indonesian': 751190, 'italian': 1943704, 'japanese': 1479698,
    'korean': 728295, 'portuguese': 1159764, 'spanish': 2073043, 'chinese': 1509727
}

def _live_wiki_articles(lang: str) -> Optional[int]:
    """Attempt a live fetch of article counts via MediaWiki API. Returns None if unavailable."""
    try:
        import requests  # local import to keep dependency optional
        code = _LANG_CODES.get(lang.lower())
        if not code:
            return None
        url = f"https://{code}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "meta": "siteinfo",
            "siprop": "statistics",
            "format": "json",
            "formatversion": "2"
        }
        headers = {"User-Agent": "macro-features/1.0 (research-contact@example.com)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return int(data["query"]["statistics"]["articles"])
    except Exception:
        return None

# ------------------------------------------------------------------
# Syllable inventory table (lowercased keys)
# ------------------------------------------------------------------
_SYLLABLES = {
    "basque": 2082,
    "english": 6949,
    "cantonese": 1298,
    "catalan": 3600,
    "finnish": 3844,
    "french": 2949,
    "german": 5100,
    "hungarian": 4325,
    "italian": 2729,
    "japanese": 643,
    "korean": 1104,
    "chinese": 1274,
    "serbian": 3831,
    "spanish": 2778,
    "thai": 2438,
    "turkish": 3260,
    "vietnamese": 2776,
    "hebrew": 2000,
    "hindi": 3500,
    "indonesian": 2200,
    "portuguese": 3000,
}

# =========================
#  Public API (5 functions)
# =========================

def check_languages_family(language1: str, language2: str) -> bool:
    """Return True if two languages share the same family (case-insensitive)."""
    fam1 = _LANG2FAMILY.get(language1.lower(), ("Unknown", "Unknown"))[0]
    fam2 = _LANG2FAMILY.get(language2.lower(), ("Unknown", "Unknown"))[0]
    return fam1 != "Unknown" and fam1 == fam2

def check_languages_genus(language1: str, language2: str) -> bool:
    """Return True if two languages share the same genus/sub-branch (case-insensitive)."""
    gen1 = _LANG2FAMILY.get(language1.lower(), ("Unknown", "Unknown"))[1]
    gen2 = _LANG2FAMILY.get(language2.lower(), ("Unknown", "Unknown"))[1]
    return gen1 != "Unknown" and gen1 == gen2

def get_script(language: str) -> Optional[str]:
    """Return a human-friendly script label for a language using Unicode heuristic on sample text."""
    sample = _SAMPLES.get(language.lower(), "")
    return _detect_script(sample) if sample else None

def get_syllable_count(language: str) -> Optional[int]:
    """Return the approximate syllable inventory size for a language, if available."""
    return _SYLLABLES.get(language.lower())

def get_wiki_size(language: str):
    """Return article count for the language Wikipedia. Tries live fetch, falls back to cached snapshot.

    Returns an int (count) or None if unknown.

    """
    lang = language.lower()
    live = _live_wiki_articles(lang)
    if live is not None:
        return live
    return _WIKI_COUNTS_CACHE.get(lang)
