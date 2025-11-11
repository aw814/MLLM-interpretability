import re
import jieba

try:
    from sudachipy import tokenizer as sudachi_tokenizer
    from sudachipy import dictionary
    SUDACHI_AVAILABLE = True
    sudachi_dict = dictionary.Dictionary().create()
    sudachi_mode = sudachi_tokenizer.Tokenizer.SplitMode.C
except ImportError:
    SUDACHI_AVAILABLE = False
    print("Warning: Sudachi not available. Install with: pip install sudachipy sudachidict-core")

try:
    from konlpy.tag import Okt
    # Test if Java is actually available
    try:
        okt = Okt()
        okt.morphs("테스트")
        KONLPY_AVAILABLE = True
    except:
        KONLPY_AVAILABLE = False
        print("Warning: KoNLPy installed but Java not available. Using fallback for Korean.")
except ImportError:
    KONLPY_AVAILABLE = False

try:
    from pythainlp.tokenize import word_tokenize as thai_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False


def _merge_ascii_tokens(text, tokens):
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", text)
    ascii_tokens = [t.lower() for t in ascii_tokens]
    merged = list(tokens)
    for t in ascii_tokens:
        if t and t not in merged:
            merged.append(t)
    return merged


def chinese_tokenizer(text):
    """Use jieba to segment Chinese text."""
    tokens = jieba.lcut(text)
    tokens = [t.strip() for t in tokens if t.strip() and re.match(r'^[\u4e00-\u9fff]+$', t)]
    return _merge_ascii_tokens(text, tokens)


def japanese_tokenizer(text):
    """Use Sudachi to segment Japanese text."""
    if not SUDACHI_AVAILABLE:
        # Fallback: character-level tokenization
        print("Warning: Using character-level fallback for Japanese. Install Sudachi for better results.")
        tokens = []
        for char in text:
            if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
        return _merge_ascii_tokens(text, tokens if tokens else [text])
    
    try:
        text = str(text)
        if len(text.encode("utf-8")) > 48000:
            # Input too long for Sudachi; use character-level fallback directly.
            tokens = []
            for char in text:
                if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff':
                    tokens.append(char)
            return _merge_ascii_tokens(text, tokens if tokens else [text])
        # Use Sudachi tokenizer
        tokens = [m.surface() for m in sudachi_dict.tokenize(text, sudachi_mode)]
        # Keep only tokens with Japanese characters
        tokens = [t for t in tokens if t.strip() and re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', t)]
        return _merge_ascii_tokens(text, tokens if tokens else [text])
    except Exception as e:
        print(f"Warning: Sudachi tokenization failed: {e}. Using character-level fallback.")
        # Fallback to character-level
        tokens = []
        for char in text:
            if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
        return _merge_ascii_tokens(text, tokens if tokens else [text])


def korean_tokenizer(text):
    """Use KoNLPy to segment Korean text."""
    if not KONLPY_AVAILABLE:
        # Fallback: simple space splitting + filter Korean
        tokens = text.split()
        tokens = [t for t in tokens if t.strip() and re.search(r'[\uac00-\ud7af]', t)]
        return _merge_ascii_tokens(text, tokens)
    
    try:
        okt = Okt()
        tokens = okt.morphs(text)
        tokens = [t for t in tokens if t.strip() and re.match(r'^[\uac00-\ud7af]+$', t)]
        return _merge_ascii_tokens(text, tokens)
    except:
        # Fallback
        tokens = text.split()
        tokens = [t for t in tokens if t.strip() and re.search(r'[\uac00-\ud7af]', t)]
        return _merge_ascii_tokens(text, tokens)


def hindi_tokenizer(text):
    """Simple tokenizer for Hindi (Devanagari script)."""
    # Split on whitespace and punctuation, keep Devanagari
    tokens = re.findall(r'[\u0900-\u097f]+', text)
    tokens = [t for t in tokens if len(t) > 1]  # Filter single characters
    return _merge_ascii_tokens(text, tokens)


def hebrew_tokenizer(text):
    """Simple tokenizer for Hebrew."""
    # Split on whitespace and punctuation, keep Hebrew characters
    tokens = re.findall(r'[\u0590-\u05ff]+', text)
    tokens = [t for t in tokens if len(t) > 1]  # Filter single characters
    return _merge_ascii_tokens(text, tokens)


def thai_tokenizer(text):
    """Use pythainlp to segment Thai text."""
    if not PYTHAINLP_AVAILABLE:
        # Fallback: extract Thai character sequences
        tokens = re.findall(r'[\u0e00-\u0e7f]+', text)
        return _merge_ascii_tokens(text, tokens)
    
    try:
        tokens = thai_tokenize(text, engine='newmm')
        tokens = [t for t in tokens if t.strip() and re.match(r'^[\u0e00-\u0e7f]+$', t)]
        return _merge_ascii_tokens(text, tokens)
    except:
        tokens = re.findall(r'[\u0e00-\u0e7f]+', text)
        return _merge_ascii_tokens(text, tokens)


def get_stopwords_for_lang(lang: str) -> set[str]:
    lang = (lang or "").lower()
    if lang.startswith("en"):
        return {"in","which","was","the","a","an","and","or","of","to","is","are"}
    elif lang.startswith("fr"):
        return {"dans","quel","se","l","la","le","les","de","du","des","en","un","une","et","au","aux"}
    elif lang.startswith("de"):
        return {"in","welchem","sich","der","die","das","und","zu","im","am","ein","eine"}
    elif lang.startswith("es"):
        return {"en","qué","que","el","la","los","las","un","una","y","de","del","al"}
    elif lang.startswith("it"):
        return {"in","quale","che","il","la","i","gli","le","un","una","e","di","del","della"}
    elif lang.startswith("pt"):
        return {"em","qual","que","o","a","os","as","um","uma","e","de","do","da","dos","das"}
    elif lang.startswith("id"):
        return {"di","ke","dari","dan","yang","itu","ini","mana"}
    elif lang.startswith("hi"):
        return {"में","था","और","यह","ये"}
    elif lang.startswith("he"):
        return set()
    elif lang.startswith("zh"):
        return set()
    elif lang.startswith("ja"):
        return set()
    elif lang.startswith("ko"):
        return set()
    elif lang.startswith("th"):
        return set()
    else:
        return set()


def get_language_config(lang):
    """
    Return tokenizer and stop_words configuration for a given language.
    Returns: (tokenizer_function, stop_words, token_pattern)
    
    Note: scikit-learn only has built-in 'english' stop words.
    For other languages, we use None or you can provide custom lists.
    """
    # Languages requiring special tokenization (no word boundaries or complex morphology)
    if lang and lang.startswith("zh"):
        return chinese_tokenizer, None, None
    elif lang and lang.startswith("ja"):
        return japanese_tokenizer, None, None
    elif lang and lang.startswith("ko"):
        return korean_tokenizer, None, None
    elif lang and lang.startswith("hi"):
        return hindi_tokenizer, None, None
    elif lang and lang.startswith("he"):
        return hebrew_tokenizer, None, None
    elif lang and lang.startswith("th"):
        return thai_tokenizer, None, None
    
    # English has built-in stop words
    elif lang and lang.startswith("en"):
        return None, 'english', r"(?u)\b\w\w+\b"
    
    # Other European languages: use default tokenization, no stop words
    # (sklearn only has 'english' built-in)
    elif lang in ['fr', 'de', 'es', 'it', 'pt', 'id']:
        return None, None, r"(?u)\b\w\w+\b"
    
    # Default for other languages
    else:
        return None, None, r"(?u)\b\w\w+\b"
