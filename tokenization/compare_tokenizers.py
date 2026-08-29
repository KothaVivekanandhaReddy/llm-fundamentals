"""
07 - Tokenizer Comparison Lab

Compare:

    Character
    Word
    BPE
    Byte-Level BPE
    WordPiece
    Unigram

All tokenizers use the SAME corpus
and SAME test set.

Metrics:

    - token count
    - character count
    - UTF-8 byte count
    - tokens / character
    - tokens / byte
    - round-trip
"""


import sys
from pathlib import Path


# =============================================================
# IMPORT LOCAL TOKENIZERS
# =============================================================

TOKEN_DIR = Path(__file__).parent

sys.path.insert(
    0,
    str(TOKEN_DIR)
)


from importlib.util import (
    spec_from_file_location,
    module_from_spec
)


def load_class(
    filename,
    class_name
):

    path = TOKEN_DIR / filename

    spec = spec_from_file_location(
        filename,
        path
    )

    module = module_from_spec(spec)

    spec.loader.exec_module(module)

    return getattr(
        module,
        class_name
    )


CharacterTokenizer = load_class(
    "char_tokenizer.py",
    "CharacterTokenizer"
)

WordTokenizer = load_class(
    "word_tokenizer.py",
    "WordTokenizer"
)

BPETokenizer = load_class(
    "bpe_tokenizer.py",
    "BPETokenizer"
)

ByteLevelBPETokenizer = load_class(
    "byte_bpe_tokenizer.py",
    "ByteLevelBPETokenizer"
)

WordPieceTokenizer = load_class(
    "wordpiece_tokenizer.py",
    "WordPieceTokenizer"
)

UnigramTokenizer = load_class(
    "unigram_tokenizer.py",
    "UnigramTokenizer"
)


# =============================================================
# CORPUS
# =============================================================

CORPUS = (
    "The cat sat on the mat. "
    "The cat ate the rat. "
    "The dog sat on the log. "
    "The dog ate the frog. "
    "Natural language processing is the study of how computers "
    "understand and generate human language. "
    "Tokenization is the first step in any NLP pipeline. "
    "Machine learning and artificial intelligence are transforming "
    "natural language processing. "
    "Transformers are neural networks that process sequences "
    "using attention mechanisms. "
    "Language models predict the next token from previous tokens."
)


# =============================================================
# TEST SET
# =============================================================

TEST_TEXTS = [

    "The cat sat on the mat.",

    "Natural language processing",

    "tokenization pipeline",

    "unhappiness",

    "happiness unhappy happily",

    "internationalization",

    "transformer attention mechanism",

    "def fibonacci(n): return n + 1",

    "Hello, world!",

    "The answer is 42.",

    "你好世界",

    "नमस्ते दुनिया",

    "こんにちは世界",

    "🚀 AI is amazing!",

    "AI 2026 🚀 भारत"
]


# =============================================================
# TRAIN
# =============================================================

tokenizers = {

    "Character": CharacterTokenizer(),

    "Word": WordTokenizer(),

    "BPE": BPETokenizer(),

    "Byte-BPE": ByteLevelBPETokenizer(),

    "WordPiece": WordPieceTokenizer(),

    "Unigram": UnigramTokenizer()
}


print("=" * 80)
print("TRAINING TOKENIZERS")
print("=" * 80)


tokenizers["Character"].train(
    CORPUS
)

tokenizers["Word"].train(
    CORPUS
)

tokenizers["BPE"].train(
    CORPUS,
    num_merges=80
)

tokenizers["Byte-BPE"].train(
    CORPUS,
    num_merges=80
)

tokenizers["WordPiece"].train(
    CORPUS,
    vocab_size=160
)

tokenizers["Unigram"].train(
    CORPUS,
    vocab_size=160
)


# =============================================================
# TOKEN PIECES
# =============================================================

def get_pieces(
    tokenizer,
    text,
    ids
):

    if hasattr(
        tokenizer,
        "tokenize"
    ):

        return tokenizer.tokenize(
            text
        )

    return [
        tokenizer.id_to_token[token_id]
        for token_id in ids
    ]


# =============================================================
# RUN ONE TEST
# =============================================================

def run_test(
    name,
    tokenizer,
    text
):

    try:

        ids = tokenizer.encode(text)

        pieces = get_pieces(
            tokenizer,
            text,
            ids
        )

        decoded = tokenizer.decode(
            ids
        )

        return {
            "name": name,
            "ids": ids,
            "pieces": pieces,
            "count": len(ids),
            "decoded": decoded,
            "roundtrip": (
                decoded == text
            ),
            "error": None
        }

    except Exception as error:

        return {
            "name": name,
            "ids": [],
            "pieces": [],
            "count": 0,
            "decoded": None,
            "roundtrip": False,
            "error": str(error)
        }


# =============================================================
# DETAILED COMPARISON
# =============================================================

for text in TEST_TEXTS:

    print("\n")
    print("=" * 80)

    print(
        f"TEXT: {text}"
    )

    print("=" * 80)

    chars = len(text)

    byte_count = len(
        text.encode("utf-8")
    )

    print(
        f"Characters : {chars}"
    )

    print(
        f"UTF-8 bytes: {byte_count}"
    )

    for name, tokenizer in (
        tokenizers.items()
    ):

        result = run_test(
            name,
            tokenizer,
            text
        )

        print(
            f"\n{name}"
        )

        if result["error"]:

            print(
                f"  ERROR: "
                f"{result['error']}"
            )

            continue

        print(
            f"  Pieces: "
            f"{result['pieces']}"
        )

        print(
            f"  Token count: "
            f"{result['count']}"
        )

        print(
            f"  Tokens/char: "
            f"{result['count'] / chars:.2f}"
        )

        print(
            f"  Tokens/byte: "
            f"{result['count'] / byte_count:.2f}"
        )

        print(
            f"  Roundtrip: "
            f"{'PASS' if result['roundtrip'] else 'FAIL'}"
        )


# =============================================================
# SUMMARY
# =============================================================

print("\n\n")
print("=" * 100)
print("TOKEN COUNT SUMMARY")
print("=" * 100)


print(
    f"{'Text':35}"
    f"{'Char':>8}"
    f"{'Word':>8}"
    f"{'BPE':>8}"
    f"{'BBPE':>8}"
    f"{'WP':>8}"
    f"{'Uni':>8}"
)


print("-" * 100)


for text in TEST_TEXTS:

    results = []

    for tokenizer in tokenizers.values():

        result = run_test(
            "",
            tokenizer,
            text
        )

        if result["error"]:

            results.append(
                "ERR"
            )

        else:

            results.append(
                str(result["count"])
            )

    display_text = text[:34]

    print(
        f"{display_text:35}"
        f"{results[0]:>8}"
        f"{results[1]:>8}"
        f"{results[2]:>8}"
        f"{results[3]:>8}"
        f"{results[4]:>8}"
        f"{results[5]:>8}"
    )


# =============================================================
# VOCABULARY SIZES
# =============================================================

print("\n")
print("=" * 80)
print("VOCABULARY SIZES")
print("=" * 80)


for name, tokenizer in (
    tokenizers.items()
):

    print(
        f"{name:15} -> "
        f"{tokenizer.vocabulary_size()}"
    )


# =============================================================
# ROUNDTRIP SUMMARY
# =============================================================

print("\n")
print("=" * 80)
print("ROUNDTRIP SUMMARY")
print("=" * 80)


for name, tokenizer in (
    tokenizers.items()
):

    passed = 0
    failed = 0
    errors = 0

    for text in TEST_TEXTS:

        result = run_test(
            name,
            tokenizer,
            text
        )

        if result["error"]:

            errors += 1

        elif result["roundtrip"]:

            passed += 1

        else:

            failed += 1

    print(
        f"{name:15} "
        f"PASS={passed:2d} "
        f"FAIL={failed:2d} "
        f"ERROR={errors:2d}"
    )


print("\n")
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
