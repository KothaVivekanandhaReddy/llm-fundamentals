"""
Tokenizer Comparison Lab

All tokenizers:
    - use the same corpus
    - use the same test set
    - are evaluated using the same metrics
"""

import sys
from pathlib import Path
from importlib.util import (
    spec_from_file_location,
    module_from_spec
)


# ============================================================
# PATHS
# ============================================================

TOKEN_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(TOKEN_DIR)
)


CORPUS_PATH = TOKEN_DIR / "corpus.txt"

corpus = CORPUS_PATH.read_text(
    encoding="utf-8"
)


# ============================================================
# LOAD CLASSES
# ============================================================

def load_class(filename, class_name):

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


# ============================================================
# TEST SET
# ============================================================

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


# ============================================================
# CREATE TOKENIZERS
# ============================================================

tokenizers = {

    "Character": CharacterTokenizer(),

    "Word": WordTokenizer(),

    "BPE": BPETokenizer(),

    "Byte-BPE": ByteLevelBPETokenizer(),

    "WordPiece": WordPieceTokenizer(),

    "Unigram": UnigramTokenizer()
}


# ============================================================
# TRAIN
# ============================================================

print("=" * 80)
print("TRAINING TOKENIZERS")
print("=" * 80)

tokenizers["Character"].train(
    corpus
)

tokenizers["Word"].train(
    corpus
)

tokenizers["BPE"].train(
    corpus,
    num_merges=80
)

tokenizers["Byte-BPE"].train(
    corpus,
    num_merges=80
)

tokenizers["WordPiece"].train(
    corpus,
    vocab_size=160
)

tokenizers["Unigram"].train(
    corpus,
    vocab_size=160
)


# ============================================================
# TOKEN DISPLAY
# ============================================================

def get_pieces(tokenizer, text, ids):

    if hasattr(
        tokenizer,
        "token_pieces"
    ):
        return tokenizer.token_pieces(ids)

    if hasattr(
        tokenizer,
        "tokenize"
    ):
        return tokenizer.tokenize(text)

    if hasattr(
        tokenizer,
        "id_to_token"
    ):
        return [
            tokenizer.id_to_token[token_id]
            for token_id in ids
        ]

    return [
        str(token_id)
        for token_id in ids
    ]


# ============================================================
# UNKNOWN TOKEN
# ============================================================

def is_unknown_piece(piece):

    return piece in {
        "<UNK>",
        "[UNK]",
        "<unk>"
    }


def count_unknowns(pieces):

    return sum(
        is_unknown_piece(piece)
        for piece in pieces
    )


# ============================================================
# RUN ONE TEST
# ============================================================

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

        unk_count = count_unknowns(
            pieces
        )

        if decoded == text:

            status = "PASS"

        elif unk_count > 0:

            status = "LOSSY"

        else:

            status = "FAIL"

        return {

            "name": name,

            "ids": ids,

            "pieces": pieces,

            "count": len(ids),

            "decoded": decoded,

            "roundtrip": decoded == text,

            "unk_count": unk_count,

            "status": status,

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

            "unk_count": 0,

            "status": "ERROR",

            "error": str(error)
        }


# ============================================================
# RUN ALL EXPERIMENTS ONCE
# ============================================================

all_results = {}


for name, tokenizer in tokenizers.items():

    all_results[name] = {}

    for text in TEST_TEXTS:

        all_results[name][text] = run_test(
            name,
            tokenizer,
            text
        )


# ============================================================
# DETAILED RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("DETAILED COMPARISON")
print("=" * 80)


for text in TEST_TEXTS:

    chars = len(text)

    byte_count = len(
        text.encode("utf-8")
    )

    print("\n")
    print("=" * 80)
    print(f"TEXT: {text}")
    print("=" * 80)

    print(
        f"Characters : {chars}"
    )

    print(
        f"UTF-8 bytes: {byte_count}"
    )

    for name in tokenizers:

        result = all_results[name][text]

        print(
            f"\n{name}"
        )

        print(
            f"  Status: "
            f"{result['status']}"
        )

        if result["status"] == "ERROR":

            print(
                f"  Error: "
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

        if chars > 0:

            print(
                f"  Tokens/char: "
                f"{result['count'] / chars:.3f}"
            )

        if byte_count > 0:

            print(
                f"  Tokens/byte: "
                f"{result['count'] / byte_count:.3f}"
            )

        print(
            f"  UNK count: "
            f"{result['unk_count']}"
        )

        print(
            f"  Roundtrip: "
            f"{'PASS' if result['roundtrip'] else 'FAIL'}"
        )


# ============================================================
# TOKEN COUNT SUMMARY
# ============================================================

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

    for name in tokenizers:

        result = all_results[name][text]

        if result["status"] == "ERROR":

            results.append("ERR")

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


# ============================================================
# VOCABULARY SIZES
# ============================================================

print("\n")
print("=" * 80)
print("VOCABULARY SIZES")
print("=" * 80)


for name, tokenizer in tokenizers.items():

    print(
        f"{name:15} -> "
        f"{tokenizer.vocabulary_size()}"
    )


# ============================================================
# STATUS SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("STATUS SUMMARY")
print("=" * 80)


for name in tokenizers:

    results = [
        all_results[name][text]
        for text in TEST_TEXTS
    ]

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    lossy = sum(
        result["status"] == "LOSSY"
        for result in results
    )

    failed = sum(
        result["status"] == "FAIL"
        for result in results
    )

    errors = sum(
        result["status"] == "ERROR"
        for result in results
    )

    print(
        f"{name:15}"
        f"PASS={passed:2d} "
        f"LOSSY={lossy:2d} "
        f"FAIL={failed:2d} "
        f"ERROR={errors:2d}"
    )


print("\n")
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)