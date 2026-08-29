
"""
07 - Tokenizer Evaluation

Evaluates the six tokenizer implementations:

1. Character
2. Word
3. BPE
4. Byte-Level BPE
5. WordPiece
6. Unigram / SentencePiece-style

IMPORTANT:
This file does NOT change tokenizer implementations.
It only evaluates their behavior on the same test set.

Metrics:
- Token count
- Characters per token
- Bytes per token
- Tokens per character
- Tokens per byte
- UNK count
- UNK rate
- Round-trip success
- Sequence length
"""

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


TOKEN_DIR = Path(__file__).parent


# ============================================================
# LOAD TOKENIZER CLASSES
# ============================================================

def load_class(filename, class_name):

    path = TOKEN_DIR / filename

    spec = spec_from_file_location(
        filename,
        path
    )

    module = module_from_spec(spec)

    spec.loader.exec_module(module)

    return getattr(module, class_name)


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
# TRAINING CORPUS
# ============================================================

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


# ============================================================
# TEST SET
# ============================================================

TEST_TEXTS = [

    # Familiar English
    "The cat sat on the mat.",

    "Natural language processing",

    "tokenization pipeline",

    # Morphology
    "unhappiness",

    "happiness unhappy happily",

    "internationalization",

    # Technical
    "transformer attention mechanism",

    "def fibonacci(n): return n + 1",

    # Punctuation / numbers
    "Hello, world!",

    "The answer is 42.",

    # Unicode
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

tokenizers["Character"].train(CORPUS)

tokenizers["Word"].train(CORPUS)

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


# ============================================================
# TOKEN DISPLAY
# ============================================================

def get_pieces(tokenizer, text, ids):

    if hasattr(tokenizer, "tokenize"):

        return tokenizer.tokenize(text)

    if hasattr(tokenizer, "token_pieces"):

        return tokenizer.token_pieces(ids)

    return [str(token_id) for token_id in ids]


# ============================================================
# UNKNOWN TOKEN DETECTION
# ============================================================

def is_unknown_piece(piece):

    unknown_markers = {

        "<UNK>",

        "[UNK]",

        "<unk>"
    }

    return piece in unknown_markers


def count_unknowns(pieces):

    return sum(
        1
        for piece in pieces
        if is_unknown_piece(piece)
    )


# ============================================================
# EVALUATE ONE TEXT
# ============================================================

def evaluate_one(tokenizer, text):

    result = {

        "success": False,

        "token_count": 0,

        "characters": len(text),

        "bytes": len(
            text.encode("utf-8")
        ),

        "tokens_per_character": None,

        "tokens_per_byte": None,

        "characters_per_token": None,

        "bytes_per_token": None,

        "unk_count": 0,

        "unk_rate": None,

        "roundtrip": False,

        "pieces": [],

        "ids": [],

        "error": None
    }


    try:

        ids = tokenizer.encode(text)

        pieces = get_pieces(
            tokenizer,
            text,
            ids
        )

        decoded = tokenizer.decode(ids)

        token_count = len(ids)

        unk_count = count_unknowns(
            pieces
        )

        result["success"] = True

        result["token_count"] = token_count

        result["ids"] = ids

        result["pieces"] = pieces

        result["unk_count"] = unk_count

        result["roundtrip"] = (
            decoded == text
        )

        if token_count > 0:

            result["characters_per_token"] = (
                result["characters"] /
                token_count
            )

            result["bytes_per_token"] = (
                result["bytes"] /
                token_count
            )

        if result["characters"] > 0:

            result["tokens_per_character"] = (
                token_count /
                result["characters"]
            )

        if result["bytes"] > 0:

            result["tokens_per_byte"] = (
                token_count /
                result["bytes"]
            )

        if token_count > 0:

            result["unk_rate"] = (
                unk_count /
                token_count
            )

    except Exception as error:

        result["error"] = str(error)


    return result


# ============================================================
# DETAILED EVALUATION
# ============================================================

print("\n")
print("=" * 80)
print("DETAILED EVALUATION")
print("=" * 80)


all_results = {}


for name, tokenizer in tokenizers.items():

    all_results[name] = {}

    print("\n")
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    for text in TEST_TEXTS:

        result = evaluate_one(
            tokenizer,
            text
        )

        all_results[name][text] = result

        print("\n" + "-" * 70)

        print(
            f"Text: {text}"
        )

        if not result["success"]:

            print(
                f"ERROR: {result['error']}"
            )

            continue

        print(
            f"Pieces: {result['pieces']}"
        )

        print(
            f"Token count: "
            f"{result['token_count']}"
        )

        print(
            f"Characters: "
            f"{result['characters']}"
        )

        print(
            f"UTF-8 bytes: "
            f"{result['bytes']}"
        )

        print(
            f"Tokens/character: "
            f"{result['tokens_per_character']:.3f}"
        )

        print(
            f"Tokens/byte: "
            f"{result['tokens_per_byte']:.3f}"
        )

        print(
            f"Characters/token: "
            f"{result['characters_per_token']:.3f}"
        )

        print(
            f"Bytes/token: "
            f"{result['bytes_per_token']:.3f}"
        )

        print(
            f"UNK count: "
            f"{result['unk_count']}"
        )

        print(
            f"UNK rate: "
            f"{result['unk_rate']:.3f}"
        )

        print(
            f"Roundtrip: "
            f"{'PASS' if result['roundtrip'] else 'FAIL'}"
        )


# ============================================================
# AGGREGATE METRICS
# ============================================================

print("\n\n")
print("=" * 100)
print("AGGREGATE EVALUATION")
print("=" * 100)


summary = {}


for name in tokenizers:

    results = [
        result
        for result in all_results[name].values()
        if result["success"]
    ]

    if not results:

        continue

    total_tokens = sum(
        result["token_count"]
        for result in results
    )

    total_characters = sum(
        result["characters"]
        for result in results
    )

    total_bytes = sum(
        result["bytes"]
        for result in results
    )

    total_unk = sum(
        result["unk_count"]
        for result in results
    )

    passed = sum(
        result["roundtrip"]
        for result in results
    )

    summary[name] = {

        "texts": len(results),

        "total_tokens": total_tokens,

        "total_characters": total_characters,

        "total_bytes": total_bytes,

        "tokens_per_character": (
            total_tokens /
            total_characters
        ),

        "tokens_per_byte": (
            total_tokens /
            total_bytes
        ),

        "characters_per_token": (
            total_characters /
            total_tokens
        ),

        "bytes_per_token": (
            total_bytes /
            total_tokens
        ),

        "unk_count": total_unk,

        "unk_rate": (
            total_unk /
            total_tokens
        ),

        "roundtrip_pass": passed,

        "roundtrip_total": len(results),

        "roundtrip_rate": (
            passed /
            len(results)
        )
    }


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n")

print(
    f"{'Tokenizer':15}"
    f"{'Tokens':>10}"
    f"{'Tok/Char':>10}"
    f"{'Tok/Byte':>10}"
    f"{'Char/Tok':>10}"
    f"{'Byte/Tok':>10}"
    f"{'UNK':>8}"
    f"{'UNK %':>9}"
    f"{'RT %':>8}"
)

print("-" * 100)


for name, data in summary.items():

    print(
        f"{name:15}"
        f"{data['total_tokens']:10d}"
        f"{data['tokens_per_character']:10.3f}"
        f"{data['tokens_per_byte']:10.3f}"
        f"{data['characters_per_token']:10.3f}"
        f"{data['bytes_per_token']:10.3f}"
        f"{data['unk_count']:8d}"
        f"{data['unk_rate'] * 100:8.2f}%"
        f"{data['roundtrip_rate'] * 100:7.2f}%"
    )


# ============================================================
# VOCABULARY SIZES
# ============================================================

print("\n")
print("=" * 80)
print("VOCABULARY SIZES")
print("=" * 80)


for name, tokenizer in tokenizers.items():

    if hasattr(
        tokenizer,
        "vocabulary_size"
    ):

        size = tokenizer.vocabulary_size()

    elif hasattr(
        tokenizer,
        "vocab"
    ):

        size = len(tokenizer.vocab)

    else:

        size = "UNKNOWN"

    print(
        f"{name:15} -> {size}"
    )


# ============================================================
# BEST / WORST
# ============================================================

print("\n")
print("=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)


if summary:

    longest = max(
        summary,
        key=lambda name:
            summary[name]["total_tokens"]
    )

    shortest = min(
        summary,
        key=lambda name:
            summary[name]["total_tokens"]
    )

    lowest_unk = min(
        summary,
        key=lambda name:
            summary[name]["unk_rate"]
    )

    best_roundtrip = max(
        summary,
        key=lambda name:
            summary[name]["roundtrip_rate"]
    )

    print(
        f"Longest sequences : {longest}"
    )

    print(
        f"Shortest sequences: {shortest}"
    )

    print(
        f"Lowest UNK rate   : {lowest_unk}"
    )

    print(
        f"Best roundtrip    : {best_roundtrip}"
    )


print("\n")
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print("""
Interpret the results conceptually:

Character
    → simple
    → long sequences
    → vocabulary depends on characters
    → poor unseen-character handling

Word
    → very short sequences
    → severe OOV problem
    → <UNK> can hide information loss

BPE
    → learns reusable subword pieces
    → handles unseen words better
    → vocabulary/sequence trade-off

Byte-BPE
    → starts from bytes
    → robust Unicode coverage
    → no character-level OOV problem

WordPiece
    → subword segmentation
    → continuation pieces such as ##ing
    → often produces [UNK] when segmentation fails

Unigram
    → probabilistic subword model
    → chooses among possible segmentations
    → segmentation can differ from BPE
""")
