"""
Tokenizer Evaluation

Evaluates:
    Character
    Word
    BPE
    Byte-Level BPE
    WordPiece
    Unigram

All tokenizers:
    - use the same corpus
    - use the same test set

Evaluation is separated into:

    ENCODE
        SUCCESS / ERROR

    COVERAGE
        FULL / PARTIAL

    LOSSLESS
        YES / NO

    EFFICIENCY
        tokens / character
        tokens / byte
        characters / token
        bytes / token

    UNK
        count / rate
"""


from pathlib import Path
from importlib.util import (
    spec_from_file_location,
    module_from_spec
)


# ============================================================
# PATH
# ============================================================

TOKEN_DIR = Path(__file__).resolve().parent

CORPUS_PATH = TOKEN_DIR / "corpus.txt"

corpus = CORPUS_PATH.read_text(
    encoding="utf-8"
)


# ============================================================
# LOAD TOKENIZER CLASS
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


tokenizers["Character"].train(corpus)

tokenizers["Word"].train(corpus)

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
# GET TOKEN PIECES
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

UNKNOWN_MARKERS = {
    "<UNK>",
    "[UNK]",
    "<unk>"
}


def count_unknowns(pieces):

    return sum(
        piece in UNKNOWN_MARKERS
        for piece in pieces
    )


# ============================================================
# EVALUATE ONE TEXT
# ============================================================

def evaluate_one(tokenizer, text):

    characters = len(text)

    byte_count = len(
        text.encode("utf-8")
    )

    result = {

        "encode_success": False,

        "coverage": "UNKNOWN",

        "lossless": False,

        "token_count": 0,

        "characters": characters,

        "bytes": byte_count,

        "tokens_per_character": None,

        "tokens_per_byte": None,

        "characters_per_token": None,

        "bytes_per_token": None,

        "unk_count": 0,

        "unk_rate": 0.0,

        "pieces": [],

        "ids": [],

        "decoded": None,

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


        result["encode_success"] = True

        result["ids"] = ids

        result["pieces"] = pieces

        result["decoded"] = decoded

        result["token_count"] = token_count

        result["unk_count"] = unk_count


        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        if unk_count == 0:

            result["coverage"] = "FULL"

        else:

            result["coverage"] = "PARTIAL"


        # ----------------------------------------------------
        # Losslessness
        # ----------------------------------------------------

        result["lossless"] = (
            decoded == text
            and unk_count == 0
        )


        # ----------------------------------------------------
        # Efficiency
        # ----------------------------------------------------

        if token_count > 0:

            result["tokens_per_character"] = (
                token_count /
                characters
            )

            result["tokens_per_byte"] = (
                token_count /
                byte_count
            )

            result["characters_per_token"] = (
                characters /
                token_count
            )

            result["bytes_per_token"] = (
                byte_count /
                token_count
            )

            result["unk_rate"] = (
                unk_count /
                token_count
            )


    except Exception as error:

        result["encode_success"] = False

        result["coverage"] = "ERROR"

        result["lossless"] = False

        result["error"] = str(error)


    return result


# ============================================================
# RUN ALL TESTS
# ============================================================

all_results = {}


for name, tokenizer in tokenizers.items():

    all_results[name] = {}

    for text in TEST_TEXTS:

        all_results[name][text] = evaluate_one(
            tokenizer,
            text
        )


# ============================================================
# DETAILED RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("DETAILED RESULTS")
print("=" * 80)


for text in TEST_TEXTS:

    print("\n")
    print("=" * 80)

    print(
        f"TEXT: {text}"
    )

    print("=" * 80)

    print(
        f"Characters : {len(text)}"
    )

    print(
        f"UTF-8 bytes: "
        f"{len(text.encode('utf-8'))}"
    )


    for name in tokenizers:

        result = all_results[name][text]

        print("\n" + "-" * 60)

        print(name)

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        print(
            f"Encode     : "
            f"{'SUCCESS' if result['encode_success'] else 'ERROR'}"
        )


        if not result["encode_success"]:

            print(
                f"Error      : "
                f"{result['error']}"
            )

            continue


        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        print(
            f"Coverage   : "
            f"{result['coverage']}"
        )


        # ----------------------------------------------------
        # Lossless
        # ----------------------------------------------------

        print(
            f"Lossless   : "
            f"{'YES' if result['lossless'] else 'NO'}"
        )


        # ----------------------------------------------------
        # Tokens
        # ----------------------------------------------------

        print(
            f"Pieces     : "
            f"{result['pieces']}"
        )

        print(
            f"Token count: "
            f"{result['token_count']}"
        )


        # ----------------------------------------------------
        # UNK
        # ----------------------------------------------------

        print(
            f"UNK count  : "
            f"{result['unk_count']}"
        )

        print(
            f"UNK rate   : "
            f"{result['unk_rate']:.3f}"
        )


        # ----------------------------------------------------
        # Efficiency
        # ----------------------------------------------------

        print(
            f"Tokens/char: "
            f"{result['tokens_per_character']:.3f}"
        )

        print(
            f"Tokens/byte: "
            f"{result['tokens_per_byte']:.3f}"
        )

        print(
            f"Chars/token: "
            f"{result['characters_per_token']:.3f}"
        )

        print(
            f"Bytes/token: "
            f"{result['bytes_per_token']:.3f}"
        )


# ============================================================
# AGGREGATE
# ============================================================

print("\n\n")
print("=" * 110)
print("AGGREGATE EVALUATION")
print("=" * 110)


print(
    f"{'Tokenizer':15}"
    f"{'Tokens':>10}"
    f"{'Tok/Char':>10}"
    f"{'Tok/Byte':>10}"
    f"{'Char/Tok':>10}"
    f"{'Byte/Tok':>10}"
    f"{'UNK':>8}"
    f"{'UNK %':>9}"
    f"{'Lossless':>10}"
    f"{'Coverage':>12}"
)


print("-" * 110)


aggregate = {}


for name in tokenizers:

    results = [
        all_results[name][text]
        for text in TEST_TEXTS
    ]


    successful = [
        result
        for result in results
        if result["encode_success"]
    ]


    total_tokens = sum(
        result["token_count"]
        for result in successful
    )


    total_characters = sum(
        result["characters"]
        for result in successful
    )


    total_bytes = sum(
        result["bytes"]
        for result in successful
    )


    total_unk = sum(
        result["unk_count"]
        for result in successful
    )


    lossless_count = sum(
        result["lossless"]
        for result in successful
    )


    full_coverage_count = sum(
        result["coverage"] == "FULL"
        for result in successful
    )


    encode_errors = sum(
        not result["encode_success"]
        for result in results
    )


    aggregate[name] = {

        "total_tokens": total_tokens,

        "tokens_per_character": (
            total_tokens /
            total_characters
        ) if total_characters else 0,

        "tokens_per_byte": (
            total_tokens /
            total_bytes
        ) if total_bytes else 0,

        "characters_per_token": (
            total_characters /
            total_tokens
        ) if total_tokens else 0,

        "bytes_per_token": (
            total_bytes /
            total_tokens
        ) if total_tokens else 0,

        "unk": total_unk,

        "unk_rate": (
            total_unk /
            total_tokens
        ) if total_tokens else 0,

        "lossless": lossless_count,

        "coverage": full_coverage_count,

        "errors": encode_errors
    }


    print(
        f"{name:15}"
        f"{total_tokens:10d}"
        f"{aggregate[name]['tokens_per_character']:10.3f}"
        f"{aggregate[name]['tokens_per_byte']:10.3f}"
        f"{aggregate[name]['characters_per_token']:10.3f}"
        f"{aggregate[name]['bytes_per_token']:10.3f}"
        f"{total_unk:8d}"
        f"{aggregate[name]['unk_rate'] * 100:8.2f}%"
        f"{lossless_count:>6}/{len(TEST_TEXTS):<3}"
        f"{full_coverage_count:>7}/{len(TEST_TEXTS):<3}"
    )


# ============================================================
# VOCABULARY
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
# ENCODE / COVERAGE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ENCODE / COVERAGE SUMMARY")
print("=" * 80)


for name in tokenizers:

    data = aggregate[name]

    print(
        f"{name:15}"
        f" Encode errors={data['errors']:2d}"
        f"  Full coverage={data['coverage']:2d}/{len(TEST_TEXTS)}"
        f"  Lossless={data['lossless']:2d}/{len(TEST_TEXTS)}"
    )


# ============================================================
# EFFICIENCY OBSERVATION
# ============================================================

print("\n")
print("=" * 80)
print("EFFICIENCY OBSERVATION")
print("=" * 80)


lowest_token_count = min(
    aggregate,
    key=lambda name:
        aggregate[name]["total_tokens"]
)


highest_chars_per_token = max(
    aggregate,
    key=lambda name:
        aggregate[name]["characters_per_token"]
)


lowest_unk = min(
    aggregate,
    key=lambda name:
        aggregate[name]["unk_rate"]
)


best_lossless = max(
    aggregate,
    key=lambda name:
        aggregate[name]["lossless"]
)


print(
    f"Fewest tokens       : "
    f"{lowest_token_count}"
)

print(
    f"Highest chars/token  : "
    f"{highest_chars_per_token}"
)

print(
    f"Lowest UNK rate      : "
    f"{lowest_unk}"
)

print(
    f"Most lossless tests  : "
    f"{best_lossless}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

#How did you evaluate the tokenizers?"

#I used the same corpus and same 15 test strings for every tokenizer.
#I measured:
#token count
#character count
#UTF-8 byte count
#tokens per character
#tokens per byte
#characters per token
#bytes per token
#UNK count/rate
#coverage
#encoding errors
#round-trip losslessness

