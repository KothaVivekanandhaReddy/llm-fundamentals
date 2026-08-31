
"""
04 - Byte-Level BPE (BBPE)

Goal:
    Understand Byte-Level BPE.

Difference from Step 3:

    Character BPE:

        text
          ↓
        characters
          ↓
        BPE merges

    Byte-Level BPE:

        text
          ↓
        UTF-8 bytes
          ↓
        BPE merges


Why bytes?

UTF-8 can represent essentially any Unicode text.

Examples:

    English
    中文
    हिन्दी
    日本語
    العربية
    emojis 🚀

All ultimately become sequences of bytes.

Base vocabulary:

    0 ... 255

Then BPE learns additional tokens:

    256+
"""


from collections import Counter


class ByteLevelBPETokenizer:

    def __init__(self):

        # -----------------------------------------------------
        # Vocabulary
        #
        # token ID -> bytes
        # -----------------------------------------------------

        self.vocab = {}

        # -----------------------------------------------------
        # Merge rules
        #
        # (token_a, token_b) -> new_token_id
        # -----------------------------------------------------

        self.merges = {}

        # -----------------------------------------------------
        # Merge priority
        #
        # pair -> rank
        #
        # Lower rank = earlier learned merge
        # -----------------------------------------------------

        self.merge_ranks = {}
        self.id_to_token = {}


    # =========================================================
    # INITIALIZE BYTE VOCABULARY
    # =========================================================

    def _initialize_vocab(self):

        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }



    # =========================================================
    # MERGE PAIR
    # =========================================================

    def _merge_pair(
        self,
        tokens,
        pair,
        new_token
    ):

        merged = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):

                merged.append(new_token)

                i += 2

            else:

                merged.append(tokens[i])

                i += 1

        return merged
    
    def _get_pairs(self, tokens):

        pairs = Counter()

        for i in range(len(tokens) - 1):

            pair = (
                tokens[i],
                tokens[i + 1]
            )

            pairs[pair] += 1

        return pairs

    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        text,
        num_merges=40
    ):

        # Reset state.
        self.vocab = {}
        self.merges = {}
        self.merge_ranks = {}

        # Base vocabulary = 256 possible bytes.
        self._initialize_vocab()

        # -----------------------------------------------------
        # Convert training text to UTF-8 bytes.
        # -----------------------------------------------------

        tokens = list(
            text.encode("utf-8")
        )

        print(
            f"Initial byte sequence length: "
            f"{len(tokens)}"
        )

        # =====================================================
        # BPE TRAINING
        # =====================================================

        for merge_index in range(num_merges):

            pairs = self._get_pairs(tokens)

            if not pairs:
                break

            # Most frequent pair.
            best_pair = max(
                pairs,
                key=pairs.get
            )

            # New token ID.
            new_token = (
                256 + merge_index
            )

            # Merge rule.
            self.merges[best_pair] = new_token

            # Merge priority.
            self.merge_ranks[
                best_pair
            ] = merge_index

            # Build byte representation.
            self.vocab[new_token] = (
                self.vocab[best_pair[0]]
                +
                self.vocab[best_pair[1]]
            )

            # Apply merge to training sequence.
            tokens = self._merge_pair(
                tokens,
                best_pair,
                new_token
            )

            print(
                f"Merge {merge_index + 1:2d}: "
                f"{best_pair} "
                f"-> token {new_token} "
                f"(frequency={pairs[best_pair]})"
            )

        return self

    # =========================================================
    # FIND BEST APPLICABLE MERGE
    # =========================================================

    def _get_best_pair(self, tokens):

        candidates = []

        for i in range(len(tokens) - 1):

            pair = (
                tokens[i],
                tokens[i + 1]
            )

            if pair in self.merge_ranks:

                candidates.append(
                    (
                        self.merge_ranks[pair],
                        pair
                    )
                )

        if not candidates:
            return None

        # Lowest rank = highest priority.
        candidates.sort(
            key=lambda x: x[0]
        )

        return candidates[0][1]

    # =========================================================
    # ENCODE
    # =========================================================

    def encode(self, text):

        # UTF-8 → bytes → integer byte IDs.
        tokens = list(
            text.encode("utf-8")
        )

        # Apply learned merges.
        while True:

            best_pair = self._get_best_pair(
                tokens
            )

            if best_pair is None:
                break

            new_token = self.merges[
                best_pair
            ]

            tokens = self._merge_pair(
                tokens,
                best_pair,
                new_token
            )

        return tokens

    # =========================================================
    # DECODE
    # =========================================================

    def decode(self, token_ids):

        byte_sequence = b"".join(
            self.vocab[token_id]
            for token_id in token_ids
        )

        return byte_sequence.decode(
            "utf-8",
            errors="replace"
        )

    # =========================================================
    # GET TOKEN BYTES
    # =========================================================

    def token_bytes(self, token_id):

        return self.vocab[token_id]

    # =========================================================
    # GET TOKEN DISPLAY
    # =========================================================

    def token_pieces(self, token_ids):

        pieces = []

        for token_id in token_ids:

            raw_bytes = self.vocab[
                token_id
            ]

            pieces.append(
                raw_bytes.decode(
                    "utf-8",
                    errors="replace"
                )
            )

        return pieces

    # =========================================================
    # VOCABULARY SIZE
    # =========================================================

    def vocabulary_size(self):

        return len(self.vocab)


from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent / "corpus.txt"

corpus = CORPUS_PATH.read_text(encoding="utf-8")

# =============================================================
# TRAIN
# =============================================================

print("=" * 70)
print("BYTE-LEVEL BPE TRAINING")
print("=" * 70)

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    corpus,
    num_merges=40
)


# =============================================================
# VOCABULARY
# =============================================================

print("\n" + "=" * 70)
print("VOCABULARY")
print("=" * 70)

print(
    f"Base vocabulary: 256 bytes"
)

print(
    f"Learned merges: "
    f"{len(tokenizer.merges)}"
)

print(
    f"Final vocabulary size: "
    f"{tokenizer.vocabulary_size()}"
)


# =============================================================
# TEST SENTENCES
# =============================================================

test_sentences = [

    "The cat sat on the mat.",

    "Natural language processing",

    "tokenization pipeline",

    "unhappiness",

    "Hello, world!",

    "你好世界",

    "नमस्ते दुनिया",

    "こんにちは世界",

    "🚀 AI is amazing!"
]


for sentence in test_sentences:

    print("\n" + "-" * 70)

    print(
        f"Text: {sentence}"
    )

    token_ids = tokenizer.encode(
        sentence
    )

    pieces = tokenizer.token_pieces(
        token_ids
    )

    decoded = tokenizer.decode(
        token_ids
    )

    raw_bytes = len(
        sentence.encode("utf-8")
    )

    print(
        f"Pieces: {pieces}"
    )

    print(
        f"Token IDs: {token_ids}"
    )

    print(
        f"Token count: "
        f"{len(token_ids)}"
    )

    print(
        f"UTF-8 bytes: "
        f"{raw_bytes}"
    )

    print(
        f"Tokens / byte: "
        f"{len(token_ids) / raw_bytes:.2f}"
    )

    print(
        f"Decoded: {decoded}"
    )

    print(
        f"Roundtrip: "
        f"{'PASS' if decoded == sentence else 'FAIL'}"
    )


# =============================================================
# RAW BYTE DEMONSTRATION
# =============================================================

print("\n" + "=" * 70)
print("UTF-8 BYTE DEMONSTRATION")
print("=" * 70)

examples = [
    "A",
    "é",
    "你",
    "न",
    "🚀"
]

for text in examples:

    raw = text.encode("utf-8")

    print(
        f"{text!r:8} -> "
        f"{list(raw)}"
    )


# =============================================================
# LEARNED MERGES
# =============================================================

print("\n" + "=" * 70)
print("LEARNED MERGES")
print("=" * 70)

for rank, (pair, token_id) in enumerate(
    tokenizer.merges.items()
):

    left = tokenizer.vocab[
        pair[0]
    ]

    right = tokenizer.vocab[
        pair[1]
    ]

    merged = tokenizer.vocab[
        token_id
    ]

    left_display = left.decode(
        "utf-8",
        errors="replace"
    )

    right_display = right.decode(
        "utf-8",
        errors="replace"
    )

    merged_display = merged.decode(
        "utf-8",
        errors="replace"
    )

    print(
        f"{rank:3d}: "
        f"{left_display!r} + "
        f"{right_display!r} "
        f"-> {merged_display!r}"
    )

