
"""
03 - Byte/Character BPE Tokenizer

Goal:
    Understand the original BPE idea used for
    subword tokenization.

Pipeline:

    Text
      ↓
    Characters
      ↓
    Count adjacent pairs
      ↓
    Find most frequent pair
      ↓
    Merge pair
      ↓
    Repeat
      ↓
    Learned subword vocabulary


Example:

    "low low lower"

    Initially:

    l o w   l o w   l o w e r

    Frequent pair:

    ("l", "o")

    becomes:

    "lo"

    Then:

    ("lo", "w")

    becomes:

    "low"


This implementation is intentionally educational.
It starts from characters rather than UTF-8 bytes.
Byte-Level BPE is implemented separately in Step 4.
"""

from collections import Counter


class BPETokenizer:

    def __init__(self):

        # Token -> integer ID
        self.vocab = {}

        # Pair -> merged token
        self.merges = {}

        # Pair -> merge priority
        self.merge_ranks = {}

        # Token ID -> token string
        self.id_to_token = {}

    # =========================================================
    # PAIR COUNTING
    # =========================================================

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
    # MERGE A PAIR
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

            # Check whether current and next token
            # form the pair we want to merge.
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

    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        text,
        num_merges=40
    ):

        # Reset tokenizer state.
        self.vocab = {}
        self.merges = {}
        self.merge_ranks = {}
        self.id_to_token = {}

        # -----------------------------------------------------
        # INITIAL VOCABULARY
        # -----------------------------------------------------

        characters = sorted(set(text))

        for token_id, character in enumerate(
            characters
        ):

            self.vocab[character] = token_id
            self.id_to_token[token_id] = character

        # Convert training text into character tokens.
        tokens = list(text)

        # -----------------------------------------------------
        # BPE TRAINING LOOP
        # -----------------------------------------------------

        for merge_index in range(num_merges):

            pairs = self._get_pairs(tokens)

            if not pairs:
                break

            # Select the most frequent pair.
            best_pair = max(
                pairs,
                key=pairs.get
            )

            # Create a new subword.
            new_token = (
                best_pair[0]
                + best_pair[1]
            )

            # Assign new vocabulary ID.
            new_id = len(self.vocab)

            # Add token to vocabulary.
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token

            # Store merge rule.
            self.merges[best_pair] = new_token

            # Earlier merges have higher priority.
            self.merge_ranks[best_pair] = merge_index

            # Apply merge to training sequence.
            tokens = self._merge_pair(
                tokens,
                best_pair,
                new_token
            )

            print(
                f"Merge {merge_index + 1:2d}: "
                f"{best_pair} -> "
                f"'{new_token}' "
                f"(frequency={pairs[best_pair]})"
            )

        return self

    # =========================================================
    # FIND HIGHEST PRIORITY APPLICABLE MERGE
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

        # Smaller rank = earlier learned merge
        # = higher priority.
        candidates.sort(
            key=lambda x: x[0]
        )

        return candidates[0][1]

    # =========================================================
    # ENCODE
    # =========================================================

    def encode(self, text):

        # Start with characters.
        tokens = list(text)

        # Unknown characters cannot be represented
        # because this tokenizer was trained on characters
        # from the training corpus.
        for token in tokens:

            if token not in self.vocab:

                raise ValueError(
                    f"Unknown character: {repr(token)}"
                )

        # Repeatedly apply learned merges.
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

        return [
            self.vocab[token]
            for token in tokens
        ]

    # =========================================================
    # DECODE
    # =========================================================

    def decode(self, token_ids):

        tokens = [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

        return "".join(tokens)

    # =========================================================
    # GET TOKEN PIECES
    # =========================================================

    def tokenize(self, text):

        token_ids = self.encode(text)

        return [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

    # =========================================================
    # VOCABULARY SIZE
    # =========================================================

    def vocabulary_size(self):

        return len(self.vocab)


# =============================================================
# TRAINING CORPUS
# =============================================================

corpus = (
    "The cat sat on the mat. "
    "The cat ate the rat. "
    "The dog sat on the log. "
    "The dog ate the frog. "
    "Natural language processing is the study of how computers "
    "understand and generate human language. "
    "Tokenization is the first step in any NLP pipeline."
)


# =============================================================
# TRAIN TOKENIZER
# =============================================================

print("=" * 70)
print("BPE TRAINING")
print("=" * 70)

tokenizer = BPETokenizer()

tokenizer.train(
    corpus,
    num_merges=40
)


# =============================================================
# VOCABULARY INFORMATION
# =============================================================

print("\n" + "=" * 70)
print("VOCABULARY")
print("=" * 70)

print(
    f"Vocabulary size: "
    f"{tokenizer.vocabulary_size()}"
)

print(
    f"Learned merges: "
    f"{len(tokenizer.merges)}"
)


# =============================================================
# TEST SENTENCES
# =============================================================

test_sentences = [

    "The cat sat on the mat.",

    "Natural language processing",

    "tokenization pipeline",

    "unhappiness",

    "Hello, world!"
]


for sentence in test_sentences:

    print("\n" + "-" * 70)

    print(
        f"Text: {sentence}"
    )

    try:

        token_ids = tokenizer.encode(
            sentence
        )

        pieces = tokenizer.tokenize(
            sentence
        )

        decoded = tokenizer.decode(
            token_ids
        )

        print(
            f"Pieces: {pieces}"
        )

        print(
            f"Token IDs: {token_ids}"
        )

        print(
            f"Token count: {len(token_ids)}"
        )

        print(
            f"Character count: {len(sentence)}"
        )

        print(
            f"Compression ratio: "
            f"{len(token_ids) / len(sentence):.2f}"
        )

        print(
            f"Decoded: {decoded}"
        )

        print(
            f"Roundtrip: "
            f"{'PASS' if decoded == sentence else 'FAIL'}"
        )

    except ValueError as error:

        print(
            f"ERROR: {error}"
        )


# =============================================================
# SHOW MERGE TABLE
# =============================================================

print("\n" + "=" * 70)
print("LEARNED BPE MERGES")
print("=" * 70)

for rank, (pair, new_token) in enumerate(
    tokenizer.merges.items()
):

    print(
        f"{rank:3d}: "
        f"{repr(pair[0])} + "
        f"{repr(pair[1])} "
        f"-> {repr(new_token)}"
    )
