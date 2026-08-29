"""
06 - Unigram Tokenizer

Goal:
    Implement an educational Unigram tokenizer.

SentencePiece supports multiple algorithms.
Here we implement the Unigram model.

Core idea:

    BPE:
        Start small
        ↓
        Merge frequent pieces

    Unigram:
        Start with many candidate pieces
        ↓
        Assign probabilities
        ↓
        Find the best segmentation

Example:

    "unhappiness"

    Possible segmentations:

        un | happiness
        unh | appiness
        un | happi | ness
        u | n | happiness
        ...

    The tokenizer chooses the segmentation
    with the highest probability.

We use dynamic programming (Viterbi)
to find that segmentation efficiently.

This is an EDUCATIONAL implementation.
It is not a reproduction of the full
SentencePiece training algorithm.
"""

from collections import Counter
import math
import re


class UnigramTokenizer:

    def __init__(self):

        # token -> probability
        self.probabilities = {}

        # token -> integer ID
        self.vocab = {}

        # integer ID -> token
        self.id_to_token = {}

        # Unknown token
        self.unk_token = "<UNK>"

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def _split_words(self, text):

        """
        Split text into words and punctuation.

        Example:

            "Hello, world!"

        becomes:

            ["Hello", ",", "world", "!"]
        """

        return re.findall(
            r"\w+|[^\w\s]",
            text,
            flags=re.UNICODE
        )

    # =========================================================
    # CREATE CANDIDATE VOCABULARY
    # =========================================================

    def _build_candidates(
        self,
        words,
        max_piece_length=6
    ):

        """
        Generate candidate substrings from
        the training corpus.

        Example:

            "hello"

        produces candidates such as:

            h
            he
            hel
            hell
            hello
            e
            el
            ell
            ello
            ...

        We limit the maximum piece length.
        """

        counts = Counter()

        for word in words:

            length = len(word)

            for start in range(length):

                for end in range(
                    start + 1,
                    min(
                        length,
                        start + max_piece_length
                    ) + 1
                ):

                    piece = word[
                        start:end
                    ]

                    counts[piece] += 1

        return counts

    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        text,
        vocab_size=100
    ):

        words = self._split_words(text)

        # -----------------------------------------------------
        # Count candidate substrings.
        # -----------------------------------------------------

        candidate_counts = (
            self._build_candidates(words)
        )

        # -----------------------------------------------------
        # Select most common candidates.
        # -----------------------------------------------------

        candidates = (
            candidate_counts
            .most_common(vocab_size)
        )

        # -----------------------------------------------------
        # Calculate total frequency.
        # -----------------------------------------------------

        total_count = sum(
            count
            for _, count
            in candidates
        )

        # -----------------------------------------------------
        # Create vocabulary.
        # -----------------------------------------------------

        self.vocab = {
            self.unk_token: 0
        }

        self.probabilities = {
            self.unk_token: 1e-10
        }

        for piece, count in candidates:

            if piece == self.unk_token:
                continue

            token_id = len(self.vocab)

            self.vocab[
                piece
            ] = token_id

            probability = (
                count / total_count
            )

            self.probabilities[
                piece
            ] = probability

        # Reverse vocabulary.
        self.id_to_token = {
            token_id: token
            for token, token_id
            in self.vocab.items()
        }

        return self

    # =========================================================
    # VITERBI / BEST SEGMENTATION
    # =========================================================

    def _segment_word(self, word):

        """
        Find the highest-probability segmentation.

        Dynamic programming:

            dp[i] =
                best score for word[:i]

        We try every vocabulary piece
        that can begin at position i.
        """

        n = len(word)

        # Best score up to each position.
        dp = [
            float("-inf")
            for _ in range(n + 1)
        ]

        # Backpointer.
        backpointer = [
            None
            for _ in range(n + 1)
        ]

        dp[0] = 0.0

        # -----------------------------------------------------
        # Dynamic programming.
        # -----------------------------------------------------

        for i in range(n):

            if dp[i] == float("-inf"):
                continue

            for piece in self.probabilities:

                if piece == self.unk_token:
                    continue

                if not word.startswith(
                    piece,
                    i
                ):
                    continue

                j = i + len(piece)

                score = (
                    dp[i]
                    +
                    math.log(
                        self.probabilities[piece]
                    )
                )

                if score > dp[j]:

                    dp[j] = score

                    backpointer[j] = (
                        i,
                        piece
                    )

        # -----------------------------------------------------
        # Could not fully segment word.
        # -----------------------------------------------------

        if backpointer[n] is None:

            return None

        # -----------------------------------------------------
        # Reconstruct best path.
        # -----------------------------------------------------

        pieces = []

        position = n

        while position > 0:

            previous_position, piece = (
                backpointer[position]
            )

            pieces.append(piece)

            position = previous_position

        pieces.reverse()

        return pieces

    # =========================================================
    # ENCODE
    # =========================================================

    def encode(self, text):

        words = self._split_words(text)

        token_ids = []

        for word in words:

            pieces = self._segment_word(
                word
            )

            if pieces is None:

                token_ids.append(
                    self.vocab[
                        self.unk_token
                    ]
                )

                continue

            for piece in pieces:

                token_ids.append(
                    self.vocab[piece]
                )

        return token_ids

    # =========================================================
    # TOKENIZE
    # =========================================================

    def tokenize(self, text):

        words = self._split_words(text)

        output = []

        for word in words:

            pieces = self._segment_word(
                word
            )

            if pieces is None:

                output.append(
                    self.unk_token
                )

            else:

                output.extend(pieces)

        return output

    # =========================================================
    # DECODE
    # =========================================================

    def decode(self, token_ids):

        pieces = [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

        text = ""

        for piece in pieces:

            if piece == self.unk_token:

                if text:
                    text += " "

                text += self.unk_token

            elif piece in ".,!?;:%)]}":

                text += piece

            elif text:

                text += " " + piece

            else:

                text = piece

        return text

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
    "Tokenization is the first step in any NLP pipeline. "
    "Machine learning and artificial intelligence are transforming "
    "natural language processing."
)


# =============================================================
# TRAIN
# =============================================================

print("=" * 70)
print("UNIGRAM TOKENIZER TRAINING")
print("=" * 70)

tokenizer = UnigramTokenizer()

tokenizer.train(
    corpus,
    vocab_size=120
)


# =============================================================
# VOCABULARY
# =============================================================

print("\n" + "=" * 70)
print("VOCABULARY")
print("=" * 70)

print(
    f"Vocabulary size: "
    f"{tokenizer.vocabulary_size()}"
)

for token_id, token in (
    tokenizer.id_to_token.items()
):

    probability = (
        tokenizer.probabilities[token]
    )

    print(
        f"{token_id:4d} -> "
        f"{repr(token):20s} "
        f"P={probability:.8f}"
    )


# =============================================================
# TEST SENTENCES
# =============================================================

test_sentences = [

    "The cat sat on the mat.",

    "Natural language processing",

    "tokenization pipeline",

    "unhappiness",

    "Hello world",

    "machine learning",

    "artificial intelligence"
]


for sentence in test_sentences:

    print("\n" + "-" * 70)

    print(
        f"Text: {sentence}"
    )

    pieces = tokenizer.tokenize(
        sentence
    )

    token_ids = tokenizer.encode(
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
        f"Token count: "
        f"{len(token_ids)}"
    )

    print(
        f"Decoded: "
        f"{decoded}"
    )
