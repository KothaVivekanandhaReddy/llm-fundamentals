
"""
05 - WordPiece Tokenizer

Educational WordPiece implementation.

Training:
    Character pieces
        ↓
    Count pairs
        ↓
    WordPiece score
        ↓
    Merge best pair

Encoding:
    Longest matching vocabulary piece.

Continuation pieces are displayed with ##.

Example:

    playing

    play | ##ing
"""


from collections import Counter
import re


class WordPieceTokenizer:

    def __init__(self):

        self.vocab = {
            "[UNK]": 0
        }

        self.id_to_token = {
            0: "[UNK]"
        }

        self.unk_token = "[UNK]"

    # =========================================================
    # SPLIT TEXT
    # =========================================================

    def _split_text(self, text):

        return re.findall(
            r"\s+|[\w]+|[^\w\s]",
            text,
            flags=re.UNICODE
        )

    # =========================================================
    # PAIRS
    # =========================================================

    def _pairs(self, sequence):

        counts = Counter()

        for i in range(len(sequence) - 1):

            counts[
                (sequence[i], sequence[i + 1])
            ] += 1

        return counts

    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        text,
        vocab_size=160
    ):

        self.vocab = {
            "[UNK]": 0
        }

        words = [
            x
            for x in self._split_text(text)
            if not x.isspace()
        ]

        # Character vocabulary.
        for word in words:

            for char in word:

                if char not in self.vocab:

                    self.vocab[char] = len(
                        self.vocab
                    )

        # Character sequences per word.
        sequences = [
            list(word)
            for word in words
        ]

        # -----------------------------------------------------
        # WordPiece training
        # -----------------------------------------------------

        while len(self.vocab) < vocab_size:

            token_counts = Counter()

            pair_counts = Counter()

            for sequence in sequences:

                token_counts.update(
                    sequence
                )

                pair_counts.update(
                    self._pairs(sequence)
                )

            if not pair_counts:
                break

            best_pair = None
            best_score = -1

            for pair, count in pair_counts.items():

                left, right = pair

                score = (
                    count
                    /
                    (
                        token_counts[left]
                        *
                        token_counts[right]
                    )
                )

                if score > best_score:

                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            merged = (
                best_pair[0]
                +
                best_pair[1]
            )

            if merged in self.vocab:
                break

            self.vocab[merged] = len(
                self.vocab
            )

            sequences = [
                self._merge(
                    sequence,
                    best_pair,
                    merged
                )
                for sequence in sequences
            ]

        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
        }

        return self

    # =========================================================
    # MERGE
    # =========================================================

    def _merge(
        self,
        sequence,
        pair,
        merged
    ):

        result = []

        i = 0

        while i < len(sequence):

            if (
                i < len(sequence) - 1
                and sequence[i] == pair[0]
                and sequence[i + 1] == pair[1]
            ):

                result.append(merged)
                i += 2

            else:

                result.append(
                    sequence[i]
                )

                i += 1

        return result

    # =========================================================
    # ENCODE WORD
    # =========================================================

    def _encode_word(self, word):

        pieces = []

        start = 0

        while start < len(word):

            found = None

            for end in range(
                len(word),
                start,
                -1
            ):

                candidate = word[
                    start:end
                ]

                if candidate in self.vocab:

                    found = candidate
                    break

            if found is None:

                return None

            pieces.append(found)

            start += len(found)

        return pieces

    # =========================================================
    # TOKENIZE
    # =========================================================

    def tokenize(self, text):

        pieces = []

        for part in self._split_text(text):

            if part.isspace():

                # Preserve whitespace explicitly.
                pieces.extend(
                    list(part)
                )

                continue

            word_pieces = self._encode_word(
                part
            )

            if word_pieces is None:

                pieces.append(
                    self.unk_token
                )

                continue

            for index, piece in enumerate(
                word_pieces
            ):

                if index == 0:

                    pieces.append(piece)

                else:

                    pieces.append(
                        "##" + piece
                    )

        return pieces

    # =========================================================
    # ENCODE
    # =========================================================

    def encode(self, text):

        output = []

        for part in self._split_text(text):

            if part.isspace():

                # Whitespace token IDs are represented
                # using their character IDs.
                for char in part:

                    if char not in self.vocab:

                        self.vocab[char] = len(
                            self.vocab
                        )

                        self.id_to_token[
                            self.vocab[char]
                        ] = char

                    output.append(
                        self.vocab[char]
                    )

                continue

            pieces = self._encode_word(
                part
            )

            if pieces is None:

                output.append(
                    self.vocab[
                        self.unk_token
                    ]
                )

            else:

                output.extend(
                    self.vocab[piece]
                    for piece in pieces
                )

        return output

    # =========================================================
    # DECODE
    # =========================================================

    def decode(self, token_ids):

        result = ""

        previous_was_word = False

        for token_id in token_ids:

            piece = self.id_to_token[
                token_id
            ]

            if piece == self.unk_token:

                result += self.unk_token
                previous_was_word = False

            elif piece.isspace():

                result += piece
                previous_was_word = False

            else:

                # IDs don't contain ##; that's only
                # a display convention.
                if previous_was_word:

                    result += piece

                else:

                    result += piece

                previous_was_word = True

        return result

    # =========================================================
    # VOCAB SIZE
    # =========================================================

    def vocabulary_size(self):

        return len(self.vocab)

from pathlib import Path
if __name__ == "__main__":

    

    CORPUS_PATH = Path(__file__).resolve().parent / "corpus.txt"
    corpus = CORPUS_PATH.read_text(encoding="utf-8")

    tokenizer = WordPieceTokenizer()

    tokenizer.train(
        corpus,
        vocab_size=160
    )

    tests = [
        "The cat sat on the mat.",
        "Natural language processing",
        "unhappiness",
        "Hello world"
    ]

    print("=" * 70)
    print("WORDPIECE")
    print("=" * 70)

    for text in tests:

        pieces = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print("\nText:", text)
        print("Pieces:", pieces)
        print("IDs:", ids)
        print("Token count:", len(ids))
        print(
            "Roundtrip:",
            "PASS" if decoded == text else "FAIL"
        )
