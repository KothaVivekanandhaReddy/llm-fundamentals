
"""
02 - Word-Level Tokenizer

Goal:
    Understand whole-word tokenization.

Text
    ↓
Words
    ↓
Token IDs

Example:

    "The cat sat"

    The | cat | sat
     ↓     ↓     ↓
     0     1     2


Important limitation:
    Words not seen during training become <UNK>.
"""

import re


class WordTokenizer:

    def __init__(self):

        self.vocab = {}
        self.id_to_token = {}

        # Special token for unknown words.
        self.unk_token = "<UNK>"

    # ---------------------------------------------------------
    # TOKENIZE TEXT
    # ---------------------------------------------------------

    def _tokenize(self, text):

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

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text):

        tokens = self._tokenize(text)

        # Unique tokens in the training corpus.
        unique_tokens = sorted(set(tokens))

        # Reserve ID 0 for unknown tokens.
        self.vocab = {
            self.unk_token: 0
        }

        # Assign IDs to known words/tokens.
        for idx, token in enumerate(
            unique_tokens,
            start=1
        ):

            self.vocab[token] = idx

        # Reverse vocabulary.
        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
        }

        return self

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text):

        tokens = self._tokenize(text)

        encoded = []

        for token in tokens:

            token_id = self.vocab.get(
                token,
                self.vocab[self.unk_token]
            )

            encoded.append(token_id)

        return encoded

    # ---------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------

    def decode(self, token_ids):

        tokens = [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

        text = ""

        for token in tokens:

            # Punctuation should not have a
            # space before it.
            if token in ".,!?;:%)]}":

                text += token

            elif text:

                text += " " + token

            else:

                text = token

        return text

    # ---------------------------------------------------------
    # VOCABULARY
    # ---------------------------------------------------------

    def vocabulary_size(self):

        return len(self.vocab)


from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent / "corpus.txt"

corpus = CORPUS_PATH.read_text(encoding="utf-8")


# =============================================================
# TRAIN
# =============================================================

tokenizer = WordTokenizer()

tokenizer.train(corpus)


print("=" * 60)
print("WORD-LEVEL TOKENIZER")
print("=" * 60)

print(
    f"Vocabulary size: "
    f"{tokenizer.vocabulary_size()}"
)

print(
    f"Vocabulary:\n"
    f"{tokenizer.vocab}"
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

    "🚀 AI is amazing!"
]


for sentence in test_sentences:

    print("\n" + "-" * 60)

    print(f"Text: {sentence}")

    encoded = tokenizer.encode(sentence)

    decoded = tokenizer.decode(encoded)

    print(
        f"Tokens: "
        f"{tokenizer._tokenize(sentence)}"
    )

    print(
        f"Token IDs: "
        f"{encoded}"
    )

    print(
        f"Token count: "
        f"{len(encoded)}"
    )

    print(
        f"Decoded: "
        f"{decoded}"
    )

    print(
        f"Roundtrip: "
        f"{'PASS' if decoded == sentence else 'FAIL'}"
    )


# =============================================================
# UNKNOWN TOKEN DEMONSTRATION
# =============================================================

print("\n" + "=" * 60)
print("UNKNOWN TOKEN DEMONSTRATION")
print("=" * 60)

unknown_text = "quantum computing"

print(f"Input: {unknown_text}")

print(
    f"Tokens: "
    f"{tokenizer._tokenize(unknown_text)}"
)

encoded = tokenizer.encode(unknown_text)

print(
    f"Token IDs: "
    f"{encoded}"
)

print(
    f"Decoded: "
    f"{tokenizer.decode(encoded)}"
)

print(
    "\nNotice how words that were not present "
    "in the training vocabulary become <UNK>."
)


# =============================================================
# VOCABULARY ANALYSIS
# =============================================================

print("\n" + "=" * 60)
print("VOCABULARY")
print("=" * 60)

for token_id, token in tokenizer.id_to_token.items():

    print(
        f"{token_id:4d} -> {repr(token)}"
    )
