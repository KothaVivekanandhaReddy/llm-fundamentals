
"""
01 - Character-Level Tokenizer

Goal:
    Understand the simplest possible tokenizer.

Text
    ↓
Characters
    ↓
Token IDs

Example:

    "Hello"

    H e l l o
    ↓ ↓ ↓ ↓ ↓
    0 1 2 2 3
"""


class CharacterTokenizer:

    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text):

        # Find every unique character.
        characters = sorted(set(text))

        # Assign an integer ID to every character.
        self.vocab = {
            char: idx
            for idx, char in enumerate(characters)
        }

        # Reverse mapping.
        self.id_to_token = {
            idx: char
            for char, idx in self.vocab.items()
        }

        return self

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text):

        return [
            self.vocab[char]
            for char in text
        ]

    # ---------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------

    def decode(self, tokens):

        return "".join(
            self.id_to_token[token]
            for token in tokens
        )

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

tokenizer = CharacterTokenizer()

tokenizer.train(corpus)


print("=" * 60)
print("CHARACTER-LEVEL TOKENIZER")
print("=" * 60)

print(
    f"Vocabulary size: "
    f"{tokenizer.vocabulary_size()}"
)

print(
    f"Vocabulary: "
    f"{list(tokenizer.vocab.keys())}"
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

    try:

        encoded = tokenizer.encode(sentence)

        decoded = tokenizer.decode(encoded)

        print(f"Tokens: {encoded}")

        print(
            f"Token count: "
            f"{len(encoded)}"
        )

        print(
            f"Roundtrip: "
            f"{'PASS' if decoded == sentence else 'FAIL'}"
        )

    except KeyError as error:

        print(
            f"UNKNOWN CHARACTER: "
            f"{error}"
        )


# =============================================================
# CHARACTER ANALYSIS
# =============================================================

print("\n" + "=" * 60)
print("CHARACTER VOCABULARY")
print("=" * 60)

for token_id, character in tokenizer.id_to_token.items():

    display = repr(character)

    print(
        f"{token_id:3d} -> {display}"
    )
