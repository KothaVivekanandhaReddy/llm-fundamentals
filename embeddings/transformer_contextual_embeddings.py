import numpy as np


class ContextualEmbeddingDemo:

    def __init__(self, vocab_size, embedding_dim, seed=42):

        np.random.seed(seed)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Token embedding matrix
        self.token_embeddings = (
            np.random.randn(
                vocab_size,
                embedding_dim
            ) * 0.1
        )

        # Projection matrices for self-attention
        self.W_Q = (
            np.random.randn(
                embedding_dim,
                embedding_dim
            ) * 0.1
        )

        self.W_K = (
            np.random.randn(
                embedding_dim,
                embedding_dim
            ) * 0.1
        )

        self.W_V = (
            np.random.randn(
                embedding_dim,
                embedding_dim
            ) * 0.1
        )

    def positional_encoding(self, sequence_length):

        positions = np.arange(sequence_length)[:, None]

        dimensions = np.arange(
            self.embedding_dim
        )[None, :]

        angle_rates = 1 / np.power(
            10000,
            (2 * (dimensions // 2))
            / self.embedding_dim
        )

        angles = positions * angle_rates

        encoding = np.zeros(
            (sequence_length, self.embedding_dim)
        )

        encoding[:, 0::2] = np.sin(
            angles[:, 0::2]
        )

        encoding[:, 1::2] = np.cos(
            angles[:, 1::2]
        )

        return encoding

    def softmax(self, x):

        x = x - np.max(
            x,
            axis=-1,
            keepdims=True
        )

        exp_x = np.exp(x)

        return exp_x / np.sum(
            exp_x,
            axis=-1,
            keepdims=True
        )

    def self_attention(self, X):

        # Query
        Q = X @ self.W_Q

        # Key
        K = X @ self.W_K

        # Value
        V = X @ self.W_V

        # Attention scores
        scores = (
            Q @ K.T
        ) / np.sqrt(
            self.embedding_dim
        )

        # Attention weights
        attention_weights = self.softmax(
            scores
        )

        # Contextual representations
        output = (
            attention_weights @ V
        )

        return output, attention_weights

    def encode(self, token_ids):

        sequence_length = len(token_ids)

        # --------------------------------------------------
        # 1. Token embeddings
        # --------------------------------------------------

        token_vectors = self.token_embeddings[
            token_ids
        ]

        # --------------------------------------------------
        # 2. Positional encoding
        # --------------------------------------------------

        positions = self.positional_encoding(
            sequence_length
        )

        # --------------------------------------------------
        # 3. Combine token + position
        # --------------------------------------------------

        X = token_vectors + positions

        # --------------------------------------------------
        # 4. Self-attention
        # --------------------------------------------------

        contextual_vectors, attention_weights = (
            self.self_attention(X)
        )

        return (
            contextual_vectors,
            attention_weights
        )


if __name__ == "__main__":

    print("=" * 60)
    print("TRANSFORMER CONTEXTUAL EMBEDDINGS")
    print("=" * 60)

    # ------------------------------------------------------
    # Tiny vocabulary
    # ------------------------------------------------------

    vocabulary = {
        "the": 0,
        "bank": 1,
        "river": 2,
        "money": 3,
        "flows": 4
    }

    reverse_vocabulary = {
        index: word
        for word, index in vocabulary.items()
    }

    model = ContextualEmbeddingDemo(
        vocab_size=len(vocabulary),
        embedding_dim=8,
        seed=42
    )

    # ------------------------------------------------------
    # Sentence 1
    # ------------------------------------------------------

    sentence_1 = [
        "the",
        "bank",
        "river"
    ]

    ids_1 = [
        vocabulary[word]
        for word in sentence_1
    ]

    vectors_1, attention_1 = model.encode(
        ids_1
    )

    # ------------------------------------------------------
    # Sentence 2
    # ------------------------------------------------------

    sentence_2 = [
        "the",
        "bank",
        "money"
    ]

    ids_2 = [
        vocabulary[word]
        for word in sentence_2
    ]

    vectors_2, attention_2 = model.encode(
        ids_2
    )

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print("\nSentence 1:")
    print(" ".join(sentence_1))

    print("\nAttention weights:")
    print(np.round(attention_1, 3))

    print("\nContextual embedding for 'bank':")
    print(
        np.round(
            vectors_1[1],
            4
        )
    )

    print("\nSentence 2:")
    print(" ".join(sentence_2))

    print("\nAttention weights:")
    print(np.round(attention_2, 3))

    print("\nContextual embedding for 'bank':")
    print(
        np.round(
            vectors_2[1],
            4
        )
    )

    # ------------------------------------------------------
    # Compare bank representations
    # ------------------------------------------------------

    bank_1 = vectors_1[1]
    bank_2 = vectors_2[1]

    similarity = np.dot(
        bank_1,
        bank_2
    ) / (
        np.linalg.norm(bank_1)
        * np.linalg.norm(bank_2)
    )

    print("\n" + "=" * 60)
    print("CONTEXT EFFECT")
    print("=" * 60)

    print(
        "\nCosine similarity between "
        "'bank' representations:"
    )

    print(
        f"{similarity:.4f}"
    )