import numpy as np


class TrainableEmbedding:

    def __init__(self, vocab_size, embedding_dim, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        self.weights = np.random.randn(
            vocab_size,
            embedding_dim
        )

    def lookup(self, token_id):
        return self.weights[token_id]

    def update(self, token_id, gradient):
        self.weights[token_id] -= (
            self.learning_rate * gradient
        )


if __name__ == "__main__":

    np.random.seed(42)

    embedding = TrainableEmbedding(
        vocab_size=5,
        embedding_dim=3,
        learning_rate=0.1
    )

    token_id = 2

    before = embedding.lookup(token_id).copy()

    # Artificial gradient for demonstration.
    # In a real model this comes from the loss/backpropagation.
    gradient = np.array([0.5, -0.2, 0.8])

    embedding.update(token_id, gradient)

    after = embedding.lookup(token_id)

    print("=" * 60)
    print("TRAINABLE EMBEDDING")
    print("=" * 60)

    print("\nToken ID:")
    print(token_id)

    print("\nEmbedding before update:")
    print(before)

    print("\nGradient:")
    print(gradient)

    print("\nEmbedding after update:")
    print(after)

    print("\nChange:")
    print(after - before)