import numpy as np


class EmbeddingModel:

    def __init__(self, vocab_size, embedding_dim, learning_rate=0.1):
        self.learning_rate = learning_rate

        self.embeddings = np.random.randn(
            vocab_size,
            embedding_dim
        )

    def forward(self, input_id, target_id):
        input_vector = self.embeddings[input_id]
        target_vector = self.embeddings[target_id]

        score = np.dot(input_vector, target_vector)

        return score

    def train_step(self, input_id, target_id):

        input_vector = self.embeddings[input_id]
        target_vector = self.embeddings[target_id]

        # Similarity score
        score = np.dot(input_vector, target_vector)

        # We want related tokens to have a high score.
        # Simple loss:
        loss = 0.5 * (1.0 - score) ** 2

        # dLoss/dScore
        dscore = score - 1.0

        # Gradients
        grad_input = dscore * target_vector
        grad_target = dscore * input_vector

        # Update both embedding vectors
        self.embeddings[input_id] -= (
            self.learning_rate * grad_input
        )

        self.embeddings[target_id] -= (
            self.learning_rate * grad_target
        )

        return loss, score


if __name__ == "__main__":

    np.random.seed(42)

    model = EmbeddingModel(
        vocab_size=5,
        embedding_dim=3,
        learning_rate=0.01
    )

    input_id = 1
    target_id = 2

    print("=" * 60)
    print("EMBEDDING TRAINING")
    print("=" * 60)

    initial_score = model.forward(
        input_id,
        target_id
    )

    print("\nInitial score:")
    print(initial_score)

    print("\nTraining...")

    for step in range(100):

        loss, score = model.train_step(
            input_id,
            target_id
        )

        if step % 10 == 0:
            print(
                f"Step {step:3d} | "
                f"Loss: {loss:.6f} | "
                f"Score: {score:.6f}"
            )

    final_score = model.forward(
        input_id,
        target_id
    )

    print("\nFinal score:")
    print(final_score)

    print("\nFinal input embedding:")
    print(model.embeddings[input_id])

    print("\nFinal target embedding:")
    print(model.embeddings[target_id])