import numpy as np


class SkipGramDataset:

    def __init__(self, text, window_size=1):

        self.text = text.lower()
        self.window_size = window_size

        self.tokens = self.tokenize(self.text)

        self.word_to_id = self.build_vocabulary()

        self.id_to_word = {
            idx: word
            for word, idx in self.word_to_id.items()
        }

        self.training_pairs = self.create_training_pairs()

    def tokenize(self, text):

        return text.split()

    def build_vocabulary(self):

        vocabulary = sorted(set(self.tokens))

        return {
            word: idx
            for idx, word in enumerate(vocabulary)
        }

    def create_training_pairs(self):

        token_ids = [
            self.word_to_id[token]
            for token in self.tokens
        ]

        pairs = []

        for center_index in range(len(token_ids)):

            start = max(
                0,
                center_index - self.window_size
            )

            end = min(
                len(token_ids),
                center_index + self.window_size + 1
            )

            center_id = token_ids[center_index]

            for context_index in range(start, end):

                if context_index == center_index:
                    continue

                context_id = token_ids[context_index]

                pairs.append(
                    (center_id, context_id)
                )

        return pairs


class SkipGramModel:

    def __init__(self, vocab_size, embedding_dim):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Input embedding matrix
        # Shape: (V, D)
        self.W_in = (
            np.random.randn(
                vocab_size,
                embedding_dim
            ) * 0.01
        )

        # Output/context matrix
        # Shape: (D, V)
        self.W_out = (
            np.random.randn(
                embedding_dim,
                vocab_size
            ) * 0.01
        )

    def forward(self, center_id):

        # Embedding lookup
        hidden = self.W_in[center_id]

        # Project embedding into vocabulary space
        logits = hidden @ self.W_out

        return hidden, logits

    def softmax(self, logits):

        shifted_logits = (
            logits - np.max(logits)
        )

        exp_logits = np.exp(
            shifted_logits
        )

        probabilities = (
            exp_logits /
            np.sum(exp_logits)
        )

        return probabilities

    def cross_entropy_loss(
        self,
        probabilities,
        target_id
    ):

        return -np.log(
            probabilities[target_id]
        )

    def backward(
        self,
        center_id,
        hidden,
        probabilities,
        target_id
    ):

        # Gradient of softmax + cross entropy
        d_logits = probabilities.copy()

        d_logits[target_id] -= 1

        # Gradient of W_out
        d_W_out = np.outer(
            hidden,
            d_logits
        )

        # Gradient flowing into hidden
        d_hidden = (
            d_logits @ self.W_out.T
        )

        # Gradient of W_in
        d_W_in = np.zeros_like(
            self.W_in
        )

        # Only selected embedding row
        # receives the gradient.
        d_W_in[center_id] = d_hidden

        return d_W_in, d_W_out

    def train_step(
        self,
        center_id,
        target_id,
        learning_rate=0.1
    ):

        # -------------------------
        # Forward
        # -------------------------

        hidden, logits = self.forward(
            center_id
        )

        probabilities = self.softmax(
            logits
        )

        # -------------------------
        # Loss
        # -------------------------

        loss = self.cross_entropy_loss(
            probabilities,
            target_id
        )

        # -------------------------
        # Backward
        # -------------------------

        d_W_in, d_W_out = self.backward(
            center_id,
            hidden,
            probabilities,
            target_id
        )

        # -------------------------
        # Gradient descent
        # -------------------------

        self.W_in -= (
            learning_rate * d_W_in
        )

        self.W_out -= (
            learning_rate * d_W_out
        )

        return loss
    def train(
        self,
        training_pairs,
        epochs=100,
        learning_rate=0.1
    ):

        history = []

        for epoch in range(epochs):

            total_loss = 0.0

            for center_id, target_id in training_pairs:

                loss = self.train_step(
                    center_id,
                    target_id,
                    learning_rate
                )

                total_loss += loss

            average_loss = (
                total_loss /
                len(training_pairs)
            )

            history.append(average_loss)

            if (
                epoch == 0
                or (epoch + 1) % 10 == 0
            ):
                print(
                    f"Epoch {epoch + 1:3d} "
                    f"| Loss: {average_loss:.6f}"
                )

        return history


if __name__ == "__main__":

    # ==================================================
    # DATASET
    # ==================================================

    text = "the cat sat on the mat"

    dataset = SkipGramDataset(
        text,
        window_size=1
    )

    # ==================================================
    # MODEL
    # ==================================================

    model = SkipGramModel(
        vocab_size=len(
            dataset.word_to_id
        ),
        embedding_dim=4
    )

    print("=" * 60)
    print("WORD2VEC SKIP-GRAM")
    print("=" * 60)

    print("\nVocabulary size:")
    print(model.vocab_size)

    print("\nEmbedding dimension:")
    print(model.embedding_dim)

    print("\nW_in shape:")
    print(model.W_in.shape)

    print("\nW_out shape:")
    print(model.W_out.shape)

        # ==================================================
    # FULL TRAINING
    # ==================================================

    print("\nTraining pairs:")

    for center_id, target_id in dataset.training_pairs:

        center_word = dataset.id_to_word[
            center_id
        ]

        target_word = dataset.id_to_word[
            target_id
        ]

        print(
            f"{center_word:>5} → "
            f"{target_word:<5}"
        )

    print("\nNumber of training pairs:")
    print(len(dataset.training_pairs))

    print("\n--- Training ---")

    history = model.train(
        dataset.training_pairs,
        epochs=100,
        learning_rate=0.1
    )

    print("\n--- Training Complete ---")

    print("Initial loss:")
    print(history[0])

    print("\nFinal loss:")
    print(history[-1])