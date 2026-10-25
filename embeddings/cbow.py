#Model	    Input	                    Prediction target
#Skip-gram	One center word	            Surrounding context words
#CBOW	    Surrounding context words	One center word

# Continuous Bag of Words (CBOW) : it predicts a missing target word based on its surrounding context words.
# A word embedding model that predicts a target word based on its surrounding context words. 
# It takes the context words as input and learns to predict the center word. The model uses a neural network architecture to learn 
# the embeddings for words in a given vocabulary.

# ex : the cat sat on the mat
# With a context window of 1, one CBOW training example is: 
# Input: ["the", "sat"] → Target: "cat"
# The model looks up the embeddings for the and sat, averages them, and uses the resulting vector to predict cat.
# mathematically for context embeddings v1 v2 .... vc : h = 1/c sigma(i=1 to c) vi and z = hWout 
# Here, h is the averaged context representation and z contains vocabulary logits.

import numpy as np


class CBOWDataset:
    """Create CBOW context-target training examples."""

    def __init__(self, text, window_size=1):
        self.text = text.lower()
        self.window_size = window_size

        self.tokens = self.tokenize(self.text)
        self.word_to_id = self.build_vocabulary()
        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }
        self.training_examples = self.create_training_examples()

    @staticmethod
    def tokenize(text):
        return text.split()

    def build_vocabulary(self):
        vocabulary = sorted(set(self.tokens))
        return {
            word: idx for idx, word in enumerate(vocabulary)
        }

    def create_training_examples(self):
        token_ids = [
            self.word_to_id[token] for token in self.tokens
        ]
        examples = []

        for center_index, target_id in enumerate(token_ids):
            start = max(0, center_index - self.window_size)
            end = min(
                len(token_ids),
                center_index + self.window_size + 1,
            )

            context_ids = [
                token_ids[i]
                for i in range(start, end)
                if i != center_index
            ]

            # The first and last positions can have partial context.
            # Skip only positions that have no context at all.
            if context_ids:
                examples.append((context_ids, target_id))

        return examples


class CBOWModel:
    """CBOW with mean pooling, full softmax, and manual backpropagation."""

    def __init__(self, vocab_size, embedding_dim, seed=42):
        rng = np.random.default_rng(seed)

        # Input embedding matrix: (vocab_size, embedding_dim)
        self.W_in = (
            rng.standard_normal((vocab_size, embedding_dim)) * 0.01
        )

        # Output weight matrix: (embedding_dim, vocab_size)
        self.W_out = (
            rng.standard_normal((embedding_dim, vocab_size)) * 0.01
        )

    def forward(self, context_ids):
        context_ids = np.asarray(context_ids, dtype=np.int64)

        if context_ids.size == 0:
            raise ValueError("context_ids must not be empty.")

        # Look up context embeddings: (context_size, embedding_dim)
        context_embeddings = self.W_in[context_ids]

        # Mean-pool the context: (embedding_dim,)
        hidden = np.mean(context_embeddings, axis=0)

        # Vocabulary logits: (vocab_size,)
        logits = hidden @ self.W_out

        # Numerically stable softmax
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits)

        return hidden, probabilities

    @staticmethod
    def cross_entropy_loss(probabilities, target_id):
        probability = np.clip(
            probabilities[target_id], 1e-12, 1.0
        )
        return float(-np.log(probability))

    def train_step(self, context_ids, target_id, learning_rate=0.01):
        """
        Perform one SGD update.

        All gradients are calculated using the same pre-update
        parameter values. Repeated context IDs accumulate gradients.
        """
        context_ids = np.asarray(context_ids, dtype=np.int64)

        # Forward pass
        hidden, probabilities = self.forward(context_ids)
        loss = self.cross_entropy_loss(probabilities, target_id)

        # dL/dlogits for softmax + cross-entropy
        d_logits = probabilities.copy()
        d_logits[target_id] -= 1.0

        # dL/dW_out: (embedding_dim, vocab_size)
        d_W_out = np.outer(hidden, d_logits)

        # dL/dhidden: (embedding_dim,)
        d_hidden = self.W_out @ d_logits

        # Backpropagate through mean pooling
        d_context_embedding = d_hidden / len(context_ids)

        # Accumulate gradients for the input embedding matrix.
        # This handles repeated word IDs correctly.
        d_W_in = np.zeros_like(self.W_in)

        for word_id in context_ids:
            d_W_in[word_id] += d_context_embedding

        # Apply both updates only after computing all gradients.
        self.W_in -= learning_rate * d_W_in
        self.W_out -= learning_rate * d_W_out

        return loss

    def evaluate_loss(self, training_examples):
        """Evaluate mean cross-entropy without updating parameters."""
        if not training_examples:
            raise ValueError("training_examples must not be empty.")

        total_loss = 0.0

        for context_ids, target_id in training_examples:
            _, probabilities = self.forward(context_ids)
            total_loss += self.cross_entropy_loss(
                probabilities, target_id
            )

        return total_loss / len(training_examples)

    def train(
        self,
        training_examples,
        epochs=100,
        learning_rate=0.01,
        seed=42,
    ):
        if not training_examples:
            raise ValueError("training_examples must not be empty.")

        rng = np.random.default_rng(seed)
        history = []

        for epoch in range(epochs):
            # Shuffle the example order reproducibly each epoch.
            order = rng.permutation(len(training_examples))

            for index in order:
                context_ids, target_id = training_examples[index]
                self.train_step(
                    context_ids,
                    target_id,
                    learning_rate=learning_rate,
                )

            # Evaluate the complete dataset using fixed parameters.
            epoch_loss = self.evaluate_loss(training_examples)
            history.append(epoch_loss)

            if epoch == 0 or (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Evaluation loss: {epoch_loss:.6f}"
                )

        return history

    @staticmethod
    def cosine_similarity(a, b):
        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 0.0

        return float(np.dot(a, b) / denominator)

    def most_similar(self, word, dataset, top_k=3):
        if word not in dataset.word_to_id:
            raise ValueError(f"Word not in vocabulary: {word}")

        word_id = dataset.word_to_id[word]
        query_vector = self.W_in[word_id]
        scores = []

        for candidate, candidate_id in dataset.word_to_id.items():
            if candidate == word:
                continue

            score = self.cosine_similarity(
                query_vector,
                self.W_in[candidate_id],
            )
            scores.append((candidate, score))

        return sorted(
            scores,
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]


if __name__ == "__main__":
    TEXT = "the cat sat on the mat"
    WINDOW_SIZE = 1
    EMBEDDING_DIM = 4
    EPOCHS = 1000
    LEARNING_RATE = 0.1
    SEED = 42

    dataset = CBOWDataset(TEXT, window_size=WINDOW_SIZE)

    print("=" * 60)
    print("CBOW MODEL TRAINING")
    print("=" * 60)
    print(f"Vocabulary size: {len(dataset.word_to_id)}")
    print(f"Training examples: {len(dataset.training_examples)}")
    print(f"Vocabulary: {dataset.word_to_id}")

    model = CBOWModel(
        vocab_size=len(dataset.word_to_id),
        embedding_dim=EMBEDDING_DIM,
        seed=SEED,
    )

    # Evaluate before training using the entire dataset.
    initial_loss = model.evaluate_loss(dataset.training_examples)

    history = model.train(
        dataset.training_examples,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )

    # Evaluate after training using the same examples and metric.
    final_loss = model.evaluate_loss(dataset.training_examples)

    loss_reduction = (
        (initial_loss - final_loss) / initial_loss * 100
        if initial_loss != 0
        else 0.0
    )

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Initial evaluation loss: {initial_loss:.6f}")
    print(f"Final evaluation loss:   {final_loss:.6f}")
    print(f"Loss reduction:          {loss_reduction:.2f}%")

    print("\nInput embedding matrix shape:", model.W_in.shape)
    print("Output weight matrix shape:", model.W_out.shape)

    print("\nLearned input embeddings:")
    for word, word_id in dataset.word_to_id.items():
        print(f"{word:>5}: {model.W_in[word_id]}")

    print("\nMost similar words (cosine similarity):")
    for word in dataset.word_to_id:
        print(f"{word}: {model.most_similar(word, dataset)}")
