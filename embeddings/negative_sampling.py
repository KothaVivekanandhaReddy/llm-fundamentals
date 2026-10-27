#Negative sampling : changes a slow multi-class prediction task into a fast binary classification task
# Instead of calculating probabilities for every single word in the entire vocabulary, the model learns to tell the difference 
# between a real word pair and a few fake (random) word pairs.

# Why It Is Needed :
#  The Softmax Problem: 
#  Standard Word2Vec models use a softmax function over the whole vocabulary. If your dictionary has 3 million words, the model 
#  must update millions of weights for every single training step. This makes training very slow.

# The Solution: Negative sampling skips the full vocabulary calculation. It updates only a tiny fraction of the weights during 
# each step, making training fast and scalable for large text files

# How It Works :
# 1. Positive Pairs: The model takes a real word pair that appears together in a sentence (like "bark" and "dog") and labels it as 1 (true).
# 2. Negative Pairs:  The model picks a small number (k) of random words from the vocabulary that do not appear with the center word (like "bark" and "table") 
# and labels them as 0 (false).
# Binary Classification: The model uses logistic regression to maximize the score of the real pair and minimize the scores of the fake pairs.

# Choosing Negative Samples :
# Word2Vec does not pick random words uniformly. If it did, common words like "the" or "is" would be chosen as negative samples all the time. Instead, 
# it uses a unigram distribution raised to the 3/4 power (3/4). This gives rare words a higher chance of being picked as negative examples than they would have 
# normally, which helps the model learn better representations.

# key change :
# Full Softmax :
#center word
#   ↓
#embedding
#   ↓
#scores for ALL vocabulary words
#   ↓
#softmax(V classes)

# Negative Sampling :
# center word
# ↓
#embedding
#  ↓
#positive context + k negative contexts
#   ↓
#binary classification

"""
Word2Vec Negative Sampling — NumPy implementation from scratch.

Objective:
    L = -log(sigmoid(v_c . u_w))
        - sum_i log(sigmoid(-v_n_i . u_w))

where:
    v_c   : input embedding of the center word
    u_w   : output embedding of the positive context word
    u_n_i : output embedding of a negative sample

This is educational implementation and uses uniform negative sampling.
It does not implement the unigram^0.75 distribution used by
the original Word2Vec negative-sampling implementation.
"""

import numpy as np

from word2vec_skipgram import SkipGramDataset


# ============================================================
# NUMERICAL UTILITIES
# ============================================================

def sigmoid(x):
    """Numerically stable sigmoid for a scalar."""
    if x >= 0:
        exp_neg_x = np.exp(-x)
        return 1.0 / (1.0 + exp_neg_x)

    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def cosine_similarity(a, b):
    """Cosine similarity with protection against zero vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    denominator = norm_a * norm_b

    if denominator < 1e-12:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ============================================================
# NEGATIVE SAMPLING MODEL
# ============================================================

class NegativeSamplingModel:

    def __init__(self, vocab_size, embedding_dim, seed=42):
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")

        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Model-owned RNG: initialization, shuffling, and sampling
        # are reproducible without relying on global NumPy state.
        self.rng = np.random.default_rng(seed)

        # Input embeddings: one row per center word.
        self.W_in = (
            self.rng.standard_normal((vocab_size, embedding_dim))
            * 0.01
        )

        # Output embeddings: one column per context word.
        self.W_out = (
            self.rng.standard_normal((embedding_dim, vocab_size))
            * 0.01
        )

    # --------------------------------------------------------
    # NEGATIVE SAMPLING LOSS
    # --------------------------------------------------------

    def _loss_and_gradients(
        self,
        center_id,
        positive_id,
        negative_ids,
    ):
        """
        Compute loss and gradients at the current parameters.

        No model parameters are modified in this method.

        Returns:
            loss
            d_center
            d_positive_vector
            d_negative_vectors
        """

        center = self.W_in[center_id].copy()
        positive_vector = self.W_out[:, positive_id].copy()

        # -----------------------------------------------
        # Positive pair: maximize sigmoid(center · positive)
        # -----------------------------------------------

        positive_score = float(
            np.dot(center, positive_vector)
        )

        # Stable equivalent of -log(sigmoid(score)):
        # log(1 + exp(-score))
        positive_loss = float(
            np.logaddexp(0.0, -positive_score)
        )

        positive_gradient = (
            sigmoid(positive_score) - 1.0
        )

        d_center = (
            positive_gradient * positive_vector
        )

        d_positive_vector = (
            positive_gradient * center
        )

        # -----------------------------------------------
        # Negative pairs: minimize sigmoid(center · negative)
        # -----------------------------------------------

        negative_loss = 0.0
        d_negative_vectors = []

        for negative_id in negative_ids:
            negative_vector = (
                self.W_out[:, negative_id].copy()
            )

            negative_score = float(
                np.dot(center, negative_vector)
            )

            # Stable equivalent of -log(sigmoid(-score)):
            # log(1 + exp(score))
            negative_loss += float(
                np.logaddexp(0.0, negative_score)
            )

            negative_gradient = sigmoid(
                negative_score
            )

            d_center += (
                negative_gradient * negative_vector
            )

            d_negative_vectors.append(
                negative_gradient * center
            )

        loss = positive_loss + negative_loss

        return (
            loss,
            d_center,
            d_positive_vector,
            d_negative_vectors,
        )

    # --------------------------------------------------------
    # SINGLE TRAINING STEP
    # --------------------------------------------------------

    def train_step(
        self,
        center_id,
        positive_id,
        negative_ids,
        learning_rate=0.1,
    ):
        """
        Apply one SGD update for a positive pair and its negatives.

        Returns the loss computed before the parameter update.
        """

        if not 0 <= center_id < self.vocab_size:
            raise ValueError("Invalid center_id.")

        if not 0 <= positive_id < self.vocab_size:
            raise ValueError("Invalid positive_id.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        negative_ids = np.asarray(
            negative_ids,
            dtype=np.int64,
        ).reshape(-1)

        if negative_ids.size == 0:
            raise ValueError(
                "At least one negative sample is required."
            )

        if np.any(negative_ids < 0) or np.any(
            negative_ids >= self.vocab_size
        ):
            raise ValueError("Invalid negative word ID.")

        # A negative sample must not be the positive target.
        if np.any(negative_ids == positive_id):
            raise ValueError(
                "Negative samples must exclude the positive target."
            )

        (
            loss,
            d_center,
            d_positive_vector,
            d_negative_vectors,
        ) = self._loss_and_gradients(
            center_id,
            positive_id,
            negative_ids,
        )

        # -----------------------------------------------
        # Apply SGD updates
        # -----------------------------------------------

        self.W_in[center_id] -= (
            learning_rate * d_center
        )

        self.W_out[:, positive_id] -= (
            learning_rate * d_positive_vector
        )

        for negative_id, gradient in zip(
            negative_ids,
            d_negative_vectors,
        ):
            self.W_out[:, negative_id] -= (
                learning_rate * gradient
            )

        return loss

    # --------------------------------------------------------
    # NEGATIVE SAMPLING
    # --------------------------------------------------------

    def sample_negatives(
        self,
        positive_id,
        negative_samples=3,
    ):
        """
        Uniformly sample negative IDs without replacement.

        The positive target is excluded.

        Note:
            The center word is not automatically excluded.
            This is an explicit sampling policy for this
            educational implementation.
        """

        if not 0 <= positive_id < self.vocab_size:
            raise ValueError("Invalid positive_id.")

        if negative_samples < 1:
            raise ValueError(
                "negative_samples must be at least 1."
            )

        candidates = np.concatenate(
            (
                np.arange(positive_id),
                np.arange(positive_id + 1, self.vocab_size),
            )
        )

        if negative_samples > len(candidates):
            raise ValueError(
                f"Requested {negative_samples} negative samples, "
                f"but only {len(candidates)} candidates exist."
            )

        return self.rng.choice(
            candidates,
            size=negative_samples,
            replace=False,
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    def evaluate_loss(
        self,
        pairs,
        negative_samples=3,
        eval_seed=12345,
    ):
        """
        Evaluate mean loss with fixed negative samples.

        The same evaluation seed and pair order produce the
        same negative samples at every evaluation point.

        Parameters are not modified.
        """

        pairs = list(pairs)

        if not pairs:
            raise ValueError("Cannot evaluate an empty dataset.")

        eval_rng = np.random.default_rng(eval_seed)

        all_ids = np.arange(self.vocab_size)
        total_loss = 0.0

        for center_id, positive_id in pairs:
            candidates = all_ids[
                all_ids != positive_id
            ]

            if negative_samples < 1:
                raise ValueError(
                    "negative_samples must be at least 1."
                )

            if negative_samples > len(candidates):
                raise ValueError(
                    "Too many negative samples for vocabulary size."
                )

            negative_ids = eval_rng.choice(
                candidates,
                size=negative_samples,
                replace=False,
            )

            loss, _, _, _ = self._loss_and_gradients(
                center_id,
                positive_id,
                negative_ids,
            )

            total_loss += loss

        return total_loss / len(pairs)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    def train(
        self,
        pairs,
        epochs=100,
        negative_samples=3,
        learning_rate=0.1,
        eval_seed=12345,
    ):
        """
        Train using shuffled positive pairs and freshly sampled
        negatives.

        Returns fixed-negative evaluation loss after each epoch.
        """

        pairs = list(pairs)

        if not pairs:
            raise ValueError("Cannot train on an empty dataset.")

        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        if negative_samples < 1:
            raise ValueError(
                "negative_samples must be at least 1."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        losses = []

        initial_loss = self.evaluate_loss(
            pairs,
            negative_samples=negative_samples,
            eval_seed=eval_seed,
        )

        print(f"Initial evaluation loss: {initial_loss:.6f}")

        for epoch in range(epochs):
            # Shuffle the positive training pairs.
            order = self.rng.permutation(len(pairs))

            epoch_training_loss = 0.0

            for pair_index in order:
                center_id, positive_id = pairs[pair_index]

                negative_ids = self.sample_negatives(
                    positive_id,
                    negative_samples=negative_samples,
                )

                step_loss = self.train_step(
                    center_id=center_id,
                    positive_id=positive_id,
                    negative_ids=negative_ids,
                    learning_rate=learning_rate,
                )

                epoch_training_loss += step_loss

            mean_training_loss = (
                epoch_training_loss / len(pairs)
            )

            evaluation_loss = self.evaluate_loss(
                pairs,
                negative_samples=negative_samples,
                eval_seed=eval_seed,
            )

            losses.append(evaluation_loss)

            if (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or epoch == epochs - 1
            ):
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Training loss: {mean_training_loss:.6f} | "
                    f"Evaluation loss: {evaluation_loss:.6f}"
                )

        final_loss = losses[-1]

        print("\nTraining summary")
        print(f"Initial evaluation loss: {initial_loss:.6f}")
        print(f"Final evaluation loss:   {final_loss:.6f}")

        if initial_loss > 0:
            reduction = (
                (initial_loss - final_loss)
                / initial_loss
                * 100.0
            )
            print(f"Evaluation loss reduction: {reduction:.2f}%")

        return losses

    # --------------------------------------------------------
    # EMBEDDING ACCESS
    # --------------------------------------------------------

    def get_embedding(self, word_id):
        return self.W_in[word_id].copy()


# ============================================================
# MAIN
# ============================================================

def main():
    text = "the cat sat on the mat"

    dataset = SkipGramDataset(
        text=text,
        window_size=1,
    )

    vocab_size = len(dataset.word_to_id)

    print("=" * 65)
    print("WORD2VEC NEGATIVE SAMPLING")
    print("=" * 65)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Training pairs: {len(dataset.training_pairs)}")
    print(f"Vocabulary: {dataset.word_to_id}")

    print("\nTraining pairs:")
    for center_id, context_id in dataset.training_pairs:
        center_word = dataset.id_to_word[center_id]
        context_word = dataset.id_to_word[context_id]
        print(f"  {center_word:>5} -> {context_word:<5}")

    embedding_dim = 4
    epochs = 100
    negative_samples = 3
    learning_rate = 0.1

    model = NegativeSamplingModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seed=42,
    )

    print("\n--- Training ---")

    losses = model.train(
        pairs=dataset.training_pairs,
        epochs=epochs,
        negative_samples=negative_samples,
        learning_rate=learning_rate,
        eval_seed=12345,
    )

    print("\n" + "=" * 65)
    print("LEARNED INPUT EMBEDDINGS")
    print("=" * 65)

    embeddings = {}

    for word, word_id in dataset.word_to_id.items():
        vector = model.get_embedding(word_id)
        embeddings[word] = vector
        print(f"{word:>5}: {vector}")

    print("\n" + "=" * 65)
    print("COSINE SIMILARITY")
    print("=" * 65)

    words = list(dataset.word_to_id.keys())

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word_a = words[i]
            word_b = words[j]

            similarity = cosine_similarity(
                embeddings[word_a],
                embeddings[word_b],
            )

            print(
                f"{word_a:>5} <-> {word_b:<5}: "
                f"{similarity:.4f}"
            )

    print("\nEmbedding matrix shapes:")
    print(f"W_in:  {model.W_in.shape}")
    print(f"W_out: {model.W_out.shape}")


if __name__ == "__main__":
    main()