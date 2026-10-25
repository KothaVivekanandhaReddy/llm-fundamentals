# ==============================================================================
# GLOVE (GLOBAL VECTORS FOR WORD REPRESENTATION) - OVERVIEW & SUMMARY
# ==============================================================================
# 1. THE CORE IDEA: BRIDGING TWO WORLDS : 
# ------------------------------------------------------------------------------
# Before GloVe, word embeddings were built using two distinct methodologies:
# * Global Matrix Factorization (LSA):
#   - Scans the entire corpus to count global word co-occurrences.
#   - Excellent at capturing broad thematic/global statistical information.
#   - Weak at capturing local semantic subtleties and word analogies.
# * Local Window Methods (Word2Vec - CBOW / Skip-gram):
#   - Uses a streaming/sliding window to predict localized word context.
#   - Excellent at learning structural analogies (e.g., King - Man + Woman = Queen).
#   - Computationally inefficient as it repeatedly scans identical word patterns.
#
# GloVe's Solution:
# Combines both worlds. It trains directly on a precomputed global word-word 
# co-occurrence matrix, achieving high computational efficiency while 
# retaining the sharp analogical capabilities of local window methods.


# 2. HOW GLOVE WORKS: RATIOS OF CO-OCCURRENCE PROBABILITIES
# ------------------------------------------------------------------------------
# Rather than predicting raw probabilities, GloVe trains on the *ratio* of 
# how words co-occur. This highlights relative semantic relationships:
#
# Example with three words: [ice], [steam], and a probe word [solid]
# * P(solid | ice)   -> High (Solid frequently appears near ice)
# * P(solid | steam) -> Low  (Solid rarely appears near steam)
#
# Ratio = P(solid | ice) / P(solid | steam) -> Very Large (>> 1)
#
# This large ratio clearly indicates that "solid" is structurally and 
# semantically related to "ice", but not to "steam". 
# The GloVe objective forces the dot product of two word vectors to equal 
# the logarithm of their global co-occurrence probability.


# 3. GLOVE VS. WORD2VEC (DIRECT COMPARISON)
# ------------------------------------------------------------------------------
# +------------------+-----------------------------+---------------------------+
# | Feature          | Word2Vec (CBOW / Skip-gram) | GloVe                     |
# +------------------+-----------------------------+---------------------------+
# | Training Type    | Predictive (Guessing words) | Count-based (Matrix fact.)|
# | Data Usage       | Online / Local text stream  | Offline / Global matrix   |
# | Training Speed   | Slower on giant datasets    | Faster on massive text    |
# | Word Weighting   | Equal weight per window     | Weighted least squares    |
# +------------------+-----------------------------+---------------------------+
# 
# * Speed Context: Word2Vec loops over text sentence-by-sentence. GloVe builds 
#   the matrix once, then only processes unique word-word pairs.
# * Weighting Context: GloVe uses a custom weighting function so ultra-frequent 
#   stop words (e.g., "the", "is") do not dominate and corrupt the loss function.


# 4. QUICK CONCISE SUMMARY (3 POINTS)
# ------------------------------------------------------------------------------
# * GloVe leverages global statistics by precomputing how often every word 
#   appears next to every other word across the entire corpus.
# * It isolates meaning through ratios of co-occurrence probabilities, revealing 
#   how words relate to one target concept relative to another.
# * It offers identical utility to Word2Vec downstream, but is often much 
#   cleaner and more efficient to train on massive, static data like Wikipedia.

import numpy as np


class GloVeDataset:
    def __init__(self, text, window_size=1):
        self.text = text.lower()
        self.window_size = window_size

        self.tokens = self.tokenize(self.text)
        self.word_to_id = self.build_vocabulary()
        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }

        self.cooccurrence_matrix = self.build_cooccurrence_matrix()

    def tokenize(self, text):
        return text.split()

    def build_vocabulary(self):
        vocabulary = sorted(set(self.tokens))
        return {
            word: idx
            for idx, word in enumerate(vocabulary)
        }

    def build_cooccurrence_matrix(self):
        vocab_size = len(self.word_to_id)

        matrix = np.zeros(
            (vocab_size, vocab_size),
            dtype=np.float64,
        )

        token_ids = [
            self.word_to_id[token]
            for token in self.tokens
        ]

        for center_index, center_id in enumerate(token_ids):
            start = max(0, center_index - self.window_size)
            end = min(
                len(token_ids),
                center_index + self.window_size + 1,
            )

            for context_index in range(start, end):
                if context_index == center_index:
                    continue

                context_id = token_ids[context_index]
                distance = abs(context_index - center_index)

                # Distance weighting: closer words contribute more.
                matrix[center_id, context_id] += 1.0 / distance

        return matrix
class GloVeObjective:
    def __init__(self, cooccurrence_matrix, x_max=100.0, alpha=0.75):
        self.X = np.asarray(cooccurrence_matrix, dtype=np.float64)

        if self.X.ndim != 2 or self.X.shape[0] != self.X.shape[1]:
            raise ValueError("Co-occurrence matrix must be square.")

        if np.any(self.X < 0):
            raise ValueError("Co-occurrence counts cannot be negative.")

        if x_max <= 0:
            raise ValueError("x_max must be positive.")

        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        self.x_max = float(x_max)
        self.alpha = float(alpha)

        # Train only on observed co-occurrences.
        self.rows, self.cols = np.nonzero(self.X)
        self.counts = self.X[self.rows, self.cols]

        # Precompute the targets and weights.
        self.log_counts = np.log(self.counts)
        self.weights = self.weighting_function(self.counts)

    def weighting_function(self, counts):
        counts = np.asarray(counts, dtype=np.float64)

        return np.where(
            counts < self.x_max,
            (counts / self.x_max) ** self.alpha,
            1.0,
        )

    def loss(self, word_vectors, context_vectors, word_bias, context_bias):
        predictions = (
            np.sum(
                word_vectors[self.rows] * context_vectors[self.cols],
                axis=1,
            )
            + word_bias[self.rows]
            + context_bias[self.cols]
        )

        errors = predictions - self.log_counts

        return float(np.sum(self.weights * errors**2))
    def train(
        self,
        embedding_dim=4,
        epochs=100,
        learning_rate=0.05,
        seed=42,
        log_every=20,
    ):
        vocab_size = self.X.shape[0]
        rng = np.random.default_rng(seed)

        # Word and context embeddings are separate parameter matrices.
        word_vectors = rng.normal(
            0.0, 0.1, size=(vocab_size, embedding_dim)
        )
        context_vectors = rng.normal(
            0.0, 0.1, size=(vocab_size, embedding_dim)
        )

        word_bias = np.zeros(vocab_size, dtype=np.float64)
        context_bias = np.zeros(vocab_size, dtype=np.float64)

        initial_loss = self.loss(
            word_vectors,
            context_vectors,
            word_bias,
            context_bias,
        )

        history = []

        for epoch in range(1, epochs + 1):
            # Visit observed co-occurrence pairs in a shuffled order.
            order = rng.permutation(len(self.counts))

            for position in order:
                i = self.rows[position]
                j = self.cols[position]

                count = self.counts[position]
                weight = self.weights[position]
                target = self.log_counts[position]

                # Prediction and residual for this word-context pair.
                prediction = (
                    np.dot(word_vectors[i], context_vectors[j])
                    + word_bias[i]
                    + context_bias[j]
                )
                error = prediction - target

                # Gradients must use the parameter values from before
                # either embedding vector is updated.
                grad_word = (
                    2.0 * weight * error * context_vectors[j].copy()
                )
                grad_context = (
                    2.0 * weight * error * word_vectors[i].copy()
                )
                grad_bias = 2.0 * weight * error

                # SGD updates.
                word_vectors[i] -= learning_rate * grad_word
                context_vectors[j] -= learning_rate * grad_context

                word_bias[i] -= learning_rate * grad_bias
                context_bias[j] -= learning_rate * grad_bias

            epoch_loss = self.loss(
                word_vectors,
                context_vectors,
                word_bias,
                context_bias,
            )
            history.append(epoch_loss)

            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Loss: {epoch_loss:.8f}"
                )

        final_loss = history[-1]

        print(f"\nInitial loss: {initial_loss:.8f}")
        print(f"Final loss:   {final_loss:.8f}")

        if initial_loss > 0:
            reduction = (
                (initial_loss - final_loss) / initial_loss
            ) * 100
            print(f"Loss reduction: {reduction:.2f}%")

        return {
            "word_vectors": word_vectors,
            "context_vectors": context_vectors,
            "word_bias": word_bias,
            "context_bias": context_bias,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "history": history,
        }
    @staticmethod
    def cosine_similarity(vector_a, vector_b):
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        denominator = norm_a * norm_b

        if denominator < 1e-12:
            return 0.0

        return float(np.dot(vector_a, vector_b) / denominator)

    @staticmethod
    def nearest_neighbors(word, word_to_id, word_vectors, top_k=3):
        if word not in word_to_id:
            raise ValueError(f"Unknown word: {word}")

        query_id = word_to_id[word]
        query_vector = word_vectors[query_id]

        scores = []

        for candidate, candidate_id in word_to_id.items():
            if candidate == word:
                continue

            score = GloVeObjective.cosine_similarity(
                query_vector,
                word_vectors[candidate_id],
            )

            scores.append((candidate, score))

        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]


def check_glove_gradients(objective, seed=123, epsilon=1e-6):
    rng = np.random.default_rng(seed)

    vocab_size = objective.X.shape[0]
    embedding_dim = 3

    word_vectors = rng.normal(
        0.0, 0.1, size=(vocab_size, embedding_dim)
    )
    context_vectors = rng.normal(
        0.0, 0.1, size=(vocab_size, embedding_dim)
    )
    word_bias = rng.normal(0.0, 0.1, size=vocab_size)
    context_bias = rng.normal(0.0, 0.1, size=vocab_size)

    # Choose one observed word-context pair.
    i = objective.rows[0]
    j = objective.cols[0]

    # Compute the analytical gradient of the FULL objective
    # with respect to word_vectors[i, 0].
    analytical_word_grad = 0.0

    for position in range(len(objective.counts)):
        row = objective.rows[position]
        col = objective.cols[position]

        if row != i:
            continue

        weight = objective.weights[position]
        target = objective.log_counts[position]

        prediction = (
            np.dot(word_vectors[row], context_vectors[col])
            + word_bias[row]
            + context_bias[col]
        )

        error = prediction - target

        analytical_word_grad += (
            2.0
            * weight
            * error
            * context_vectors[col, 0]
        )

    # Compute the analytical gradient of the FULL objective
    # with respect to context_vectors[j, 0].
    analytical_context_grad = 0.0

    for position in range(len(objective.counts)):
        row = objective.rows[position]
        col = objective.cols[position]

        if col != j:
            continue

        weight = objective.weights[position]
        target = objective.log_counts[position]

        prediction = (
            np.dot(word_vectors[row], context_vectors[col])
            + word_bias[row]
            + context_bias[col]
        )

        error = prediction - target

        analytical_context_grad += (
            2.0
            * weight
            * error
            * word_vectors[row, 0]
        )

    # Compute the analytical gradient of the FULL objective
    # with respect to word_bias[i].
    analytical_bias_grad = 0.0

    for position in range(len(objective.counts)):
        row = objective.rows[position]
        col = objective.cols[position]

        if row != i:
            continue

        weight = objective.weights[position]
        target = objective.log_counts[position]

        prediction = (
            np.dot(word_vectors[row], context_vectors[col])
            + word_bias[row]
            + context_bias[col]
        )

        error = prediction - target

        analytical_bias_grad += 2.0 * weight * error

    def total_loss():
        return objective.loss(
            word_vectors,
            context_vectors,
            word_bias,
            context_bias,
        )

    # Check one word-vector coordinate.
    original = word_vectors[i, 0]

    word_vectors[i, 0] = original + epsilon
    loss_plus = total_loss()

    word_vectors[i, 0] = original - epsilon
    loss_minus = total_loss()

    word_vectors[i, 0] = original

    numerical_word_grad = (
        loss_plus - loss_minus
    ) / (2.0 * epsilon)

    # Check one context-vector coordinate.
    original = context_vectors[j, 0]

    context_vectors[j, 0] = original + epsilon
    loss_plus = total_loss()

    context_vectors[j, 0] = original - epsilon
    loss_minus = total_loss()

    context_vectors[j, 0] = original

    numerical_context_grad = (
        loss_plus - loss_minus
    ) / (2.0 * epsilon)

    # Check one word-bias coordinate.
    original = word_bias[i]

    word_bias[i] = original + epsilon
    loss_plus = total_loss()

    word_bias[i] = original - epsilon
    loss_minus = total_loss()

    word_bias[i] = original

    numerical_bias_grad = (
        loss_plus - loss_minus
    ) / (2.0 * epsilon)

    checks = [
        (
            "word vector",
            analytical_word_grad,
            numerical_word_grad,
        ),
        (
            "context vector",
            analytical_context_grad,
            numerical_context_grad,
        ),
        (
            "word bias",
            analytical_bias_grad,
            numerical_bias_grad,
        ),
    ]

    print("\n" + "=" * 60)
    print("GLOVE GRADIENT CHECK")
    print("=" * 60)

    for name, analytical, numerical in checks:
        absolute_error = abs(analytical - numerical)

        print(f"{name}:")
        print(f"  Analytical: {analytical:.10f}")
        print(f"  Numerical:  {numerical:.10f}")
        print(f"  Abs error:  {absolute_error:.3e}")

        if not np.isclose(
            analytical,
            numerical,
            rtol=1e-4,
            atol=1e-7,
        ):
            raise AssertionError(
                f"Gradient check failed for {name}"
            )

    print("\nAll checked gradients passed.")
if __name__ == "__main__":
    text = "the cat sat on the mat"

    dataset = GloVeDataset(
        text=text,
        window_size=1,
    )
    #text = """
        #the cat sat on the mat
       # the cat ate fish
       # the cat sat near the window
       # the dog sat on the mat
       # the dog ate food
      #  the dog sat near the door
      #  the cat sat near the door
      #  the dog sat on the floor
     #   the cat ate food
    #    the dog ate fish
    #"""

    #dataset = GloVeDataset(text, window_size=2)

    print("=" * 60)
    print("GLOVE CO-OCCURRENCE MATRIX VALIDATION")
    print("=" * 60)

    print("\nVocabulary:")
    print(dataset.word_to_id)

    print("\nVocabulary size:")
    print(len(dataset.word_to_id))

    print("\nCo-occurrence matrix:")
    print(dataset.cooccurrence_matrix)

    print("\nMatrix shape:")
    print(dataset.cooccurrence_matrix.shape)

    print("\nMatrix sum:")
    print(dataset.cooccurrence_matrix.sum())

    print("\nNon-zero entries:")
    print(np.count_nonzero(dataset.cooccurrence_matrix))

    print("\nCo-occurrence counts by word:")
    for word, word_id in dataset.word_to_id.items():
        row = dataset.cooccurrence_matrix[word_id]

        counts = {
            dataset.id_to_word[index]: float(value)
            for index, value in enumerate(row)
            if value > 0
        }

        print(f"{word:>5}: {counts}")

    print(
        "\nMatrix symmetric:",
        np.allclose(
            dataset.cooccurrence_matrix,
            dataset.cooccurrence_matrix.T,
        ),
    )
    print("\n" + "=" * 60)
    print("GLOVE OBJECTIVE VALIDATION")
    print("=" * 60)

    objective = GloVeObjective(
        dataset.cooccurrence_matrix,
        x_max=100.0,
        alpha=0.75,
    )

    print("Observed co-occurrence entries:", len(objective.counts))
    print("Log-count targets:", objective.log_counts)
    print("Weights:", objective.weights)

    vocab_size = len(dataset.word_to_id)
    embedding_dim = 4

    rng = np.random.default_rng(42)

    word_vectors = rng.normal(
        0.0, 0.1, size=(vocab_size, embedding_dim)
    )
    context_vectors = rng.normal(
        0.0, 0.1, size=(vocab_size, embedding_dim)
    )

    word_bias = np.zeros(vocab_size)
    context_bias = np.zeros(vocab_size)

    initial_loss = objective.loss(
        word_vectors,
        context_vectors,
        word_bias,
        context_bias,
    )

    print("Initial objective loss:", initial_loss)
    print("Word-vector shape:", word_vectors.shape)
    print("Context-vector shape:", context_vectors.shape)
    print("\n" + "=" * 60)
    print("GLOVE TRAINING EXPERIMENT")
    print("=" * 60)
    check_glove_gradients(objective)
    result = objective.train(
        embedding_dim=4,
        epochs=100,
        learning_rate=0.05,
        seed=42,
        log_every=20,
    )

    print("\nLearned word vectors:")
    for word, word_id in dataset.word_to_id.items():
        print(f"{word:>5}: {result['word_vectors'][word_id]}")
    print("\n" + "=" * 60)
    print("GLOVE EMBEDDING EVALUATION")
    print("=" * 60)

    word_vectors = result["word_vectors"]

    test_pairs = [
        ("cat", "mat"),
        ("cat", "sat"),
        ("cat", "the"),
        ("on", "the"),
    ]

    print("\nCosine similarity:")
    for word_a, word_b in test_pairs:
        id_a = dataset.word_to_id[word_a]
        id_b = dataset.word_to_id[word_b]

        similarity = GloVeObjective.cosine_similarity(
            word_vectors[id_a],
            word_vectors[id_b],
        )

        print(f"{word_a:>5} <-> {word_b:<5}: {similarity:.4f}")

    print("\nNearest neighbors:")
    for word in ["cat", "mat", "the"]:
        neighbors = GloVeObjective.nearest_neighbors(
            word,
            dataset.word_to_id,
            word_vectors,
            top_k=3,
        )

        print(f"\n{word}:")
        for neighbor, score in neighbors:
            print(f"  {neighbor:<5} {score:.4f}")