#implementation from scratch with NumPy. The important pieces are:

#1.Input embedding matrix W_in
#2.Output embedding matrix W_out

#3.Forward pass:
#context token → W_in → vector

#4.Score every vocabulary token:
#score = input_vector · W_out[token]

#5.Softmax
#6.Cross-entropy loss
#7.Backpropagation
#8.Update W_in and W_out
#9.Print loss during training

import numpy as np


# ============================================================
# DATA
# ============================================================

tokens = [
    "the",
    "cat",
    "sat",
    "on",
    "mat"
]

vocab = {token: i for i, token in enumerate(tokens)}
vocab_size = len(vocab)

pairs = [
    ("the", "cat"),
    ("the", "sat"),
    ("cat", "the"),
    ("cat", "sat"),
    ("cat", "on"),
    ("sat", "the"),
    ("sat", "cat"),
    ("sat", "on"),
    ("sat", "the"),
    ("on", "cat"),
    ("on", "sat"),
    ("on", "the"),
    ("on", "mat"),
    ("the", "sat"),
    ("the", "on"),
    ("the", "mat"),
    ("mat", "on"),
    ("mat", "the"),
]


# ============================================================
# CONFIGURATION
# ============================================================

EMBED_DIM = 3
LEARNING_RATE = 0.05
STEPS = 300

np.random.seed(42)


# ============================================================
# PARAMETERS
# ============================================================

W_in = np.random.randn(
    vocab_size,
    EMBED_DIM
) * 0.1

W_out = np.random.randn(
    vocab_size,
    EMBED_DIM
) * 0.1


# ============================================================
# SOFTMAX
# ============================================================

def softmax(x):

    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


# ============================================================
# SIMILARITY REPORT
# ============================================================

def print_similarities():

    print("\nCosine similarities:")

    for word_a, word_b in [
        ("cat", "sat"),
        ("cat", "mat"),
        ("cat", "the"),
        ("sat", "mat"),
        ("the", "on")
    ]:

        a = W_in[vocab[word_a]]
        b = W_in[vocab[word_b]]

        similarity = cosine_similarity(a, b)

        print(
            f"{word_a:>3} ↔ {word_b:<3} : "
            f"{similarity:.4f}"
        )


# ============================================================
# INITIAL SIMILARITY
# ============================================================

print("=" * 60)
print("SKIP-GRAM FROM SCRATCH")
print("=" * 60)

print("\nVocabulary:")
print(vocab)

print("\nInitial similarity:")
print_similarities()


# ============================================================
# TRAINING
# ============================================================

print("\nTraining...")

for step in range(STEPS):

    total_loss = 0.0

    for target_word, context_word in pairs:

        target_id = vocab[target_word]
        context_id = vocab[context_word]

        # ----------------------------------------------------
        # 1. INPUT EMBEDDING
        # ----------------------------------------------------

        input_vector = W_in[target_id]

        # ----------------------------------------------------
        # 2. SCORE ALL VOCABULARY TOKENS
        # ----------------------------------------------------

        scores = W_out @ input_vector

        # ----------------------------------------------------
        # 3. SOFTMAX
        # ----------------------------------------------------

        probabilities = softmax(scores)

        # ----------------------------------------------------
        # 4. CROSS-ENTROPY LOSS
        # ----------------------------------------------------

        loss = -np.log(
            probabilities[context_id] + 1e-10
        )

        total_loss += loss

        # ----------------------------------------------------
        # 5. GRADIENT OF SOFTMAX CROSS-ENTROPY
        # ----------------------------------------------------

        grad_scores = probabilities.copy()

        grad_scores[context_id] -= 1.0

        # ----------------------------------------------------
        # 6. GRADIENT W_OUT
        # ----------------------------------------------------

        grad_W_out = np.outer(
            grad_scores,
            input_vector
        )

        # ----------------------------------------------------
        # 7. GRADIENT W_IN
        # ----------------------------------------------------

        grad_input = (
            grad_scores @ W_out
        )

        # ----------------------------------------------------
        # 8. UPDATE
        # ----------------------------------------------------

        W_out -= (
            LEARNING_RATE
            * grad_W_out
        )

        W_in[target_id] -= (
            LEARNING_RATE
            * grad_input
        )

    if step % 50 == 0:

        average_loss = (
            total_loss / len(pairs)
        )

        print(
            f"Step {step:3d} | "
            f"Loss: {average_loss:.6f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("\nLearned embeddings:")

for word, index in vocab.items():

    print(
        f"{word:>3}: "
        f"{W_in[index]}"
    )

print("\nFinal similarity:")
print_similarities()

print("\nTraining complete.")
