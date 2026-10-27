import numpy as np


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid for a scalar."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))

    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def negative_sampling_loss(
    center_vector: np.ndarray,
    positive_context_vector: np.ndarray,
    negative_context_vectors: np.ndarray,
) -> float:
    """Compute SGNS loss for one positive pair and K negative pairs."""

    positive_score = float(
        np.dot(center_vector, positive_context_vector)
    )

    positive_loss = np.logaddexp(0.0, -positive_score)

    negative_scores = negative_context_vectors @ center_vector

    negative_loss = np.logaddexp(
        0.0,
        negative_scores,
    ).sum()

    return float(positive_loss + negative_loss)


def negative_sampling_gradients(
    center_vector: np.ndarray,
    positive_context_vector: np.ndarray,
    negative_context_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return gradients with respect to:
    1. center_vector
    2. positive_context_vector
    3. negative_context_vectors
    """

    positive_score = float(
        np.dot(center_vector, positive_context_vector)
    )

    positive_coefficient = sigmoid(positive_score) - 1.0

    grad_center = (
        positive_coefficient * positive_context_vector
    )

    grad_positive_context = (
        positive_coefficient * center_vector
    )

    negative_scores = negative_context_vectors @ center_vector

    negative_coefficients = np.array(
        [sigmoid(float(score)) for score in negative_scores]
    )

    grad_center += (
        negative_coefficients @ negative_context_vectors
    )

    grad_negative_contexts = (
        negative_coefficients[:, None] * center_vector[None, :]
    )

    return (
        grad_center,
        grad_positive_context,
        grad_negative_contexts,
    )