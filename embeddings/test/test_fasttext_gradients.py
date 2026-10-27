import numpy as np

from embeddings.fasttext_objective import (
    negative_sampling_loss,
    negative_sampling_gradients,
)


def numerical_gradient(
    loss_fn,
    parameter: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Estimate gradients using central finite differences."""

    gradient = np.zeros_like(parameter, dtype=np.float64)

    for index in np.ndindex(parameter.shape):
        original_value = parameter[index]

        parameter[index] = original_value + epsilon
        loss_plus = loss_fn()

        parameter[index] = original_value - epsilon
        loss_minus = loss_fn()

        parameter[index] = original_value

        gradient[index] = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

    return gradient


def test_fasttext_objective_gradients():
    center = np.array([0.2, -0.1, 0.4], dtype=np.float64)

    positive = np.array(
        [0.3, 0.2, -0.2],
        dtype=np.float64,
    )

    negatives = np.array(
        [
            [-0.2, 0.1, 0.3],
            [0.1, -0.3, 0.2],
        ],
        dtype=np.float64,
    )

    analytical = negative_sampling_gradients(
        center,
        positive,
        negatives,
    )

    numerical_center = numerical_gradient(
        lambda: negative_sampling_loss(
            center,
            positive,
            negatives,
        ),
        center,
    )

    numerical_positive = numerical_gradient(
        lambda: negative_sampling_loss(
            center,
            positive,
            negatives,
        ),
        positive,
    )

    numerical_negatives = numerical_gradient(
        lambda: negative_sampling_loss(
            center,
            positive,
            negatives,
        ),
        negatives,
    )

    numerical = (
        numerical_center,
        numerical_positive,
        numerical_negatives,
    )

    names = (
        "center vector",
        "positive context vector",
        "negative context vectors",
    )

    for name, analytic_grad, numeric_grad in zip(
        names,
        analytical,
        numerical,
    ):
        max_error = np.max(
            np.abs(analytic_grad - numeric_grad)
        )

        print(f"{name}: maximum absolute error = {max_error:.3e}")

        assert np.allclose(
            analytic_grad,
            numeric_grad,
            rtol=1e-5,
            atol=1e-7,
        ), f"Gradient check failed for {name}"