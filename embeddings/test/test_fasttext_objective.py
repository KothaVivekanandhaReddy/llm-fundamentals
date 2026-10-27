import numpy as np

from embeddings.fasttext_objective import (
    negative_sampling_loss,
    negative_sampling_gradients,
)


def test_loss_is_finite():
    center = np.array([0.2, -0.1, 0.4])
    positive = np.array([0.3, 0.2, -0.2])
    negatives = np.array([
        [-0.2, 0.1, 0.3],
        [0.1, -0.3, 0.2],
    ])

    loss = negative_sampling_loss(center, positive, negatives)

    assert np.isfinite(loss)
    assert loss > 0


def test_gradient_shapes():
    center = np.array([0.2, -0.1, 0.4])
    positive = np.array([0.3, 0.2, -0.2])
    negatives = np.array([
        [-0.2, 0.1, 0.3],
        [0.1, -0.3, 0.2],
    ])

    grad_center, grad_positive, grad_negatives = (
        negative_sampling_gradients(center, positive, negatives)
    )

    assert grad_center.shape == center.shape
    assert grad_positive.shape == positive.shape
    assert grad_negatives.shape == negatives.shape


def test_gradient_values_are_finite():
    center = np.array([0.2, -0.1, 0.4])
    positive = np.array([0.3, 0.2, -0.2])
    negatives = np.array([
        [-0.2, 0.1, 0.3],
        [0.1, -0.3, 0.2],
    ])

    gradients = negative_sampling_gradients(
        center,
        positive,
        negatives,
    )

    for gradient in gradients:
        assert np.isfinite(gradient).all()