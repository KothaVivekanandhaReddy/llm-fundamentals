import numpy as np

from negative_sampling import NegativeSamplingModel


def sigmoid(x):
    """Numerically stable sigmoid for scalar inputs."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))

    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def loss_and_gradients(model, center_id, positive_id, negative_ids):
    """
    Calculate negative-sampling loss and gradients without
    modifying model parameters.

    This makes the analytical gradients testable using
    finite differences.
    """
    center = model.W_in[center_id].copy()
    positive = model.W_out[:, positive_id].copy()

    positive_score = np.dot(center, positive)

    # Stable positive-pair loss: log(1 + exp(-score))
    loss = np.logaddexp(0.0, -positive_score)

    positive_gradient = sigmoid(positive_score) - 1.0

    d_center = positive_gradient * positive
    d_positive = positive_gradient * center

    d_negative = {}

    for negative_id in negative_ids:
        negative = model.W_out[:, negative_id].copy()
        negative_score = np.dot(center, negative)

        # Stable negative-pair loss: log(1 + exp(score))
        loss += np.logaddexp(0.0, negative_score)

        negative_gradient = sigmoid(negative_score)

        d_center += negative_gradient * negative

        if negative_id not in d_negative:
            d_negative[negative_id] = np.zeros_like(negative)

        d_negative[negative_id] += negative_gradient * center

    d_W_in = np.zeros_like(model.W_in)
    d_W_out = np.zeros_like(model.W_out)

    d_W_in[center_id] = d_center
    d_W_out[:, positive_id] += d_positive

    for negative_id, gradient in d_negative.items():
        d_W_out[:, negative_id] += gradient

    return loss, d_W_in, d_W_out


def numerical_gradient(model, center_id, positive_id,
                       negative_ids, parameter, index, epsilon=1e-5):
    """Estimate one parameter's gradient using central differences."""
    original = parameter[index]

    try:
        parameter[index] = original + epsilon
        loss_plus, _, _ = loss_and_gradients(
            model, center_id, positive_id, negative_ids
        )

        parameter[index] = original - epsilon
        loss_minus, _, _ = loss_and_gradients(
            model, center_id, positive_id, negative_ids
        )
    finally:
        parameter[index] = original

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def main():
    model = NegativeSamplingModel(
        vocab_size=5,
        embedding_dim=4,
        seed=42,
    )

    # Fixed IDs ensure both gradient calculations use identical samples.
    center_id = 0
    positive_id = 1
    negative_ids = np.array([2, 3, 4])

    loss, d_W_in, d_W_out = loss_and_gradients(
        model, center_id, positive_id, negative_ids
    )

    checks = [
        ("W_in", model.W_in, d_W_in, (0, 0)),
        ("W_in", model.W_in, d_W_in, (0, 2)),
        ("W_out", model.W_out, d_W_out, (1, 1)),
        ("W_out", model.W_out, d_W_out, (2, 2)),
        ("W_out", model.W_out, d_W_out, (3, 4)),
    ]

    print("=" * 65)
    print("NEGATIVE SAMPLING: NUMERICAL GRADIENT CHECK")
    print("=" * 65)
    print(f"Fixed-sample loss: {loss:.10f}\n")

    passed = 0

    for name, parameter, analytical, index in checks:
        numerical = numerical_gradient(
            model,
            center_id,
            positive_id,
            negative_ids,
            parameter,
            index,
        )

        analytic_value = analytical[index]
        error = abs(analytic_value - numerical)

        # Relative error is useful when gradient magnitudes vary.
        relative_error = error / max(
            1e-8, abs(analytic_value) + abs(numerical)
        )

        success = (
            error < 1e-7
            or relative_error < 1e-5
        )

        if success:
            passed += 1

        print(
            f"{name}{index}: "
            f"analytic={analytic_value:+.10e}, "
            f"numerical={numerical:+.10e}, "
            f"abs_error={error:.3e}, "
            f"{'PASS' if success else 'FAIL'}"
        )

    print(f"\nResult: {passed}/{len(checks)} checks passed.")

    if passed != len(checks):
        raise SystemExit("Gradient check failed.")

    print("All sampled gradients passed.")


if __name__ == "__main__":
    main()