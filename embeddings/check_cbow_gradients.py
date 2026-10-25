import numpy as np

from cbow import CBOWDataset, CBOWModel


def compute_analytic_gradients(model, training_examples):
    """
    Calculate analytical gradients of mean cross-entropy loss
    over the complete dataset without updating parameters.
    """
    d_W_in = np.zeros_like(model.W_in)
    d_W_out = np.zeros_like(model.W_out)

    for context_ids, target_id in training_examples:
        hidden, probabilities = model.forward(context_ids)

        # Gradient of cross-entropy + softmax.
        d_logits = probabilities.copy()
        d_logits[target_id] -= 1.0

        # Gradient with respect to output weights.
        d_W_out += np.outer(hidden, d_logits)

        # Gradient with respect to the pooled context vector.
        d_hidden = model.W_out @ d_logits

        # Backpropagate through mean pooling.
        d_context = d_hidden / len(context_ids)

        # Accumulate gradients for context word embeddings.
        for word_id in context_ids:
            d_W_in[word_id] += d_context

    # Match evaluate_loss(), which returns mean loss.
    n = len(training_examples)
    d_W_in /= n
    d_W_out /= n

    return d_W_in, d_W_out


def compute_numerical_gradient(
    model,
    training_examples,
    parameter_name,
    index,
    epsilon=1e-5,
):
    """
    Estimate one gradient using central finite differences:

    df/dx ~= [f(x + epsilon) - f(x - epsilon)] / (2 * epsilon)
    """
    parameter = getattr(model, parameter_name)
    original_value = parameter[index]

    try:
        parameter[index] = original_value + epsilon
        loss_plus = model.evaluate_loss(training_examples)

        parameter[index] = original_value - epsilon
        loss_minus = model.evaluate_loss(training_examples)

    finally:
        # Restore the original parameter even if evaluation fails.
        parameter[index] = original_value

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def check_gradient(
    model,
    training_examples,
    parameter_name,
    index,
    analytic_gradient,
    epsilon=1e-5,
):
    numerical_gradient = compute_numerical_gradient(
        model,
        training_examples,
        parameter_name,
        index,
        epsilon,
    )

    analytic_value = analytic_gradient[index]

    absolute_error = abs(
        analytic_value - numerical_gradient
    )

    relative_error = absolute_error / max(
        1e-12,
        abs(analytic_value) + abs(numerical_gradient),
    )

    print(f"\nParameter: {parameter_name}{index}")
    print(f"Analytical gradient: {analytic_value:.10f}")
    print(f"Numerical gradient:  {numerical_gradient:.10f}")
    print(f"Absolute error:      {absolute_error:.3e}")
    print(f"Relative error:      {relative_error:.3e}")

    # Numerical differentiation has finite precision, so use a tolerance.
    passed = np.isclose(
        analytic_value,
        numerical_gradient,
        rtol=1e-4,
        atol=1e-7,
    )

    print("Result:", "PASS" if passed else "FAIL")
    return passed


def main():
    np.set_printoptions(precision=8, suppress=True)

    dataset = CBOWDataset(
        "the cat sat on the mat",
        window_size=1,
    )

    model = CBOWModel(
        vocab_size=len(dataset.word_to_id),
        embedding_dim=4,
        seed=42,
    )

    examples = dataset.training_examples

    print("=" * 60)
    print("CBOW FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 60)
    print(f"Vocabulary size: {len(dataset.word_to_id)}")
    print(f"Training examples: {len(examples)}")
    print(f"Initial mean loss: {model.evaluate_loss(examples):.10f}")

    # Compute analytical gradients once, using unchanged parameters.
    d_W_in, d_W_out = compute_analytic_gradients(
        model,
        examples,
    )

    # Check several entries from both parameter matrices.
    checks = [
        ("W_in", (0, 0), d_W_in),
        ("W_in", (2, 1), d_W_in),
        ("W_in", (4, 3), d_W_in),
        ("W_out", (0, 0), d_W_out),
        ("W_out", (1, 2), d_W_out),
        ("W_out", (3, 4), d_W_out),
    ]

    results = []

    for parameter_name, index, analytical_gradient in checks:
        results.append(
            check_gradient(
                model=model,
                training_examples=examples,
                parameter_name=parameter_name,
                index=index,
                analytic_gradient=analytical_gradient,
            )
        )

    print("\n" + "=" * 60)
    print(f"PASSED: {sum(results)}/{len(results)} checks")

    if all(results):
        print("SUCCESS: All sampled gradients match numerically.")
    else:
        print("FAILURE: Investigate the gradient calculation.")

    print("=" * 60)


if __name__ == "__main__":
    main()
