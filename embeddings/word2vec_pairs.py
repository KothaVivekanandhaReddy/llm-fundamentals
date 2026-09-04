from collections import defaultdict


def create_context_pairs(tokens, window_size=2):

    pairs = []

    for i, target in enumerate(tokens):

        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)

        for j in range(start, end):

            if i == j:
                continue

            context = tokens[j]

            pairs.append((target, context))

    return pairs


if __name__ == "__main__":

    text = "the cat sat on the mat"

    tokens = text.split()

    pairs = create_context_pairs(
        tokens,
        window_size=2
    )

    print("=" * 60)
    print("CONTEXT / TARGET PAIRS")
    print("=" * 60)

    print("\nTokens:")
    print(tokens)

    print("\nTraining pairs:")

    for target, context in pairs:
        print(
            f"Target: {target:>4}  "
            f"Context: {context}"
        )

    print("\nTotal pairs:", len(pairs))