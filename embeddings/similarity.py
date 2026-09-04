import numpy as np


def dot_product(a, b):
    return np.dot(a, b)


def vector_norm(a):
    return np.linalg.norm(a)


def cosine_similarity(a, b):
    dot = dot_product(a, b)

    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return dot / (norm_a * norm_b)


if __name__ == "__main__":

    print("=" * 60)
    print("VECTOR SIMILARITY")
    print("=" * 60)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([-1.0, -2.0, -3.0])
    d = np.array([3.0, 0.0, 0.0])

    print("\nVectors:")
    print("A:", a)
    print("B:", b)
    print("C:", c)
    print("D:", d)

    print("\nDot products:")
    print("A · B =", dot_product(a, b))
    print("A · C =", dot_product(a, c))
    print("A · D =", dot_product(a, d))

    print("\nNorms:")
    print("||A|| =", vector_norm(a))
    print("||B|| =", vector_norm(b))

    print("\nCosine similarities:")
    print("cos(A, B) =", cosine_similarity(a, b))
    print("cos(A, C) =", cosine_similarity(a, c))
    print("cos(A, D) =", cosine_similarity(a, d))