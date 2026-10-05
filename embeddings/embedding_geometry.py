#To understand the geometry behind embeddings before training them.

# Vector
#  ↓
#Dot Product
#  ↓
#Magnitude / L2 Norm
#  ↓
#Euclidean Distance
# ↓
#Cosine Similarity
#  ↓
#L2 Normalization

import numpy as np


def dot_product(a, b):
    return np.dot(a, b)

#linalg : submodules found in scientific computing libraries like NumPy  and SciPy. 
#provide standard matrix operations like dot products, determinants, inverses, eigenvalues, and solving linear equations (Ax = b).
# norm : computes the magnitude (length) of a vector or the norm of a matrix
def vector_norm(a):
    return np.linalg.norm(a)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def normalize(a):
    norm = np.linalg.norm(a)

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")

    return a / norm


if __name__ == "__main__":

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    c = np.array([-1.0, -2.0, -3.0])
    d = np.array([3.0, 0.0, 0.0])

    print("=" * 60)
    print("EMBEDDING GEOMETRY")
    print("=" * 60)

    print("\nVectors:")
    print("A:", a)
    print("B:", b)
    print("C:", c)
    print("D:", d)

    print("\n--- Dot Product ---")

    print("A · B:", dot_product(a, b))
    print("A · C:", dot_product(a, c))
    print("A · D:", dot_product(a, d))

    print("\n--- Vector Norm ---")

    print("||A||:", vector_norm(a))
    print("||B||:", vector_norm(b))
    print("||C||:", vector_norm(c))

    print("\n--- Euclidean Distance ---")

    print("A ↔ B:", euclidean_distance(a, b))
    print("A ↔ C:", euclidean_distance(a, c))
    print("A ↔ D:", euclidean_distance(a, d))

    print("\n--- Cosine Similarity ---")

    print("A ↔ B:", cosine_similarity(a, b))
    print("A ↔ C:", cosine_similarity(a, c))
    print("A ↔ D:", cosine_similarity(a, d))

    print("\n--- Normalization ---")

    normalized_a = normalize(a)

    print("Original A:", a)
    print("Normalized A:", normalized_a)
    print("Norm:", vector_norm(normalized_a))

    print("\n--- Scale Invariance ---")

    scaled_a = a * 10

    print("A:", a)
    print("10A:", scaled_a)

    print("Euclidean distance:", euclidean_distance(a, scaled_a))
    print("Cosine similarity:", cosine_similarity(a, scaled_a))

    #Normalization : is scale-invariant, meaning that scaling a vector does not change its direction. 
    #This property is crucial in many machine learning applications, especially in embeddings, where we care about the direction 
    #of vectors rather than their magnitude.
    normalized_a = normalize(a)
    normalized_b = normalize(b)

    print("\nNormalized vectors:")
    print(normalized_a)
    print(normalized_b)

    print("\nNorms:")
    print(vector_norm(normalized_a))
    print(vector_norm(normalized_b))

    print("\nDot product after normalization:")
    print(dot_product(normalized_a, normalized_b))

    print("\nCosine similarity:")
    print(cosine_similarity(a, b))