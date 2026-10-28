"""Reusable evaluation utilities for static word embeddings."""

import numpy as np


def nearest_neighbors(
    query_word,
    vocabulary,
    embeddings,
    top_k=5,
):
    """
    Return the top-k nearest words using cosine similarity.

    Parameters
    ----------
    query_word : str
        Word whose nearest neighbors will be retrieved.
    vocabulary : sequence[str]
        Vocabulary ordered to match the rows of embeddings.
    embeddings : numpy.ndarray
        Embedding matrix with shape (vocabulary_size, embedding_dim).
    top_k : int
        Maximum number of neighbors to return.

    Returns
    -------
    list[tuple[str, float]]
        (word, cosine_similarity) pairs sorted by descending similarity.

    Notes
    -----
    The query word is excluded from the results.
    Zero-norm candidate vectors are excluded.
    Similarity ties are resolved alphabetically for reproducibility.
    """

    if not isinstance(query_word, str) or not query_word:
        raise ValueError("query_word must be a non-empty string.")

    if isinstance(top_k, bool) or not isinstance(
        top_k, (int, np.integer)
    ):
        raise TypeError("top_k must be an integer.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    vocabulary = list(vocabulary)
    embeddings = np.asarray(embeddings, dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")

    if len(vocabulary) != embeddings.shape[0]:
        raise ValueError(
            "Vocabulary size must match the number of embedding rows."
        )

    if embeddings.shape[1] == 0:
        raise ValueError("Embedding dimension must be greater than zero.")

    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Vocabulary must not contain duplicate words.")

    if not all(isinstance(word, str) for word in vocabulary):
        raise TypeError("Every vocabulary entry must be a string.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must contain only finite values.")

    word_to_id = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    if query_word not in word_to_id:
        raise KeyError(f"Unknown query word: {query_word!r}")

    query_id = word_to_id[query_word]
    query_vector = embeddings[query_id]

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        raise ValueError(
            f"Query word {query_word!r} has a zero-norm embedding."
        )

    # Vectorized cosine similarity against the entire matrix.
    norms = np.linalg.norm(embeddings, axis=1)

    similarities = (
        embeddings @ query_vector
    ) / (norms * query_norm + 1e-12)

    # Exclude the query word itself and zero-norm candidates.
    valid = norms > 0
    valid[query_id] = False

    candidate_ids = np.flatnonzero(valid)

    if candidate_ids.size == 0:
        return []

    # Sort by descending similarity, then alphabetically for ties.
    ranked_ids = sorted(
        candidate_ids.tolist(),
        key=lambda idx: (
            -float(similarities[idx]),
            vocabulary[idx],
        ),
    )

    return [
        (vocabulary[idx], float(similarities[idx]))
        for idx in ranked_ids[:top_k]
    ]

def pair_similarity(word_a, word_b, vocabulary, embeddings):
    """
    Calculate cosine similarity between two words.

    Parameters
    ----------
    word_a, word_b : str
        Words whose embeddings will be compared.
    vocabulary : sequence[str]
        Vocabulary ordered to match the rows of embeddings.
    embeddings : numpy.ndarray
        Matrix with shape (vocabulary_size, embedding_dim).

    Returns
    -------
    float
        Cosine similarity in the range [-1, 1].

    Raises
    ------
    KeyError
        If either word is absent from the vocabulary.
    ValueError
        If the embedding matrix is invalid or either vector is zero.
    """

    if not isinstance(word_a, str) or not word_a:
        raise ValueError("word_a must be a non-empty string.")

    if not isinstance(word_b, str) or not word_b:
        raise ValueError("word_b must be a non-empty string.")

    vocabulary = list(vocabulary)
    embeddings = np.asarray(embeddings, dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")

    if len(vocabulary) != embeddings.shape[0]:
        raise ValueError(
            "Vocabulary size must match the number of embedding rows."
        )

    if embeddings.shape[1] == 0:
        raise ValueError("Embedding dimension must be greater than zero.")

    if not all(isinstance(word, str) for word in vocabulary):
        raise TypeError("Every vocabulary entry must be a string.")

    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Vocabulary must not contain duplicate words.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must contain only finite values.")

    word_to_id = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    for word in (word_a, word_b):
        if word not in word_to_id:
            raise KeyError(f"Unknown word: {word!r}")

    vector_a = embeddings[word_to_id[word_a]]
    vector_b = embeddings[word_to_id[word_b]]

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0:
        raise ValueError(f"Word {word_a!r} has a zero-norm embedding.")

    if norm_b == 0:
        raise ValueError(f"Word {word_b!r} has a zero-norm embedding.")

    score = np.dot(vector_a, vector_b) / (norm_a * norm_b)

    # Protect against tiny floating-point excursions beyond [-1, 1].
    return float(np.clip(score, -1.0, 1.0))


def evaluate_similarity_pairs(word_pairs, vocabulary, embeddings):
    """
    Evaluate cosine similarity for a collection of word pairs.

    Returns
    -------
    list[tuple[str, str, float]]
        Each result contains (word_a, word_b, cosine_similarity).

    Raises
    ------
    ValueError or KeyError
        If a pair contains invalid or unknown words, or its embedding
        cannot be evaluated.
    """

    results = []

    for word_a, word_b in word_pairs:
        score = pair_similarity(
            word_a,
            word_b,
            vocabulary,
            embeddings,
        )

        results.append((word_a, word_b, score))

    return results

from collections.abc import Callable, Sequence

import numpy as np


VectorProvider = Callable[[str], np.ndarray]


def build_embedding_matrix(
    vocabulary: Sequence[str],
    get_vector: VectorProvider,
) -> np.ndarray:
    """
    Build an embedding matrix using a model-independent vector provider.

    Rows follow the exact order of `vocabulary`.
    """
    if not vocabulary:
        raise ValueError("Vocabulary must not be empty.")

    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Vocabulary must not contain duplicates.")

    if any(not isinstance(word, str) or not word for word in vocabulary):
        raise ValueError("Every vocabulary entry must be a non-empty string.")

    vectors = []

    for word in vocabulary:
        vector = np.asarray(get_vector(word), dtype=np.float64)

        if vector.ndim != 1:
            raise ValueError(
                f"Vector for {word!r} must be one-dimensional."
            )

        if vector.size == 0:
            raise ValueError(f"Vector for {word!r} must not be empty.")

        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Vector for {word!r} contains non-finite values.")

        vectors.append(vector)

    dimensions = {vector.shape[0] for vector in vectors}

    if len(dimensions) != 1:
        raise ValueError("All embedding vectors must have the same dimension.")

    return np.stack(vectors, axis=0)


def evaluate_embedding_space(
    vocabulary: Sequence[str],
    get_vector: VectorProvider,
    similarity_pairs: Sequence[tuple[str, str]],
    query_words: Sequence[str],
    top_k: int = 5,
) -> dict:
    """
    Run the same word-pair and nearest-neighbor evaluations
    against any embedding model.

    The provider must return a vector for each requested word.
    """
    vocabulary = list(vocabulary)

    embeddings = build_embedding_matrix(vocabulary, get_vector)

    pair_results = evaluate_similarity_pairs(
        similarity_pairs,
        vocabulary,
        embeddings,
    )

    neighbor_results = {
        word: nearest_neighbors(
            word,
            vocabulary,
            embeddings,
            top_k=top_k,
        )
        for word in query_words
    }

    return {
        "vocabulary": vocabulary,
        "embeddings": embeddings,
        "similarity_pairs": pair_results,
        "nearest_neighbors": neighbor_results,
        "embedding_dim": embeddings.shape[1],
        "vocabulary_size": len(vocabulary),
    }