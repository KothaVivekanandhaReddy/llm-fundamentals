import numpy as np
import pytest

from embeddings.evaluation import nearest_neighbors


@pytest.fixture
def sample_embeddings():
    vocabulary = [
        "cat",
        "kitten",
        "dog",
        "car",
        "vehicle",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],    # cat
            [0.9, 0.1],    # kitten: close to cat
            [0.8, 0.2],    # dog: also close to cat geometrically
            [0.0, 1.0],    # car
            [0.1, 0.9],    # vehicle: close to car
        ]
    )

    return vocabulary, embeddings


def test_nearest_neighbor_ranking(sample_embeddings):
    vocabulary, embeddings = sample_embeddings

    results = nearest_neighbors(
        "cat",
        vocabulary,
        embeddings,
        top_k=2,
    )

    assert [word for word, _ in results] == [
        "kitten",
        "dog",
    ]

    assert results[0][1] > results[1][1]


def test_query_word_is_excluded(sample_embeddings):
    vocabulary, embeddings = sample_embeddings

    results = nearest_neighbors(
        "cat",
        vocabulary,
        embeddings,
        top_k=5,
    )

    assert "cat" not in [word for word, _ in results]


def test_similarity_scores_are_bounded(sample_embeddings):
    vocabulary, embeddings = sample_embeddings

    results = nearest_neighbors(
        "cat",
        vocabulary,
        embeddings,
        top_k=4,
    )

    for _, score in results:
        assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9


def test_unknown_query_raises_error(sample_embeddings):
    vocabulary, embeddings = sample_embeddings

    with pytest.raises(KeyError):
        nearest_neighbors(
            "unknown",
            vocabulary,
            embeddings,
        )


def test_zero_norm_query_raises_error():
    vocabulary = ["cat", "dog"]
    embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="zero-norm"):
        nearest_neighbors("cat", vocabulary, embeddings)


def test_zero_norm_candidates_are_excluded():
    vocabulary = ["cat", "dog", "unknown"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 0.0],
        ]
    )

    results = nearest_neighbors(
        "cat",
        vocabulary,
        embeddings,
        top_k=5,
    )

    assert [word for word, _ in results] == ["dog"]


def test_ties_are_deterministic():
    vocabulary = ["cat", "zebra", "apple"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    results = nearest_neighbors(
        "cat",
        vocabulary,
        embeddings,
        top_k=2,
    )

    assert [word for word, _ in results] == [
        "apple",
        "zebra",
    ]


def test_invalid_top_k_raises_error(sample_embeddings):
    vocabulary, embeddings = sample_embeddings

    with pytest.raises(ValueError):
        nearest_neighbors(
            "cat",
            vocabulary,
            embeddings,
            top_k=0,
        )


def test_vocabulary_matrix_mismatch_raises_error():
    with pytest.raises(ValueError, match="Vocabulary size"):
        nearest_neighbors(
            "cat",
            ["cat", "dog"],
            np.array([[1.0, 0.0]]),
        )