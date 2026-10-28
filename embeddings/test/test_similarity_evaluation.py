
import numpy as np
import pytest

from embeddings.evaluation import (
    pair_similarity,
    evaluate_similarity_pairs,
)


@pytest.fixture
def sample_data():
    vocabulary = ["cat", "kitten", "car", "vehicle", "opposite"]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
        ]
    )

    return vocabulary, embeddings


def test_identical_word_similarity_is_one(sample_data):
    vocabulary, embeddings = sample_data

    score = pair_similarity(
        "cat", "cat", vocabulary, embeddings
    )

    assert score == pytest.approx(1.0)


def test_similar_direction_has_high_similarity(sample_data):
    vocabulary, embeddings = sample_data

    score = pair_similarity(
        "cat", "kitten", vocabulary, embeddings
    )

    assert score > 0.99


def test_orthogonal_vectors_have_zero_similarity(sample_data):
    vocabulary, embeddings = sample_data

    score = pair_similarity(
        "cat", "car", vocabulary, embeddings
    )

    assert score == pytest.approx(0.0)


def test_opposite_vectors_have_negative_similarity(sample_data):
    vocabulary, embeddings = sample_data

    score = pair_similarity(
        "cat", "opposite", vocabulary, embeddings
    )

    assert score == pytest.approx(-1.0)


def test_unknown_word_raises_error(sample_data):
    vocabulary, embeddings = sample_data

    with pytest.raises(KeyError, match="Unknown word"):
        pair_similarity(
            "cat", "unknown", vocabulary, embeddings
        )


def test_zero_norm_vector_raises_error():
    vocabulary = ["cat", "dog"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="zero-norm"):
        pair_similarity(
            "cat", "dog", vocabulary, embeddings
        )


def test_evaluate_multiple_pairs(sample_data):
    vocabulary, embeddings = sample_data

    pairs = [
        ("cat", "kitten"),
        ("cat", "car"),
        ("cat", "opposite"),
    ]

    results = evaluate_similarity_pairs(
        pairs, vocabulary, embeddings
    )

    assert len(results) == 3
    assert results[0][:2] == ("cat", "kitten")
    assert results[1][:2] == ("cat", "car")
    assert results[2][:2] == ("cat", "opposite")

    assert results[0][2] > results[1][2] > results[2][2]


def test_invalid_embedding_values_raise_error():
    vocabulary = ["cat", "dog"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [np.nan, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        pair_similarity(
            "cat", "dog", vocabulary, embeddings
        )
