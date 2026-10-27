import pytest

from embeddings.fasttext import character_ngrams
from embeddings.fasttext import SubwordVocabulary
from embeddings.fasttext import SubwordEmbeddings

#test_boundary_markers_are_used & test_multiple_ngram_lengths: Confirms that your sliding window extracts exact chunks and properly 
# sweeps through both low and high bounds (e.g., pulling out both 3-character and 4-character slices).
def test_boundary_markers_are_used(): 
    result = character_ngrams("cat", min_n=3, max_n=3)

    assert result == ["<ca", "cat", "at>"]


def test_multiple_ngram_lengths():
    result = character_ngrams("cat", min_n=3, max_n=4)

    assert result == [
        "<ca",
        "cat",
        "at>",
        "<cat",
        "cat>",
    ]

#test_short_word: Validates a  edge-case. When a word is shorter than your min_n target (like "a" with min_n=3), 
# your code handles it without crashing by gracefully returning the complete wrapped string ["<a>"]

def test_short_word():
    result = character_ngrams("a", min_n=3, max_n=3)

    assert result == ["<a>"]

#test_duplicate_ngrams_are_preserved: Confirms that repeating character patterns (like "aaa") are kept as duplicates. 
# This is vital because the frequency of subwords impacts the final aggregated word embedding vector.
def test_duplicate_ngrams_are_preserved():
    result = character_ngrams("aaaa", min_n=3, max_n=3)

    assert result == ["<aa", "aaa", "aaa", "aa>"]


def test_empty_word_is_rejected():
    with pytest.raises(ValueError):
        character_ngrams("")


def test_invalid_ngram_range_is_rejected():
    with pytest.raises(ValueError):
        character_ngrams("cat", min_n=5, max_n=3)


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        character_ngrams(123)

# • Validates that the internal forward mapping index (ngram_to_id) and backward mapping list (id_to_ngram) stay perfectly aligned in scale.
def test_subword_vocabulary_builds_ids():
    vocab = SubwordVocabulary(min_n=3, max_n=3)
    vocab.fit(["cat", "car"])

    assert len(vocab) > 0
    assert len(vocab.ngram_to_id) == len(vocab.id_to_ngram)

# • Confirms that initializing separate standalone processes over the exact same language files generates twin identical mapping IDs.
def test_subword_ids_are_deterministic():
    words = ["cat", "car", "dog"]

    vocab_a = SubwordVocabulary(3, 3).fit(words)
    vocab_b = SubwordVocabulary(3, 3).fit(words)

    assert vocab_a.ngram_to_id == vocab_b.ngram_to_id

#  Confirms that matching subword sequences inside different tokens (like the prefix <ca in both cat and car) map to the exact same
#  tracking slot. This enables the model to share morphological features across related vocabulary words.
def test_shared_ngrams_have_shared_ids():
    vocab = SubwordVocabulary(3, 3).fit(["cat", "car"])

    assert vocab.ngram_to_id["<ca"] == vocab.ngram_to_id["<ca"]
    assert vocab.ngram_to_id["<ca"] in vocab.encode_word("cat")
    assert vocab.ngram_to_id["<ca"] in vocab.encode_word("car")


def test_word_mapping_preserves_repeated_ngrams():
    vocab = SubwordVocabulary(3, 3).fit(["aaaa"])

    ids = vocab.encode_word("aaaa")
    ngrams = [
        vocab.id_to_ngram[idx]
        for idx in ids
    ]

    assert ngrams.count("aaa") == 2

#  Proves your OOV mitigation strategy. When the completely unseen word "there" is evaluated against a model that only knows "where"
#  your code successfully extracts and processes the shared subwords ("her", "ere", etc.) within valid index boundaries without generating an execution crash.
def test_unknown_word_returns_known_subwords():
    vocab = SubwordVocabulary(3, 3).fit(["where"])

    ids = vocab.encode_word("there")

    assert isinstance(ids, list)
    assert all(0 <= idx < len(vocab) for idx in ids)

import numpy as np
import pytest


def test_embedding_table_shape():
    vocab = SubwordVocabulary(3, 3).fit(["cat", "car"])

    model = SubwordEmbeddings(
        vocabulary=vocab,
        embedding_dim=8,
        seed=42,
    )

    assert model.embeddings.shape == (len(vocab), 8)


def test_word_vector_is_sum_of_subword_vectors():
    vocab = SubwordVocabulary(3, 3).fit(["cat"])

    model = SubwordEmbeddings(
        vocabulary=vocab,
        embedding_dim=4,
        seed=42,
    )

    ids = vocab.encode_word("cat")
    expected = model.embeddings[ids].sum(axis=0)

    actual = model.get_word_vector("cat")

    np.testing.assert_allclose(actual, expected)


def test_word_vector_has_expected_dimension():
    vocab = SubwordVocabulary(3, 3).fit(["cat", "car"])

    model = SubwordEmbeddings(vocab, embedding_dim=16)

    vector = model.get_word_vector("cat")

    assert vector.shape == (16,)


def test_same_seed_produces_same_embeddings():
    vocab = SubwordVocabulary(3, 3).fit(["cat", "car"])

    model_a = SubwordEmbeddings(vocab, embedding_dim=8, seed=42)
    model_b = SubwordEmbeddings(vocab, embedding_dim=8, seed=42)

    np.testing.assert_array_equal(
        model_a.embeddings,
        model_b.embeddings,
    )


def test_unknown_word_uses_known_subwords():
    vocab = SubwordVocabulary(3, 3).fit(["where"])

    model = SubwordEmbeddings(vocab, embedding_dim=8, seed=42)

    vector = model.get_word_vector("there")

    assert vector.shape == (8,)
    assert np.isfinite(vector).all()


def test_word_with_no_known_subwords_raises_error():
    vocab = SubwordVocabulary(3, 3).fit(["cat"])

    model = SubwordEmbeddings(vocab, embedding_dim=8, seed=42)

    with pytest.raises(ValueError, match="No known subword"):
        model.get_word_vector("xyz")
def test_train_step_returns_finite_loss():
    vocab = SubwordVocabulary(3, 3).fit(["cat", "car"])

    model = SubwordEmbeddings(
        vocab,
        embedding_dim=4,
        seed=42,
    )

    positive = np.array([0.1, -0.2, 0.3, 0.4])

    negatives = np.array([
        [-0.2, 0.1, 0.2, 0.3],
        [0.3, -0.1, 0.4, -0.2],
    ])

    loss = model.train_step(
        "cat",
        positive,
        negatives,
        learning_rate=0.01,
    )

    assert np.isfinite(loss)
    assert loss > 0


def test_train_step_updates_parameters():
    vocab = SubwordVocabulary(3, 3).fit(["cat", "car"])

    model = SubwordEmbeddings(
        vocab,
        embedding_dim=4,
        seed=42,
    )

    positive = np.array([0.1, -0.2, 0.3, 0.4])

    negatives = np.array([
        [-0.2, 0.1, 0.2, 0.3],
        [0.3, -0.1, 0.4, -0.2],
    ])

    embeddings_before = model.embeddings.copy()
    positive_before = positive.copy()
    negatives_before = negatives.copy()

    model.train_step(
        "cat",
        positive,
        negatives,
        learning_rate=0.01,
    )

    assert not np.array_equal(
        model.embeddings,
        embeddings_before,
    )

    assert not np.array_equal(
        positive,
        positive_before,
    )

    assert not np.array_equal(
        negatives,
        negatives_before,
    )


def test_train_step_rejects_invalid_learning_rate():
    vocab = SubwordVocabulary(3, 3).fit(["cat"])

    model = SubwordEmbeddings(vocab, embedding_dim=4)

    positive = np.ones(4)
    negatives = np.ones((2, 4))

    with pytest.raises(ValueError, match="learning_rate"):
        model.train_step(
            "cat",
            positive,
            negatives,
            learning_rate=0,
        )
def test_train_step_reduces_loss_on_repeated_example():
    from embeddings.fasttext_objective import negative_sampling_loss

    vocabulary = SubwordVocabulary(min_n=3, max_n=3)
    vocabulary.fit(["cat", "dog"])

    model = SubwordEmbeddings(
        vocabulary=vocabulary,
        embedding_dim=4,
        seed=42,
    )

    word = "cat"
    positive_context = np.array([0.2, -0.1, 0.3, 0.1], dtype=float)
    negative_contexts = np.array(
        [
            [-0.2, 0.1, 0.1, -0.3],
            [0.1, 0.2, -0.1, 0.2],
        ],
        dtype=float,
    )

    center = model.get_word_vector(word)
    initial_loss = negative_sampling_loss(
        center,
        positive_context.copy(),
        negative_contexts.copy(),
    )

    updated_loss = model.train_step(
        word=word,
        positive_context_vector=positive_context,
        negative_context_vectors=negative_contexts,
        learning_rate=0.01,
    )

    final_loss = negative_sampling_loss(
        model.get_word_vector(word),
        positive_context,
        negative_contexts,
    )

    assert np.isfinite(updated_loss)
    assert final_loss < initial_loss
def test_word_vocabulary_builds_deterministic_ids():
    from embeddings.fasttext import WordVocabulary

    vocab = WordVocabulary().fit(["cat", "dog", "cat"])

    assert len(vocab) == 2
    assert vocab.encode("cat") == 0
    assert vocab.encode("dog") == 1


def test_word_vocabulary_rejects_unknown_word():
    import pytest
    from embeddings.fasttext import WordVocabulary

    vocab = WordVocabulary().fit(["cat", "dog"])

    with pytest.raises(ValueError, match="Unknown word"):
        vocab.encode("bird")


def test_context_embedding_table_has_expected_shape():
    from embeddings.fasttext import (
        SubwordVocabulary,
        SubwordEmbeddings,
    )

    words = ["the", "cat", "sat", "the"]

    subword_vocab = SubwordVocabulary().fit(words)
    model = SubwordEmbeddings(
        vocabulary=subword_vocab,
        embedding_dim=8,
        seed=42,
    )

    model.initialize_context_embeddings(words, seed=43)

    assert model.context_embeddings.shape == (
        3,
        8,
    )


def test_context_embeddings_are_reproducible():
    from embeddings.fasttext import (
        SubwordVocabulary,
        SubwordEmbeddings,
    )

    words = ["the", "cat", "sat"]

    subword_vocab = SubwordVocabulary().fit(words)

    model_a = SubwordEmbeddings(
        subword_vocab,
        embedding_dim=8,
        seed=42,
    )
    model_b = SubwordEmbeddings(
        subword_vocab,
        embedding_dim=8,
        seed=42,
    )

    model_a.initialize_context_embeddings(words, seed=43)
    model_b.initialize_context_embeddings(words, seed=43)

    assert np.array_equal(
        model_a.context_embeddings,
        model_b.context_embeddings,
    )
def test_train_pair_updates_persistent_context_embeddings():
    from embeddings.fasttext import (
        SubwordVocabulary,
        SubwordEmbeddings,
    )

    words = ["the", "cat", "sat", "mat"]

    subword_vocab = SubwordVocabulary().fit(words)

    model = SubwordEmbeddings(
        vocabulary=subword_vocab,
        embedding_dim=8,
        seed=42,
    )
    model.initialize_context_embeddings(words, seed=43)

    input_before = model.embeddings.copy()
    context_before = model.context_embeddings.copy()

    loss = model.train_pair(
        center_word="cat",
        positive_context_word="sat",
        negative_context_words=["mat"],
        learning_rate=0.025,
    )

    assert np.isfinite(loss)
    assert not np.array_equal(input_before, model.embeddings)
    assert not np.array_equal(
        context_before,
        model.context_embeddings,
    )


def test_train_pair_requires_initialized_context_embeddings():
    import pytest
    from embeddings.fasttext import (
        SubwordVocabulary,
        SubwordEmbeddings,
    )

    words = ["the", "cat", "sat"]
    subword_vocab = SubwordVocabulary().fit(words)

    model = SubwordEmbeddings(
        vocabulary=subword_vocab,
        embedding_dim=8,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="Initialize context embeddings",
    ):
        model.train_pair(
            center_word="cat",
            positive_context_word="sat",
            negative_context_words=["the"],
        )
def test_generate_context_pairs_with_window_one():
    from embeddings.fasttext import (
        WordVocabulary,
        generate_context_pairs,
    )

    words = ["the", "cat", "sat"]
    vocab = WordVocabulary().fit(words)

    pairs = generate_context_pairs(
        words,
        vocab,
        window_size=1,
    )

    expected = [
        (vocab.encode("the"), vocab.encode("cat")),
        (vocab.encode("cat"), vocab.encode("the")),
        (vocab.encode("cat"), vocab.encode("sat")),
        (vocab.encode("sat"), vocab.encode("cat")),
    ]

    assert pairs == expected


def test_generate_context_pairs_with_window_two():
    from embeddings.fasttext import (
        WordVocabulary,
        generate_context_pairs,
    )

    words = ["a", "b", "c"]
    vocab = WordVocabulary().fit(words)

    pairs = generate_context_pairs(
        words,
        vocab,
        window_size=2,
    )

    assert len(pairs) == 6


def test_generate_context_pairs_rejects_invalid_window():
    import pytest
    from embeddings.fasttext import (
        WordVocabulary,
        generate_context_pairs,
    )

    words = ["the", "cat"]
    vocab = WordVocabulary().fit(words)

    with pytest.raises(ValueError, match="window_size"):
        generate_context_pairs(
            words,
            vocab,
            window_size=0,
        )


def test_generate_context_pairs_rejects_empty_corpus():
    import pytest
    from embeddings.fasttext import (
        WordVocabulary,
        generate_context_pairs,
    )

    vocab = WordVocabulary().fit(["the", "cat"])

    with pytest.raises(ValueError, match="Corpus words"):
        generate_context_pairs(
            [],
            vocab,
            window_size=1,
        )
def test_train_corpus_returns_one_loss_per_epoch():
    from embeddings.fasttext import (
        SubwordVocabulary,
        SubwordEmbeddings,
    )

    words = (
        "the cat sat on the mat "
        "the cat sat on the mat"
    ).split()

    subword_vocab = SubwordVocabulary().fit(words)

    model = SubwordEmbeddings(
        vocabulary=subword_vocab,
        embedding_dim=8,
        seed=42,
    )

    losses = model.train_corpus(
        words,
        epochs=3,
        window_size=1,
        negative_samples=2,
        learning_rate=0.025,
        seed=123,
    )

    assert len(losses) == 3
    assert all(np.isfinite(loss) for loss in losses)
    assert model.context_embeddings.shape == (
        len(set(words)),
        8,
    )


def test_train_corpus_is_reproducible():
    from embeddings.fasttext import (
        SubwordVocabulary,
        SubwordEmbeddings,
    )

    words = (
        "the cat sat on the mat "
        "the cat sat on the mat"
    ).split()

    subword_vocab = SubwordVocabulary().fit(words)

    model_a = SubwordEmbeddings(
        subword_vocab,
        embedding_dim=8,
        seed=42,
    )
    model_b = SubwordEmbeddings(
        subword_vocab,
        embedding_dim=8,
        seed=42,
    )

    losses_a = model_a.train_corpus(
        words,
        epochs=3,
        window_size=1,
        negative_samples=2,
        seed=123,
    )
    losses_b = model_b.train_corpus(
        words,
        epochs=3,
        window_size=1,
        negative_samples=2,
        seed=123,
    )

    assert np.allclose(losses_a, losses_b)
    assert np.allclose(
        model_a.embeddings,
        model_b.embeddings,
    )
    assert np.allclose(
        model_a.context_embeddings,
        model_b.context_embeddings,
    )


def test_negative_sampler_excludes_positive_context():
    from embeddings.fasttext import sample_negative_word_ids

    rng = np.random.default_rng(123)

    sampled = sample_negative_word_ids(
        positive_context_id=2,
        vocabulary_size=5,
        negative_samples=100,
        rng=rng,
    )

    assert len(sampled) == 100
    assert all(word_id != 2 for word_id in sampled)
    assert all(0 <= word_id < 5 for word_id in sampled)