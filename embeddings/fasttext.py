# ==============================================================================
# FASTTEXT EMBEDDINGS - CORE SUMMARY & COMPARISON
# ==============================================================================

# 1.DEFINITION EXPLANATION : 
# ------------------------------------------------------------------------------
# * Subword n-gram Architecture:
#   Unlike Word2Vec or GloVe which treat words as atomic, indivisible tokens, 
#   FastText breaks each word down into a bag of character n-grams bounded by 
#   angular brackets (e.g., for n=3, "cat" becomes "<ca", "cat", "at>", plus 
#   the full word "<cat>"). The final vector representation of a word is the 
#   mathematical sum of its individual subword n-gram vectors.
#
# * Out-of-Vocabulary (OOV) & Morphological Resilience:
#   By looking inside the word structure, FastText can natively generate highly 
#   accurate embedding vectors for unseen words, typos, or rare variations 
#   (e.g., "catitude") during inference. It achieves this by aggregating the 
#   familiar subword prefixes, roots, and suffixes it learned during training.


# 2. EMBEDDING MODELS COMPARISON SUMMARY
# ------------------------------------------------------------------------------
# +-------------------+-------------------------+-------------------------+-------------------------+
# | Feature           | Word2Vec (CBOW/SGNS)    | GloVe                   | FastText                |
# +-------------------+-------------------------+-------------------------+-------------------------+
# | Basic Token Unit  | Full Words (Atomic)     | Full Words (Atomic)     | Character n-grams       |
# | Core Architecture | Predictive Local Window | Global Count-Based Matrix| Subword Predictive Loop |
# | OOV Support       | Fails (Returns <UNK>)   | Fails (Returns <UNK>)   | Natively Synthesizes    |
# | Typo Robustness   | Low (Treats as new word)| Low (Treats as new word)| High (Shared n-grams)   |
# | Footprint & Speed | Fast, Low Memory        | Fast, Low Memory        | Slow, Heavy Memory      |
# +-------------------+-------------------------+-------------------------+-------------------------+

# FastText adds boundary markers to a word and extracts character n-grams.
from random import seed

import numpy as np
def character_ngrams( #function that precisely mirrors how FastText breaks down words into smaller chunks before embedding them
    word: str,
    min_n: int = 3,
    max_n: int = 6,
) -> list[str]:
    """Extract character n-grams from a word with boundary markers."""

    if not isinstance(word, str):
        raise TypeError("word must be a string")

    if not word:
        raise ValueError("word must not be empty")

    if min_n < 1:
        raise ValueError("min_n must be >= 1")

    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")

    marked_word = f"<{word}>" #Boundary markers are added to the word to ensure that the n-grams capture the start and end of the word,
                              #which is important for distinguishing between different words that may share similar internal character sequences.
    ngrams = []

    for n in range(min_n, max_n + 1): #moves step-by-step through the character sequence to capture all possible context fragments.
        for start in range(len(marked_word) - n + 1):
            ngrams.append(marked_word[start : start + n])

    return ngrams

# Subword Vocabulary : 
# Extract character n-grams from each word.
#Build a deterministic vocabulary of unique n-grams.
#Assign each n-gram a stable integer ID.
#Map each word to its n-gram IDs.
#Preserve repeated n-gram occurrences in a word's mapping.

class SubwordVocabulary:
    """Build and manage a vocabulary of character n-grams."""

    def __init__(self, min_n: int = 3, max_n: int = 6):
        if min_n < 1:
            raise ValueError("min_n must be >= 1")

        if max_n < min_n:
            raise ValueError("max_n must be >= min_n")

        self.min_n = min_n
        self.max_n = max_n

        self.ngram_to_id: dict[str, int] = {}
        self.id_to_ngram: list[str] = []
        self.word_to_ids: dict[str, list[int]] = {}

    # extracts all n-grams from your entire text collection and bundles them into a unique set.
    def fit(self, words: list[str]) -> "SubwordVocabulary":
        """Build a deterministic vocabulary from training words."""

        unique_ngrams = set()

        for word in words:
            ngrams = character_ngrams(
                word,
                min_n=self.min_n,
                max_n=self.max_n,
            )
            unique_ngrams.update(ngrams)

        self.id_to_ngram = sorted(unique_ngrams)

        self.ngram_to_id = {
            ngram: idx
            for idx, ngram in enumerate(self.id_to_ngram)
        }

        self.word_to_ids = {}

        for word in words:
            ngrams = character_ngrams(
                word,
                min_n=self.min_n,
                max_n=self.max_n,
            )

            self.word_to_ids[word] = [
                self.ngram_to_id[ngram]
                for ngram in ngrams
            ]

        return self
    
    #  breaks an incoming word down into n-grams and looks up their mapped integer IDs.
    def encode_word(self, word: str) -> list[int]:
        """Map a word to the IDs of its character n-grams."""

        ngrams = character_ngrams(
            word,
            min_n=self.min_n,
            max_n=self.max_n,
        )

        return [
            self.ngram_to_id[ngram]
            for ngram in ngrams
            if ngram in self.ngram_to_id
        ]

    def __len__(self) -> int:
        return len(self.id_to_ngram)

class WordVocabulary:
    """Build a deterministic vocabulary of whole words."""

    def __init__(self):
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: list[str] = []

    def fit(self, words: list[str]) -> "WordVocabulary":
        """Build a sorted vocabulary from training words."""
        self.id_to_word = sorted(set(words))
        self.word_to_id = {
            word: idx
            for idx, word in enumerate(self.id_to_word)
        }
        return self

    def encode(self, word: str) -> int:
        """Convert a word to its vocabulary ID."""
        if word not in self.word_to_id:
            raise ValueError(f"Unknown word: {word!r}")

        return self.word_to_id[word]

    def __len__(self) -> int:
        return len(self.id_to_word)

class SubwordEmbeddings:
    """Construct word vectors from learned character n-gram vectors."""

    def __init__(
        self,
        vocabulary: SubwordVocabulary,
        embedding_dim: int = 50,
        seed: int = 42,
    ):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim

        rng = np.random.default_rng(seed)

        self.embeddings = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(len(vocabulary), embedding_dim),
        )
        self.context_vocabulary = None
        self.context_embeddings = None

    def get_word_vector(self, word: str) -> np.ndarray:
        """Sum the vectors of the word's known character n-grams."""

        subword_ids = self.vocabulary.encode_word(word)

        if not subword_ids:
            raise ValueError(
                f"No known subword n-grams found for word: {word!r}"
            )

        return self.embeddings[subword_ids].sum(axis=0)
    
    def train_step(
        self,
        word: str,
        positive_context_vector: np.ndarray,
        negative_context_vectors: np.ndarray,
        learning_rate: float = 0.025,
    ) -> float:
        """Perform one SGNS update for a center word."""

        from embeddings.fasttext_objective import (
            negative_sampling_loss,
            negative_sampling_gradients,
        )

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        subword_ids = self.vocabulary.encode_word(word)

        if not subword_ids:
            raise ValueError(
                f"No known subword n-grams found for word: {word!r}"
            )

        expected_shape = (self.embedding_dim,)

        if positive_context_vector.shape != expected_shape:
            raise ValueError(
                "positive_context_vector has an invalid shape"
            )

        if negative_context_vectors.ndim != 2:
            raise ValueError(
                "negative_context_vectors must be a 2D array"
            ) 

        if negative_context_vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                "negative_context_vectors have an invalid dimension"
            )

    # Build the center word representation.
        center_vector = self.embeddings[subword_ids].sum(axis=0)

    # Calculate the objective before modifying parameters.
        loss = negative_sampling_loss(
            center_vector,
            positive_context_vector,
            negative_context_vectors,
        )

    # Calculate gradients using the current parameter values.
        (
            grad_center,
            grad_positive,
            grad_negatives,
        ) = negative_sampling_gradients(
            center_vector,
            positive_context_vector,
            negative_context_vectors,
        )

    # Update every subword occurrence.
    # Repeated IDs intentionally receive repeated updates.
        for subword_id in subword_ids:
            self.embeddings[subword_id] -= (
                learning_rate * grad_center
            )

    # Update the positive context vector.
        positive_context_vector -= (
            learning_rate * grad_positive
        ) 

    # Update each negative context vector.
        negative_context_vectors -= (
            learning_rate * grad_negatives
        )

        return loss
    def initialize_context_embeddings(
        self,
        words: list[str],
        seed: int = 43,
        ) -> None:
        """Initialize whole-word context embeddings from a corpus."""

        if not words:
            raise ValueError("Corpus words cannot be empty.")

        self.context_vocabulary = WordVocabulary().fit(words)

        rng = np.random.default_rng(seed)

        self.context_embeddings = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(
                len(self.context_vocabulary),
                self.embedding_dim,
            ),
        )
    def train_pair(
        self,
        center_word: str,
        positive_context_word: str,
        negative_context_words: list[str],
        learning_rate: float = 0.025,
    ) -> float:
        """Train one center-positive-context pair with negative sampling."""

        from embeddings.fasttext_objective import (
            negative_sampling_loss,
            negative_sampling_gradients,
        )

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.context_vocabulary is None or self.context_embeddings is None:
            raise ValueError(
                "Initialize context embeddings before training."
            )

        subword_ids = self.vocabulary.encode_word(center_word)
        if not subword_ids:
            raise ValueError(
                f"No known subwords for center word: {center_word!r}"
            )

        if not negative_context_words:
             raise ValueError(
                "At least one negative context word is required."
            )

        context_vocab = self.context_vocabulary

        positive_id = context_vocab.encode(positive_context_word)
        negative_ids = [
            context_vocab.encode(word)
            for word in negative_context_words
        ]

    # Copy the vectors so every gradient uses the same
    # pre-update parameter values.
        center_vector = self.embeddings[subword_ids].sum(axis=0).copy()
        positive_vector = self.context_embeddings[positive_id].copy()
        negative_vectors = self.context_embeddings[negative_ids].copy()

        loss = negative_sampling_loss(
            center_vector,
            positive_vector,
            negative_vectors,
        )

        grad_center, grad_positive, grad_negatives = (
            negative_sampling_gradients(
                center_vector,
                positive_vector,
                negative_vectors,
            )
        )

    # Update input subword vectors.
    # Repeated n-gram IDs represent repeated occurrences,
    # so each occurrence receives its gradient contribution.
        for subword_id in subword_ids:
            self.embeddings[subword_id] -= (
                 learning_rate * grad_center
            )

    # Update the persistent positive context vector.
        self.context_embeddings[positive_id] -= (
            learning_rate * grad_positive
        )

    # Update the persistent negative context vectors.
    # Explicit row updates are important: NumPy advanced
    # indexing returns copies rather than writable row views.
        for negative_id, grad_negative in zip(
            negative_ids,
            grad_negatives,
        ):
            self.context_embeddings[negative_id] -= (
                learning_rate * grad_negative
            )

        return float(loss)
    def train_corpus(
        self,
        words: list[str],
        epochs: int = 10,
        window_size: int = 2,
        negative_samples: int = 5,
        learning_rate: float = 0.025,
        seed: int = 123,
    ) -> list[float]:
        """Train FastText SGNS on a tokenized corpus."""

        from embeddings.fasttext import (
            WordVocabulary,
            generate_context_pairs,
            sample_negative_word_ids,
        )

        if not words:
            raise ValueError("Corpus words cannot be empty.")

        if epochs <= 0:
           raise ValueError("epochs must be positive.")

        if learning_rate <= 0:
           raise ValueError("learning_rate must be positive.")

        if negative_samples <= 0:
           raise ValueError("negative_samples must be positive.")

        if window_size <= 0:
            raise ValueError("window_size must be positive.")

     # Build the whole-word vocabulary and persistent context table.
        self.context_vocabulary = WordVocabulary().fit(words)
 
        rng = np.random.default_rng(seed)

        self.context_embeddings = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(
                len(self.context_vocabulary),
                self.embedding_dim,
            ),
        )

    # Every corpus word must be representable by the subword vocabulary.
        for word in self.context_vocabulary.id_to_word:
            if not self.vocabulary.encode_word(word):
                raise ValueError(
                    f"No known subwords for corpus word: {word!r}"
                )

    # Generate positive center-context examples.
        pairs = generate_context_pairs(
            words,
            self.context_vocabulary,
            window_size=window_size,
        )

        if not pairs:
            raise ValueError(
                 "No center-context pairs generated. "
                "Provide a longer corpus or a larger window."
            )

        epoch_losses: list[float] = []

        for _ in range(epochs):
             # Shuffle pair order each epoch.
            order = rng.permutation(len(pairs))
            total_loss = 0.0

            for pair_index in order:
                center_id, positive_context_id = pairs[pair_index]

                center_word = self.context_vocabulary.id_to_word[
                    center_id
                ]
                positive_context_word = (
                    self.context_vocabulary.id_to_word[
                        positive_context_id
                    ]
                )

                negative_ids = sample_negative_word_ids(
                    positive_context_id=positive_context_id,
                    vocabulary_size=len(self.context_vocabulary),
                    negative_samples=negative_samples,
                    rng=rng,
                )

                negative_words = [
                    self.context_vocabulary.id_to_word[negative_id]
                    for negative_id in negative_ids
                ]

                loss = self.train_pair(
                    center_word=center_word,
                    positive_context_word=positive_context_word,
                    negative_context_words=negative_words,
                    learning_rate=learning_rate,
                )

                total_loss += loss

            epoch_losses.append(total_loss / len(pairs))

        return epoch_losses


    
def generate_context_pairs(
    words: list[str],
    word_vocabulary: WordVocabulary,
    window_size: int = 2,
) -> list[tuple[int, int]]:
    """Generate center-context pairs within a fixed window."""

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if not words:
        raise ValueError("Corpus words cannot be empty.")

    word_ids = [
        word_vocabulary.encode(word)
        for word in words
    ]

    pairs: list[tuple[int, int]] = []

    for center_position, center_id in enumerate(word_ids):
        left = max(0, center_position - window_size)
        right = min(
            len(word_ids),
            center_position + window_size + 1,
        )

        for context_position in range(left, right):
            if context_position == center_position:
                continue

            context_id = word_ids[context_position]
            pairs.append((center_id, context_id))

    return pairs
def sample_negative_word_ids(
    positive_context_id: int,
    vocabulary_size: int,
    negative_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample negative word IDs, excluding the positive context word."""

    if vocabulary_size < 2:
        raise ValueError(
            "Negative sampling requires at least two unique words."
        )

    if not 0 <= positive_context_id < vocabulary_size:
        raise ValueError("positive_context_id is out of range.")

    if negative_samples <= 0:
        raise ValueError("negative_samples must be positive.")

    candidates = np.concatenate(
        (
            np.arange(positive_context_id),
            np.arange(positive_context_id + 1, vocabulary_size),
        )
    )

    sampled_ids = rng.choice(
        candidates,
        size=negative_samples,
        replace=True,
    )

    return sampled_ids.tolist()