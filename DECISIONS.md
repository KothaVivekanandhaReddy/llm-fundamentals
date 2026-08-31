Important decisions and why made them.
# Technical Decisions

## LLM Fundamentals — Tokenization

---

## 1. Purpose of the Tokenization 

The tokenization module is an educational laboratory for understanding how different tokenization strategies convert raw text into model-readable token IDs.

The objective is not to reproduce production tokenizers exactly.

The objective is to:

* implement the core mechanisms ourselves
* run controlled experiments
* compare tokenizer behavior
* understand trade-offs
* connect tokenizer design to LLM behavior

---

## 2. Tokenizers Implemented

This covers six approaches:

1. Character Tokenization
2. Word Tokenization
3. BPE
4. Byte-Level BPE
5. WordPiece
6. Unigram / SentencePiece-style tokenization

Each tokenizer is implemented separately so that its mechanism can be studied independently.

---

## 3. Character Tokenization

Character tokenization represents text as individual characters.

Example:```hello```
becomes:
```
h | e | l | l | o
```

Decision:

Use character tokenization as the simplest baseline.

Purpose:

* establish the basic tokenization concept
* demonstrate vocabulary construction
* demonstrate long sequence lengths
* demonstrate unseen-character/OOV behavior

---

## 4. Word Tokenization

Word tokenization represents text primarily as complete words.

Example:

```
The cat sat
```

becomes:

```
The | cat | sat
```

Decision:

Use word tokenization as the second baseline.

Purpose:

* demonstrate short sequences
* demonstrate vocabulary growth
* demonstrate the out-of-vocabulary problem
* show why modern LLMs generally require subword methods

---

## 5. BPE

Byte Pair Encoding is implemented as a learned subword tokenizer.

Core mechanism:

```
characters
    ↓
count adjacent pairs
    ↓
select most frequent pair
    ↓
merge pair
    ↓
repeat
```

Decision:

Implement BPE manually rather than using a library.

Purpose:

* understand the actual merge process
* inspect learned merges
* understand how frequent patterns become reusable tokens
* observe behavior on unseen words

---

## 6. Byte-Level BPE

Byte-Level BPE starts from the 256 possible byte values rather than Unicode characters.

Core mechanism:

```
text
  ↓
UTF-8 bytes
  ↓
byte sequences
  ↓
BPE merges
```

Decision:

Include Byte-Level BPE separately from ordinary BPE.

Purpose:

* understand byte-level fallback
* demonstrate robust handling of arbitrary Unicode
* compare Unicode behavior against character-level approaches
* connect the implementation to modern LLM tokenizers

---

## 7. WordPiece

WordPiece is implemented separately from BPE.

The training objective uses a WordPiece-style pair score rather than simply selecting the most frequent pair.

Encoding uses subword matching and continuation-piece notation.

Example:

```
playing
```

can conceptually become:

```
play | ##ing
```

Decision:

Keep WordPiece as a separate implementation rather than treating it as "BPE with different names."

Purpose:

* understand the difference between merge objectives
* understand continuation pieces
* understand subword vocabulary construction
* compare WordPiece segmentation against BPE

---

## 8. Unigram

The Unigram tokenizer is implemented as a probabilistic subword tokenizer.

Core mechanism:

```
candidate pieces
      ↓
probabilities
      ↓
possible segmentations
      ↓
Viterbi / dynamic programming
      ↓
best segmentation
```

Decision:

Implement an educational Unigram tokenizer rather than attempting to reproduce the complete production SentencePiece training algorithm.

Important distinction:

SentencePiece is a tokenizer framework that can support different algorithms.

This lab specifically focuses on the Unigram approach.

---

## 9. Same Training Corpus

All six tokenizers are trained using the same small English corpus.

Decision:

Keep the training corpus controlled across experiments.

Reason:

This makes algorithmic differences easier to observe.

Limitation:

The corpus is intentionally tiny and therefore does not represent production tokenizer training.

---

## 10. Same Test Set

The comparison includes:

* familiar English
* unseen English words
* morphological variants
* technical terminology
* programming text
* punctuation
* numbers
* Chinese
* Hindi
* Japanese
* emoji
* mixed Unicode/English text

Decision:

Use the same test texts for every tokenizer.

Reason:

This provides a controlled comparison of coverage and segmentation behavior.

---

## 11. Round-Trip Testing

Every tokenizer is tested using:

```
text
  ↓
encode
  ↓
token IDs
  ↓
decode
  ↓
reconstructed text
```

The reconstructed text is compared with the original.

Decision:

Round-trip correctness is treated as an explicit evaluation metric.

Reason:

A low token count is meaningless if information has been lost.

---

## 12. UNK Must Be Evaluated Separately

A tokenizer producing:

```
<UNK>
```

must not automatically be interpreted as efficient tokenization.

Example:

```
internationalization
    ↓
<UNK>
```

may produce only one token, but the original information was not represented faithfully.

Decision:

Track:

* UNK count
* UNK rate
* round-trip success

alongside token count.

---

## 13. Token Count Is Not the Only Metric

The evaluation includes:

* total token count
* tokens per character
* tokens per byte
* characters per token
* bytes per token
* UNK count
* UNK rate
* round-trip rate
* vocabulary size

Decision:

Do not rank tokenizers solely by token count.

Reason:

Token count must be interpreted together with vocabulary coverage and information preservation.

---

## 14. Byte-Level Unicode Behavior

Byte-Level BPE is expected to handle arbitrary UTF-8 text more robustly than the current closed-vocabulary character/BPE implementations.

The experiment includes:

* Chinese
* Hindi
* Japanese
* emoji

Decision:

Keep Unicode tests in the evaluation suite.

Reason:

Unicode coverage is an important practical tokenizer property.

---

## 15. Comparison Script Architecture

The comparison/evaluation script does not modify tokenizer algorithms.

It adapts to the existing tokenizer interfaces.

For example:

* normal tokenizers expose `tokenize()`
* Byte-Level BPE exposes `token_pieces()`

Decision:

Keep tokenizer implementations independent from the evaluation framework.

Reason:

The experiment should compare implementations rather than force them into an identical internal architecture.

---

## 16. Educational vs Production Scope

These implementations are educational.

They are intended to explain:

* vocabulary construction
* merge algorithms
* segmentation
* OOV behavior
* Unicode representation
* token sequence length
* probabilistic segmentation

They are not intended to replace production implementations such as those used by modern LLM libraries.

Production systems additionally involve:

* optimized data structures
* efficient training
* normalization
* pre-tokenization
* special tokens
* serialization
* highly optimized inference-time encoding

---
# Decisions

## Use One Shared Corpus

All tokenizer implementations use the same `corpus.txt`.

### Reason

A tokenizer comparison is only meaningful when the training data is controlled.

---

## Use One Shared Test Set

All tokenizer implementations are evaluated on the same 15 test strings.

The test set intentionally includes:

- normal English
- unseen words
- morphological variants
- punctuation
- numbers
- source-code-like text
- Chinese
- Hindi
- Japanese
- emoji
- mixed Unicode text

### Reason

This exposes differences in vocabulary coverage, segmentation, and Unicode handling.

---

## Keep Tokenizer Implementations From Scratch

The tokenizer implementations are educational implementations rather than wrappers around production tokenizer libraries.

### Reason

The objective is to understand the internal mechanics:

```text
text
→ representation
→ tokenization
→ vocabulary
→ token IDs
→ decoding

