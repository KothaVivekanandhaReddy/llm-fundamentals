What changed/was completed.
# Changelog

All notable changes and experiments in this repository are recorded here.
This repository is a learning and experimentation lab. Entries describe what was implemented, tested, observed, or changed.

---

## 2026-08-29

### Tokenization — Initial Experiments

- Created the `tokenization/` workspace.
- Implemented character-level tokenization experiments.
- Implemented word-level and wordpiece-style tokenizer experiments.
- Implemented unigram tokenizer experiments.
- Implemented a BPE tokenizer from scratch.
- Added pair-frequency counting for BPE.
- Added iterative pair merging.
- Added learned merge rules.
- Added vocabulary construction.
- Added token → ID mapping.
- Added ID → token mapping.
- Added encoding and decoding.
- Added round-trip testing.
- Added byte-level BPE experimentation.
- Added tokenizer comparison experiments.
- Added tokenizer evaluation experiments.
- Added a small text corpus for controlled experiments.
- Tested tokenization behavior on repeated words, unknown words, spaces, newlines, punctuation, and other text patterns.
- Observed how corpus composition affects learned BPE merges.
- Observed how naive character-level BPE can merge whitespace and newline characters with neighboring text.
- Tested unknown-token behavior.
- Compared token counts and tokenizer behavior across different approaches.
- Started documenting tokenizer design decisions and evaluation criteria.

### Documentation

- Updated `README.md`.
- Added tokenizer-specific documentation.
- Added `DECISIONS.md` entries for tokenizer design choices.
- Added `FLOW.md` entries documenting the experiment progression.

---

## Current State

The tokenizer work is still experimental.

The implementations are intentionally simple and are being used to understand tokenizer mechanics before relying on production tokenizer libraries.

## 2026-08-31

### Implemented

- Character-level tokenizer from scratch
- Word-level tokenizer from scratch
- BPE tokenizer from scratch
- Byte-Level BPE tokenizer from scratch
- WordPiece tokenizer from scratch
- Unigram tokenizer from scratch
- Shared `corpus.txt` for all tokenizer experiments
- Shared test set for controlled comparison
- Token encoding and decoding
- Vocabulary inspection
- Token-piece inspection
- Round-trip testing
- Unicode testing
- UTF-8 byte analysis
- Pre-tokenization experiment
- Tokenizer comparison script
- Aggregate tokenizer evaluation

### Evaluation Metrics

- Token count
- Character count
- UTF-8 byte count
- Tokens / character
- Tokens / byte
- Characters / token
- Bytes / token
- UNK count
- UNK rate
- Coverage
- Losslessness
- Encoding errors

### Observations

- Character tokenization is simple but produces long sequences and fails on unseen characters.
- Word tokenization produces short sequences but suffers heavily from OOV words.
- BPE reduces sequence length by learning reusable subword merges.
- Byte-Level BPE provides full byte-level coverage for the current test set.
- WordPiece provides subword segmentation but can produce `[UNK]`.
- Unigram produces alternative probabilistic subword segmentations but currently has lower coverage with the learned vocabulary.
- Token count alone is not a sufficient measure of tokenizer quality.
- `<UNK>` can make word-level tokenization appear artificially efficient because information is lost.
- Encode errors and lossy encoding must be evaluated separately.
- Byte-level tokenization avoids the unseen-character problem observed in the character and standard BPE implementations.

### Current Experimental Result

Using the same corpus and 15 test cases:

- Character: 12/15 lossless
- Word: 2/15 lossless
- BPE: 12/15 lossless
- Byte-BPE: 15/15 lossless
- WordPiece: 12/15 lossless
- Unigram: 6/15 lossless

These results are specific to this implementation, corpus, vocabulary size, and test set.

### Status

Tokenizer implementation and comparison experiments completed.