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

