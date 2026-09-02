# LLM Learning Log

## Tokenization

- Built 6 tokenizers from scratch:
  - Character
  - Word
  - BPE
  - Byte-Level BPE
  - WordPiece
  - Unigram

- Tested all on the same 15-text test set.

### Comparison

| Tokenizer | Tokens | Tok/Char | UNK | Lossless |
|---|---:|---:|---:|---:|
| Character | 229 | 1.000 | 0% | 12/15 |
| Word | 62 | 0.228 | 43.55% | 2/15 |
| BPE | 153 | 0.668 | 0% | 12/15 |
| Byte-BPE | 235 | 0.864 | 0% | **15/15** |
| WordPiece | 247 | 0.908 | 2.02% | 12/15 |
| Unigram | 166 | 0.610 | 18.07% | 6/15 |

- Word tokenizer produced the fewest tokens but had **43.55% UNK rate**.
- Byte-BPE handled all 15 test cases losslessly.
- Unicode, code, URLs, numbers and punctuation were tested.
- Fixed whitespace, Unicode and decoding bugs during implementation.

### Benchmark

- Dataset: **3.71 MB / 64,000 lines**
- Scratch Byte-BPE: **0.28 MB/s**
- Hugging Face Tokenizer: **2.31 MB/s**
- Hugging Face was **8.34× faster** on this benchmark.
- Main reason: optimized native implementation vs Python-level loops.

---

## Embeddings

- Next: build embeddings from scratch.
- Inspect vectors.
- Calculate cosine similarity.
- Train embeddings and observe how the vectors change.