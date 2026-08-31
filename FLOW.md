```md
# Tokenization Flow

## Overall

```text
Corpus
  ↓
Tokenizer Training
  ↓
Learn Vocabulary / Merges / Probabilities
  ↓
Test Text
  ↓
Tokenization
  ↓
Token Pieces
  ↓
Token IDs
  ↓
Decoding
  ↓
Reconstructed Text
  ↓
Evaluation