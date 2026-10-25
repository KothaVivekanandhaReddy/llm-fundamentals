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

# Embeddings Learning Flow

## Embedding Lookup

Token IDs
    ↓
Embedding Matrix
    ↓
Select rows using token IDs
    ↓
Dense vectors

File:
embeddings/embeddings.py


## Embedding Geometry

Dense vectors
    ↓
Dot Product
    ↓
Vector Norm
    ↓
Euclidean Distance
    ↓
Cosine Similarity
    ↓
L2 Normalization

File:
embeddings/embedding_geometry.py

## Word2Vec Skip-Gram

Corpus
    ↓
Tokenization
    ↓
Vocabulary
    ↓
Context-window pairs
    ↓
Center token ID
    ↓
W_in lookup
    ↓
Center embedding
    ↓
W_out projection
    ↓
Logits
    ↓
Softmax
    ↓
Cross-entropy
    ↓
Backpropagation
    ↓
Gradient descent
    ↓
Updated embeddings
File : word2vec_skipgram.py

