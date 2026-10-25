#Token IDs
#   ↓
#Embedding Matrix
#   ↓
#Rows selected by IDs
#   ↓
#Dense vectors

#Example: 
#vocab_size = 10 #total number of unique tokens
#embedding_dim = 4 #size of the dense vector used to represent each individual token
# means we create matrix of size 10 * 4 . row0 [...] row1[..] ... row9[...]
# if input = [2,7,4] output is : [2,7,4] rows selction.. 
# ID 2 → [0.6465, 0.1108, 0.7404, -0.6706]
# ID 7 → [0.7804, 1.2249, -1.6501, -2.4221]
# ID 4 → [0.4112, 1.1136, -0.1082, 0.5393]
#Embedding lookup is fundamentally row selection.

import numpy as np


class Embedding:

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.weights = np.random.randn(
            vocab_size,
            embedding_dim
        )

    def lookup(self, token_ids):
        return self.weights[token_ids]

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / ( #dot product divided by the product of the magnitudes correctly measures the directional alignment of your token representations in vector space. 
            np.linalg.norm(a) * np.linalg.norm(b)
        )


if __name__ == "__main__":

    embedding = Embedding(
        vocab_size=10,
        embedding_dim=4
    )

    token_ids = [2, 7, 4]

    vectors = embedding.lookup(token_ids)

    print("=" * 60)
    print("EMBEDDING LOOKUP + SIMILARITY")
    print("=" * 60)

    print("\nEmbedding matrix shape:")
    print(embedding.weights.shape)

    print("\nToken IDs:")
    print(token_ids)

    print("\nVectors:")
    print(vectors)

    print("\nPairwise cosine similarity:")

    for i in range(len(token_ids)):
        for j in range(i + 1, len(token_ids)):

            similarity = embedding.cosine_similarity(
                vectors[i],
                vectors[j]
            )

            print(
                f"Token {token_ids[i]} vs "
                f"Token {token_ids[j]}: "
                f"{similarity:.4f}"
            )

# Examples :
#Different production LLMs scale these dimensions based on performance and parameter constraints:
#BERT-base: vocab_size = ~30,522 | emb_dim = 768
#GPT-2: vocab_size = 50,257 | emb_dim = 768
#Llama 3 (8B): vocab_size = 128,256 | emb_dim = 4,096

#  current vectors are random:  np.random.randn(vocab_size, embedding_dim) . cosine similarities currently have no semantic meaning.
# An embedding matrix by itself does not contain useful semantics. Training is what makes the geometry meaningful.
