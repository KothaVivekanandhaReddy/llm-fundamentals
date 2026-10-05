An embedding is a way of representing real-world objects—like words, sentences, images, or audio—as a dense list of numbers (a vector) in a multi-dimensional space.

Computers cannot inherently understand the meaning of text or images, but they excel at processing numbers. By translating data into these mathematical coordinates, machine learning models can calculate relationships, allowing semantically similar concepts to sit close together in the vector space

embeddings are not just an abstract NLP concept.
They directly affect:
parameter count
RAM/VRAM
model size
inference memory
training cost

## Key Properties:
# Dimensionality Reduction:
 They compress massive, sparse data formats (like millions of one-hot encoded vocabulary words) into compact, continuous, and efficient numeric arrays.
# Semantic Meaning: 
Instead of just comparing literal characters, embeddings capture contextual relationships and concepts.

## Common Types
# Word Embeddings:
 Converts individual words into vectors. Older static algorithms like Google Word2Vec or Stanford GloVe gave each word a single fixed vector. Modern transformers (like BERT or GPT architectures) generate dynamic, contextual embeddings where the vector changes depending on the surrounding sentence structure.
# Sentence/Document Embeddings: 
Encodes full paragraphs or entire documents into a single vector, preserving the overarching topic or theme. 
# Multimodal Embeddings: 
Maps different data formats—such as an image and its text description—into the exact same vector space so an AI can find a visual photo based on a written prompt.

## Progress
Embedding Fundamentals :

Implemented basic embedding lookup using NumPy.

Concept:

Token IDs → Embedding Matrix → Row Selection → Dense Vectors

Implemented:
- Embedding matrix initialization
- Token ID lookup
- Batch token lookup
- Cosine similarity
- Pairwise similarity

Important:
The initial embedding matrix is randomly initialized, so the vectors do not contain learned semantic meaning yet.

### Embedding Geometry 

Implemented the mathematical operations used to compare embedding vectors.

Implemented:
- Dot product
- L2 norm / vector magnitude
- Euclidean distance
- Cosine similarity
- L2 normalization

Key observation:
Cosine similarity is invariant to vector magnitude and measures directional similarity.

Normalized vectors have unit length, allowing cosine similarity to be computed as the dot product of normalized vectors.