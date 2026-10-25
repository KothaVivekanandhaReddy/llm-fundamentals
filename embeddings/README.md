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

## Word2Vec Skip-Gram :

Implemented a Skip-Gram Word2Vec model from scratch using NumPy.

Pipeline:

Corpus
→ Tokenization
→ Vocabulary
→ Context-window pairs
→ Embedding lookup
→ Output projection
→ Softmax
→ Cross-entropy
→ Backpropagation
→ Gradient descent
→ Learned embeddings

Implemented:
- Skip-Gram training pair generation
- Input embedding matrix
- Output/context matrix
- Forward pass
- Softmax
- Cross-entropy loss
- Analytical gradients
- Backpropagation
- Gradient descent
- Multi-epoch training
- Cosine similarity between learned vectors

Experiment:

 corpus:
"The cat sat on the mat"

Training pairs:
10

Embedding dimension:
4

Loss:
1.6094 → 0.8507

Important observation:
The learned vectors encode distributional relationships from context. The tiny corpus is insufficient for meaningful general-purpose semantic embeddings.

## Word2Vec Negative Sampling 

Implemented Negative Sampling from scratch using NumPy.

Instead of calculating a softmax over the entire vocabulary, training uses:

- 1 positive context word
- Multiple negative context words
- Sigmoid binary classification
- Analytical gradients
- Gradient descent

Objective:

Positive pair:
center · positive → HIGH

Negative pair:
center · negative → LOW

Experiment:
- Vocabulary: 5
- Training pairs: 10
- Negative samples: 3
- Embedding dimension: 4
- Loss: 2.7726 → 1.2888

Important:
Negative sampling makes Skip-Gram training substantially more efficient for large vocabularies because each update only evaluates a small number of sampled words.

## Word2Vec CBOW

Implemented a Continuous Bag of Words (CBOW) model from scratch using NumPy.

**Pipeline:**

Corpus  
→ Tokenization  
→ Vocabulary  
→ Context-window generation  
→ Context embedding lookup  
→ Mean context embedding  
→ Output projection  
→ Softmax  
→ Cross-entropy  
→ Backpropagation  
→ Gradient descent  
→ Learned embeddings

**Implemented:**
- CBOW context-target pair generation
- Vocabulary construction
- Input embedding matrix
- Output weight matrix
- Context embedding lookup
- Mean pooling over context embeddings
- Forward pass
- Numerically stable softmax
- Cross-entropy loss
- Analytical gradients
- Backpropagation
- Gradient descent
- Multi-epoch training
- Deterministic initialization and shuffled training examples
- Cosine similarity and nearest-word lookup

**Experiment:**
- Corpus: `"the cat sat on the mat"`
- Vocabulary size: 5
- Training examples: 6
- Context window: 1
- Embedding dimension: 4
- Epochs: 100
- Learning rate: 0.1
- Initial single-example loss: 1.6031
- Epoch 1 average training loss: 1.6111
- Epoch 100 average training loss: 0.4890

**Important observation:**

CBOW predicts a target word from the mean of its surrounding context embeddings. Training loss decreased substantially, indicating that the model learned to predict the training targets more effectively.

The small corpus is insufficient to establish meaningful semantic quality or generalization.

---

## Word2Vec: Full Softmax vs. Negative Sampling

Implemented and explored two training objectives for Word2Vec using NumPy.

### Full Softmax

The model calculates scores across the entire vocabulary and applies softmax to produce a probability distribution over candidate words.

Implemented in:
- Skip-Gram with full softmax
- CBOW with full softmax

### Negative Sampling

The model learns to distinguish observed center-context pairs from sampled negative pairs.

Implemented:
- Positive and negative word-pair scoring
- Sigmoid-based binary classification
- Positive-pair loss
- Negative-pair loss
- Analytical gradients
- Backpropagation
- Gradient descent
- Multi-epoch training
- Learned embedding extraction
- Cosine similarity evaluation

**Experiment:**
- Vocabulary size: 5
- Positive training pairs: 10
- Negative samples per positive pair: 3
- Embedding dimension: 4
- Epochs: 100
- Initial epoch loss: 2.7726
- Final epoch loss: 1.2888

**Important observation:**

Negative sampling avoids computing a full vocabulary-wide softmax distribution for each training example. Its training objective instead uses one positive pair and a small number of negative samples.

Our current implementation uses simplified uniform negative sampling. It does not yet implement the standard Word2Vec unigram distribution raised to the 3/4 power.

The loss values from full softmax and negative sampling are not directly comparable because the objectives differ.

---

## Word2Vec — Key Learnings

Implemented three educational baselines:
1. Skip-Gram with full softmax
2. CBOW with full softmax
3. Skip-Gram with negative sampling

**Key conceptual differences:**
- Skip-Gram predicts surrounding context words from a center word.
- CBOW predicts a center word from the average of surrounding context embeddings.
- Negative sampling trains the model to distinguish observed word pairs from sampled negative pairs.

**Key observations:**
- Training loss decreased across all three experiments.
- Embeddings are learned through optimization rather than being meaningful at random initialization.
- Cosine similarity measures the directional relationship between vectors, but high similarity does not necessarily establish semantic equivalence.
- A tiny corpus is useful for validating implementation mechanics, not for evaluating general-purpose semantic representations.
- Training loss alone cannot establish embedding quality.
- Meaningful comparisons require controlled experiments, shared evaluation data, and appropriate quality and performance metrics.

**Current limitations:**
- Small training corpus and vocabulary.
- No held-out evaluation set