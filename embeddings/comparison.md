## Experiment: Word2Vec Skip-gram — Full Softmax Baseline

### Configuration
- Implementation: NumPy from scratch
- Corpus: `the cat sat on the mat`
- Vocabulary size: 5
- Embedding dimension: 4
- Context window: 1
- Training pairs: 10
- Epochs: 100
- Learning rate: 0.1
- Training order: Shuffled every epoch
- Random seed: 42
- Output layer: Full softmax
- Loss: Cross-entropy

### Results
- Initial training loss: 1.609425
- Final training loss: 0.843993
- Training loss reduction: 47.56%
- Input embedding matrix shape: (5, 4)
- Output weight matrix shape: (4, 5)

### Selected Cosine Similarities
- cat ↔ on: 0.9994
- cat ↔ mat: 0.6530
- cat ↔ sat: -0.8048

### Observations
- Training loss decreased, indicating improved context-word prediction on the training pairs.
- The tiny corpus is insufficient for meaningful semantic evaluation.
- High cosine similarity does not necessarily indicate semantic equivalence.
- Full softmax computes scores across the entire vocabulary for every training example.
- This implementation is a learning baseline, not a production-quality embedding model.

### Limitations
- Extremely small corpus and vocabulary.
- No held-out validation set.
- No systematic hyperparameter tuning.
- No comparison with a production library yet.
- Similarity scores are not evidence of semantic generalization.

### Status
Skip-gram full-softmax baseline implemented and initial training behavior validated.

## Experiment: CBOW — Full Softmax Baseline

### Configuration
- Implementation: NumPy from scratch
- Corpus: `the cat sat on the mat`
- Vocabulary size: 5
- Embedding dimension: 4
- Context window: 1
- Training examples: 6
- Epochs: 100
- Learning rate: 0.1
- Random seed: 42
- Output layer: Full softmax
- Loss: Cross-entropy

### Results
- Initial loss: 1.603148
- final loss: 0.488989
- Loss reduction from epoch 1 to epoch 100: approximately 69.65%
- Input embedding matrix shape: (5, 4)
- Output weight matrix shape: (4, 5)

### Selected Cosine Similarities
- cat ↔ mat: 0.8261
- cat ↔ on: 0.5966
- cat ↔ sat: -0.3968

### Observations
- Training loss decreased substantially.
- CBOW predicts a target word from the mean of its surrounding context embeddings.
- Contexts near sentence boundaries contain fewer words.
- Similarity values are highly dependent on this tiny corpus and training setup.
- Full softmax calculates scores for every vocabulary word for each example.

### Limitations
- Extremely small corpus and vocabulary.
- No held-out validation set.
- No systematic hyperparameter tuning.
- No production-library comparison yet.
- Similarity scores do not establish semantic generalization.

### Status
CBOW dataset generation, full-softmax training, and cosine-similarity evaluation implemented and validated on the toy corpus.

## Experiment: Skip-gram with Negative Sampling

### Configuration
- Implementation: NumPy from scratch
- Corpus: `the cat sat on the mat`
- Vocabulary size: 5
- Embedding dimension: 4
- Context window: 1
- Positive training pairs: 10
- Negative samples per positive pair: 3
- Epochs: 100
- Learning rate: 0.1
- Random seed: 42
- Objective: Binary logistic loss for positive and negative pairs

### Results
- Epoch 1 loss: 2.7726
- Epoch 100 loss: 1.2888
- Loss reduction from epoch 1 to epoch 100: approximately 53.51%

### Selected Cosine Similarities
- cat ↔ mat: 0.8582
- cat ↔ on: 0.9956
- cat ↔ sat: -0.0943
- sat ↔ the: 0.8200

### Observations
- Training loss decreased substantially, with fluctuations across epochs.
- Negative sampling updates the center embedding and the selected positive and negative output vectors.
- Unlike full softmax, each training step does not calculate a normalized probability distribution over the entire vocabulary.
- Similarity values from this tiny corpus are not reliable evidence of semantic quality.

### Implementation Limitations
- Negative words are sampled uniformly, not from the unigram distribution raised to the 3/4 power.
- Sampling is without replacement.
- Training pairs are not shuffled each epoch.
- No held-out evaluation set.
- No fair performance benchmark against full softmax yet.

### Status
Core negative-sampling objective and manual gradient updates implemented and trained. Standard Word2Vec sampling improvements and production comparisons remain outstanding.

## Word2Vec Conclusions

### Implementations Completed
1. Skip-gram with full softmax
2. CBOW with full softmax
3. Skip-gram with negative sampling

### Conceptual Comparison

**Skip-gram**
- Uses a center word to predict surrounding context words.
- Produces a training example for each center-context relationship.
- Full softmax computes scores across the vocabulary for each prediction.

**CBOW**
- Averages the embeddings of surrounding context words.
- Predicts the center word from the aggregated context representation.
- Full softmax computes scores across the vocabulary for each prediction.

**Negative Sampling**
- Trains the model to score observed word pairs higher than sampled negative pairs.
- Avoids computing a full vocabulary-wide softmax distribution.
- Standard Word2Vec commonly samples negatives according to a unigram distribution raised to the 3/4 power.
- Our current implementation uses simplified uniform sampling and is not yet equivalent to the standard sampling procedure.

### Experimental Observations
- All three implementations demonstrated decreasing training loss on the toy corpus.
- The corpus `the cat sat on the mat` is useful for validating tensor shapes, training loops, and gradient updates.
- The corpus is too small to establish semantic quality or generalization.
- Cosine similarities varied across training objectives and initialization.
- Training losses from different objectives cannot be compared directly without accounting for their different definitions.

### Current Limitations
- Very small corpus and vocabulary.
- No held-out evaluation set.
- No controlled comparison using the same quality metric.
- No reliable throughput or memory benchmark yet.
- Negative sampling does not yet use the standard unigram^0.75 sampling distribution.
- No production-library comparison yet.

### Conclusions
- Skip-gram and CBOW differ in how they construct the prediction task.
- Full softmax explicitly scores the entire vocabulary.
- Negative sampling replaces full-vocabulary classification with a positive-versus-negative training objective.
- A lower training loss alone does not prove better embeddings.
- Meaningful comparisons require a larger shared corpus, controlled configurations, reproducible seeds, and common evaluation metrics.

## GloVe — Global Co-occurrence Embeddings

### Configuration
- Corpus: 10 short sentences about cats, dogs, food, and locations
- Vocabulary size: 13
- Context window: 2
- Embedding dimension: 4
- Training epochs: 100
- Optimizer: Stochastic Gradient Descent (SGD)
- Co-occurrence weighting: Distance-weighted counts, using `1 / distance`
- GloVe weighting parameters: `x_max = 100`, `alpha = 0.75`

### Co-occurrence Matrix Validation
- Matrix shape: `(13, 13)`
- Matrix sum: `152.0`
- Non-zero entries: `77`
- Matrix symmetric: `True`

### Objective and Training Results
- Initial objective loss: `4.84237008`
- Final objective loss: `0.40448898`
- Loss reduction: `91.65%`

Loss was recorded every 20 epochs, with values:
- Epoch 1: `3.95951034`
- Epoch 20: `0.71161599`
- Epoch 40: `0.54775807`
- Epoch 60: `0.48183843`
- Epoch 80: `0.43991799`
- Epoch 100: `0.40448898`

Note: the initial loss is measured before training, while epoch 1 is measured after the first epoch.

### Gradient Check
Analytical gradients were compared against central finite-difference estimates.

| Parameter | Analytical gradient | Numerical gradient | Absolute error |
|---|---:|---:|---:|
| Word vector | -0.0127468936 | -0.0127468938 | 1.585e-10 |
| Context vector | 0.0909276658 | 0.0909276658 | 2.812e-11 |
| Word bias | -0.5150989864 | -0.5150989866 | 1.679e-10 |

Result: all three selected gradient checks passed within the configured tolerance. This validates the tested derivatives, not every parameter or training update.

### Cosine Similarity Results
- `cat ↔ mat`: `0.1688`
- `cat ↔ sat`: `-0.6423`
- `cat ↔ the`: `-0.5947`
- `on ↔ the`: `-0.6063`

### Nearest Neighbors
- `cat`: `dog (0.7951)`, `food (0.6527)`, `near (0.6457)`
- `mat`: `floor (0.8513)`, `door (0.7536)`, `dog (0.5132)`
- `the`: `sat (0.7940)`, `window (0.3242)`, `ate (0.0318)`

### Interpretation and Limitations
1. The training objective decreased substantially, indicating that the model learned to fit the observed co-occurrence statistics.
2. The gradient check provides evidence that the tested analytical derivatives match numerical estimates.
3. Nearest-neighbor results are exploratory; the corpus is too small and artificial to establish robust semantic quality.
4. GloVe's objective differs from the full-softmax and negative-sampling objectives used in the Word2Vec experiments. Raw losses must not be compared directly across these methods.
5. This is an educational NumPy implementation, not a production-optimized GloVe trainer.

### Gradient Verification
Analytical gradients were compared against central finite-difference estimates with fixed negative samples.

| Parameter | Analytical gradient | Numerical gradient | Absolute error | Result |
|---|---:|---:|---:|---|
| `W_in(0, 0)` | `+6.6029251903e-03` | `+6.6029252066e-03` | `1.634e-11` | PASS |
| `W_in(0, 2)` | `-1.5173792608e-03` | `-1.5173792667e-03` | `5.865e-12` | PASS |
| `W_out(1, 1)` | `+5.2002255725e-03` | `+5.2002255835e-03` | `1.102e-11` | PASS |
| `W_out(2, 2)` | `+3.7520341904e-03` | `+3.7520341944e-03` | `4.044e-12` | PASS |
| `W_out(3, 4)` | `+4.7029401229e-03` | `+4.7029401262e-03` | `3.285e-12` | PASS |

**Result:** 5/5 sampled gradient checks passed. This validates the tested derivatives, not every parameter or all possible training updates.
