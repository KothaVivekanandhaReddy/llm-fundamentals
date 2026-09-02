Tokenization is the process of breaking raw, unstructured text down into smaller, manageable chunks called tokens
These individual tokens are mapped to unique numerical IDs.

## TYPES

1. word level : Whole individual words.      simple; preserves full word meaning.      Out of vocabulary 
ex :word like "Geschwindigkeitsbegrenzung"? Word-level requires a massive vocabulary to cover every word in every language. Miss a word and you get the dreaded [UNK] token -- the model's way of saying "I have no idea what this is." English alone has over a million word forms. Add code, URLs, scientific notation, and 100 other languages and you need an infinite vocabulary.

2. char level : Every single letter,symbol,and space.  ex : ["U", "n", "h", "a", "p", "..."]
Character-level tokenization goes the other direction. "hello" becomes ["h", "e", "l", "l", "o"]. Vocabulary is tiny (a few hundred characters). No unknown tokens ever. But sequences become extremely long. A sentence that would be 10 word-level tokens becomes 50 character-level tokens. The model must learn that "t", "h", "e" together mean "the" -- burning attention capacity on something a human learns at age three.

3. subword level : Meaningful word fragments, prefixes, and suffixes.   ex : ["Un", "happi", "ness"]
Subword tokenization finds the sweet spot. Common words stay whole: "the" is one token. Rare words decompose into meaningful pieces: "unhappiness" becomes ["un", "happi", "ness"]. Vocabulary stays manageable (30K to 128K tokens). Sequences stay short. Unknown tokens essentially disappear because any word can be built from subword pieces.


## Modern Subword Tokenization Algorithms
1. Byte Pair Encoding (BPE) :
BPE is a greedy compression algorithm repurposed for tokenization. The idea is simple enough to fit on an index card.
Start with individual characters. Count every adjacent pair in the training corpus. Merge the most frequent pair into a new token. Repeat until you reach your target vocabulary size. 
Used by: OpenAI's GPT Models, Meta's LLaMA, and RoBERTa.

2. Byte-level BPE (BBPE) (GPT-2, GPT-3, GPT-4):
Standard BPE operates on Unicode characters. Byte-level BPE operates on raw bytes (0-255). This gives you a base vocabulary of exactly 256, handles any language or encoding, and never produces an unknown token.

GPT-2 introduced this approach. The base vocabulary covers every possible byte. BPE merges build on top of that. OpenAI's tiktoken library implements byte-level BPE with these vocabulary sizes:

GPT-2: 50,257 tokens
GPT-3.5/GPT-4: ~100,256 tokens 
GPT-4o: 200,019 tokens 
Used by: Modern foundational language models to achieve seamless native multilingual support.

3. WordPiece (BERT) : 
Very similar to BPE, but instead of choosing the absolute most frequent pair to merge, it calculates a mathematical likelihood score. It chooses the pair that maximizes the statistical probability of the training data, prioritizing words that are highly expected to sit next to each other.
BPE asks: "Which pair appears most often?" WordPiece asks: "Which pair appears together more often than you would expect by chance?

WordPiece also uses a "##" prefix for continuation subwords:
"unhappiness" -> ["un", "##happi", "##ness"]
"embedding"   -> ["em", "##bed", "##ding"]
The "##" prefix tells you this piece continues a previous token. BERT uses WordPiece with a vocabulary of 30,522 tokens. Every BERT variant -- DistilBERT, RoBERTa's tokenizer is actually BPE, but BERT itself is WordPiece.
Used by: Google’s BERT and DistilBERT.

4. SentencePiece (Llama, T5)
SentencePiece treats the input as a raw stream of Unicode characters, including whitespace. No pre-tokenization step. No language-specific rules about word boundaries. This makes it genuinely language-agnostic -- it works on Chinese, Japanese, Thai, and other languages where spaces do not separate words.

SentencePiece supports two algorithms:
BPE mode: same merge logic as standard BPE, applied to raw character sequences
Unigram mode: starts with a large vocabulary and iteratively removes tokens that least affect the overall likelihood. The reverse of BPE -- prune instead of merge.
Llama 2 uses SentencePiece BPE with a vocabulary of 32,000 tokens. T5 uses SentencePiece Unigram with 32,000 tokens. Note: Llama 3 switched to a tiktoken-based byte-level BPE tokenizer with 128,256 tokens.


5. Unigram  :
Unlike BPE or WordPiece which start small and build up, Unigram starts with a massive vocabulary of complete words and common sentences, then systematically prunes away the least useful or least probable tokens until it reaches the target vocabulary size.Used by: T5 (Text-to-Text Transfer Transformer) and SentencePiece implementations.

## Vocabulary Size Tradeoffs
This is a real engineering decision with measurable consequences.
graph LR
    subgraph Small["Small Vocab (32K)\ne.g., BERT, T5"]
        S1["More tokens per text"]
        S2["Longer sequences"]
        S3["Smaller embedding matrix"]
        S4["Better rare-word handling"]
    end
    subgraph Large["Large Vocab (128K+)\ne.g., Llama 3, GPT-4o"]
        L1["Fewer tokens per text"]
        L2["Shorter sequences"]
        L3["Larger embedding matrix"]
        L4["Faster inference"]
    end

Concrete numbers. For a 128K vocabulary with 4,096-dimensional embeddings, the embedding matrix alone is 128,000 x 4,096 = 524 million parameters. For a 32K vocabulary, it is 131 million parameters. That is a 400M parameter difference from the tokenizer choice alone.

But larger vocabularies compress text more aggressively. The same English paragraph that takes 100 tokens with a 32K vocabulary might take 70 tokens with a 128K vocabulary. That means 30% fewer forward passes during generation. For a model serving millions of requests, that is a direct reduction in compute cost.

The trend is clear: vocabulary sizes are growing. GPT-2 used 50,257. GPT-4 uses ~100K. Llama 3 uses 128K. GPT-4o uses 200K.

Model	Vocab Size	Tokenizer Type	Avg Tokens per English Word
BERT	30,522	WordPiece	~1.4
GPT-2	50,257	Byte-level BPE	~1.3
Llama 2	32,000	SentencePiece BPE	~1.4
GPT-4	~100,256	Byte-level BPE	~1.2
Llama 3	128,256	Byte-level BPE (tiktoken)	~1.1
GPT-4o	200,019	Byte-level BPE	~1.0

The Multilingual Tax
Tokenizers trained primarily on English are brutal to other languages. Korean text in GPT-2's tokenizer averages 2-3 tokens per word. Chinese can be worse. This means a Korean user effectively has a context window that is half the size of an English user's -- paying the same price for less information density.

This is why Llama 3 quadrupled its vocabulary from 32K to 128K. More tokens dedicated to non-English scripts means fairer compression across languages.


