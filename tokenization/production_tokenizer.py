# Save this code exactly as: tokenization/production_tokenizer.py
import os
import tempfile
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, NFKC

class ProductionTokenizer:
    def __init__(self, vocab_size: int = 1500):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.normalizer = Sequence([NFKC()])
        self.tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "<|endoftext|>"]

    def train_on_corpus(self, lines: list[str]):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            initial_alphabet=ByteLevelPreTokenizer.alphabet()
        )
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
            for line in lines:
                tmp.write(line + "\n")
            tmp_path = tmp.name
        try:
            self.tokenizer.train([tmp_path], trainer)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
