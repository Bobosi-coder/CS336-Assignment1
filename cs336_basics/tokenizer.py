from collections.abc import Iterable
import json
import os
from typing import Iterator, Self
import regex
from tests.common import gpt2_bytes_to_unicode



class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens. 
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.reverse_vocab = {v : k for k, v in self.vocab.items()}

        if self.special_tokens:
            # 按照special token 长度降序排列，方便之后匹配先匹配到更长的
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

            # 将新遇到的 special token 加入到词表中
            for word in self.special_tokens:
                token = word.encode('utf-8')
                if token not in self.reverse_vocab:
                    token_id = len(self.vocab)
                    self.vocab[token_id] = token
                    self.reverse_vocab[token] = token_id
        
        self.merges_ranks = {pair : rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None) -> Self:
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary 
        and list of merges (in the same format that your BPE training code output) 
        and (optionally) a list of special tokens. 
        """
        # deal with vocab, convert gpt-2 style unicode to bytes string 
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            reverse_vocab_gpt2_format = json.load(f)

        vocab = {int(token_id) : cls.printable_string_to_bytes(token_str) 
                 for token_str , token_id in reverse_vocab_gpt2_format.items()}

        # deal with merges list
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token1_str, token2_str = line.split(" ")
                token1 = cls.printable_string_to_bytes(token1_str)
                token2 = cls.printable_string_to_bytes(token2_str)
                merges.append((token1, token2))
        
        return cls(vocab = vocab, merges = merges, special_tokens = special_tokens)

    
    @classmethod
    def printable_string_to_bytes(cls, string : str) -> bytes:
        """
        Convert GPT-2-style printable unicode stringar token into byte.
        """
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v : k for k, v in byte_encoder.items()}

        return bytes(byte_decoder[ch] for ch in string)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokenized_rel = []

        # 当用户没有指定任何special token
        if not self.special_tokens:
            segments = [text]
        else:
            pattern = "|".join(regex.escape(tok) for tok in self.special_tokens)
            segments = regex.split(f"({pattern})", text)
        
        for segment in segments:
            if segment == "":
                continue

            if self.special_tokens and segment in self.special_tokens:
                tokenized_rel.append(self.reverse_vocab[segment.encode('utf-8')])
            else:
                for match in regex.finditer(PAT, segment):
                    word = match.group()
                    word_bytes = word.encode('utf-8')
                    segment_token_id = self.bpe_merge(word_bytes)
                    tokenized_rel.extend(segment_token_id)
        return tokenized_rel

    
    def bpe_merge(self, word_bytes : bytes) -> list[int]:
        tokens = [bytes([b]) for b in word_bytes]

        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) -1 )]

            megerable_pairs = [
                pair for pair in pairs
                if pair in self.merges_ranks
            ]
            
            if not megerable_pairs:
                break

            best_pair = min(
                megerable_pairs,
                key = lambda pair : self.merges_ranks[pair]
            )

            merged_token = best_pair[0] + best_pair[1]

            new_tokens : list[bytes] = []
            i = 0
            while i < len(tokens):
                if (
                i < len(tokens) - 1
                and tokens[i] == best_pair[0]
                and tokens[i + 1] == best_pair[1]
            ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
        return [self.reverse_vocab[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable ofstrings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files 
        that we cannot directly load into memory.
        这里默认原始文本中每个line的结束末尾都是安全边界。不存在夸unicode code point
        """
        for line in iterable:
            yield from self.encode(line)        



    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        tokens_rel = [self.vocab[id] for id in ids]
        
        return b"".join(tokens_rel).decode('utf-8', errors= 'replace')
