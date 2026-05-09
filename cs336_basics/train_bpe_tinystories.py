"""
This is for the solution of 
Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
"""


from cs336_basics.train_bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode
import json
import cProfile
import io
import os
import pstats
import time
import tracemalloc
from pathlib import Path

from cs336_basics.train_bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode

def bytes_to_printable_string(token : bytes) -> str:
    """
    Convert arbitrary byte token into GPT-2-style printable unicode string.
    This avoids UnicodeDecodeError for byte-level BPE tokens that are not valid UTF-8.
    """
    byte_encoder = gpt2_bytes_to_unicode()
    return "".join(byte_encoder[b] for b in token)


def serialize_vocab(vocab: dict[int, bytes],
                    output_path: str | os.PathLike) -> None:
    """
    Save vocab as GPT-2-style JSON:
        printable_token_string -> token_id
    """
    reverse_vocab = {
        bytes_to_printable_string(token_bytes): token_id
        for token_id, token_bytes in vocab.items()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reverse_vocab, f, ensure_ascii=False, indent=2)

def serialize_merge(
        merges: list[tuple[bytes, bytes]],
        output_path: str | os.PathLike
) -> None:
    """
    Save merges as one merge per line:
        token1 token2
    Tokens are converted to GPT-2-style printable unicode strings.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for token1, token2 in merges:
            token1_str = bytes_to_printable_string(token1)
            token2_str = bytes_to_printable_string(token2)

            f.write(f"{token1_str} {token2_str}\n")

def get_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    longest_token_id, longest_token_bytes = max(
        vocab.items(),
        key= lambda item: len(item[1])
    )
    return longest_token_id, longest_token_bytes

def run_training() -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 32_000
    special_tokens =['<|endoftext|>']

    vocab, merges = train_bpe(
        input_path= input_path,
        vocab_size = vocab_size,
        special_tokens=special_tokens
    )

    return vocab, merges


def main() -> None:
    output_dir = Path('artifact')
    output_dir.mkdir(exist_ok=True)

    profile_output_path = output_dir / 'cprofile_owt_train_bpe.txt'
    vocab_output_path = output_dir / 'owt_vocab.json'
    merge_output_path = output_dir / 'owt_merges.txt'

    profiler = cProfile.Profile()

    tracemalloc.start()
    start_time = time.perf_counter()

    profiler.enable()
    vocab, merges = run_training()
    profiler.disable()

    end_time = time.perf_counter()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_seconds = end_time - start_time

    serialize_vocab(vocab, vocab_output_path)
    serialize_merge(merges, merge_output_path)

    longest_token_id, longest_token_bytes = get_longest_token(vocab)
    longest_token_printable = bytes_to_printable_string(longest_token_bytes)

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream= stats_stream)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(50)

    profile_text = stats_stream.getvalue()

    with open(profile_output_path, 'w', encoding='utf-8') as f:
        f.write(profile_text)

    print("Training finished.")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    print(f"Tracemalloc current memory: {current_memory / 1024 / 1024:.2f} MB")
    print(f"Tracemalloc peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print()
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print()
    print(f"Longest token id: {longest_token_id}")
    print(f"Longest token byte length: {len(longest_token_bytes)}")
    print(f"Longest token printable form: {longest_token_printable!r}")
    print()
    print(f"Saved vocab to: {vocab_output_path}")
    print(f"Saved merges to: {merge_output_path}")
    print(f"Saved cProfile report to: {profile_output_path}")
    print()
    print("Top cProfile results by cumulative time:")
    print(profile_text)


if __name__ == "__main__":
    main()
    
