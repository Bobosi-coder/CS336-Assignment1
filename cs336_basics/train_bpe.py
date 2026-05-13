import os
import regex as re
from multiprocessing import Pool
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tqdm import tqdm


def init_vocab(
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], set[int]]:
    """
    初始化vocab, vocab 是从token id 到 bytes 的对应关系
    一开始只有 0-255 字节，并且每个字节都和 id 直接映射

    Return:
        返回vocab 的 dict 和 special token ids 的set
    """

    vocab = {i: bytes([i]) for i in range(256)}
    special_tokens_ids = set()

    if special_tokens is not None:
        for token in special_tokens:
            token_id = len(vocab)
            vocab[token_id] = token.encode("utf-8")
            special_tokens_ids.add(token_id)
    return vocab, special_tokens_ids


# Pre-tokenization
def pre_tokenize(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_token = special_tokens[0]

    cpu_count = os.cpu_count() or 1
    num_workers = max(1, min(8, cpu_count - 1))  # 减少一个核心给系统，防止卡死

    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, num_workers, special_token.encode("utf-8")
        )

    # multiprocessing, create jobs parameter for sub-process
    jobs = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        if start == end:
            continue
        jobs.append((input_path, start, end, PAT, special_tokens))

    words_count = Counter()
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(pre_tokenize_chunk, jobs):
            words_count.update(result)

    return dict(words_count)


def pre_tokenize_chunk(job) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, regex_pattern, special_tokens = job

    # split the chunk with the special token
    words_count_sub = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")  # 先转为字符再匹配

        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments_no_special_token = re.split(pattern, chunk)

        for segment in segments_no_special_token:
            for match in re.finditer(regex_pattern, segment):
                word = match.group()
                word_in_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                words_count_sub[word_in_bytes] += 1

    return dict(words_count_sub)


def bpe_state(
    words_count: dict[tuple[bytes, ...], int],
) -> tuple[
    list[tuple[list[bytes], int]],
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], set[int]],
]:
    """
    构造倒排索引、words 表和 pair_count
    words 表是将 words_count 字典转为 list， 从而实现之后能够按照 idx 查询结果
    """
    words = []
    pair_counts = Counter()
    pair_to_word_ids = (
        {}
    )  # 倒排索引， (bytes, bytes) -> (words_idx_1, words_idx_2, ...)

    for idx, (word_bytes, value) in enumerate(words_count.items()):
        words.append((list(word_bytes), value))

        i = 0
        while i < (len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            pair_counts[pair] += value
            if pair not in pair_to_word_ids:
                pair_to_word_ids[pair] = set()
            pair_to_word_ids[pair].add(idx)

            i += 1

    return words, dict(pair_counts), pair_to_word_ids


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab, special_tokens_ids = init_vocab(special_tokens)

    words_count = pre_tokenize(input_path, special_tokens)

    words, pair_counts, pair_to_word_ids = bpe_state(words_count)

    loop_times = vocab_size - len(vocab)
    merges = []
    for pos in tqdm(range(loop_times), desc="Training process"):
        max_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        token_id = len(vocab)
        vocab[token_id] = max_pair[0] + max_pair[1]

        merges.append(max_pair)

        # 增量改写 state 阶段得到的结果
        words_idxs = list(pair_to_word_ids[max_pair])

        for idx in words_idxs:
            word_entry = words[
                idx
            ]  # word_entry 是每一条 ([word_byte_1, word_byte_2, ...], word出现次数)
            word_bytes, value = word_entry

            # 先将新合并的 pair 之前的 word 里所有的 pair 的计数都减去相应的值
            pos = 0
            while pos < len(word_bytes) - 1:
                pair = (word_bytes[pos], word_bytes[pos + 1])
                pair_counts[pair] -= value

                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                if pair in pair_to_word_ids:
                    pair_to_word_ids[pair].discard(idx)
                    if not pair_to_word_ids[pair]:
                        del pair_to_word_ids[pair]
                pos += 1
            # 开始合并新的 pair
            pos = 0
            new_word_bytes = []
            while pos < len(word_bytes):
                if (
                    pos < len(word_bytes) - 1
                    and word_bytes[pos] == max_pair[0]
                    and word_bytes[pos + 1] == max_pair[1]
                ):
                    new_word_bytes.append(max_pair[0] + max_pair[1])
                    pos += 2
                else:
                    new_word_bytes.append(word_bytes[pos])
                    pos += 1
            words[idx] = (
                new_word_bytes,
                value,
            )  # 更换 words 表中的单词 bytes， 将新的pair合并到一起

            # 将新合并的pair所在的 word 里所有 pair 进行重新计数
            j = 0
            while j < len(new_word_bytes) - 1:
                pair = (new_word_bytes[j], new_word_bytes[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + value
                if pair not in pair_to_word_ids:
                    pair_to_word_ids[pair] = set()
                pair_to_word_ids[pair].add(idx)

                j += 1

    return vocab, merges
