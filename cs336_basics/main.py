"""
CS336 Assignment 1 — 统一入口 main.py

通过 YAML 配置文件 + --mode 驱动整个 pipeline，所有超参数写在配置里方便调参。

用法示例:
    python -m cs336_basics.main --mode train_tokenizer --config configs/tinystories_base.yaml
    python -m cs336_basics.main --mode encode_data      --config configs/tinystories_base.yaml
    python -m cs336_basics.main --mode train_lm         --config configs/tinystories_base.yaml
    python -m cs336_basics.main --mode generate         --config configs/tinystories_base.yaml
    python -m cs336_basics.main --mode all              --config configs/tinystories_base.yaml

支持的 mode:
    - train_tokenizer: 训练 BPE，保存 vocab.json / merges.txt
    - encode_data:     加载已有 tokenizer，把 train/val txt 编码成 .npy
    - train_lm:        加载 .npy token 数据，启动 training loop
    - generate:        加载 tokenizer + checkpoint，生成文本
    - all:             train_tokenizer -> encode_data -> train_lm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from cs336_basics.train_bpe import train_bpe
from cs336_basics.train_bpe_tinystories import serialize_vocab, serialize_merge
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.training_loop import training_loop
from cs336_basics.generating_text import decoding   


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #

TORCH_DTYPE_MAP = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}

NP_DTYPE_MAP = {
    'int16': np.int16,
    'uint16': np.uint16,
    'int32': np.int32,
    'int64': np.int64,
}


def load_config(config_path: str) -> argparse.Namespace:
    """读取 YAML 配置，返回一个支持 args.xxx 属性访问的 Namespace。

    用 Namespace（而不是 dict）是为了让原本接收 argparse `args` 的
    training_loop 不用任何改动就能继续工作。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    return argparse.Namespace(**cfg_dict)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def find_latest_checkpoint(checkpoint_dir: str) -> Path | None:
    """在 checkpoint 目录里找迭代次数最大的 ckpt_it_*.pt。"""
    ckpts = list(Path(checkpoint_dir).glob('ckpt_it_*.pt'))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split('_')[-1]))


# --------------------------------------------------------------------------- #
# Mode: train_tokenizer
# --------------------------------------------------------------------------- #

def run_train_tokenizer(args: argparse.Namespace) -> None:
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)

    print(f'[train_tokenizer] training BPE on {args.tokenizer_train_path} '
          f'(vocab_size={args.vocab_size}) ...')

    vocab, merges = train_bpe(
        input_path=args.tokenizer_train_path,
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token],
    )

    serialize_vocab(vocab, args.vocab_path)
    serialize_merge(merges, args.merges_path)

    print(f'[train_tokenizer] vocab  -> {args.vocab_path} (size={len(vocab)})')
    print(f'[train_tokenizer] merges -> {args.merges_path} (n={len(merges)})')


# --------------------------------------------------------------------------- #
# Mode: encode_data
# --------------------------------------------------------------------------- #

def run_encode_data(args: argparse.Namespace) -> None:
    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=[args.special_token],
    )

    token_dtype = NP_DTYPE_MAP[args.token_dtype]

    jobs = [
        ('train', args.raw_train_path, args.train_path),
        ('val',   args.raw_val_path,   args.val_path),
    ]

    for split, raw_path, out_path in jobs:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        print(f'[encode_data] encoding {split}: {raw_path} -> {out_path}')

        # 注意: encode_iterable 接收的是「行的可迭代对象」（文件句柄），
        # 而不是路径字符串。这里打开文件句柄传进去。
        # np.fromiter 边产生边写入，避免先建一个超大 Python list 再转换。
        with open(raw_path, 'r', encoding='utf-8') as f:
            tokens = np.fromiter(
                tokenizer.encode_iterable(f),
                dtype=token_dtype,
            )

        np.save(out_path, tokens)
        print(f'[encode_data]   {split}: {tokens.shape[0]} tokens saved.')


# --------------------------------------------------------------------------- #
# Mode: train_lm
# --------------------------------------------------------------------------- #

def run_train_lm(args: argparse.Namespace) -> None:
    # training_loop 已经包办了 device/dtype 选择、模型/优化器构建、
    # checkpoint 续训、wandb 记录等全部逻辑，这里直接把配置喂进去即可。
    training_loop(args)


# --------------------------------------------------------------------------- #
# Mode: generate
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_generate(args: argparse.Namespace) -> None:
    device = resolve_device()
    dtype = TORCH_DTYPE_MAP[args.model_dtype]

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=[args.special_token],
    )

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        use_rope=args.use_rope,
        theta=args.theta,
        device=device,
        dtype=dtype,
    )
    model.to(device)

    # 选 checkpoint：优先 resume_from，否则取目录里迭代数最大的那个。
    ckpt_path = args.resume_from or find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f'找不到 checkpoint：resume_from 为空，且 {args.checkpoint_dir} 下没有 ckpt_it_*.pt'
        )
    print(f'[generate] loading checkpoint: {ckpt_path}')
    # 生成阶段不需要 optimizer 状态，直接加载 model 权重即可。
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    # eos id：优先用 tokenizer 里 special token 的真实 id，配置里的 eos_token_id 兜底。
    eos_id = tokenizer.reverse_vocab.get(
        args.special_token.encode('utf-8'),
        getattr(args, 'eos_token_id', 256),
    )

    # prompt 编码成 (1, T) 的 batch，喂给 decoding。
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = decoding(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )

    # decoding 在命中 eos 后会把 eos 也 append 进序列（之后还会向右填充 eos）。
    # 解码前截到第一个 eos 之前，否则会把 <|endoftext|> 字面量打印出来。
    out_ids = generated[0].tolist()
    if eos_id in out_ids:
        out_ids = out_ids[:out_ids.index(eos_id)]

    text = tokenizer.decode(out_ids)
    print('=' * 30, 'GENERATION', '=' * 30)
    print(text)
    print('=' * 72)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

# 每个 mode 对应一串按顺序执行的步骤；all 即把前三步串起来。
MODE_DISPATCH = {
    'train_tokenizer': [run_train_tokenizer],
    'encode_data':     [run_encode_data],
    'train_lm':        [run_train_lm],
    'generate':        [run_generate],
    'all':             [run_train_tokenizer, run_encode_data, run_train_lm],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='CS336 Assignment 1 — unified entry (config-driven).'
    )
    parser.add_argument(
        '--mode', type=str, required=True, choices=list(MODE_DISPATCH.keys()),
        help='要执行的阶段',
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='YAML 配置文件路径，例如 configs/tinystories_base.yaml',
    )
    cli = parser.parse_args()

    args = load_config(cli.config)

    print('=' * 10,
          f'CS336 assignment1 by Dawei Sun | mode = {cli.mode}',
          '=' * 10)

    for step in MODE_DISPATCH[cli.mode]:
        step(args)


if __name__ == '__main__':
    main()