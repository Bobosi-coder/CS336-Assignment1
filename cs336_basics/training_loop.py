from random import sample
from xml.etree.ElementInclude import default_loader

import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from torchgen import context
import os
import time
import typing
import wandb
import argparse

from cs336_basics.training_components import AdamW, cross_entropy_loss, gradient_clipping, learning_rate_schedule
from cs336_basics.transformer import TransformerLM

def get_batch(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str | None = None
        )-> tuple[torch.Tensor, torch.Tensor]:
    
    dataset_length = len(dataset)

    sample_begin_position = np.random.randint(0, dataset_length - context_length, 
                                              batch_size)
    
    sequences = [dataset[i : i + context_length + 1] for i in sample_begin_position]
    batch_tensor = torch.tensor(np.stack(sequences), device=device, dtype= torch.long)

    train_tensors = batch_tensor[:, :context_length]
    ans_tensors = batch_tensor[:, 1: ]

    return (train_tensors, ans_tensors)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    state ={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(state, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: nn.Module, optimizer: torch.optim.Optimizer) -> int:   
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    iteration = state_dict['iteration']

    return iteration


def training_loop():
    parser = argparse.ArgumentParser(description= "Transformer LM pre-training script")

    # model 本身的超参数
    parser.add_argument('--d_model', type=int, default=512, help = 'embedding 的维度大小')
    parser.add_argument('--num_heads', type = int, default=16, help='mha 的多头数量')
    parser.add_argument('--d_ff', type = int, default=1344, help='ffn 层扩大的向量维度') # 默认值为8/3 × 512 向上取整到 64 的倍数
    parser.add_argument('--num_layers', type = int, default= 4, help='attention block 层数')
    parser.add_argument('--context_length', type=int, default=256, help='最长序列长度')
    parser.add_argument('--vocab_size', type = int, default=10000, help='tokenizer 词表大小')
    parser.add_argument('--theta',type = float, default=10000.0, help='RoPE 的频率超参数')
    parser.add_argument('--use_rope', action='store_true', help = '启用旋转位置编码 (默认: 关闭，需输入后开始)')

    # AdamW 超参数
    parser.add_argument('--lr', type = float, default=3e-4, help='默认学习率')
    parser.add_argument('--beta_1', type = float, default= 0.9, help='AdamW 一阶 momentum 折扣因子')
    parser.add_argument('--beta_2', type = float, default = 0.999, help= 'AdamW 二阶 momentum 折扣因子')
    parser.add_argument('--eps', type = float, default=1e-8, help='除以二阶 momentum 时候加上的微小变量，避免除以0')
    parser.add_argument('--weight_decay', type = float, default=0.1, help='参数衰退权重')

    # 余弦退火
    parser.add_argument('--max_learning_rate', type = float, default = 3e-4, help='scheduler 能带达到的最大学习率')
    parser.add_argument('--min_learning_rate', type = float, default= 3e-5, help='scheduler 达到的最小学习率')
    parser.add_argument('--warmup_iters', type = int, default = 2000, help='warmup 迭代次数')
    parser.add_argument('--cosine_cycle_iters', type=int, default = 10000, help='余弦退火 迭代次数')

    # training
    parser.add_argument('--max_grad_norm', type = float, default=1.0, help='grad clipping 最大 norm')
    parser.add_argument('--batch_size', type = int, help='训练数据 batch 数量')
    parser.add_argument('--max_iters', type = int, help='最大训练次数迭代')
    parser.add_argument('--eval_interval', type = int, default=500, help ='模型评估记录训练次数')
    parser.add_argument('--save_interval', type = int, default=1000, help='每 save_interval 次保存一次')
    parser.add_argument('--resume_from', type = str, default=None, help='如果想继续从某个点开始训练，传入checkpoint的文件路径')

    # IO
    parser.add_argument('--train_path', type = str, help='训练数据路径')
    parser.add_argument('--val_path', type=str, help='验证数据集路径')
    parser.add_argument('--model_dtype', type = str, default='float32', help = "数据格式")

    # Project name
    parser.add_argument('--proj_name', type = str, help='项目名称')


    args = parser.parse_args()
    proj_name = args.proj_name

    # 判断 device 类型  
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 得到 dtype 类型
    model_dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16':
  torch.float16}
    model_dtype = model_dtype_map[args.model_dtype]

    # 设置随机 seed 
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(42)
    else:
        torch.manual_seed(42)
    np.random.seed(42)

    train_data = np.load(args.train_path, mmap_mode='r')
    val_data = np.load(args.val_path, mmap_mode='r') # 这里的dtype在使用tokenizer训练并保存好数据后再决定

    model = TransformerLM(
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff = args.d_ff,
            vocab_size = args.vocab_size,
            context_length=args.context_length,
            num_layers= args.num_layers,
            use_rope = args.use_rope,
            theta=args.theta,
            device = device,
            dtype = model_dtype
            #怎么处理 device 和 dtype？
        )
    model.to(device)

    opt = AdamW(model.parameters(), args.lr, 
                (args.beta_1, args.beta_2), 
                args.eps,args.weight_decay)

    if args.resume_from:
        it = load_checkpoint(args.resume_from, model, opt) + 1
    else:
        it = 1
        
    with wandb.init(project=proj_name, config = args) as run:
        batch_size = args.batch_size
        context_length = args.context_length
        max_iters = args.max_iters
        max_learning_rate = args.max_learning_rate
        min_learning_rate = args.min_learning_rate
        warmup_iters = args.warmup_iters
        cosine_cycle_iters = args.cosine_cycle_iters

        start_time = time.perf_counter()

        for it in range(it, max_iters + 1):
            train_seq, ground_truth_seq = get_batch(train_data, batch_size=batch_size, 
                                                context_length=context_length, device = device)
            train_seq = train_seq.to(device)
            ground_truth_seq = ground_truth_seq.to(device)

            opt.zero_grad()
            logits = model(train_seq)
            loss = cross_entropy_loss(logits, ground_truth_seq)

            loss.backward()
            gradient_clipping(model.parameters(),args.max_grad_norm)
            opt.param_groups[0]['lr'] = learning_rate_schedule(it, max_learning_rate,
                                                               min_learning_rate, warmup_iters,
                                                               cosine_cycle_iters)
            opt.step()

            if it % args.eval_interval == 0:
                with torch.no_grad():
                    model.eval()
                    val_seq, val_ground_truth_seq = get_batch(val_data, batch_size=batch_size,
                                                            context_length=context_length, device = device)
                    val_seq = val_seq.to(device)
                    val_ground_truth_seq = val_ground_truth_seq.to(device)

                    val_logits = model(val_seq)
                    val_loss = cross_entropy_loss(val_logits, val_ground_truth_seq)
                    
                    wall_clock_time = time.perf_counter() - start_time

                    tokens_processed = it * args.batch_size * args.context_length
                    tokens_per_second = tokens_processed / wall_clock_time

                    print(f'iteration = {it}, loss = {loss}, val_loss = {val_loss}')
                    run.log({'loss': loss.item(), 
                            'val_loss': val_loss.item(),
                            'step': it,
                            'lr': opt.param_groups[0]['lr'],
                            'wall_clock_time_sec':wall_clock_time,
                            "tokens_per_second": tokens_per_second,},
                            )
                    model.train()

            if it % args.save_interval == 0:
                os.makedirs('./checkpoint/', exist_ok=True)
                checkpoint_path = f'./checkpoint/ckpt_it_{it}.pt'
                save_checkpoint(model, opt, it, checkpoint_path)
                print('='* 10, f'The model has been saved at {it} iterations', '='*10)

