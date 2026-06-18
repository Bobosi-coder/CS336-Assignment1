from ast import arg
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
    state_dict = torch.load(src, map_location=model.device)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    iteration = state_dict['iteration']

    return iteration


def training_loop(args):
    
    proj_name = args.proj_name

    # 判断 device 类型  
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('=' * 10,f'The current device is {device}','=' * 10)

    # 得到 dtype 类型
    model_dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16':
  torch.float16}
    model_dtype = model_dtype_map[args.model_dtype]

    print('=' * 10,f'The current dtype of model is {model_dtype}',
          '=' * 10)

    # 设置随机 seed 
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = np.load(args.train_path, mmap_mode='r')
    val_data = np.load(args.val_path, mmap_mode='r') 

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
        )
    model.to(device)

    opt = AdamW(model.parameters(), args.lr, 
                (args.beta_1, args.beta_2), 
                args.eps,args.weight_decay)

    if args.resume_from:
        it = load_checkpoint(args.resume_from, model, opt) + 1
    else:
        it = 1
        
    with wandb.init(project=proj_name, name = args.run_name, config = args) as run:
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
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                checkpoint_path = args.checkpoint_dir + f'/ckpt_it_{it}.pt'
                save_checkpoint(model, opt, it, checkpoint_path)
                print('='* 10, f'The model has been saved at {it} iterations', '='*10)

