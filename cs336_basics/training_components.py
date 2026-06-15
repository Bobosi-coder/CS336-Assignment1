from typing import Callable

from networkx import second_order_centrality
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
from cs336_basics.pre_norm_transformer_blocks import *
from cs336_basics.transformer import *
import einops


def cross_entropy_loss(predicted_logits : torch.Tensor,
                       target : torch.Tensor) -> torch.Tensor:
    expanded_target = target.unsqueeze(-1) # target必须首先和predicted logits 对齐维度
    max_val, _ = torch.max(predicted_logits, dim=-1, keepdim=True)
    
    # 2. 全局原位平移，确保所有指数项的最大值死死卡在 e^0 = 1.0，绝不溢出
    shifted_logits = predicted_logits - max_val
    
    # 3. 计算稳定的对数分母
    log_exp_sum = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    
    # 4. 注意：必须从平移后的 shifted_logits 中去 gather 正确位置的得分
    targeted_shifted_logits = torch.gather(shifted_logits, dim=-1, index=expanded_target)
    
    # 5. 代数消去：(o_target - m) - log(sum(exp(o_j - m)))
    log_probs_target = targeted_shifted_logits - log_exp_sum
    
    # 6. 对所有前置的 Batch 维度一网打尽求全局平均
    return torch.mean(-log_probs_target)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-5):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr" : lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr * grad # Update weight tensor in-place. 
                state["t"] = t + 1 # Increment iteration number.
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr:float = 1e-3, betas:tuple[float, float] = (0.9, 0.999), eps = 1e-8,
                 weight_decay: float = 0.01 ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr" : lr, 'beta_1' : betas[0],
                    'beta_2': betas[1], 'eps' : eps,
                    'weight_decay' : weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 得到历史数据
                state  = self.state[p]
                first_order_momentum = state.get('momentum_1order', torch.zeros_like(p))
                second_order_momentum = state.get('momentum_2order', torch.zeros_like(p))
                t = state.get('t', 1)

                # 得到梯度，在学习率中乘以一阶和二阶 momentum 的标准化项
                grad = p.grad.data
                lr_corrected = lr * math.sqrt(1 - beta_2**t)/(1 - beta_1**t)
                
                # weight_decay 和梯度下降解耦
                p.data -= lr * weight_decay * p.data

                # 更新 momentum
                first_order_momentum = beta_1 * first_order_momentum + (1 - beta_1) * grad
                second_order_momentum = beta_2 * second_order_momentum + (1 - beta_2) * grad **2
                
                # 更新参数
                p.data = p.data - lr_corrected * first_order_momentum/(torch.sqrt(second_order_momentum) + eps)

                # 更新历史数据  
                state['momentum_1order'] = first_order_momentum
                state['momentum_2order'] = second_order_momentum
                state["t"] = t + 1

        return loss


def learning_rate_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int):
    
    assert warmup_iters != 0
    assert cosine_cycle_iters != warmup_iters


    if it < warmup_iters:
        return it/warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + (
                0.5 * ( 
                    1 + math.cos( (it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))
                    * (max_learning_rate - min_learning_rate))
        return lr
    else:
        return min_learning_rate

