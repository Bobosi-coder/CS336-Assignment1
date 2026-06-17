import torch
from yaml import Token
from cs336_basics.pre_norm_transformer_blocks import softmax
from cs336_basics.tokenizer import Tokenizer

def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    max_value, _ = torch.max(logits, dim = -1, keepdim=True)
    scaled_logits = (logits - max_value) / temperature
    scaled_logits_exp = torch.exp(scaled_logits)
    scaled_logits_exp_sum = torch.sum(scaled_logits_exp, dim = -1, keepdim=True)
    return scaled_logits_exp / scaled_logits_exp_sum

def top_p_sampling(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_indices =  torch.sort(probs, dim = -1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

    sorted_indices_to_remove = cumulative_probs > p

    # 将剔除掩码向右平移一位。
    # 必须保证累加刚好超过 top_p 的那第一个词被保留下来, 第一位永远是 0（保留概率最大的词）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., : -1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim = -1,
        index = sorted_indices,
        src = sorted_indices_to_remove
    )

    cloned_probs = probs.clone()
    cloned_probs[indices_to_remove] = 0.0

    # 重新归一化：由于砍掉了一部分词，剩下词的概率之和小于 1.0，
    # 必须除以它们现在的总和，使其重新恢复为合法的概率空间
    cloned_probs = cloned_probs / (torch.sum(cloned_probs, dim=-1, keepdim=True) + 1e-10)

    return cloned_probs

def decoding(
        model: torch.nn.Module,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        context_length: int,
        temperature: float,
        top_p: float,
        eos_token_id: int = 256
) -> torch.Tensor:
    # 保持输入的克隆，后续自回归会将新生成的词源源不断地 append 进这个矩阵
    generated = prompt_tokens.clone()

    model.eval()

    batch_size = prompt_tokens.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt_tokens.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_seq = generated[:, -context_length:]
            outputs = model(input_seq)
            next_token_logits = outputs[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim = -1, keepdim=True)
            else:
                probs_q = temperature_scaling(next_token_logits, temperature)
                final_prob = top_p_sampling(probs_q, top_p)
                next_token = torch.multinomial(final_prob, num_samples=1)
            
            # next_token 的形状是 batch_size * 1
            next_token = torch.where(finished.unsqueeze(-1), eos_token_id, next_token)
            finished |= (next_token.squeeze(-1) == eos_token_id)

            generated = torch.cat([generated, next_token], dim = -1)

            if finished.all():
                break
    
    return generated