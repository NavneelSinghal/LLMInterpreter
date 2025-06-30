import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from torch.nn import functional as F
import einops

class KVCache:
    def __init__(self, batch_size: int, num_heads: int, head_dim: int,
                 max_seq_len: int, device: torch.device, dtype: torch.dtype):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.k_cache = torch.zeros(
            (batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )
        
        self.current_seq_len = 0

    def append(self, current_k: torch.Tensor, current_v: torch.Tensor):
        num_new_tokens = current_k.shape[2]
        if num_new_tokens == 0:
            return

        assert current_k.shape[0] == self.batch_size, "Batch size mismatch during append."
        assert current_k.shape[1] == self.num_heads, "Number of heads mismatch during append."
        assert current_k.shape[3] == self.head_dim, "Head dimension mismatch during append."
        assert current_k.device == self.device, f"Device mismatch: Cache is on {self.device}, current_k is on {current_k.device}."
        assert current_k.dtype == self.dtype, f"Dtype mismatch: Cache is {self.dtype}, current_k is {current_k.dtype}."
        assert current_v.shape == current_k.shape, "Shapes of current_k and current_v must match."

        if self.current_seq_len + num_new_tokens > self.max_seq_len:
            raise ValueError(
                f"KV Cache full. Max pre-allocated length is {self.max_seq_len}, "
                f"current stored length is {self.current_seq_len}, trying to add {num_new_tokens} tokens."
                "Consider increasing max_seq_len for KVCache."
            )

        start_idx = self.current_seq_len
        end_idx = self.current_seq_len + num_new_tokens
        
        self.k_cache[:, :, start_idx:end_idx, :] = current_k
        self.v_cache[:, :, start_idx:end_idx, :] = current_v
        self.current_seq_len = end_idx

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k_cache[:, :, :self.current_seq_len, :],
            self.v_cache[:, :, :self.current_seq_len, :]
        )

    def get_current_length(self) -> int:
        return self.current_seq_len

    def truncate_suffix(self, new_target_length: int):
        if new_target_length < 0:
            new_target_length = 0
        self.current_seq_len = min(new_target_length, self.current_seq_len)

    def reset(self):
        self.current_seq_len = 0
    
def norm(x):
    return F.rms_norm(x, (x.size(-1),), eps=1e-6).type_as(x)


class RoPE(nn.Module):
    def __init__(self, d_head, rope_base, max_len):
        super().__init__()
        assert d_head % 4 == 0, "d_head must be divisible by 4 for the original RoPE structure"
        self.d_head = d_head
        self.d_head_half = d_head // 2

        inv_freq_part1 = rope_base ** (-torch.arange(0, d_head // 4, dtype=torch.float32) / (d_head // 4))
        inv_freq_part2 = torch.zeros(d_head // 4, dtype=torch.float32)
        inv_freq = torch.cat((inv_freq_part1, inv_freq_part2), dim=0)

        t = torch.arange(max_len, dtype=torch.float32)
        rope_theta = torch.einsum('i, j -> i j', t, inv_freq) 
        
        rope_cos = torch.cos(rope_theta)
        rope_sin = torch.sin(rope_theta)
        
        self.register_buffer('rope_cos', rope_cos.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('rope_sin', rope_sin.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x, offset: int = 0):
        B, n_heads, T_curr, d_head_val = x.size()
        assert d_head_val == self.d_head
        
        x1 = x[..., :self.d_head_half]
        x2 = x[..., self.d_head_half:]
        
        cos = self.rope_cos[:, :, offset:offset + T_curr, :]
        sin = self.rope_sin[:, :, offset:offset + T_curr, :]
        
        y1 = cos * x1 - sin * x2
        y2 = sin * x1 + cos * x2
        
        return torch.cat((y1, y2), dim=-1).type_as(x)

class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, rope_base: float, max_len: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, dtype=torch.bfloat16)
        self.rope = RoPE(self.head_dim, rope_base, max_len)

    def forward(self, 
                x: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                pos_offset: int = 0
               ) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        current_q, current_k, current_v = einops.rearrange(qkv,
                                                           'b t (three head dim) -> three b head t dim',
                                                           three=3, head=self.num_heads, dim=self.head_dim)
        current_q, current_k = self.rope(current_q, offset=pos_offset), self.rope(current_k, offset=pos_offset)

        past_kv_seq_len = 0
        is_using_cache_with_history = False
        if kv_cache is not None:
            past_kv_seq_len = kv_cache.get_current_length()
            kv_cache.append(current_k, current_v)
            effective_k, effective_v = kv_cache.get_kv()
            is_using_cache_with_history = (past_kv_seq_len > 0)
        else:
            effective_k = current_k
            effective_v = current_v
        
        q_seq_len = current_q.shape[2]
        kv_seq_len = effective_k.shape[2]

        attn_mask_for_sdpa = None
        is_causal_for_sdpa = False

        if not is_using_cache_with_history:
            if q_seq_len > 1:
                is_causal_for_sdpa = True
        else:
            is_causal_for_sdpa = False
            if q_seq_len > 1:
                attn_mask_for_sdpa = torch.tril(
                    torch.ones(q_seq_len, kv_seq_len, device=current_q.device, dtype=torch.bool),
                    diagonal=past_kv_seq_len
                )
        if q_seq_len == 0:
            context = torch.zeros_like(current_q)
        else:
            context = F.scaled_dot_product_attention(
                current_q, effective_k, effective_v,
                attn_mask=attn_mask_for_sdpa,
                is_causal=is_causal_for_sdpa
            )

        context = context.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(context)

        return output

class Block(nn.Module):
    def __init__(self, d_model, n_heads, rope_base, max_len):
        super().__init__()
        self.attn = Attention(d_model, n_heads, rope_base, max_len)
        self.linear1 = nn.Linear(d_model, 4 * d_model, bias=False, dtype=torch.bfloat16)
        self.linear2 = nn.Linear(4 * d_model, d_model, bias=False, dtype=torch.bfloat16)

    def forward(self,
                x: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                pos_offset: int = 0):
        normed_x = norm(x)
        attn_output = self.attn(normed_x, kv_cache, pos_offset=pos_offset)
        x = x + attn_output
        normed_x = norm(x)
        mlp_out = self.linear2(F.relu(self.linear1(normed_x)).square())
        x = x + mlp_out
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, rope_base=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=torch.bfloat16)
        self.transformer_blocks = nn.ModuleList([
            Block(d_model, n_heads, rope_base, max_len) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=torch.bfloat16)

    def forward(self,
                x_indices: torch.Tensor,
                kv_cache: Optional[list[Optional[KVCache]]] = None,
                pos_offset: int = 0):

        x_emb = self.embedding(x_indices)

        if kv_cache is None:
            kv_cache = [None for i in range(len(self.transformer_blocks))]
        for i, block in enumerate(self.transformer_blocks):
            x_emb = block(x_emb, kv_cache=kv_cache[i], pos_offset=pos_offset)

        final_norm_out = norm(x_emb)
        logits = self.lm_head(final_norm_out)
        logits = logits.to(torch.float32)

        return logits

    @torch.no_grad()
    def generate_naive(self, 
                 prompt_tokens: torch.Tensor,
                 num_tokens_to_generate: int, 
                 temperature: float = 1.0, 
                 top_k: Optional[int] = None):
        
        self.eval()
        B, T_prompt = prompt_tokens.size()
        device = prompt_tokens.device

        if B == 0:
            return torch.empty((0, T_prompt + num_tokens_to_generate), dtype=torch.long, device=device)

        if T_prompt == 0 and num_tokens_to_generate > 0:
            raise ValueError(
                "Cannot start generation from an empty prompt (T_prompt=0). "
                "Provide a start token in prompt_tokens."
            )
        
        if num_tokens_to_generate == 0:
            return prompt_tokens

        all_tokens_list = [prompt_tokens]
        
        next_token_logits = None

        if T_prompt > 0:
            logits_prompt = self.forward(
                prompt_tokens, 
                pos_offset=0
            )
            next_token_logits = logits_prompt[:, -1, :] 

        input_for_fwd = prompt_tokens

        for i in range(num_tokens_to_generate):
            current_total_seq_len = prompt_tokens.size(1) + i 

            if current_total_seq_len >= self.max_len:
                print(f"Warning: Sequence length {current_total_seq_len} trying to exceed model max_len {self.max_len}. Stopping generation.")
                break
            
            if next_token_logits is None:
                print(f"Error: next_token_logits is None at generation step {i}. This indicates an unhandled start case.")
                break 

            if temperature == 0.0:
                newly_sampled_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(scaled_logits, dim=-1)
                newly_sampled_token = torch.multinomial(probs, num_samples=1)
            
            all_tokens_list.append(newly_sampled_token)
            
            if i == num_tokens_to_generate - 1:
                break

            input_for_fwd = torch.cat((input_for_fwd, newly_sampled_token), dim=-1)

            logits_step = self.forward(
                input_for_fwd,
                pos_offset=0
            )
            next_token_logits = logits_step[:, -1, :]
            
        final_generated_tokens = torch.cat(all_tokens_list, dim=1)
        return final_generated_tokens

    @torch.no_grad()
    def generate(self, 
                 prompt_tokens: torch.Tensor,
                 num_tokens_to_generate: int, 
                 temperature: float = 1.0, 
                 top_k: Optional[int] = None):
        
        self.eval()
        B, T_prompt = prompt_tokens.size()
        device = prompt_tokens.device
        dtype = torch.bfloat16

        if B == 0:
            return torch.empty((0, T_prompt + num_tokens_to_generate), dtype=torch.long, device=device)

        if T_prompt == 0 and num_tokens_to_generate > 0:
            raise ValueError(
                "Cannot start generation from an empty prompt (T_prompt=0) with this KV cache implementation. "
                "Provide at least one start token in prompt_tokens."
            )
        
        if num_tokens_to_generate == 0:
            return prompt_tokens

        kv_caches_list = [
            KVCache(
                batch_size=B,
                num_heads=self.n_heads,
                head_dim=self.d_model // self.n_heads,
                max_seq_len=self.max_len,
                device=device,
                dtype=dtype
            ) for _ in range(self.n_layers)
        ]

        all_tokens_list = [prompt_tokens]
        
        next_token_input = prompt_tokens
        current_pos_offset = 0

        if T_prompt > 0:
            logits_prompt = self.forward(
                prompt_tokens, 
                kv_cache=kv_caches_list,
                pos_offset=0
            )
            next_token_logits = logits_prompt[:, -1, :] 
            current_pos_offset = T_prompt
            next_token_input = prompt_tokens[:, -1:]
        else:
            next_token_logits = None

        for i in range(num_tokens_to_generate):
            if current_pos_offset >= self.max_len:
                print(f"Warning: KV cache current length {current_pos_offset} attempting to exceed model max_len {self.max_len}. Stopping generation.")
                break
            
            if next_token_logits is None and T_prompt == 0:
                print(f"Error: next_token_logits is None at generation step {i} with T_prompt=0. Unhandled.")
                break
            elif next_token_logits is None:
                 print(f"Error: next_token_logits is None at generation step {i}. This indicates an unhandled start case.")
                 break


            if temperature == 0.0:
                newly_sampled_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(scaled_logits, dim=-1)
                newly_sampled_token = torch.multinomial(probs, num_samples=1)
            
            all_tokens_list.append(newly_sampled_token)
            
            if i == num_tokens_to_generate - 1:
                break

            next_token_input = newly_sampled_token
            
            logits_step = self.forward(
                next_token_input,
                kv_cache=kv_caches_list,
                pos_offset=current_pos_offset
            )
            next_token_logits = logits_step[:, -1, :]
            current_pos_offset += 1
            
        final_generated_tokens = torch.cat(all_tokens_list, dim=1)
        return final_generated_tokens
