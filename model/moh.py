"""
Standalone Mixture-of-Heads (MoH) Attention Module
===================================================
从 language_model_moh.py 中剥离的独立 MoH 系统，可插入 MiniMind 使用。

核心思路: 将 Q 头分为 shared heads (始终激活) 和 expert heads (通过路由选择性激活)。
每个 token 只激活 1 个 expert Q head，在保持计算量的前提下增加注意力多样性。

特性:
  - L2 Norm 路由 (基于 Q head 范数选择, 零额外参数)
  - 推理 top-1 优化路径 (跳过未选中的 expert heads)
  - 负载均衡损失 (鼓励各 expert head 均匀使用)
  - GPU-native 利用率统计
  - 兼容 MiniMind 的 GQA + RoPE + FlashAttention

用法 (插入 MiniMind):
    from model.moh import MoHConfig, MoHAttention

    attn = MoHAttention(minimind_config)

    # forward: 接口与原生 Attention 一致
    output, past_kv = attn(x, position_embeddings, past_key_value, use_cache, attention_mask)
    # 额外可通过 attn.aux_loss 获取负载均衡损失
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


# ═══════════════════════════════════════════════════════════════════════════════
# MoH Attention
# ═══════════════════════════════════════════════════════════════════════════════

class MoHAttention(nn.Module):
    """
    Mixture-of-Heads Attention — 在 Q 头上做稀疏路由。

    结构:
      Q heads = shared_heads (始终激活) + num_experts (稀疏激活)
      每个 token 从 num_experts 中选 routed_head 个激活

    Args:
        hidden_size:     隐藏层维度
        num_heads:       总 Q 头数
        num_kv_heads:    KV 头数
        head_dim:        每头维度
        shared_heads:    始终激活的 Q 头数 (默认=num_kv_heads)
        routed_head:     每个 token 激活的 expert head 数 (默认=1)
        balance_loss_weight: 负载均衡损失权重
        dropout:         dropout 率
        flash_attn:      是否使用 FlashAttention
        rms_norm_eps:    QK Norm epsilon
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        head_dim: int = 96,
        shared_heads: int = 4,
        routed_head: int = 1,
        balance_loss_weight: float = 0.01,
        dropout: float = 0.0,
        flash_attn: bool = True,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = num_heads
        self.n_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout

        # GQA
        self.n_rep = num_heads // num_kv_heads

        # MoH
        self.shared_heads = shared_heads
        self.num_experts = num_heads - shared_heads
        self.routed_head = routed_head
        self.balance_loss_weight = balance_loss_weight

        # 投影层
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK Norm
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # FlashAttention
        self.flash = hasattr(F, 'scaled_dot_product_attention') and flash_attn

        # 累积统计
        self.aux_loss = torch.zeros(1).squeeze()
        self._expert_util_sum = None
        self._expert_util_cnt = 0

        self._norm_eps = 1e-6

    # ── L2 Norm 路由 ────────────────────────────────────────────────────────

    def _l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        """高效 L2 范数: ||x|| = sqrt(sum(x²))"""
        return torch.sqrt(torch.mul(x, x).sum(dim=-1) + self._norm_eps)

    # ── 负载均衡损失 ────────────────────────────────────────────────────────

    def _balance_loss(self, expert_norms: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失 (P 对方差 from uniform)"""
        if not self.training or self.balance_loss_weight <= 0:
            return expert_norms.new_zeros(1).squeeze()
        e = expert_norms.float()
        P_soft = F.softmax(e, dim=-1)
        P_global = P_soft.mean(dim=(0, 1))
        uniform = 1.0 / self.num_experts
        L_b = ((P_global - uniform) ** 2).mean()
        return (self.balance_loss_weight * L_b).to(expert_norms.dtype)

    # ── 专家利用率 ──────────────────────────────────────────────────────────

    def _expert_utilization(self, keep_indices: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """GPU-native 专家利用率统计"""
        n_votes = B * T * self.routed_head
        if n_votes == 0:
            return torch.zeros(self.num_experts, device=keep_indices.device, dtype=torch.float32)
        flat = keep_indices.reshape(-1)
        counts = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)
        ones = torch.ones_like(flat, dtype=torch.float32)
        counts.index_add_(0, flat, ones)
        return counts / n_votes

    # ── 推理优化路径 ────────────────────────────────────────────────────────

    def _forward_inference_top1(
        self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None
    ):
        """推理路径: 只计算 shared heads + 1 selected expert, 节省计算"""
        bsz, seq_len, _ = x.shape
        cos, sin = position_embeddings

        # 1. 投影
        xq = self.q_proj(x)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 2. 路由: L2 norm → argmax
        q_view = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        expert_q = q_view[:, :, self.shared_heads:]
        expert_norms = self._l2_norm(expert_q)
        best_expert = torch.argmax(expert_norms, dim=-1)  # (B, T)

        # 3. 构造 Q: shared + 1 selected expert
        shared_q = q_view[:, :, :self.shared_heads]  # (B, T, S, D)
        idx_exp = best_expert.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.head_dim)
        selected_q = torch.gather(expert_q, 2, idx_exp).squeeze(2)  # (B, T, D)
        q = torch.cat([shared_q, selected_q.unsqueeze(2)], dim=2)  # (B, T, S+1, D)

        actual_n_heads = self.shared_heads + 1

        # 4. QK Norm + RoPE (保持 [B, T, H, D] 兼容 MiniMind convention)
        q_reshaped = q.reshape(bsz * seq_len * actual_n_heads, self.head_dim)
        q_reshaped = self.q_norm(q_reshaped)
        q = q_reshaped.view(bsz, seq_len, actual_n_heads, self.head_dim)
        xk = self.k_norm(xk)
        q, xk = apply_rotary_pos_emb(q, xk, cos, sin)
        q = q.transpose(1, 2)  # (B, H, T, D) for attention

        # 5. KV Cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 6. GQA Expansion (适配 shared+1 heads)
        if actual_n_heads % self.n_kv_heads != 0:
            # 退化为全量计算
            q_orig = q.view(bsz, actual_n_heads, -1, self.head_dim)
            # Pad to full n_heads
            return self._forward_training(x, position_embeddings, past_key_value, use_cache, attention_mask)

        n_kv_groups = actual_n_heads // self.n_kv_heads
        kv_b = xk  # (B, T_kv, n_kv_heads, D)

        # Repeat KV
        k_exp = kv_b[:, :, :, None, :].expand(bsz, -1, self.n_kv_heads, n_kv_groups, self.head_dim)
        k_exp = k_exp.reshape(bsz, -1, actual_n_heads, self.head_dim).transpose(1, 2)
        v_exp = xv[:, :, :, None, :].expand(bsz, -1, self.n_kv_heads, n_kv_groups, self.head_dim)
        v_exp = v_exp.reshape(bsz, -1, actual_n_heads, self.head_dim).transpose(1, 2)

        # 7. Attention
        is_causal = past_key_value is None
        if self.flash and seq_len > 1 and attention_mask is None:
            output = F.scaled_dot_product_attention(
                q, k_exp, v_exp, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
            )
        else:
            scores = (q @ k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(q)) @ v_exp

        # 8. Output
        output = output.transpose(1, 2).reshape(bsz, seq_len, actual_n_heads * self.head_dim)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv

    # ── 训练路径 ────────────────────────────────────────────────────────────

    def _forward_training(
        self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None
    ):
        """训练路径: 计算所有 heads, 然后 mask 未选中的 expert heads"""
        bsz, seq_len, _ = x.shape
        cos, sin = position_embeddings

        # 1. 投影
        xq = self.q_proj(x)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 2. MoH 路由决策
        q_view = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        expert_norms = self._l2_norm(q_view[:, :, self.shared_heads:])
        _, keep_indices = torch.topk(expert_norms, k=self.routed_head, dim=-1)

        # 负载均衡 + 利用率
        load_balance_aux = self._balance_loss(expert_norms)
        expert_util = self._expert_utilization(keep_indices, bsz, seq_len)

        # 3. QK Norm + RoPE
        # QK Norm (在 [B, T, H, D] 下做 RoPE，兼容 MiniMind apply_rotary_pos_emb)
        q = q_view  # (B, T, H, D)
        q_reshaped = q.reshape(bsz * seq_len * self.n_heads, self.head_dim)
        q_reshaped = self.q_norm(q_reshaped)
        q = q_reshaped.view(bsz, seq_len, self.n_heads, self.head_dim)

        xk = self.k_norm(xk)
        q, xk = apply_rotary_pos_emb(q, xk, cos, sin)
        q = q.transpose(1, 2)  # (B, H, T, D)

        # 4. KV Cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 5. GQA Expansion
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 6. Attention
        is_causal = past_key_value is None
        if self.flash and seq_len > 1 and attention_mask is None:
            y = F.scaled_dot_product_attention(
                q, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
            )
        else:
            scores = (q @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            y = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(q)) @ xv

        y = y.transpose(1, 2).contiguous()  # (B, T, H, D)

        # 7. MoH Mask: gather-scatter 只保留被选中的 expert heads
        if self.num_experts > 0 and self.routed_head > 0:
            y_shared = y[:, :, :self.shared_heads, :]
            y_expert = y[:, :, self.shared_heads:, :]

            gather_idx = keep_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            selected = torch.gather(y_expert, dim=2, index=gather_idx)

            y_expert_new = torch.zeros_like(y_expert)
            y_expert_new.scatter_(dim=2, index=gather_idx, src=selected)
            y = torch.cat([y_shared, y_expert_new], dim=2)

        # 8. Output Projection
        y = y.reshape(bsz, seq_len, self.n_heads * self.head_dim)
        y = self.resid_dropout(self.o_proj(y))

        # 存储 aux_loss + 累积利用率
        self.aux_loss = load_balance_aux
        if self._expert_util_sum is None:
            self._expert_util_sum = expert_util.clone()
        else:
            self._expert_util_sum += expert_util
        self._expert_util_cnt += 1

        return y, past_kv

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        if not self.training and self.routed_head == 1:
            return self._forward_inference_top1(x, position_embeddings, past_key_value, use_cache, attention_mask)
        return self._forward_training(x, position_embeddings, past_key_value, use_cache, attention_mask)

    # ── 统计方法 ────────────────────────────────────────────────────────────

    def reset_moh_stats(self):
        self._expert_util_sum = None
        self._expert_util_cnt = 0

    def get_moh_stats(self):
        if self._expert_util_sum is None or self._expert_util_cnt == 0:
            return None
        return (self._expert_util_sum / self._expert_util_cnt).cpu()


# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm (for QK Norm)
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))).type_as(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MoH Module Self-Test")
    print("=" * 60)

    # 模拟 precompute_freqs_cis (dim=head_dim=96)
    dim, end, rope_base = 96, 128, 1e6
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))  # [48]
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()  # [128, 48]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [128, 96]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [128, 96]

    # 模拟 MiniMind 配置: hidden=768, n_heads=8, n_kv_heads=4, head_dim=96
    attn = MoHAttention(
        hidden_size=768, num_heads=8, num_kv_heads=4, head_dim=96,
        shared_heads=4, routed_head=1, balance_loss_weight=0.01,
    ).cuda()

    x = torch.randn(2, 64, 768).cuda()
    freqs_cos, freqs_sin = freqs_cos[:64].cuda(), freqs_sin[:64].cuda()
    pe = (freqs_cos, freqs_sin)

    # [1] 训练模式
    attn.train()
    out, pkv = attn(x, pe)
    print(f"\n[1] Train mode:")
    print(f"    out: {out.shape}, aux_loss: {attn.aux_loss.item():.6f}")

    # [2] 推理模式 (top1 优化)
    attn.eval()
    out2, pkv2 = attn(x, pe)
    print(f"\n[2] Inference (top1):")
    print(f"    out: {out2.shape}, std: {out2.std().item():.4f}")

    # [3] 梯度检查
    attn.train()
    out3, _ = attn(x, pe)
    (out3.sum() + attn.aux_loss).backward()
    grad_norms = {}
    for name, p in attn.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    if grad_norms:
        print(f"\n[3] Gradient check:")
        print(f"    params with grad: {len(grad_norms)}, max_grad: {max(grad_norms.values()):.4f}")
    else:
        print("\n[3] ⚠️ No gradients!")

    # [4] 参数统计
    total = sum(p.numel() for p in attn.parameters())
    print(f"\n[4] Params: {total:,}")

    print(f"\n✅ All tests passed!")
