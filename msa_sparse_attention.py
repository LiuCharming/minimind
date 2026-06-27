"""
MiniMax Sparse Attention (MSA) — PyTorch 参考实现
================================================================

基于 MiniMax-AI/MSA 仓库的算法思想，用纯 PyTorch 实现两阶段稀疏注意力：

  阶段1（Proxy）：用廉价的代理 Q（1 个 KV head）估算每个 KV block 的重要性
  阶段2（TopK 选择）：选出 top-k 个得分最高的 KV block
  阶段3（稀疏注意力）：只对选中的 block 做精确注意力计算

核心数据结构：
  kv_block_indexes: [total_q, num_heads, topK]  →  每个 query token 选中的 KV block 索引

硬件无关，可在任何 GPU/CPU 上运行。仅供学习理解 MSA 算法。

用法：
  python msa_sparse_attention.py
"""

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# RoPE（旋转位置编码）
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """标准的 RoPE，用于给 Q 和 K 注入位置信息"""

    def __init__(self, dim: int, max_seq_len: int = 131072, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            position_ids: [batch, seq_len]
        Returns:
            rotated x, same shape
        """
        batch, num_heads, seq_len, dim = x.shape
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq.to(x.device)
        emb = torch.cat([freqs, freqs], dim=-1)                    # [B, S, dim]
        cos = emb.cos().unsqueeze(1)                               # [B, 1, S, dim]
        sin = emb.sin().unsqueeze(1)

        x_rot = x.reshape(*x.shape[:-1], dim // 2, 2)
        x1, x2 = x_rot[..., 0], x_rot[..., 1]
        rotated = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        return x * cos + rotated * sin


# ============================================================================
# 辅助函数
# ============================================================================

def _reshape_for_attention(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """[batch, seq, d_model] -> [batch, heads, seq, head_dim]"""
    B, S, _ = x.shape
    return x.view(B, S, num_heads, head_dim).transpose(1, 2)


def _reshape_from_attention(
    x: torch.Tensor, d_model: int
) -> torch.Tensor:
    """[batch, heads, seq, head_dim] -> [batch, seq, d_model]"""
    B, H, S, D = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, H * D)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV head 重复 n_rep 次以匹配 Q head 数量（GQA）"""
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, S, D)
    return x.reshape(B, H_kv * n_rep, S, D)


# ============================================================================
# MSA：两阶段稀疏注意力
# ============================================================================

class MiniMaxSparseAttention(nn.Module):
    """
    MiniMax Sparse Attention (MSA)

    ┌─────────────────────────────────────────────────────────┐
    │  阶段1：Proxy 评分                                       │
    │    proxy_Q = Q[:, :1, :, :]    只用 1 个 head 做代理     │
    │    proxy_K = K[:, :1, :, :]    对应的 1 个 KV head       │
    │    scores = proxy_Q @ proxy_K^T                         │
    │    block_max = max(scores) per KV block (page_size=128)  │
    │                                                         │
    │  阶段2：TopK 选择                                        │
    │    topk_idx = argtopk(block_max, k=topk)                │
    │    → kv_block_indexes: [total_q, H, topK]                │
    │                                                         │
    │  阶段3：稀疏注意力                                       │
    │    对每个 query token，只 gather top-k 个 KV block        │
    │    sparse_K = gather(K, kv_block_indexes)               │
    │    sparse_V = gather(V, kv_block_indexes)               │
    │    out = softmax(Q @ sparse_K^T) @ sparse_V             │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,        # GQA: KV head 少于 Q head
        head_dim: int = 128,
        page_size: int = 128,          # KV block 大小
        topk: int = 16,               # 每个 query 选中多少个 KV block
        force_begin_blocks: int = 1,   # 强制保留前 N 个 block（sink token）
        force_end_blocks: int = 1,     # 强制保留最后 N 个 block（局部窗口）
        dropout: float = 0.0,
        max_seq_len: int = 131072,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.topk = topk
        self.force_begin_blocks = force_begin_blocks
        self.force_end_blocks = force_end_blocks
        self.n_rep = num_heads // num_kv_heads  # GQA 重复因子

        # QKV 投影
        self.W_Q = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.W_O = nn.Linear(num_heads * head_dim, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def _compute_block_max_scores(
        self,
        q: torch.Tensor,                        # [B, H_proxy, S_q, D]
        k: torch.Tensor,                        # [B, H_proxy, S_kv, D]
        attention_mask: Optional[torch.Tensor],  # [B, 1, S_q, S_kv]
    ) -> torch.Tensor:
        """
        阶段1：用代理 Q / K 计算每个 KV block 的最大注意力分数。

        Returns:
            block_max: [B, H_proxy, S_q, num_blocks]
        """
        B, H_proxy, S_q, D = q.shape
        _, _, S_kv, _ = k.shape

        # 计算代理注意力分数
        scale = D ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S_q, S_kv]

        if attention_mask is not None:
            scores = scores + attention_mask

        # 以 block 为单位取 max
        num_blocks = (S_kv + self.page_size - 1) // self.page_size

        # 将 scores 按 page_size 分组
        pad_len = num_blocks * self.page_size - S_kv
        if pad_len > 0:
            scores = F.pad(scores, (0, pad_len), value=float("-inf"))

        scores_blocks = scores.view(B, H_proxy, S_q, num_blocks, self.page_size)
        block_max = scores_blocks.max(dim=-1).values  # [B, H, S_q, num_blocks]

        return block_max

    def _select_topk_blocks(
        self,
        block_max: torch.Tensor,            # [B, H_proxy, S_q, num_blocks]
        num_valid_blocks_per_seq: int,
    ) -> torch.Tensor:
        """
        阶段2：从 block_max 中选出 top-k 个 KV block。

        强制保留 force_begin_blocks 和 force_end_blocks。

        Returns:
            kv_block_indexes: [B, H_proxy, S_q, topk]  int64, 升序排列
        """
        B, H_proxy, S_q, num_blocks = block_max.shape
        device = block_max.device
        topk = self.topk
        fb = self.force_begin_blocks
        fe = self.force_end_blocks

        n_valid = min(num_valid_blocks_per_seq, num_blocks)

        # 强制保留的 block 索引
        forced_indices = set()
        for i in range(min(fb, n_valid)):
            forced_indices.add(i)
        for i in range(max(0, n_valid - fe), n_valid):
            forced_indices.add(i)

        forced_indices = sorted(forced_indices)
        num_forced = len(forced_indices)
        num_free = topk - num_forced

        # 创建输出
        kv_block_indexes = torch.full(
            (B, H_proxy, S_q, topk), -1, dtype=torch.int64, device=device
        )

        # 对每个 (batch, head, query) 独立处理
        for b in range(B):
            for h in range(H_proxy):
                scores_row = block_max[b, h]  # [S_q, num_blocks]

                for t in range(S_q):
                    token_scores = scores_row[t].clone()

                    # 屏蔽掉强制保留的 block，避免重复选择
                    mask = torch.ones(num_blocks, dtype=torch.bool, device=device)
                    for fi in forced_indices:
                        mask[fi] = False
                    token_scores[~mask] = float("-inf")

                    # 选 top-(num_free) 个自由 block
                    if num_free > 0:
                        _, free_topk = torch.topk(
                            token_scores, min(num_free, n_valid - num_forced), dim=-1
                        )
                        selected = list(forced_indices) + free_topk.tolist()
                    else:
                        selected = list(forced_indices)

                    # 排序以保证升序
                    selected = sorted(selected[:topk])

                    # 填充
                    for i, idx in enumerate(selected):
                        kv_block_indexes[b, h, t, i] = idx

        return kv_block_indexes

    def _gather_kv_by_blocks(
        self,
        k: torch.Tensor,                    # [B, H_kv, S_kv, D]
        v: torch.Tensor,                    # [B, H_kv, S_kv, D]
        kv_block_indexes: torch.Tensor,     # [B, H, S_q, topk]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        阶段3（前半）：根据 kv_block_indexes 从 K、V 中 gather 出稀疏的 KV tensor。

        Returns:
            sparse_K: [B, H_kv, S_q, topk * page_size, D]
            sparse_V: same shape
        """
        B, H_kv, S_kv, D = k.shape
        _, _, S_q, topk = kv_block_indexes.shape

        # 补零对齐到 page_size 的整数倍
        num_blocks = (S_kv + self.page_size - 1) // self.page_size
        pad_len = num_blocks * self.page_size - S_kv
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        k_blocks = k.view(B, H_kv, num_blocks, self.page_size, D)
        v_blocks = v.view(B, H_kv, num_blocks, self.page_size, D)

        sparse_K = torch.zeros(B, H_kv, S_q, topk * self.page_size, D,
                               dtype=k.dtype, device=k.device)
        sparse_V = torch.zeros_like(sparse_K)

        # kv_block_indexes: [B, H, S_q, topk]，H 可能 != H_kv
        # 取第一个 head 的索引（所有 head 共享同样的选择）
        gather_idx = kv_block_indexes[:, 0, :, :]  # [B, S_q, topk]

        for b in range(B):
            for t in range(S_q):
                idx = gather_idx[b, t]  # [topk]
                for i in range(topk):
                    if idx[i] >= 0:
                        blk = idx[i].item()
                        start = i * self.page_size
                        end = start + self.page_size
                        sparse_K[b, :, t, start:end] = k_blocks[b, :, blk]
                        sparse_V[b, :, t, start:end] = v_blocks[b, :, blk]

        return sparse_K, sparse_V

    def forward(
        self,
        hidden_states: torch.Tensor,            # [B, S, d_model]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_sparse: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, 1, seq_len, seq_len] 或 None
            position_ids: [batch, seq_len]
            use_sparse: 是否使用稀疏注意力。False 则退化为标准 dense attention。

        Returns:
            output: [batch, seq_len, d_model]
        """
        B, S, _ = hidden_states.shape
        device = hidden_states.device

        if position_ids is None:
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # ── Q、K、V 投影 ──
        Q = self.W_Q(hidden_states)  # [B, S, num_heads * D]
        K = self.W_K(hidden_states)  # [B, S, num_kv_heads * D]
        V = self.W_V(hidden_states)

        Q = _reshape_for_attention(Q, self.num_heads, self.head_dim)     # [B, H, S, D]
        K = _reshape_for_attention(K, self.num_kv_heads, self.head_dim)  # [B, H_kv, S, D]
        V = _reshape_for_attention(V, self.num_kv_heads, self.head_dim)

        # ── RoPE ──
        Q = self.rope(Q, position_ids)
        K = self.rope(K, position_ids)

        # ── 标准 Dense Attention（fallback） ──
        if not use_sparse:
            K_expanded = repeat_kv(K, self.n_rep)
            V_expanded = repeat_kv(V, self.n_rep)

            scale = self.head_dim ** -0.5
            scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V_expanded)
            out = _reshape_from_attention(out, self.d_model)
            return self.W_O(out)

        # ── MSA 稀疏注意力 ──

        # 阶段1：代理评分（只用 1 个 head）
        proxy_Q = Q[:, :1, :, :]   # [B, 1, S, D]
        proxy_K = K[:, :1, :, :]   # [B, 1, S, D]

        block_max = self._compute_block_max_scores(proxy_Q, proxy_K, attention_mask)
        # block_max: [B, 1, S, num_blocks]

        num_valid_blocks = (S + self.page_size - 1) // self.page_size

        # 阶段2：选 top-k blocks
        kv_block_indexes = self._select_topk_blocks(block_max, num_valid_blocks)
        # kv_block_indexes: [B, 1, S, topk]

        # 扩展到所有 Q head
        kv_block_indexes = kv_block_indexes.expand(B, self.num_heads, S, self.topk)

        # 阶段3：gather 稀疏 KV
        K_expanded_full = repeat_kv(K, self.n_rep)    # [B, H, S, D]
        V_expanded_full = repeat_kv(V, self.n_rep)

        sparse_K, sparse_V = self._gather_kv_by_blocks(
            K_expanded_full, V_expanded_full, kv_block_indexes
        )
        # sparse_K: [B, H, S, topk * page_size, D]

        # 阶段3（后半）：稀疏注意力计算
        scale = self.head_dim ** -0.5
        Q_expanded = Q.unsqueeze(3)  # [B, H, S, 1, D]

        scores_sparse = torch.matmul(
            Q_expanded, sparse_K.transpose(-2, -1)
        ).squeeze(3)  # [B, H, S, topk*page]
        scores_sparse = scores_sparse * scale

        # 对无效 block（索引为 -1 的）做 mask
        invalid_mask = (kv_block_indexes == -1)
        # [B, H, S, topk] -> [B, H, S, topk, page] -> [B, H, S, topk*page]
        invalid_mask = invalid_mask.unsqueeze(-1).expand(
            B, self.num_heads, S, self.topk, self.page_size
        ).reshape(B, self.num_heads, S, self.topk * self.page_size)

        scores_sparse = scores_sparse.masked_fill(invalid_mask, float("-inf"))

        attn_weights = F.softmax(scores_sparse, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights.unsqueeze(3), sparse_V).squeeze(3)
        # out: [B, H, S, D]

        out = _reshape_from_attention(out, self.d_model)
        return self.W_O(out)


# ============================================================================
# 测试 & 对比
# ============================================================================

def test_msa():
    """测试 MSA 的正确性和性能"""
    print("=" * 70)
    print("MiniMax Sparse Attention (MSA) — PyTorch 参考实现")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 配置
    B = 2
    S = 2048
    d_model = 1024
    num_heads = 8
    num_kv_heads = 2
    page_size = 128
    topk = 16

    msa = MiniMaxSparseAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=d_model // num_heads,
        page_size=page_size,
        topk=topk,
    ).to(device)

    x = torch.randn(B, S, d_model, device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
    )
    attn_mask = torch.where(causal_mask, float("-inf"), 0.0).unsqueeze(0).unsqueeze(0)

    print(f"\n输入: {x.shape}")
    print(f"模型参数: d_model={d_model}, heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"序列长度: {S}, page_size={page_size}, topk={topk}")
    print(f"选中的 token 数: {topk} * {page_size} = {topk * page_size}")
    print(f"总 KV token 数: {S}")
    print(f"稀疏度: {topk * page_size / S * 100:.1f}%")

    # ── Dense Attention（参考基准） ──
    print("\n--- Dense Attention ---")

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.time()
            out_dense = msa(x, attn_mask, position_ids, use_sparse=False)
            torch.cuda.synchronize()
            elapsed_dense = (time.time() - t0) * 1000
        else:
            t0 = time.time()
            out_dense = msa(x, attn_mask, position_ids, use_sparse=False)
            elapsed_dense = (time.time() - t0) * 1000
        print(f"  耗时: {elapsed_dense:.1f} ms")

    print(f"  输出: {out_dense.shape}")

    # ── MSA Sparse Attention ──
    print("\n--- MSA Sparse Attention ---")

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.time()
            out_sparse = msa(x, attn_mask, position_ids, use_sparse=True)
            torch.cuda.synchronize()
            elapsed_sparse = (time.time() - t0) * 1000
        else:
            t0 = time.time()
            out_sparse = msa(x, attn_mask, position_ids, use_sparse=True)
            elapsed_sparse = (time.time() - t0) * 1000
        print(f"  耗时: {elapsed_sparse:.1f} ms")

    print(f"  输出: {out_sparse.shape}")
    print(f"  加速比: {elapsed_dense / elapsed_sparse:.2f}x" if elapsed_sparse > 0 else "")

    # ── 误差分析 ──
    print("\n--- 误差分析 ---")
    cos_sim = F.cosine_similarity(
        out_dense.reshape(-1), out_sparse.reshape(-1), dim=0
    )
    mse = F.mse_loss(out_dense, out_sparse)
    print(f"  Cosine Similarity: {cos_sim:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  (注：稀疏注意力会有一定误差，topk 越大越接近 dense)")

    # ── 理论分析 ──
    print("\n--- FLOPs 理论分析 ---")
    dense_flops = B * num_heads * S * S * (d_model // num_heads) * 2
    proxy_flops = B * 1 * S * S * (d_model // num_heads) * 2
    sparse_flops = B * num_heads * S * (topk * page_size) * (d_model // num_heads) * 2
    total_sparse_flops = proxy_flops + sparse_flops

    print(f"  Dense  FLOPs:        {dense_flops / 1e9:.2f} G")
    print(f"  Sparse FLOPs (总计):  {total_sparse_flops / 1e9:.2f} G")
    print(f"    - Proxy 阶段:       {proxy_flops / 1e9:.2f} G  ({proxy_flops / total_sparse_flops * 100:.1f}%)")
    print(f"    - 稀疏注意力:        {sparse_flops / 1e9:.2f} G  ({sparse_flops / total_sparse_flops * 100:.1f}%)")
    print(f"  FLOP 比值:           {total_sparse_flops / dense_flops * 100:.1f}%")

    # ── KV Cache 对比 ──
    print("\n--- KV Cache 说明 ---")
    kv_bytes_per_token = 2 * num_kv_heads * (d_model // num_heads) * 2  # BF16, K+V
    print(f"  每 token KV Cache: {kv_bytes_per_token} bytes")
    print(f"  总共 (S={S}):       {kv_bytes_per_token * S / 1024:.1f} KB")
    print(f"  (MSA 不减少 KV Cache 存储，只减少计算量)")
    print(f"  如需减少 KV Cache，请结合 MLA (Multi-head Latent Attention)")

    # ── MSA vs MLA vs Dense 对比总结 ──
    print("\n" + "=" * 70)
    print("技术对比总结")
    print("=" * 70)
    print(f"  {'方案':<20} {'KV Cache':<15} {'计算量':<15} {'精度':<10}")
    print(f"  {'─' * 60}")
    print(f"  {'Dense (MHA)':<20} {'2 * d_model':<15} {'100%':<15} {'无损':<10}")
    print(f"  {'GQA':<20} {'2 * d_kv':<15} {'~100%':<15} {'≈无损':<10}")
    print(f"  {'MLA':<20} {'d_c + d_rope':<15} {'~120%':<15} {'≈无损':<10}")
    print(f"  {'MSA (本实现)':<20} {'2 * d_model':<15} {'{:.0f}%'.format(total_sparse_flops/dense_flops*100):<15} {'少量损失':<10}")
    print(f"  MLA + MSA 叠加    {'d_c + d_rope':<15} {'<<100%':<15} {'≈无损':<10}")


if __name__ == "__main__":
    test_msa()
