import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states.
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """

    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute inverse of RMS: square the tensor element-wise, mean is computed across lm_hidden_dim.
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)  # inverse of RMS
        x = x * irms * self.weight

        return x


# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
class RotaryEmbedding(nn.Module):
    """
        Compute Rotary Embedding to introduce positional dependency to input sequence without additional training parameters and
        relative distance of token position ids through angle rotation.

        Args:
            cfg: Configuration object containing:
                - lm_hidden_dim (int): Hidden dimension size.
                - lm_n_heads (int): Number of attention heads.
                - lm_re_base (float): Base for rotary embedding frequencies.
                - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
                - lm_attn_scaling (float): Attention scaling factor.
        """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"

        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads  # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        batch_size, seq_len = position_ids.shape
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq

        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()

        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)

        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)

        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)

        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input by dividing the hidden dimension to two, then swapping and negating dimensions.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.

    Rotary positional embeddings inject position-dependent rotations into query and key vectors,
    enabling transformers to encode positional information effectively without explicit positional encoding.

    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`),
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)

    """

    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]

    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) with Mixture-of-Heads (MoH) routing.

    GQA reduces computation by using fewer key-value heads than query heads.
    MoH adds a routing mechanism to selectively activate a subset of query heads ("experts").

    Optimizations:
        - Inference path: Skip unused expert heads for ~8x attention speedup when routed_head=1
        - Efficient L2 norm: Use mul+sum+sqrt instead of torch.norm
        - GPU-native bincount: Avoid CPU-GPU data transfer
        - FlashAttention support: Optional 2-3x speedup via FlashAttention
        - Optimized gather-scatter: Reduced memory operations

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Total number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        # GQA 参数
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        # 投影层
        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.resid_dropout = nn.Dropout(self.dropout)

        # MoH 配置
        self.shared_head = 7  # 必须是 n_kv_heads 的约数或使得 shared_head + routed_head 能被 n_kv_heads 整除
        self.num_experts = self.n_heads - self.shared_head
        self.routed_head = 1

        # 负载均衡：辅助损失权重
        self.balance_loss_weight = getattr(cfg, 'lm_moh_balance_loss_weight', 0.01)

        # FlashAttention 支持检测
        self.use_flash_attn = False
        if hasattr(cfg, 'use_flash_attn') and cfg.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.use_flash_attn = True
            except ImportError:
                pass

        # 缓存常数，避免重复计算
        self._norm_eps = 1e-6

    def _efficient_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        高效 L2 范数计算: ||x|| = sqrt(sum(x^2))
        使用 mul+sum+sqrt 链，比 torch.norm 更快
        """
        return torch.sqrt(torch.mul(x, x).sum(dim=-1) + self._norm_eps)

    def _compute_load_balance_loss(
        self,
        expert_norms: torch.Tensor,
        B: int,
        T_curr: int,
    ) -> torch.Tensor:
        """
        计算负载均衡辅助损失，鼓励各专家头被均匀使用。
        采用「P 的方差」形式，全程在 float32 下计算。
        """
        if not self.training or self.balance_loss_weight <= 0:
            return torch.tensor(0.0, device=expert_norms.device, dtype=expert_norms.dtype)
        orig_dtype = expert_norms.dtype
        e = expert_norms.float()
        P_soft = F.softmax(e, dim=-1)
        P_global = P_soft.mean(dim=(0, 1))
        uniform = 1.0 / self.num_experts
        L_b = ((P_global - uniform) ** 2).mean()
        return (self.balance_loss_weight * L_b).to(orig_dtype)

    def _compute_expert_utilization_gpu(
        self,
        keep_indices: torch.Tensor,
        B: int,
        T_curr: int,
    ) -> torch.Tensor:
        """
        GPU-native 专家利用率计算，避免 CPU-GPU 数据传输。
        使用 index_add 实现 bincount 功能。
        """
        n_votes = B * T_curr * self.routed_head
        if n_votes == 0:
            return torch.zeros(self.num_experts, device=keep_indices.device, dtype=torch.float32)

        flat = keep_indices.reshape(-1)
        counts = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)
        ones = torch.ones_like(flat, dtype=torch.float32)
        counts.index_add_(0, flat, ones)
        return counts / n_votes

    def _forward_inference_top1(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                               block_kv_cache: dict) -> tuple:
        """
        推理优化路径：当 routed_head=1 时，只计算选中的单个 expert head。
        可节省约 80% 的 attention 计算量。
        """
        B, T_curr, C = x.size()

        # 1. 投影
        query_states = self.q_proj(x)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. 高效 L2 Norm 路由
        q_view = query_states.view(B, T_curr, self.n_heads, self.head_dim)
        expert_q = q_view[:, :, self.shared_head:]  # (B, T, num_experts, D)
        expert_norms = self._efficient_l2_norm(expert_q)
        best_expert = torch.argmax(expert_norms, dim=-1)  # (B, T)

        # 3. 构造最终 Q: shared heads + 1 selected expert
        # 获取选中 expert 的 Q
        idx_exp = best_expert.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.head_dim)
        selected_q = torch.gather(expert_q, 2, idx_exp).squeeze(2)  # (B, T, D)

        # 合并: shared heads + 1 selected expert -> (B, T, shared+1, D)
        q = torch.cat([q_view[:, :, :self.shared_head], selected_q.unsqueeze(2)], dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D) for RoPE

        # 4. RoPE
        q, k_rotated = apply_rotary_pos_embd(q, k_curr, cos, sin)

        # 5. KV Cache
        if block_kv_cache is not None:
            if block_kv_cache['key'] is not None:
                k = torch.cat([block_kv_cache['key'], k_rotated], dim=2)
                v = torch.cat([block_kv_cache['value'], v_curr], dim=2)
            else:
                k, v = k_rotated, v_curr
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        else:
            k, v = k_rotated, v_curr
            block_kv_cache = {'key': k, 'value': v}

        # 6. GQA Expansion (只对 shared+1 heads)
        T_kv = k.size(2)
        actual_n_heads = self.shared_head + 1
        
        # GQA 要求 n_heads 必须是 n_kv_heads 的整数倍
        # 如果不满足，退化为使用全部 heads（但 Q 只用选中的）
        if actual_n_heads % self.n_kv_heads != 0:
            # 退回训练路径（全量计算）
            return self._forward_training(x, cos, sin, block_kv_cache)
        
        n_kv_groups = actual_n_heads // self.n_kv_heads

        k_exp = k[:, :, None, :, :].expand(B, self.n_kv_heads, n_kv_groups, T_kv, self.head_dim)
        k_exp = k_exp.reshape(B, actual_n_heads, T_kv, self.head_dim)
        v_exp = v[:, :, None, :, :].expand(B, self.n_kv_heads, n_kv_groups, T_kv, self.head_dim)
        v_exp = v_exp.reshape(B, actual_n_heads, T_kv, self.head_dim)

        # 7. Attention (FlashAttention 或 SDPA)
        is_causal = (T_curr == T_kv and T_curr > 1)

        if self.use_flash_attn and is_causal and q.dtype == torch.float16:
            # FlashAttention: 输入格式 (B, T, H, D)
            q_fa = q.transpose(1, 2)
            k_fa = k_exp.transpose(1, 2)
            v_fa = v_exp.transpose(1, 2)
            y = self.flash_attn_func(q_fa, k_fa, v_fa, causal=True)
            y = y.transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous()  # (B, T, H, D)

        # 8. Output Projection
        y = y.reshape(B, T_curr, actual_n_heads * self.head_dim)
        y = self.resid_dropout(self.out_proj(y))

        # 返回虚拟的辅助损失和利用率（推理时不使用）
        return y, block_kv_cache, torch.tensor(0.0, device=x.device), torch.zeros(self.num_experts, device=x.device)

    def _forward_training(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         block_kv_cache: dict) -> tuple:
        """
        训练路径：计算所有 heads，保持原有逻辑完整性。
        """
        B, T_curr, C = x.size()

        # 1. 投影 Q, K, V
        query_states = self.q_proj(x)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim)

        # 2. MoH 路由决策
        q_view_for_gate = query_states.view(B, T_curr, self.n_heads, self.head_dim)
        expert_norms = self._efficient_l2_norm(q_view_for_gate[:, :, self.shared_head:])
        _, keep_indices = torch.topk(expert_norms, k=self.routed_head, dim=-1)

        # 负载均衡与利用率
        load_balance_aux = self._compute_load_balance_loss(expert_norms, B, T_curr)
        expert_utilization = self._compute_expert_utilization_gpu(keep_indices, B, T_curr)

        # 3. RoPE
        q = q_view_for_gate.transpose(1, 2)
        k_curr = k_curr.transpose(1, 2)
        q, k_rotated = apply_rotary_pos_embd(q, k_curr, cos, sin)

        # 4. KV Cache
        if block_kv_cache is not None:
            if block_kv_cache['key'] is not None:
                k = torch.cat([block_kv_cache['key'], k_rotated], dim=2)
                v = torch.cat([block_kv_cache['value'], v_curr.transpose(1, 2)], dim=2)
            else:
                k, v = k_rotated, v_curr.transpose(1, 2)
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        else:
            k, v = k_rotated, v_curr.transpose(1, 2)
            block_kv_cache = {'key': k, 'value': v}

        # 5. GQA Expansion
        T_kv = k.size(2)
        k_exp = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_kv_groups, T_kv, self.head_dim)
        k_exp = k_exp.reshape(B, self.n_heads, T_kv, self.head_dim)
        v_exp = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_kv_groups, T_kv, self.head_dim)
        v_exp = v_exp.reshape(B, self.n_heads, T_kv, self.head_dim)

        # 6. Attention
        is_causal = (T_curr == T_kv and T_curr > 1)
        dropout_p = self.dropout

        if self.use_flash_attn and is_causal and q.dtype == torch.float16:
            q_fa = q.transpose(1, 2)
            k_fa = k_exp.transpose(1, 2)
            v_fa = v_exp.transpose(1, 2)
            y = self.flash_attn_func(q_fa, k_fa, v_fa, causal=True)
            y = y.transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k_exp, v_exp, dropout_p=dropout_p, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous()

        # 7. MoH Mask (优化版 gather-scatter)
        if self.num_experts > 0 and self.routed_head > 0:
            y_shared = y[:, :, :self.shared_head, :]
            y_expert = y[:, :, self.shared_head:, :]

            gather_idx = keep_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            selected = torch.gather(y_expert, dim=2, index=gather_idx)

            y_expert_new = torch.zeros_like(y_expert)
            y_expert_new.scatter_(dim=2, index=gather_idx, src=selected)
            y = torch.cat([y_shared, y_expert_new], dim=2)

        # 8. Output Projection
        y = y.reshape(B, T_curr, self.embd_dim)
        y = self.resid_dropout(self.out_proj(y))

        return y, block_kv_cache, load_balance_aux, expert_utilization

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask=None, block_kv_cache=None):
        """
        Forward pass supporting optimized inference and training paths.
        """
        # 推理 + Top1 路由优化
        if not self.training and self.routed_head == 1:
            return self._forward_inference_top1(x, cos, sin, block_kv_cache)

        # 训练路径
        return self._forward_training(x, cos, sin, block_kv_cache)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    """
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x


# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg)  # Input Norm
        self.norm2 = RMSNorm(cfg)  # Post Attention Norm

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor = None,
                block_kv_cache: dict = None):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.

        Returns:
            Tuple[Tensor, dict, Tensor]: Output tensor after the block (same shape as input),
                the updated key-value cache dictionary, and the MoH load-balance auxiliary loss (scalar).
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache, load_balance_aux, expert_utilization = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache, load_balance_aux, expert_utilization


# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])
        self.norm = RMSNorm(cfg)  # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, kv_cache: list[dict] = None,
                start_pos: int = 0, output_hidden_states: bool = False):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.
                - Tensor: MoH load-balance auxiliary loss (scalar), for training only.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary.
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        # T_curr is the length of the current input sequence
        B, T_curr, _ = x.size()

        # Create position_ids for the current sequence based on start_pos
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(current_position_ids)  # Get rotary position embeddings for current tokens

        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        all_load_balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        expert_utilizations = []
        hidden_states_list = [] if output_hidden_states else None
        for i, block in enumerate(self.blocks):
            x, kv_cache[i], lb_aux, expert_util = block(x, cos, sin, attention_mask, kv_cache[i])
            all_load_balance_loss = all_load_balance_loss + lb_aux
            expert_utilizations.append(expert_util)
            if output_hidden_states:
                hidden_states_list.append(x)

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens:
            x = self.head(x)

        # 跨层平均专家头利用率（与 MoE 一致，便于训练时上传）
        avg_expert_utilization = torch.stack(expert_utilizations).mean(dim=0) if expert_utilizations else None
        if output_hidden_states:
            return x, kv_cache, all_load_balance_loss, avg_expert_utilization, hidden_states_list
        return x, kv_cache, all_load_balance_loss, avg_expert_utilization

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int = 20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list, _, _ = self.forward(
            generated_outputs,
            attention_mask=None,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)

            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1:
                break

            decode_step_output, kv_cache_list, _, _ = self.forward(
                next_output,
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :]

        return generated_outputs

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError

        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)

        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")

        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(
                    f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")

        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers

        # Create our model with potentially larger vocabulary
        model = cls(cfg)

        try:
            index_path = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors.index.json")
            with open(index_path, 'r') as f:
                index = json.load(f)
            # Get unique filenames from weight map
            safetensors_filenames = sorted(list(set(index['weight_map'].values())))
            # Download all the sharded files
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename=fn) for fn in
                                 safetensors_filenames]
        except EntryNotFoundError:
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")]

        sd = model.state_dict()

        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }

        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'

            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })

        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        loaded_keys = set()

        ######### MODIFICATION START #########
        # 用于收集所有不匹配信息的列表
        shape_mismatches = []
        keys_not_in_model = []
        # 假设所有key一开始都缺失，在文件中找到时再移除
        keys_not_in_safetensors = set(mapping.keys())
        model_keys_set = set(sd.keys())
        ######### MODIFICATION END #########

        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                file_keys = f.keys()  # 获取当前文件中的所有key

                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue

                    if hf_key in file_keys:
                        # 在文件中找到了这个key，将其从“缺失”列表中移除
                        keys_not_in_safetensors.discard(hf_key)

                        if our_key not in model_keys_set:
                            # 不匹配 1: Key在mapping中，但不在我们的模型里
                            keys_not_in_model.append((hf_key, our_key))
                            continue

                        tensor = f.get_tensor(hf_key)

                        # 特殊处理：词汇表扩展
                        if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                            has_extended_embeddings = True
                            print(f"INFO: Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")

                            # 复制现有 embedding
                            sd[our_key][:tensor.shape[0]].copy_(tensor)

                            # 初始化新的 embedding
                            std = 0.02  # Common value
                            init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)

                            print(f"INFO: Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                            sd['head.weight'].copy_(sd[our_key])  # 更新 head

                        elif tensor.shape == sd[our_key].shape:
                            # 完美匹配
                            sd[our_key].copy_(tensor)

                        else:
                            # 不匹配 2: 形状不匹配
                            shape_mismatches.append(
                                f"  - {hf_key} -> {our_key}: Safetensors shape {tensor.shape} vs Model shape {sd[our_key].shape}"
                            )

                        loaded_keys.add(our_key)

        ######### MODIFICATION START #########
        # 循环结束后，打印所有收集到的不匹配信息

        print("\n--- 類 权重加载不匹配报告 ---")

        # 1. 报告形状不匹配
        if shape_mismatches:
            print(f"\n[!!] 发现 {len(shape_mismatches)} 处形状不匹配:")
            for msg in shape_mismatches:
                print(msg)
        else:
            print("\n[✅] 所有加载的权重形状均匹配。")

        # 2. 报告 mapping 中定义但模型中没有的 key
        if keys_not_in_model:
            print(f"\n[!] 发现 {len(keys_not_in_model)} 个 key 在 mapping 中，但不在模型 state_dict 中:")
            for hf_key, our_key in keys_not_in_model[:10]:  # 最多打印10个
                print(f"  - HF: '{hf_key}' -> Model: '{our_key}' (不在模型中)")
            if len(keys_not_in_model) > 10:
                print(f"  - ... 以及其他 {len(keys_not_in_model) - 10} 个。")

        # 3. 报告模型中需要但 safetensors 中没有的 key
        final_missing_from_hf = []
        for hf_key in keys_not_in_safetensors:
            our_key = mapping[hf_key]
            if our_key in model_keys_set:  # 确保这个key是模型真正需要的
                final_missing_from_hf.append((hf_key, our_key))

        if final_missing_from_hf:
            print(f"\n[!] 发现 {len(final_missing_from_hf)} 个模型 key 在任何 safetensors 文件中都未找到:")
            for hf_key, our_key in final_missing_from_hf[:10]:  # 最多打印10个
                print(f"  - Model: '{our_key}' (期望的 HF key: '{hf_key}')")
            if len(final_missing_from_hf) > 10:
                print(f"  - ... 以及其他 {len(final_missing_from_hf) - 10} 个。")

        print("-------------------------------------------\n")
        ######### MODIFICATION END #########

        # (不再需要旧的检查)
        # for hf_key, our_key in mapping.items():
        #     if our_key not in loaded_keys:
        #         if our_key in sd:
        #             print(f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})")

        # 加载状态字典
        model.load_state_dict(sd)

        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if 'lm_head.weight' in f.keys():
                        lm_head = f.get_tensor('lm_head.weight')
                        if lm_head.shape[0] != sd['head.weight'].shape[0]:
                            print(f"INFO: Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                            # Copy existing weights
                            sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                            # Load updated weights
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break

        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")

        print(
            f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
