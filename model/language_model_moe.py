import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
from typing import Any, Dict, TYPE_CHECKING, Optional, Tuple, List
from torch.nn import Module
from torch import Tensor
from collections import defaultdict

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

MOE_TOP_K = 1
Constant = 2


class CopyExpert(torch.nn.Module):
    def __init__(self, expert):
        super(CopyExpert, self).__init__()
        pass

    def forward(self, inputs):
        return inputs


class ZeroExpert(torch.nn.Module):
    def __init__(self, expert):
        super(ZeroExpert, self).__init__()
        pass

    def forward(self, inputs):
        # 性能优化：使用 zeros_like 更快（如果不需要避免 CUDA Graph 问题）
        # 如果遇到 CUDA Graph 问题，可以改回 torch.zeros
        return torch.zeros_like(inputs)


class ConstantExpert(torch.nn.Module):
    def __init__(self, expert):
        super(ConstantExpert, self).__init__()
        self.constant = torch.nn.Parameter(
            torch.empty((expert.embd_dim)))
        torch.nn.init.normal_(self.constant)

        self.wg = torch.nn.Linear(expert.embd_dim, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        # 性能优化：减少类型转换，使用更高效的实现
        weight = self.wg(inputs)
        weight = self.softmax(weight)
        # 优化：使用广播和直接乘法，比 einsum 更快
        constant = self.constant.type_as(inputs)
        return weight[:, 0:1] * inputs + weight[:, 1:2] * constant


class LoadBalancer(nn.Module):
    def __init__(self, num_experts, expert_types, tau, balance_loss_weight, use_normalized_loss=True):
        """
        Args:
            num_experts (int): 专家总数 (N)
            expert_types (list[str]): 每个专家的类型 ('ffn', 'zero' 等)
            tau (float): 'zero' 专家的权重 (τ)
            balance_loss_weight (float): 施加到 L_b 上的最终权重
            use_normalized_loss (bool): 是否使用归一化损失（消除专家数量影响）
        """
        super().__init__()
        self.num_experts = num_experts
        self.balance_loss_weight = balance_loss_weight
        self.use_normalized_loss = use_normalized_loss

        # 根据 expert_types 创建 eta (η) 张量
        eta_values = []
        for t in expert_types:
            if t == 'ffn':
                eta_values.append(1.0)
            elif t == 'zero':
                eta_values.append(tau)
            else:
                # 默认其他类型 (如 'copy') 也按 1.0 计算
                eta_values.append(1.0)

        # 将 eta 注册为 buffer，使其能随模型移动 (例如 .to(device))
        self.register_buffer("eta", torch.tensor(eta_values, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, top_k_indices: torch.Tensor, compute_loss: bool = True) -> torch.Tensor:
        if not compute_loss:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        num_tokens, num_experts = logits.shape
        if num_tokens == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # 计算门控概率分布 P（保持梯度，以便损失能够影响路由网络）
        gates_prob = F.softmax(logits, dim=1)  # 允许梯度传播
        P = gates_prob.mean(dim=0)

        # 计算实际分配频率 f（离散选择，但用于计算损失）
        flat_indices = top_k_indices.flatten()  # [num_tokens * MOE_TOP_K]
        f = torch.bincount(flat_indices, minlength=num_experts).to(dtype=logits.dtype) / (num_tokens * MOE_TOP_K)
        
        # 关键修复：确保损失能够反向传播到路由网络
        # 标准做法：使用 P（可导）来计算损失，鼓励均匀分布
        if self.use_normalized_loss:
            # 归一化版本：使用 P 的变异系数（CV）的平方
            # 直接优化 P 的分布，因为 P 是可导的，可以影响路由网络
            P_mean = P.mean()  # 理论值 = 1/N
            P_std = P.std()
            cv_squared = (P_std / (P_mean + 1e-9)) ** 2
            L_b_unscaled = cv_squared
            
            # 增强版本：同时考虑 P 的方差，提供更强的梯度信号
            # 添加 P 的方差项，鼓励更均匀的分布
            P_variance = torch.var(P)
            # 归一化方差：var(P) / mean(P)^2 = CV^2，所以这里已经包含了
            # 但可以添加额外的惩罚项来增强效果
            # L_b_unscaled = cv_squared + 0.5 * P_variance * self.num_experts
        else:
            # 标准公式：Switch Transformer/GShard 风格
            # L = N * sum(f_i * P_i)
            # 关键：f 使用 stop_gradient，P 保持梯度
            # 这样损失可以通过 P 反向传播到 logits，影响路由网络
            f_detached = f.detach()  # f 来自离散选择，不参与梯度
            L_b_unscaled = self.num_experts * torch.sum(f_detached * P)
        
        L_b_final = self.balance_loss_weight * L_b_unscaled

        return L_b_final

def gating(
    logits: Tensor,
    moe_use_mixtral_gating: bool = True,
    moe_use_logits_norm: bool = False,
    moe_gate_norm_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    张量化路由（避免 dict / python 循环）：
    返回:
      - top_k_indices: [num_tokens, MOE_TOP_K] long
      - top_k_gates:   [num_tokens, MOE_TOP_K] same dtype as logits
      - sorted_token_indices: [num_tokens*MOE_TOP_K] long
      - sorted_expert_indices:[num_tokens*MOE_TOP_K] long (已排序)
      - sorted_gate_weights:  [num_tokens*MOE_TOP_K] same dtype as logits (已排序)
      - expert_counts: [num_experts] long (按 expert_id 统计 token 数，含 Top-K 展开后的次数)
    """
    num_experts = logits.size(1)
    num_tokens = logits.size(0)
    device = logits.device
    dtype = logits.dtype

    if num_tokens == 0:
        empty_topk_idx = torch.empty((0, MOE_TOP_K), device=device, dtype=torch.long)
        empty_topk_g = torch.empty((0, MOE_TOP_K), device=device, dtype=dtype)
        empty_sorted = torch.empty((0,), device=device, dtype=torch.long)
        empty_sorted_g = torch.empty((0,), device=device, dtype=dtype)
        expert_counts = torch.zeros((num_experts,), device=device, dtype=torch.long)
        return empty_topk_idx, empty_topk_g, empty_sorted, empty_sorted, empty_sorted_g, expert_counts

    # 可选 logits normalization（按 token 维度）
    if moe_use_logits_norm:
        target_std = moe_gate_norm_std
        logits_std = logits.std(dim=1, keepdim=True)
        logits = logits / (logits_std / target_std)

    if moe_use_mixtral_gating:
        # Top-1 时 softmax 恒为 1，直接走 argmax 快很多
        if MOE_TOP_K == 1:
            top_k_indices = torch.argmax(logits, dim=1, keepdim=True).to(torch.long)
            top_k_gates = torch.ones((num_tokens, 1), device=device, dtype=dtype)
        else:
            top_k_gates, top_k_indices = torch.topk(logits, k=MOE_TOP_K, dim=1)
            top_k_gates = F.softmax(top_k_gates, dim=1)
            top_k_indices = top_k_indices.to(torch.long)
    else:
        probs = F.softmax(logits, dim=1)
        top_k_gates, top_k_indices = torch.topk(probs, k=MOE_TOP_K, dim=1)
        top_k_indices = top_k_indices.to(torch.long)
        # 兼容旧逻辑：最后一个 expert 视为 zero expert，强行置 0 并重归一化
        top_k_gates = torch.where(
            top_k_indices == (num_experts - 1),
            torch.zeros_like(top_k_gates),
            top_k_gates,
        )
        denom = top_k_gates.sum(dim=1, keepdim=True).clamp_min(1e-9)
        top_k_gates = top_k_gates / denom

    # 展开为扁平表
    token_ids = torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(1).expand(-1, MOE_TOP_K)
    flat_token_ids = token_ids.reshape(-1).contiguous()
    flat_expert_ids = top_k_indices.reshape(-1).contiguous()
    flat_gate_w = top_k_gates.reshape(-1).contiguous()

    # 统计并排序（按 expert 聚合）
    expert_counts = torch.bincount(flat_expert_ids, minlength=num_experts)
    sort_idx = flat_expert_ids.argsort()
    sorted_token_indices = flat_token_ids.index_select(0, sort_idx)
    sorted_expert_indices = flat_expert_ids.index_select(0, sort_idx)
    sorted_gate_weights = flat_gate_w.index_select(0, sort_idx)

    return top_k_indices, top_k_gates, sorted_token_indices, sorted_expert_indices, sorted_gate_weights, expert_counts


class Router(Module):
    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 moe_use_mixtral_gating: bool,
                 moe_2layer_gate: bool,
                 moe_use_logits_norm: bool,
                 moe_gate_norm_std: float,
                 ) -> None:
        super().__init__()

        if moe_2layer_gate:
            self.wg = torch.nn.Sequential(
                torch.nn.Linear(model_dim, num_experts * 8, bias=False),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts * 8, num_experts, bias=False),
            )
        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
            nn.init.normal_(self.wg.weight, std=0.01)

        self.gate_map = torch.nn.Linear(num_experts, num_experts, bias=False)

        self.gate = gating
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std
        expert_types = ['ffn'] * num_experts
        expert_types[-1] = 'zero'  # 最后一个专家是 zero
        self.load_balancer = LoadBalancer(
            num_experts=num_experts,
            expert_types=expert_types,
            tau=0.75,
            balance_loss_weight=1
        )
        # 缓存类型转换状态，避免每次forward都检查
        self._wg_converted = False
    
    def forward(self, input: torch.Tensor, gate_residual=None, compute_loss: bool = True):
        
        logits = self.wg(input)

        if gate_residual is not None:
       
            logits = logits + gate_residual  # 使用 + 而不是 +=，避免 in-place 操作问题

        top_k_indices, top_k_gates, sorted_token_indices, sorted_expert_indices, sorted_gate_weights, expert_counts = self.gate(
            logits, self.moe_use_mixtral_gating, self.moe_use_logits_norm, self.moe_gate_norm_std
        )
        vbalance_loss = self.load_balancer(logits, top_k_indices, compute_loss=compute_loss)
        # route 表：后续 MoE 前向只需要排序后的扁平表 + counts
        route = (sorted_token_indices, sorted_expert_indices, sorted_gate_weights, expert_counts, top_k_indices, top_k_gates)
        return route, logits, vbalance_loss


class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1):
        super(Experts, self).__init__()

        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for _ in range(num_local_experts - 2 - Constant)] +
            [ConstantExpert(expert) for _ in range(Constant)] +
            [CopyExpert(expert), ZeroExpert(expert)])

    def forward(self, inputs):
        raise NotImplementedError


class MOELayer(Base):
    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_size,
                 num_local_experts: int,
                 moe_use_mixtral_gating: bool,
                 moe_feature_no_mul_topk: bool) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

    def forward(self, *input: Tensor, gate_residual=None, compute_loss: bool = True, **kwargs: Any):
        d_model = input[0].shape[-1]
        input_shape = input[0].shape
        reshaped_input = input[0].view(-1, d_model)  # view 比 reshape 更快（如果可能）
        
        route, gate_residual, balance_loss = self.gate(reshaped_input, gate_residual=gate_residual, compute_loss=compute_loss)
        if not (self.moe_use_mixtral_gating or self.moe_feature_no_mul_topk):
            reshaped_input = reshaped_input * MOE_TOP_K
        
        # 计算专家利用率统计
        num_tokens = reshaped_input.shape[0]
        sorted_token_indices, sorted_expert_indices, sorted_gate_weights, expert_counts, top_k_indices, top_k_gates = route
        denom = float(max(1, num_tokens * MOE_TOP_K))
        expert_utilization = expert_counts.to(dtype=torch.float32) / denom
        
        # 推理优化：使用批量处理路径（类似 DeepSeek）
        # 在推理时（not training 且 compute_loss=False）使用优化路径
        if not self.training and not compute_loss:
            output = self._moe_infer(reshaped_input, sorted_token_indices, expert_counts, sorted_gate_weights)
        else:
            output = self._moe_train(reshaped_input, sorted_token_indices, expert_counts, sorted_gate_weights)
        
        output = output.view(input_shape)  # view 比 reshape 更快

        return output, gate_residual , balance_loss, expert_utilization
    
    def _moe_train(
        self,
        reshaped_input: Tensor,
        sorted_token_indices: torch.Tensor,
        expert_counts: torch.Tensor,
        sorted_gate_weights: torch.Tensor,
    ) -> Tensor:
        """
        训练路径：排序 + 分段批处理（张量化路由）
        """
        num_tokens, hidden_dim = reshaped_input.shape
        device = reshaped_input.device
        dtype = reshaped_input.dtype
        if sorted_token_indices.numel() == 0:
            return torch.zeros_like(reshaped_input)

        # 安全：确保索引合法
        sorted_token_indices = sorted_token_indices.to(device=device, dtype=torch.long).contiguous()
        sorted_token_indices = sorted_token_indices.clamp(0, num_tokens - 1)
        sorted_gate_weights = sorted_gate_weights.to(device=device, dtype=dtype).contiguous()

        sorted_tokens = reshaped_input.index_select(0, sorted_token_indices)
        expert_counts_list = expert_counts.to("cpu").tolist()

        # 5) 分段批处理每个 expert
        outputs = []
        start_idx = 0
        for expert_id, n_tok in enumerate(expert_counts_list):
            if n_tok == 0:
                continue
            end_idx = start_idx + n_tok

            expert_tokens = sorted_tokens[start_idx:end_idx]
            expert_gates = sorted_gate_weights[start_idx:end_idx]

            expert_output = self.experts.experts[expert_id](expert_tokens)
            if expert_output.dtype != dtype:
                expert_output = expert_output.to(dtype)
            expert_output = expert_output * expert_gates.unsqueeze(-1)

            outputs.append(expert_output)
            start_idx = end_idx

        if len(outputs) == 0:
            return torch.zeros_like(reshaped_input)

        sorted_outputs = torch.cat(outputs, dim=0)

        # 6) 聚合回原 token 顺序（Top-K 时同一 token 会被多次累加）
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
        output.index_add_(dim=0, index=sorted_token_indices, source=sorted_outputs)
        return output
    
    @torch.no_grad()
    def _moe_infer(
        self,
        reshaped_input: Tensor,
        sorted_token_indices: torch.Tensor,
        expert_counts: torch.Tensor,
        sorted_gate_weights: torch.Tensor,
    ) -> Tensor:
        """
        推理优化路径：类似 DeepSeek 的批量处理
        通过排序和批量处理减少循环开销，提升推理性能
        
        优化策略：
        1. 收集所有 token-expert 分配对
        2. 按专家ID排序，批量处理每个专家的所有 tokens
        3. 重新排序并累加输出（处理 Top-K 情况）
        """
        num_tokens, hidden_dim = reshaped_input.shape
        device = reshaped_input.device
        dtype = reshaped_input.dtype

        if sorted_token_indices.numel() == 0:
            return torch.zeros_like(reshaped_input)

        sorted_token_indices = sorted_token_indices.to(device=device, dtype=torch.long).contiguous()
        sorted_token_indices = sorted_token_indices.clamp(0, num_tokens - 1)
        sorted_gate_weights = sorted_gate_weights.to(device=device, dtype=dtype).contiguous()

        sorted_tokens = reshaped_input.index_select(0, sorted_token_indices)
        
        # 5. 批量处理每个专家
        outputs = []
        start_idx = 0
        
        for expert_id in range(self.num_local_experts):
            num_tokens_for_expert = int(expert_counts[expert_id].item())
            if num_tokens_for_expert == 0:
                continue
            
            end_idx = start_idx + num_tokens_for_expert
            
            # 获取该专家的 tokens
            expert_tokens = sorted_tokens[start_idx:end_idx]
            
            # 调用专家（批量处理）
            expert_output = self.experts.experts[expert_id](expert_tokens)
            
            # 应用门控权重
            expert_gates = sorted_gate_weights[start_idx:end_idx]
            if expert_output.dtype != dtype:
                expert_output = expert_output.to(dtype)
            expert_output = expert_output * expert_gates.unsqueeze(-1)
            
            outputs.append(expert_output)
            start_idx = end_idx
        
        # 6. 合并所有专家输出
        if len(outputs) == 0:
            return torch.zeros_like(reshaped_input)
        
        sorted_outputs = torch.cat(outputs, dim=0)
        
        # 7. 使用 index_add_ 累加输出（处理 Top-K 情况，同一 token 可能被多个专家处理）
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
        output.index_add_(dim=0, index=sorted_token_indices, source=sorted_outputs)
        
        return output


class MOEPP(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 moe_use_mixtral_gating=True,
                 moe_2layer_gate=False,
                 moe_use_logits_norm=True,
                 moe_gate_norm_std=1,
                 moe_feature_no_mul_topk=True):
        super(MOEPP, self).__init__()

        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_2layer_gate = moe_2layer_gate
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

        experts = Experts(expert, self.num_local_experts)
        self.moe = MOELayer(Router(hidden_size,
                                   num_experts,
                                   self.moe_use_mixtral_gating,
                                   self.moe_2layer_gate,
                                   self.moe_use_logits_norm,
                                   self.moe_gate_norm_std),
                            experts,
                            self.ep_size,
                            self.num_local_experts,
                            self.moe_use_mixtral_gating,
                            self.moe_feature_no_mul_topk,
                            )

    def forward(self, hidden_states, used_token=None, gate_residual=None, compute_loss=True):
        output, gate_residual, balance_loss, expert_utilization = self.moe(hidden_states, used_token, gate_residual=gate_residual, compute_loss=compute_loss)
        return output, gate_residual , balance_loss, expert_utilization


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
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
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

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None,
                block_kv_cache=None) -> tuple[torch.Tensor, dict]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, T_curr, C), where
                        B = batch size,
                        T_curr = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, T_curr, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        is_prefill = block_kv_cache is None

        B, T_curr, C = x.size()  # T_curr is the sequence length of the current input x

        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1,
                                                                                       2)  # (B, n_heads, T_curr, head_dim)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1,
                                                                                          2)  # (B, n_kv_heads, T_curr, head_dim)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1,
                                                                                          2)  # (B, n_kv_heads, T_curr, head_dim)

        # Apply rotary embeddings to the current q and k
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        # Check if we can use cached keys and values
        if not is_prefill and block_kv_cache['key'] is not None:
            # Concatenate with cached K, V
            # k_rotated and v_curr are for the new token(s)
            k = block_kv_cache['key']
            v = block_kv_cache['value']
            k = torch.cat([k, k_rotated], dim=2)
            v = torch.cat([v, v_curr], dim=2)
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        else:
            # No cache, this is the first pass (prefill)
            k = k_rotated
            v = v_curr
            block_kv_cache = {'key': k, 'value': v}

        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)  # (B, n_heads, T_kv, head_dim)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)  # (B, n_heads, T_kv, head_dim)

        T_kv = k_exp.size(2)  # Total sequence length of keys/values

        # Prepare attention mask for SDPA or manual path
        # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            mask_for_keys = attention_mask[:, :T_kv]  # Ensure mask matches key length [B, T_kv]
            additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            # This additive_attn_mask shape is [B, 1, 1, T_kv]

        if self.sdpa and x.device.type != 'mps':
            # During decode, no additional masking needed as [1, T_kv] is naturally causal
            is_causal = (T_curr == T_kv and T_curr > 1)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=None if is_causal else additive_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention implementation
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim)  # (B, n_heads, T_curr, T_kv)
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1,
                                                                                                                 T_curr,
                                                                                                                 T_curr)
                attn = attn.masked_fill(~causal_mask_val, float('-inf'))

            if additive_attn_mask is not None:  # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                attn = attn + additive_attn_mask

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp

        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache


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

class LanguageModelMLPExpert(nn.Module):
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
        self.inter_dim = cfg.lm_inter_dim//2

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
class LanguageModelBlockMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.mlp = LanguageModelMLP(cfg)
        self.moe = MOEPP(hidden_size=cfg.lm_hidden_dim, expert= LanguageModelMLP(cfg),num_experts=6)
        self.norm1 = RMSNorm(cfg)  # Input Norm
        self.norm2 = RMSNorm(cfg)  # Post Attention Norm

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor = None,
                block_kv_cache: dict = None, gate: torch.Tensor = None, compute_loss=True):
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
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x

        res = x
        x = self.norm2(x)
        x1 = self.mlp(x)
        x, g, balance_loss, expert_utilization = self.moe(x, gate_residual=gate, compute_loss=compute_loss)
        x = res  + x + x1
        return x, block_kv_cache, g, balance_loss, expert_utilization


class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg) # Input Norm
        self.norm2 = RMSNorm(cfg) # Post Attention Norm
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor=None, block_kv_cache: dict=None):
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
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)

        # 每3层加一个MoE block，其他使用正常block
        self.blocks = nn.ModuleList([
            LanguageModelBlockMoE(cfg) if (i + 1) % 3 == 0 else LanguageModelBlock(cfg)
            for i in range(cfg.lm_n_blocks)
        ])
        # self.blocks = nn.ModuleList([
        #     LanguageModelBlock(cfg)
        #     for i in range(cfg.lm_n_blocks)
        # ])
        self.norm = RMSNorm(cfg)  # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        self.all_balance_loss = 0.0

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
                start_pos: int = 0,compute_moe_loss: bool = True):
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
        """
        self.all_balance_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
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

        g = None
        expert_utilizations = []  # 收集所有 MoE 层的专家利用率
        for i, block in enumerate(self.blocks):
            # 根据block类型调用不同的forward方法
            if isinstance(block, LanguageModelBlockMoE):
                # MoE block需要传入gate参数
                x, kv_cache[i], g, balance_loss, expert_utilization = block(x, cos, sin, attention_mask, kv_cache[i], g, compute_loss=compute_moe_loss)
                self.all_balance_loss = self.all_balance_loss + balance_loss.to(torch.float32)
                expert_utilizations.append(expert_utilization)
            else:
                # 正常block不需要gate参数
                x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
        # for i, block in enumerate(self.blocks):

        #         x, kv_cache[i], g, balance_loss = block(x, cos, sin, attention_mask, kv_cache[i], g, compute_loss=compute_moe_loss)
        #         self.all_balance_loss += balance_loss

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens:
            x = self.head(x)

        # 计算平均专家利用率（跨所有 MoE 层）
        if expert_utilizations:
            avg_expert_utilization = torch.stack(expert_utilizations).mean(dim=0)  # [num_experts]
        else:
            avg_expert_utilization = None

        return x, kv_cache, self.all_balance_loss, avg_expert_utilization

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
            start_pos=0,
            compute_moe_loss=False
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
                start_pos=current_token_start_pos,
                compute_moe_loss=False
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
        cfg.lm_re_base = getattr(hf_config, "rope_theta", cfg.lm_re_base)
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

        # =================================================================================
        # MODIFICATION START: Implement "safe loading"
        # =================================================================================

        # 1. Get the state_dict of the *new* (potentially modified) model
        new_model_state_dict = model.state_dict()

        # 2. Create a new dictionary to hold only the weights we can load
        weights_to_load = {}

        # =================================================================================

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

        # This is a temporary state_dict to hold embedding weights for initialization
        # We need this because we modify new_model_state_dict in place for embeddings
        temp_sd_for_init = model.state_dict()

        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue

                    # Check if key exists in safetensor AND in our *new* model
                    if hf_key in f.keys() and our_key in new_model_state_dict:
                        tensor = f.get_tensor(hf_key)

                        # Special handling for token embeddings if vocab sizes differ
                        if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != \
                                new_model_state_dict[our_key].shape[0]:
                            has_extended_embeddings = True
                            print(
                                f"Extending token embeddings from {tensor.shape} to {new_model_state_dict[our_key].shape}")

                            # Copy existing embeddings to the beginning of our larger embedding matrix
                            # We use temp_sd_for_init here to prepare the new weight tensor
                            temp_sd_for_init[our_key][:tensor.shape[0]].copy_(tensor)

                            # Initialize the new embeddings using the same approach as the original model
                            std = 0.02  # Common value
                            init.normal_(temp_sd_for_init[our_key][tensor.shape[0]:], mean=0.0, std=std)

                            print(
                                f"Initialized {new_model_state_dict[our_key].shape[0] - tensor.shape[0]} new token embeddings")

                            # Add the *entire* newly prepared tensor to our load dict
                            weights_to_load[our_key] = temp_sd_for_init[our_key]

                            # Update the head weights as well if they are tied
                            if cfg.lm_tie_weights:
                                temp_sd_for_init['head.weight'].copy_(temp_sd_for_init[our_key])
                                weights_to_load['head.weight'] = temp_sd_for_init['head.weight']

                        # Normal case: shapes match
                        elif tensor.shape == new_model_state_dict[our_key].shape:
                            # Add the tensor to our load dict
                            weights_to_load[our_key] = tensor
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {new_model_state_dict[our_key].shape} (SKIPPING)")

                        loaded_keys.add(our_key)

        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                # Check against the new model's state dict
                if our_key in new_model_state_dict:
                    print(f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})")

        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in new_model_state_dict:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if 'lm_head.weight' in f.keys():
                        lm_head = f.get_tensor('lm_head.weight')
                        if lm_head.shape[0] != new_model_state_dict['head.weight'].shape[0]:
                            print(
                                f"Extending LM head from {lm_head.shape} to {new_model_state_dict['head.weight'].shape}")
                            # Copy existing weights (using temp_sd_for_init as the blueprint)
                            temp_sd_for_init['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(temp_sd_for_init['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                            # Add the prepared tensor to our load dict
                            weights_to_load['head.weight'] = temp_sd_for_init['head.weight']
                        else:
                            # Shapes match, just add it
                            weights_to_load['head.weight'] = lm_head

                        lm_head_loaded = True
                        break

        # =================================================================================
        # MODIFICATION: Final "safe load" step
        # =================================================================================

        # 3. Report and load only the matching weights
        loaded_count = len(weights_to_load)
        total_count = len(new_model_state_dict)
        print(f"Loaded {loaded_count} / {total_count} matching weights from checkpoint.")

        # 4. Load the state dict with strict=False
        model.load_state_dict(weights_to_load, strict=False)

        # =================================================================================
        # END OF MODIFICATION
        # =================================================================================

        # Handle weight tying (if needed) - this should come *after* loading
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")

        print(
            f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model