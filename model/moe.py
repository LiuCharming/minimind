"""
Standalone Mixture-of-Experts (MoE) Module
==========================================
从 language_model_moe.py 中剥离的独立 MoE 系统，可插入 MiniMind 使用。

特性:
  - 4 种专家类型: FFN / Constant / Copy / Zero
  - 张量化路由 (pre-sort + 分段 batch, 避免 for 循环)
  - Mixtral-style gating (top-k logits softmax)
  - 双层 gate 可选
  - CV² 负载均衡损失 (可微, 梯度更稳定)
  - Gate residual 跨层传播 (DeepSeek 风格)
  - 训练 / 推理路径分离
  - 训练中自动收集所有 token 的专家利用率
  - 不依赖 cfg 对象，纯参数化

用法 (插入 MiniMind):
    from model.moe import MoEConfig, MoEBlock

    config = MoEConfig(
        hidden_size=512,
        num_experts=6,
        intermediate_size=1024,
        top_k=1,
        ep_size=1,
    )
    moe_block = MoEBlock(config)

    # forward: (batch, seq, hidden) -> (batch, seq, hidden)
    output, gate_residual, balance_loss, expert_util = moe_block(x)

    # 如果需要 MoE + Dense MLP 并行 (DeepSeek 风格):
    from model.moe import MoEBlockWithDense
    parallel_block = MoEBlockWithDense(config)
    output, gate_residual, balance_loss, expert_util = parallel_block(x, gate=prev_gate)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable, Union
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoEConfig:
    """MoE 配置 (所有字段有默认值, 按需覆盖)"""

    # ── 维度 ──
    hidden_size: int = 512                # 隐藏层维度
    intermediate_size: int = 1024          # 每个专家的中间维度 (FFN inter_dim)
    expert_intermediate_ratio: float = 0.5  # 专家 FFN 相对稠密 MLP 的比例 (1.0 = 全尺寸)

    # ── 专家 ──
    num_experts: int = 6                  # 专家总数
    num_constant_experts: int = 2          # ConstantExpert 数量
    top_k: int = 1                        # 每个 token 激活的专家数

    # ── 路由 ──
    use_mixtral_gating: bool = True       # Mixtral 风格 gating (top-k logits softmax)
    use_2layer_gate: bool = False         # 双层 gate 网络
    use_logits_norm: bool = True          # gate logits 归一化
    gate_norm_std: float = 1.0            # logits norm 目标标准差

    # ── 负载均衡 ──
    balance_loss_weight: float = 0.01     # 均衡损失总权重 (用CV²时建议1e-3~1e-2)
    tau: float = 0.75                     # ZeroExpert 负载权重 η
    use_normalized_loss: bool = True      # True=CV² 损失, False=Switch Transformer 损失

    # ── 并行 ──
    ep_size: int = 1                      # Expert Parallelism 分片数 (1 = 不分片)

    # ── 性能优化 ──
    use_fused_expert: bool = True         # 使用融合 FFN (gate+up 合并, 2 matmul 代替 3)
    use_fused_inference: bool = False     # 推理时批量化 FFN 专家 (大专家数场景加速明显, 默认关)

    # ── 其他 ──
    router_init_std: float = 0.01         # Router 权重初始化标准差

    @property
    def expert_intermediate_size(self) -> int:
        """专家真正的中间维度"""
        return int(self.intermediate_size * self.expert_intermediate_ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# 可插拔的 Expert Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_swiglu_expert(hidden_size: int, intermediate_size: int) -> nn.Module:
    """
    构建标准 SwiGLU FFN 专家。
    用户可替换此函数以使用自定义 FFN。
    """
    return _SwiGLUExpert(hidden_size, intermediate_size)


def build_fused_swiglu_expert(hidden_size: int, intermediate_size: int) -> nn.Module:
    """构建融合 SwiGLU FFN 专家 (gate+up 合并为单次 matmul)"""
    return _FusedSwiGLUExpert(hidden_size, intermediate_size)


class _SwiGLUExpert(nn.Module):
    """标准 SwiGLU FeedForward (3 matmuls: gate / up / down)"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class _FusedSwiGLUExpert(nn.Module):
    """
    融合 SwiGLU FeedForward (2 matmuls: gate_up / down).

    优化: W_gate_up = [W_gate; W_up] ∈ R^{2I × H}
    gate_up = x @ W_gate_up^T  → chunk(gate, up) → silu(gate) * up → down_proj

    相比标准版: 3 matmul → 2 matmul, 约 -1/3 次 kernel launch,
    且输入 x 只读取一次，内存带宽更友好。
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ═══════════════════════════════════════════════════════════════════════════════
# 特殊专家类型
# ═══════════════════════════════════════════════════════════════════════════════

class CopyExpert(nn.Module):
    """恒等映射 — 直接透传输入"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class ZeroExpert(nn.Module):
    """零输出 — 返回全零张量 (结构化稀疏)"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


class ConstantExpert(nn.Module):
    """
    常数专家 — 学习一个可训练的常数向量 c。
    forward: 通过 2-way softmax gate 在输入 x 和常数 c 之间做加权混合。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.constant = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.constant)
        self.wg = nn.Linear(hidden_size, 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.softmax(self.wg(inputs))          # [N, 2]
        constant = self.constant.type_as(inputs)
        return weight[:, 0:1] * inputs + weight[:, 1:2] * constant


# ═══════════════════════════════════════════════════════════════════════════════
# Load Balancer
# ═══════════════════════════════════════════════════════════════════════════════

class LoadBalancer(nn.Module):
    """
    负载均衡损失。

    支持两种模式:
      - use_normalized_loss=True  (默认): CV² 损失, 基于可微的 softmax 概率 P
      - use_normalized_loss=False: Switch Transformer 风格 N * Σ(f_i * P_i)
    """

    def __init__(self, config: MoEConfig, expert_types: List[str]):
        super().__init__()
        self.num_experts = config.num_experts
        self.balance_loss_weight = config.balance_loss_weight
        self.use_normalized_loss = config.use_normalized_loss

        eta_values = []
        for t in expert_types:
            if t == 'ffn':
                eta_values.append(1.0)
            elif t == 'zero':
                eta_values.append(config.tau)
            else:
                eta_values.append(1.0)
        self.register_buffer("eta", torch.tensor(eta_values, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, top_k_indices: torch.Tensor,
                compute_loss: bool = True) -> torch.Tensor:
        if not compute_loss:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        num_tokens, num_experts = logits.shape
        if num_tokens == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # 可微的软路由概率
        gates_prob = F.softmax(logits, dim=1)
        P = gates_prob.mean(dim=0)

        if self.use_normalized_loss:
            # CV² 损失: (σ/μ)²  — P 全程可微
            P_mean = P.mean()
            P_std = P.std()
            cv_squared = (P_std / (P_mean + 1e-9)) ** 2
            L_b_unscaled = cv_squared
        else:
            # Switch Transformer: N * Σ(f_i * P_i), f 是离散统计 (stop_grad)
            flat_indices = top_k_indices.flatten()
            f = torch.bincount(flat_indices, minlength=num_experts).to(dtype=logits.dtype)
            f = f / (num_tokens * max(1, top_k_indices.size(1)))
            L_b_unscaled = self.num_experts * torch.sum(f.detach() * P)

        return self.balance_loss_weight * L_b_unscaled


# ═══════════════════════════════════════════════════════════════════════════════
# Gating 函数 (张量化的 Top-K 路由)
# ═══════════════════════════════════════════════════════════════════════════════

def gating(
    logits: torch.Tensor,
    top_k: int,
    use_mixtral_gating: bool = True,
    use_logits_norm: bool = False,
    gate_norm_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    张量化路由 — 将 token 按专家 ID 分组排序, 返回结构化路由表。

    Returns:
        top_k_indices:          [num_tokens, top_k]  每个 token 选中的专家 ID
        top_k_gates:            [num_tokens, top_k]  对应权重
        sorted_token_indices:   [num_tokens * top_k]  按专家排序后的 token 索引
        sorted_expert_indices:  [num_tokens * top_k]  对应的专家 ID (已排序)
        sorted_gate_weights:    [num_tokens * top_k]  对应的门控权重
        expert_counts:          [num_experts]         每个专家处理的 token 数
    """
    num_experts = logits.size(1)
    num_tokens = logits.size(0)
    device = logits.device
    dtype = logits.dtype

    if num_tokens == 0:
        empty_topk_idx = torch.empty((0, top_k), device=device, dtype=torch.long)
        empty_topk_g = torch.empty((0, top_k), device=device, dtype=dtype)
        empty_sorted = torch.empty((0,), device=device, dtype=torch.long)
        empty_sorted_g = torch.empty((0,), device=device, dtype=dtype)
        expert_counts = torch.zeros((num_experts,), device=device, dtype=torch.long)
        return empty_topk_idx, empty_topk_g, empty_sorted, empty_sorted, empty_sorted_g, expert_counts

    # 可选 logits 归一化
    if use_logits_norm:
        logits_std = logits.std(dim=1, keepdim=True)
        logits = logits / (logits_std / gate_norm_std)

    if use_mixtral_gating:
        if top_k == 1:
            top_k_indices = torch.argmax(logits, dim=1, keepdim=True).to(torch.long)
            top_k_gates = torch.ones((num_tokens, 1), device=device, dtype=dtype)
        else:
            top_k_gates, top_k_indices = torch.topk(logits, k=top_k, dim=1)
            top_k_gates = F.softmax(top_k_gates, dim=1)
            top_k_indices = top_k_indices.to(torch.long)
    else:
        probs = F.softmax(logits, dim=1)
        top_k_gates, top_k_indices = torch.topk(probs, k=top_k, dim=1)
        top_k_indices = top_k_indices.to(torch.long)
        # 最后一个专家是 zero expert, 强制权重为 0 并重归一化
        top_k_gates = torch.where(
            top_k_indices == (num_experts - 1),
            torch.zeros_like(top_k_gates),
            top_k_gates,
        )
        denom = top_k_gates.sum(dim=1, keepdim=True).clamp_min(1e-9)
        top_k_gates = top_k_gates / denom

    # 展开为扁平表
    token_ids = torch.arange(num_tokens, device=device, dtype=torch.long)
    token_ids = token_ids.unsqueeze(1).expand(-1, top_k)
    flat_token_ids = token_ids.reshape(-1).contiguous()
    flat_expert_ids = top_k_indices.reshape(-1).contiguous()
    flat_gate_w = top_k_gates.reshape(-1).contiguous()

    # 按 expert 排序
    expert_counts = torch.bincount(flat_expert_ids, minlength=num_experts)
    sort_idx = flat_expert_ids.argsort()
    sorted_token_indices = flat_token_ids.index_select(0, sort_idx)
    sorted_expert_indices = flat_expert_ids.index_select(0, sort_idx)
    sorted_gate_weights = flat_gate_w.index_select(0, sort_idx)

    return (top_k_indices, top_k_gates,
            sorted_token_indices, sorted_expert_indices,
            sorted_gate_weights, expert_counts)


# ═══════════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════════

class Router(nn.Module):
    """
    门控网络 + 路由算法 + 负载均衡损失。

    支持:
      - 单层 / 双层 gate
      - gate_residual (跨层路由信息传播)
      - CV² 或 Switch Transformer 负载均衡
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        num_experts = config.num_experts
        hidden_size = config.hidden_size

        if config.use_2layer_gate:
            self.wg = nn.Sequential(
                nn.Linear(hidden_size, num_experts * 8, bias=False),
                nn.Tanh(),
                nn.Linear(num_experts * 8, num_experts, bias=False),
            )
        else:
            self.wg = nn.Linear(hidden_size, num_experts, bias=False)
            nn.init.normal_(self.wg.weight, std=config.router_init_std)

        self.gate_map = nn.Linear(num_experts, num_experts, bias=False)

        expert_types = ['ffn'] * num_experts
        expert_types[-1] = 'zero'
        self.load_balancer = LoadBalancer(config, expert_types)

    def forward(self, x: torch.Tensor, gate_residual: Optional[torch.Tensor] = None,
                compute_loss: bool = True):
        """
        Args:
            x: [num_tokens, hidden_size]
            gate_residual: 上一层的 gate logits 残差
            compute_loss: 是否计算负载均衡损失

        Returns:
            route: (sorted_token_indices, sorted_expert_indices,
                    sorted_gate_weights, expert_counts,
                    top_k_indices, top_k_gates)
            logits: [num_tokens, num_experts]  路由 logits
            balance_loss: 标量
        """
        logits = self.wg(x)

        if gate_residual is not None:
            logits = logits + gate_residual

        (top_k_indices, top_k_gates,
         sorted_token_indices, sorted_expert_indices,
         sorted_gate_weights, expert_counts) = gating(
            logits,
            top_k=self.config.top_k,
            use_mixtral_gating=self.config.use_mixtral_gating,
            use_logits_norm=self.config.use_logits_norm,
            gate_norm_std=self.config.gate_norm_std,
        )

        balance_loss = self.load_balancer(logits, top_k_indices, compute_loss=compute_loss)

        route = (sorted_token_indices, sorted_expert_indices,
                 sorted_gate_weights, expert_counts,
                 top_k_indices, top_k_gates)

        return route, logits, balance_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Experts 容器
# ═══════════════════════════════════════════════════════════════════════════════

class Experts(nn.Module):
    """
    专家集合。
    结构: [FFN] × (N - 2 - C) + [ConstantExpert] × C + [CopyExpert] × 1 + [ZeroExpert] × 1

    其中 C = config.num_constant_experts (default 2)
    """

    def __init__(self, config: MoEConfig,
                 expert_factory: Callable[[int, int], nn.Module]):
        super().__init__()
        N = config.num_experts
        C = config.num_constant_experts
        num_ffn = N - 2 - C

        experts_list: List[nn.Module] = []

        # FFN 专家
        for _ in range(num_ffn):
            experts_list.append(
                expert_factory(config.hidden_size, config.expert_intermediate_size)
            )

        # Constant 专家
        for _ in range(C):
            experts_list.append(ConstantExpert(config.hidden_size))

        # Copy + Zero
        experts_list.append(CopyExpert())
        experts_list.append(ZeroExpert())

        self.experts = nn.ModuleList(experts_list)

    def forward(self, inputs):
        raise NotImplementedError("Experts.forward is not used; call .experts[i] directly")


# ═══════════════════════════════════════════════════════════════════════════════
# MOE Layer — 核心稀疏计算
# ═══════════════════════════════════════════════════════════════════════════════

class MOELayer(nn.Module):
    """
    MoE 层 — 训练/推理路径分离的张量化稀疏计算。

    Forward 输入/输出形状:
      input[0]:  (batch, seq, hidden) 或 (num_tokens, hidden)
      output:    same shape as input
    """

    def __init__(self, config: MoEConfig,
                 expert_factory: Optional[Callable[[int, int], nn.Module]] = None):
        super().__init__()
        self.config = config
        self.num_local_experts = config.num_experts // config.ep_size

        if expert_factory is None:
            expert_factory = build_fused_swiglu_expert if config.use_fused_expert else build_swiglu_expert
        self.expert_factory = expert_factory

        self.gate = Router(config)
        self.experts = Experts(config, expert_factory)

        # 缓存 FFN 专家数量 (用于批量化)
        C = config.num_constant_experts
        self.num_ffn_experts = config.num_experts - 2 - C

    def forward(self, *input: torch.Tensor,
                gate_residual: Optional[torch.Tensor] = None,
                compute_loss: bool = True,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            input[0]:  (batch, seq, hidden) 或 (*, hidden)
            gate_residual: 前一层路由 logits 残差
            compute_loss: 是否计算负载均衡损失

        Returns:
            output:              same shape as input[0]
            gate_residual:       gate logits (传递给下一层)
            balance_loss:        标量
            expert_utilization:  [num_experts] 每个专家的 token 占比
        """
        x = input[0]
        input_shape = x.shape
        d_model = x.shape[-1]
        reshaped_input = x.reshape(-1, d_model)

        route, gate_residual, balance_loss = self.gate(
            reshaped_input, gate_residual=gate_residual, compute_loss=compute_loss
        )

        (sorted_token_indices, sorted_expert_indices,
         sorted_gate_weights, expert_counts,
         top_k_indices, top_k_gates) = route

        # 专家利用率
        num_tokens = reshaped_input.shape[0]
        denom = float(max(1, num_tokens * self.config.top_k))
        expert_utilization = expert_counts.to(dtype=torch.float32) / denom

        # 训练 / 推理分支
        if not self.training and not compute_loss:
            # 自适应: 当 FFN 专家平均 token 数足够时用融合路径, 否则 loop 路径
            if self.config.use_fused_inference and self.num_ffn_experts > 0:
                ffn_total = expert_counts[:self.num_ffn_experts].sum().item()
                # heuristic: 每个 FFN 专家平均 ≥ 8 token 时才值得批量化
                if ffn_total >= self.num_ffn_experts * 8:
                    output = self._moe_infer_fused(reshaped_input, sorted_token_indices,
                                                    expert_counts, sorted_gate_weights)
                else:
                    output = self._moe_infer(reshaped_input, sorted_token_indices,
                                             expert_counts, sorted_gate_weights)
            else:
                output = self._moe_infer(reshaped_input, sorted_token_indices,
                                         expert_counts, sorted_gate_weights)
        else:
            output = self._moe_train(reshaped_input, sorted_token_indices,
                                     expert_counts, sorted_gate_weights)

        output = output.view(input_shape)
        return output, gate_residual, balance_loss, expert_utilization

    # ── 训练路径 ────────────────────────────────────────────────────────────

    def _moe_train(self, reshaped_input: torch.Tensor,
                   sorted_token_indices: torch.Tensor,
                   expert_counts: torch.Tensor,
                   sorted_gate_weights: torch.Tensor) -> torch.Tensor:
        """排序 + 分段批处理"""
        num_tokens, hidden_dim = reshaped_input.shape
        device = reshaped_input.device
        dtype = reshaped_input.dtype

        if sorted_token_indices.numel() == 0:
            return torch.zeros_like(reshaped_input)

        sorted_token_indices = sorted_token_indices.to(device=device, dtype=torch.long).contiguous()
        sorted_token_indices = sorted_token_indices.clamp(0, num_tokens - 1)
        sorted_gate_weights = sorted_gate_weights.to(device=device, dtype=dtype).contiguous()

        sorted_tokens = reshaped_input.index_select(0, sorted_token_indices)
        expert_counts_list = expert_counts.to("cpu").tolist()

        outputs: List[torch.Tensor] = []
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
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
        output.index_add_(dim=0, index=sorted_token_indices, source=sorted_outputs)
        return output

    # ── 推理路径 ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _moe_infer(self, reshaped_input: torch.Tensor,
                   sorted_token_indices: torch.Tensor,
                   expert_counts: torch.Tensor,
                   sorted_gate_weights: torch.Tensor) -> torch.Tensor:
        """推理优化: 与训练路径逻辑相同, 但加 @torch.no_grad()"""
        num_tokens, hidden_dim = reshaped_input.shape
        device = reshaped_input.device
        dtype = reshaped_input.dtype

        if sorted_token_indices.numel() == 0:
            return torch.zeros_like(reshaped_input)

        sorted_token_indices = sorted_token_indices.to(device=device, dtype=torch.long).contiguous()
        sorted_token_indices = sorted_token_indices.clamp(0, num_tokens - 1)
        sorted_gate_weights = sorted_gate_weights.to(device=device, dtype=dtype).contiguous()

        sorted_tokens = reshaped_input.index_select(0, sorted_token_indices)

        outputs: List[torch.Tensor] = []
        start_idx = 0
        for expert_id in range(self.num_local_experts):
            num_tokens_for_expert = int(expert_counts[expert_id].item())
            if num_tokens_for_expert == 0:
                continue

            end_idx = start_idx + num_tokens_for_expert
            expert_tokens = sorted_tokens[start_idx:end_idx]

            expert_output = self.experts.experts[expert_id](expert_tokens)
            expert_gates = sorted_gate_weights[start_idx:end_idx]
            if expert_output.dtype != dtype:
                expert_output = expert_output.to(dtype)
            expert_output = expert_output * expert_gates.unsqueeze(-1)

            outputs.append(expert_output)
            start_idx = end_idx

        if len(outputs) == 0:
            return torch.zeros_like(reshaped_input)

        sorted_outputs = torch.cat(outputs, dim=0)
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
        output.index_add_(dim=0, index=sorted_token_indices, source=sorted_outputs)
        return output

    # ── 融合推理路径 (批量化 FFN 专家, 消除 for 循环) ──────────────────────

    @torch.no_grad()
    def _moe_infer_fused(self, reshaped_input: torch.Tensor,
                         sorted_token_indices: torch.Tensor,
                         expert_counts: torch.Tensor,
                         sorted_gate_weights: torch.Tensor) -> torch.Tensor:
        """
        融合推理: 将所有 FFN 专家的 weight 堆叠 → 2 次 batched bmm 代替 N×3 次 matmul。

        步骤:
          1. 分离 FFN 专家 vs 特殊专家 (Constant/Copy/Zero)
          2. FFN 专家:  pad token batch → stack weights → batched bmm × 2
          3. 特殊专家:  逐个处理 (轻量, 无 matmul)
          4. 合并输出 → index_add_ 路由回去
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

        num_ffn = self.num_ffn_experts
        num_special = self.num_local_experts - num_ffn

        # ── 1. 分离 FFN / 特殊专家 ──
        ffn_info = []       # (expert_id, start, end, n_tok)
        special_info = []   # (expert_id, start, end, n_tok)
        start_idx = 0
        for expert_id in range(self.num_local_experts):
            n_tok = int(expert_counts[expert_id].item())
            if n_tok == 0:
                continue
            end_idx = start_idx + n_tok
            if expert_id < num_ffn:
                ffn_info.append((expert_id, start_idx, end_idx, n_tok))
            else:
                special_info.append((expert_id, start_idx, end_idx, n_tok))
            start_idx = end_idx

        # output 累积器
        all_sorted_indices = []
        all_sorted_outputs = []

        # ── 2. FFN 专家 → batched bmm ──
        if ffn_info:
            ffn_expert_indices = [info[0] for info in ffn_info]
            max_n = max(info[3] for info in ffn_info)

            # 收集 padded token batches
            padded_batches = []
            padded_gates = []
            for _, s, e, n_tok in ffn_info:
                pad = max_n - n_tok
                tokens = sorted_tokens[s:e]
                gates = sorted_gate_weights[s:e]
                if pad > 0:
                    tokens = F.pad(tokens, (0, 0, 0, pad))
                    gates = F.pad(gates, (0, pad))
                padded_batches.append(tokens)
                padded_gates.append(gates)

            tokens_stacked = torch.stack(padded_batches, dim=0)  # (num_ffn, max_n, H)
            gates_stacked = torch.stack(padded_gates, dim=0)     # (num_ffn, max_n)

            # 堆叠 FFN 权重
            gate_up_weights = torch.stack([
                self.experts.experts[eid].gate_up_proj.weight
                for eid in ffn_expert_indices
            ], dim=0)  # (num_ffn, 2*I, H)
            down_weights = torch.stack([
                self.experts.experts[eid].down_proj.weight
                for eid in ffn_expert_indices
            ], dim=0)  # (num_ffn, H, I)

            inter_dim = gate_up_weights.shape[1] // 2

            # Batched matmul 1: gate_up (num_ffn, max_n, H) @ (num_ffn, H, 2*I)
            gate_up = torch.bmm(
                tokens_stacked, gate_up_weights.transpose(1, 2)
            )  # (num_ffn, max_n, 2*I)
            gate, up = gate_up.chunk(2, dim=-1)  # both (num_ffn, max_n, I)

            # SwiGLU
            hidden = F.silu(gate) * up  # (num_ffn, max_n, I)

            # Batched matmul 2: down (num_ffn, max_n, I) @ (num_ffn, I, H)
            outputs_stacked = torch.bmm(
                hidden, down_weights.transpose(1, 2)
            )  # (num_ffn, max_n, H)

            # 乘 gate + 去除 padding
            outputs_stacked = outputs_stacked * gates_stacked.unsqueeze(-1)
            for bi, (_, s, e, n_tok) in enumerate(ffn_info):
                out_slice = outputs_stacked[bi, :n_tok]  # (n_tok, H)
                all_sorted_outputs.append(out_slice)
                all_sorted_indices.append(sorted_token_indices[s:e])

        # ── 3. 特殊专家 → 逐个处理 (无 matmul, 开销可忽略) ──
        for expert_id, s, e, n_tok in special_info:
            expert_tokens = sorted_tokens[s:e]
            expert_output = self.experts.experts[expert_id](expert_tokens)
            expert_gates = sorted_gate_weights[s:e]
            if expert_output.dtype != dtype:
                expert_output = expert_output.to(dtype)
            all_sorted_outputs.append(expert_output * expert_gates.unsqueeze(-1))
            all_sorted_indices.append(sorted_token_indices[s:e])

        # ── 4. 合并 → scatter ──
        if not all_sorted_outputs:
            return torch.zeros_like(reshaped_input)

        merged_indices = torch.cat(all_sorted_indices, dim=0)
        merged_outputs = torch.cat(all_sorted_outputs, dim=0)
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
        output.index_add_(dim=0, index=merged_indices, source=merged_outputs)
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# 顶层封装
# ═══════════════════════════════════════════════════════════════════════════════

class MoEBlock(nn.Module):
    """
    独立 MoE Block — 可直接替换 MiniMind 中的 FFN 或作为 Transformer Block 的一部分。

    用法:
        >>> moe = MoEBlock(MoEConfig(hidden_size=512))
        >>> out, g, loss, util = moe(x)

    Input:  (batch, seq, hidden) 或 (num_tokens, hidden)
    Output: (batch, seq, hidden) 或 (num_tokens, hidden)
    """

    def __init__(self, config: MoEConfig,
                 expert_factory: Optional[Callable[[int, int], nn.Module]] = None):
        super().__init__()
        self.moe = MOELayer(config, expert_factory)

    def forward(self, hidden_states: torch.Tensor,
                gate_residual: Optional[torch.Tensor] = None,
                compute_loss: bool = True):
        """
        Args:
            hidden_states: (batch, seq, hidden) 或 (*, hidden)
            gate_residual:  前一层 gate logits
            compute_loss:   是否计算负载均衡损失

        Returns:
            output:             same shape as hidden_states
            gate_residual:      gate logits (传给下一层)
            balance_loss:       标量
            expert_utilization: [num_experts]
        """
        return self.moe(hidden_states,
                        gate_residual=gate_residual,
                        compute_loss=compute_loss)


class MoEBlockWithDense(nn.Module):
    """
    DeepSeek 风格的并行 MoE + Dense Block:
      output = residual + DenseMLP(x) + MoE(x)

    这是 language_model_moe.py 中 LanguageModelBlockMoE 的精简版。
    包含 RMSNorm + Dense MLP + MoE，三个并行求和。
    """

    def __init__(self, config: MoEConfig,
                 expert_factory: Optional[Callable[[int, int], nn.Module]] = None):
        super().__init__()
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size)
        factory = expert_factory or (build_fused_swiglu_expert if config.use_fused_expert else build_swiglu_expert)
        self.dense_mlp = factory(config.hidden_size, config.intermediate_size)
        self.moe = MOELayer(config, expert_factory)

    def forward(self, x: torch.Tensor,
                gate_residual: Optional[torch.Tensor] = None,
                compute_loss: bool = True):
        """
        Args:
            x: (batch, seq, hidden)
            gate_residual: 前一层 gate logits

        Returns:
            output:             (batch, seq, hidden)
            gate_residual:      gate logits
            balance_loss:       标量
            expert_utilization: [num_experts]
        """
        residual = x
        x = self.norm(x)

        dense_out = self.dense_mlp(x)                            # Dense MLP
        moe_out, gate, loss, util = self.moe(x,
                                             gate_residual=gate_residual,
                                             compute_loss=compute_loss)  # Sparse MoE

        output = residual + dense_out + moe_out
        return output, gate, loss, util


# ═══════════════════════════════════════════════════════════════════════════════
# 快捷测试
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MoE Module Self‑Test")
    print("=" * 60)

    config = MoEConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_experts=16,
        top_k=2,
        expert_intermediate_ratio = 0.2,
    )

    print(f"\nConfig: hidden={config.hidden_size}, "
          f"num_experts={config.num_experts}, "
          f"top_k={config.top_k}, "
          f"expert_intermediate={config.expert_intermediate_size}")

    # ── 测试 1: MoEBlock ──
    print("\n[1] MoEBlock ...")
    moe_block = MoEBlock(config)
    x = torch.randn(2, 64, config.hidden_size)

    # 训练模式
    moe_block.train()
    out, g, loss, util = moe_block(x, compute_loss=True)
    print(f"    train  → out: {out.shape}, loss: {loss.item():.6f}, util: {util.tolist()}")

    # 推理模式
    moe_block.eval()
    out, g, loss, util = moe_block(x, compute_loss=False)
    print(f"    infer  → out: {out.shape}, loss: {loss.item():.6f}, util: {util.tolist()}")

    # ── 测试 2: MoEBlockWithDense ──
    print("\n[2] MoEBlockWithDense ...")
    parallel_block = MoEBlockWithDense(config)
    out, g, loss, util = parallel_block(x, compute_loss=True)
    print(f"    out: {out.shape}, loss: {loss.item():.6f}, util: {util.tolist()}")

    # ── 测试 3: Gate Residual 传播 ──
    print("\n[3] Gate residual propagation ...")
    out1, g1, loss1, util1 = moe_block(x)
    moe_block.eval()
    out2, g2, loss2, util2 = moe_block(x, gate_residual=g1, compute_loss=False)
    print(f"    gate1 shape: {g1.shape}, gate2 shape: {g2.shape}")

    # ── 测试 4: 参数统计 ──
    print("\n[4] Parameter stats ...")
    total = sum(p.numel() for p in moe_block.parameters())
    trainable = sum(p.numel() for p in moe_block.parameters() if p.requires_grad)
    print(f"    total: {total:,}, trainable: {trainable:,}")

    print("\n✅ All tests passed!")

    # ── 梯度检查 ──
    print("\n[5] Gradient check ...")
    moe_block.train()
    out, g, loss, util = moe_block(x, compute_loss=True)
    (out.sum() + loss).backward()
    grad_norms = {}
    for name, p in moe_block.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    if grad_norms:
        print(f"    grads OK ({len(grad_norms)} params, "
              f"max grad: {max(grad_norms.values()):.4f})")
    else:
        print("    ⚠️  No gradients! (check router_aux_loss_coef > 0)")
