"""
Scaling Law 实验 — 基于 MiniMind 模型架构
==========================================

复现 Kaplan (2020) / Chinchilla (2022) 的核心发现:

    Loss(N, D) = A + B / N^α + C / D^β

其中:
    N = 非 embedding 参数量
    D = 训练 token 数
    A = 不可约损失 (irreducible loss)
    α = 模型缩放指数
    β = 数据缩放指数

实验设计:
    1. 用 MiniMindConfig 构造不同大小的模型 (参数量跨越 ~2 个数量级)
    2. 每个模型在不同数据量下训练
    3. 用最小二乘法拟合幂律曲线
    4. 可视化 + 预测

运行:
    python scaling_law_experiment.py           # 完整实验 (较耗时)
    python scaling_law_experiment.py --quick   # 快速验证 (缩小网格)
"""

import math
import os
import sys
import argparse
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 确保项目路径在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model_minimind import (
    MiniMindConfig,
    MiniMindModel,
    MiniMindForCausalLM,
    MiniMindBlock,
    Attention,
    FeedForward,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
    repeat_kv,
)

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def count_non_embedding_params(model: MiniMindForCausalLM) -> int:
    """
    计算非 embedding 参数量 (Kaplan 论文中的 N)

    排除:
      - model.embed_tokens (词嵌入)
      - lm_head (与 embedding 共享权重时, 不计入)
    """
    n = 0
    for name, param in model.named_parameters():
        if "embed_tokens" in name:
            continue
        if "lm_head" in name and model.config.tie_word_embeddings:
            continue
        n += param.numel()
    return n


def count_total_params(model: MiniMindForCausalLM) -> int:
    return sum(p.numel() for p in model.parameters())


def make_model_config(
    hidden_size: int,
    num_layers: int,
    vocab_size: int = 6400,
    max_seq_len: int = 2048,
) -> MiniMindConfig:
    """构造一个指定规模的 MiniMindConfig"""
    head_dim = 64  # 固定 head_dim, 与原始 MiniMind 一致
    num_heads = hidden_size // head_dim
    num_kv_heads = max(1, num_heads // 2)  # GQA: KV heads = Q heads 的一半

    intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64

    return MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_seq_len,
        dropout=0.0,
        flash_attn=True,
        use_moe=False,
        use_moh=False,
        use_shared_ffn=False,
        use_fff=False,
        tie_word_embeddings=True,
        rope_theta=1e4,
    )


# ══════════════════════════════════════════════════════════════
# 合成数据集 (与原始 Scaling Law 实验一致)
# ══════════════════════════════════════════════════════════════

class SyntheticLanguageDataset(Dataset):
    """
    用 1-gram 概率分布 + 马尔可夫链生成 token 序列。

    让数据有可预测但非确定的结构 — 这样不同大小的模型
    才会表现出不同的收敛行为 (纯随机 token 无法体现 scaling)。
    """

    def __init__(self, vocab_size: int, total_tokens: int, seq_len: int = 256):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_tokens = total_tokens

        rng = np.random.RandomState(42)
        # 低频 + 高频词的不均匀 Zipf 分布
        probs = 1.0 / np.arange(1, vocab_size + 1)
        probs = probs / probs.sum()

        self.transition = np.zeros((vocab_size, vocab_size))
        for i in range(vocab_size):
            row = probs.copy()
            # 自循环 + 局部循环
            row[i] += 0.1
            for j in range(max(0, i - 5), min(vocab_size, i + 5)):
                row[j] += 0.05
            row = row / row.sum()
            self.transition[i] = row

        self.n_samples = total_tokens // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        tokens = np.zeros(self.seq_len, dtype=np.int64)
        tokens[0] = rng.randint(0, self.vocab_size)
        for i in range(1, self.seq_len):
            tokens[i] = rng.choice(self.vocab_size, p=self.transition[tokens[i - 1]])
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])


# ══════════════════════════════════════════════════════════════
# 训练循环
# ══════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    config_desc: str
    n_params_non_emb: int
    n_params_total: int
    n_tokens: int
    final_loss: float
    loss_history: list
    tokens_history: list
    train_time: float


def train_minimind(
    config: MiniMindConfig,
    n_train_tokens: int,
    batch_size: int = 32,
    seq_len: int = 256,
    lr: float = 3e-4,
    device: str = "cuda",
    verbose: bool = True,
) -> TrainResult:
    """
    训练一个 MiniMind 模型, 在指定数量的 token 上做 next-token prediction。
    """
    model = MiniMindForCausalLM(config).to(device)
    model.train()

    n_params_non_emb = count_non_embedding_params(model)
    n_params_total = count_total_params(model)

    # 构造训练数据
    dataset = SyntheticLanguageDataset(
        vocab_size=config.vocab_size,
        total_tokens=n_train_tokens,
        seq_len=seq_len,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))

    n_steps = n_train_tokens // (batch_size * seq_len)
    total_tokens_seen = 0

    losses = []
    tokens_history = []

    # 每 10% 的步数记录一次 loss
    log_every = max(1, n_steps // 20)

    t0 = time.time()

    for step, (x, y) in enumerate(dataloader):
        if total_tokens_seen >= n_train_tokens:
            break

        x, y = x.to(device), y.to(device)

        # MiniMindForCausalLM.forward 接受 input_ids 和 labels
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (MiniMind 训练脚本中的惯例)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_tokens_seen += x.numel()

        if step % log_every == 0 or step >= n_steps - 1:
            losses.append(loss.item())
            tokens_history.append(total_tokens_seen)

        if step >= n_steps - 1:
            break

    train_time = time.time() - t0

    # 最后 10% 的 loss 平均值作为 final_loss (平滑)
    n_final = max(1, len(losses) // 10)
    final_loss = float(np.mean(losses[-n_final:]))

    desc = f"d={config.hidden_size} L={config.num_hidden_layers}"

    if verbose:
        print(f"  {desc} | tokens={n_train_tokens/1e6:.1f}M | "
              f"loss={final_loss:.4f} | time={train_time:.0f}s | "
              f"params={n_params_non_emb/1e6:.2f}M(non-emb)")

    # 释放显存
    del model
    torch.cuda.empty_cache()

    return TrainResult(
        config_desc=desc,
        n_params_non_emb=n_params_non_emb,
        n_params_total=n_params_total,
        n_tokens=n_train_tokens,
        final_loss=final_loss,
        loss_history=losses,
        tokens_history=tokens_history,
        train_time=train_time,
    )


# ══════════════════════════════════════════════════════════════
# 拟合 Scaling Law
# ══════════════════════════════════════════════════════════════

@dataclass
class ScalingLawFit:
    A: float        # 不可约损失
    B: float        # 模型缩放系数
    C: float        # 数据缩放系数
    alpha: float    # 模型缩放指数
    beta: float     # 数据缩放指数
    r2: float       # 拟合优度
    predictions: np.ndarray
    formula: str


def fit_scaling_law(results: list[TrainResult]) -> ScalingLawFit:
    """
    拟合 Loss(N, D) = A + B / N^α + C / D^β

    方法: 在 log-log 空间做多元线性回归 + 网格搜索最优 A
    """
    N = np.array([r.n_params_non_emb for r in results])
    D = np.array([r.n_tokens for r in results])
    L = np.array([r.final_loss for r in results])

    print(f"\n{'='*60}")
    print("拟合 Scaling Law")
    print(f"{'='*60}")
    print(f"数据点: {len(results)}")
    print(f"N 范围: {N.min()/1e6:.1f}M ~ {N.max()/1e6:.1f}M")
    print(f"D 范围: {D.min()/1e6:.1f}M ~ {D.max()/1e6:.1f}M")
    print(f"L 范围: {L.min():.4f} ~ {L.max():.4f}")

    best_r2 = -np.inf
    best_result = None

    # 网格搜索 A (不可约损失应略低于最低观测 loss)
    for A in np.linspace(L.min() * 0.3, L.min() * 0.95, 30):
        L_adj = np.maximum(L - A, 1e-8)
        log_L = np.log(L_adj)

        # 多元线性回归: log(L_adj) = log(B) - α·log(N) - β·log(D)
        #                       = const - α·log(N) - β·log(D)
        X = np.stack([
            np.ones_like(N),
            np.log(N),
            np.log(D),
        ], axis=1)

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, log_L, rcond=None)
            const, neg_alpha, neg_beta = coeffs
            alpha = -neg_alpha
            beta = -neg_beta
            B_est = np.exp(const)

            # C 的估计: 用一个合理的分解
            # Loss ≈ A + B/N^α + C/D^β
            # 用中位数来估计 C
            C_est = float(np.median(L_adj - B_est / (N ** alpha)) * np.median(D ** beta))
            C_est = max(C_est, 1e-8)

            L_pred = A + B_est / (N ** alpha) + C_est / (D ** beta)

            ss_res = np.sum((L - L_pred) ** 2)
            ss_tot = np.sum((L - L.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 > best_r2 and alpha > 0.01 and beta > 0.01:
                best_r2 = r2
                best_result = (A, B_est, C_est, alpha, beta, L_pred)
        except np.linalg.LinAlgError:
            continue

    if best_result is None:
        raise RuntimeError("无法拟合 Scaling Law, 请检查数据")

    A, B, C, alpha, beta, L_pred = best_result

    formula = f"Loss(N, D) = {A:.4f} + {B:.2e}/N^{alpha:.3f} + {C:.2e}/D^{beta:.3f}"

    return ScalingLawFit(
        A=A, B=B, C=C, alpha=alpha, beta=beta,
        r2=best_r2, predictions=L_pred, formula=formula,
    )


# ══════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════

def build_model_grid(quick: bool = False) -> list[MiniMindConfig]:
    """构造模型规模网格"""

    if quick:
        # 快速模式: 3 个模型, 每个 ~3 个数据点
        return [
            make_model_config(hidden_size=128,  num_layers=2,  vocab_size=1024, max_seq_len=512),
            make_model_config(hidden_size=256,  num_layers=4,  vocab_size=2048, max_seq_len=512),
            make_model_config(hidden_size=384,  num_layers=6,  vocab_size=4096, max_seq_len=512),
        ]
    else:
        # 完整模式: 5 个模型, 参数量跨越 ~2 个数量级
        return [
            # hidden, layers,  ~non-emb params
            make_model_config(hidden_size=64,   num_layers=2,  vocab_size=1024, max_seq_len=512),   # ~0.1M
            make_model_config(hidden_size=128,  num_layers=3,  vocab_size=1024, max_seq_len=512),   # ~0.5M
            make_model_config(hidden_size=192,  num_layers=5,  vocab_size=2048, max_seq_len=512),   # ~2M
            make_model_config(hidden_size=256,  num_layers=7,  vocab_size=4096, max_seq_len=512),   # ~5M
            make_model_config(hidden_size=384,  num_layers=9,  vocab_size=6400, max_seq_len=512),   # ~15M
        ]


def build_token_budgets(quick: bool = False) -> list[int]:
    """构造数据量网格"""
    if quick:
        return [200_000, 1_000_000, 5_000_000]
    else:
        return [
            100_000,
            300_000,
            1_000_000,
            3_000_000,
            10_000_000,
            25_000_000,
        ]


def run_scaling_law_experiment(
    quick: bool = False,
    device: str = "cuda",
    lr: float = 3e-4,
    batch_size: int = 32,
) -> list[TrainResult]:
    """
    核心实验: 在不同 (模型大小, 数据量) 组合上训练, 收集 loss
    """
    model_grid = build_model_grid(quick)
    token_budgets = build_token_budgets(quick)
    seq_len = 128  # 短序列, 加快训练

    print("=" * 70)
    print("MiniMind Scaling Law 实验")
    print("=" * 70)
    print(f"模型数量: {len(model_grid)}")
    print(f"数据预算: {[f'{t/1e6:.1f}M' for t in token_budgets]}")
    print(f"序列长度: {seq_len}")
    print(f"设备: {device}")
    print()

    results: list[TrainResult] = []

    for i, cfg in enumerate(model_grid):
        # 预估参数量
        temp_model = MiniMindForCausalLM(cfg)
        n_params = count_non_embedding_params(temp_model)
        del temp_model

        print(f"[{i+1}/{len(model_grid)}] "
              f"d={cfg.hidden_size}, L={cfg.num_hidden_layers}, "
              f"N_est≈{n_params/1e6:.2f}M")
        print("-" * 70)

        for n_tokens in token_budgets:
            # 跳过不合理组合: 小模型 + 超大数据
            if n_tokens > 5_000_000 and n_params < 500_000:
                continue
            # 大模型 + 极少数据 (欠拟合太严重, 跳过)
            if n_tokens < 500_000 and n_params > 3_000_000:
                continue

            result = train_minimind(
                config=cfg,
                n_train_tokens=n_tokens,
                batch_size=batch_size,
                seq_len=seq_len,
                lr=lr,
                device=device,
                verbose=True,
            )
            results.append(result)

        print()

    return results


# ══════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════

def plot_scaling_law(results: list[TrainResult], fit: ScalingLawFit, save_path: str = "scaling_law_minimind.png"):
    """绘制 Scaling Law 四合一图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠ matplotlib 未安装, 跳过绘图。 pip install matplotlib")
        return

    N = np.array([r.n_params_non_emb for r in results])
    D = np.array([r.n_tokens for r in results])
    L = np.array([r.final_loss for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("MiniMind Scaling Law — Loss(N, D) = A + B/N^α + C/D^β",
                 fontsize=14, fontweight='bold')

    # ── 子图 1: Loss vs Model Size (固定数据量) ──
    ax1 = axes[0, 0]
    token_bins = defaultdict(list)
    for r in results:
        token_bins[r.n_tokens].append((r.n_params_non_emb, r.final_loss))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(token_bins)))
    for (n_tok, pts), c in zip(sorted(token_bins.items()), colors):
        pts = sorted(pts)
        ax1.loglog([p[0] for p in pts], [p[1] for p in pts], 'o-', color=c,
                   label=f'D={n_tok/1e6:.1f}M', markersize=7)
    ax1.set_xlabel("Non-embedding Parameters N")
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Loss vs Model Size")
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ── 子图 2: Loss vs Data (固定模型大小) ──
    ax2 = axes[0, 1]
    param_bins = defaultdict(list)
    for r in results:
        param_bins[r.n_params_non_emb].append((r.n_tokens, r.final_loss))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(param_bins)))
    for (n_par, pts), c in zip(sorted(param_bins.items()), colors):
        pts = sorted(pts)
        ax2.loglog([p[0] for p in pts], [p[1] for p in pts], 's-', color=c,
                   label=f'N={n_par/1e6:.2f}M', markersize=7)
    ax2.set_xlabel("Training Tokens D")
    ax2.set_ylabel("Test Loss")
    ax2.set_title("Loss vs Training Data")
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ── 子图 3: 拟合 vs 实际 ──
    ax3 = axes[1, 0]
    ax3.scatter(L, fit.predictions, alpha=0.7, s=70, c='steelblue', edgecolors='white')
    lims = [min(L.min(), fit.predictions.min()) * 0.95,
            max(L.max(), fit.predictions.max()) * 1.05]
    ax3.plot(lims, lims, '--', color='red', linewidth=1.5, label='y = x')
    ax3.set_xlabel("Actual Loss")
    ax3.set_ylabel("Predicted Loss")
    ax3.set_title(f"Scaling Law Fit (R² = {fit.r2:.4f})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── 子图 4: N-D 平面上的 loss 热力图 ──
    ax4 = axes[1, 1]
    sc = ax4.scatter(np.log10(N), np.log10(D), c=L, cmap='RdYlBu_r',
                     s=120, edgecolors='black', linewidth=0.7,
                     vmin=L.min(), vmax=L.max())
    ax4.set_xlabel("log₁₀(N)")
    ax4.set_ylabel("log₁₀(D)")
    ax4.set_title("Loss Landscape")

    # 叠加等 loss 线
    n_vals = np.logspace(np.log10(N.min()) - 0.1, np.log10(N.max()) + 0.1, 50)
    d_vals = np.logspace(np.log10(D.min()) - 0.1, np.log10(D.max()) + 0.1, 50)
    NN, DD = np.meshgrid(n_vals, d_vals)
    LL = fit.A + fit.B / (NN ** fit.alpha) + fit.C / (DD ** fit.beta)
    levels = np.linspace(LL.min(), LL.max(), 6)
    contour = ax4.contour(np.log10(NN), np.log10(DD), LL, levels=levels,
                          colors='black', linewidths=0.8, alpha=0.5)
    ax4.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    plt.colorbar(sc, ax=ax4, label='Loss', shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n图像已保存至: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 结果报告
# ══════════════════════════════════════════════════════════════

def print_report(results: list[TrainResult], fit: ScalingLawFit):
    """打印完整实验报告"""
    print(f"\n{'='*70}")
    print("SCALING LAW 实验报告 — MiniMind 架构")
    print(f"{'='*70}")

    print(f"\n── 模型网格 ──")
    seen = set()
    for r in results:
        key = (r.n_params_non_emb, r.config_desc)
        if key not in seen:
            seen.add(key)
            print(f"  {r.config_desc:20s}  N(non-emb)={r.n_params_non_emb/1e6:.2f}M  "
                  f"N(total)={r.n_params_total/1e6:.2f}M")

    print(f"\n── 拟合结果 ──")
    print(f"  {fit.formula}")
    print(f"  R² = {fit.r2:.4f}")
    print()

    # 与文献对比
    print(f"── 与文献对比 ──")
    print(f"  {'':25s} {'α':>8s}  {'β':>8s}  {'α/β':>8s}")
    print(f"  {'Kaplan (2020)':25s}  {0.076:>8.3f}  {0.095:>8.3f}  {0.076/0.095:>8.2f}")
    print(f"  {'Chinchilla (2022)':25s}  {0.34:>8.3f}  {0.28:>8.3f}  {0.34/0.28:>8.2f}")
    print(f"  {'本实验 (MiniMind)':25s}  {fit.alpha:>8.3f}  {fit.beta:>8.3f}  "
          f"{fit.alpha/fit.beta:>8.2f}")
    print()

    # Chinchilla 最优性分析
    if fit.alpha > 0 and fit.beta > 0:
        ratio = fit.alpha / fit.beta
        if abs(ratio - 1.0) < 0.3:
            print(f"  α/β ≈ {ratio:.2f} → 模型和数据应近似等比缩放 (Chinchilla 风格)")
        elif ratio < 1.0:
            print(f"  α/β ≈ {ratio:.2f} → 增加数据的收益 > 增加参数的收益 (偏向 Kaplan)")
        else:
            print(f"  α/β ≈ {ratio:.2f} → 增加参数的收益 > 增加数据的收益")

    # 用拟合公式预测
    print(f"\n── 外推预测 ──")
    test_cases = [
        (10_000_000, 100_000_000, "10M params, 100M tokens"),
        (50_000_000, 500_000_000, "50M params, 500M tokens"),
        (100_000_000, 1_000_000_000, "100M params, 1B tokens"),
        (100_000_000, 10_000_000_000, "100M params, 10B tokens"),
    ]
    for n, d, label in test_cases:
        pred = fit.A + fit.B / (n ** fit.alpha) + fit.C / (d ** fit.beta)
        print(f"  {label:30s} → Loss ≈ {pred:.4f}")

    print(f"\n{'='*70}\n")


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MiniMind Scaling Law Experiment")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 (更小的模型和数据网格)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备 (cuda / cpu)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA 不可用, 回退到 CPU")
        args.device = "cpu"

    print(f"随机种子: {args.seed}")
    print(f"快速模式: {args.quick}")

    # 1) 运行实验
    results = run_scaling_law_experiment(
        quick=args.quick,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # 2) 拟合 Scaling Law
    fit = fit_scaling_law(results)

    # 3) 报告
    print_report(results, fit)

    # 4) 可视化
    plot_scaling_law(results, fit)

    # 5) 保存原始数据 (方便后续分析)
    import json
    data_path = "scaling_law_data.json"
    with open(data_path, "w") as f:
        json.dump([{
            "config": r.config_desc,
            "n_params_non_emb": r.n_params_non_emb,
            "n_params_total": r.n_params_total,
            "n_tokens": r.n_tokens,
            "final_loss": r.final_loss,
            "train_time": r.train_time,
        } for r in results], f, indent=2)
    print(f"原始数据已保存至: {data_path}")


if __name__ == "__main__":
    main()
