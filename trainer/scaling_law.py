"""
Scaling Law 实验 — 基于 Shared MoE v2

研究：不同训练数据量下模型 loss 的变化规律
      L(N) ≈ a × N^(-α) + b

使用方法:
    cd F:\minimind\trainer
    python scaling_law.py

输出:
    out/scaling_law_results.json   → 原始数据
    out/scaling_law_plot.png       → 可视化图表
    out/scaling_law_*.pth          → 各数据量下的模型权重
"""

import os
import sys
import json
import time
import argparse
import warnings
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Subset

__package__ = "trainer"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr, Logger, lm_checkpoint, setup_seed,
    init_model, SkipBatchSampler, get_supported_dtype
)

warnings.filterwarnings('ignore')


def compute_loss_on_subset(model, loader, device, autocast_ctx, max_batches=50):
    """在验证集上计算平均 loss"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with autocast_ctx:
                res = model(input_ids, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_loss += res.loss.item() * n_tokens
            total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def train_one_run(args, data_fraction, run_seed):
    """
    用指定比例的训练数据训练一个 Shared MoE v2 模型。

    参数:
        args: 全局配置
        data_fraction: 使用多少比例的数据 (0.0~1.0)
        run_seed: 随机种子

    返回:
        dict: {
            'data_fraction': float,
            'num_tokens': int,        # 实际训练 token 数
            'num_steps': int,         # 训练步数
            'final_train_loss': float,
            'final_val_loss': float,
            'train_time_min': float,
        }
    """
    setup_seed(run_seed)

    # ── 1. 构建 Shared MoE v2 配置 ──
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=True,
        moe_type='v2',
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_expert_intermediate_ratio=args.moe_expert_intermediate_ratio,
        use_shared_ffn=True,  # ✅ Shared MoE v2
        flash_attn=bool(args.flash_attn),
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = get_supported_dtype(args.dtype, device_type)
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else torch.nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # ── 2. 加载模型 ──
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ── 3. 加载 & 子采样数据 ──
    full_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    n_total = len(full_ds)
    n_use = max(int(n_total * data_fraction), 1)
    indices = torch.randperm(n_total)[:n_use].tolist()

    train_ds = Subset(full_ds, indices)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── 4. 验证集 (固定取 5% 数据，不参与训练) ──
    val_indices = torch.randperm(n_total)[:max(int(n_total * 0.05), 16)].tolist()
    val_ds = Subset(full_ds, val_indices)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    iters_per_epoch = len(loader)
    total_iters = iters_per_epoch * args.epochs
    num_tokens = n_use * args.max_seq_len * args.epochs  # 近似

    Logger(f"\n{'='*60}")
    Logger(f"[ScalingLaw] data_fraction={data_fraction:.2f}  →  {n_use}/{n_total} samples")
    Logger(f"[ScalingLaw] iters_per_epoch={iters_per_epoch}, total_iters={total_iters}")
    Logger(f"[ScalingLaw] approx tokens ≈ {num_tokens / 1e6:.1f}M")
    Logger(f"{'='*60}")

    # ── 5. 训练循环 ──
    start_time = time.time()
    final_train_loss = 0.0

    for epoch in range(args.epochs):
        for step, (input_ids, labels) in enumerate(loader):
            global_step = epoch * iters_per_epoch + step
            lr = get_lr(global_step, total_iters, args.learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = res.loss + res.aux_loss
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            final_train_loss = loss.item() * args.accumulation_steps

            if (step + 1) % args.log_interval == 0:
                Logger(f"  frac={data_fraction:.2f} | epoch={epoch+1}/{args.epochs} "
                       f"step={step+1}/{iters_per_epoch} | loss={final_train_loss:.4f} | lr={lr:.2e}")

    # ── 6. 收尾 ──
    if hasattr(scaler, '_per_optimizer_states') and scaler._per_optimizer_states:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    train_time = (time.time() - start_time) / 60

    # ── 7. 评估 ──
    val_loss = compute_loss_on_subset(model, val_loader, args.device, autocast_ctx)
    Logger(f"[ScalingLaw] frac={data_fraction:.2f} | train_loss={final_train_loss:.4f} | val_loss={val_loss:.4f} | time={train_time:.1f}min")

    # ── 8. 保存权重 ──
    save_path = f"{args.save_dir}/scaling_law_frac{data_fraction:.2f}_{lm_config.hidden_size}.pth"
    os.makedirs(args.save_dir, exist_ok=True)
    raw_model = model
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, save_path)

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return {
        'data_fraction': data_fraction,
        'n_samples': n_use,
        'num_steps': total_iters,
        'num_tokens_approx': num_tokens,
        'final_train_loss': final_train_loss,
        'final_val_loss': val_loss,
        'train_time_min': train_time,
    }


def fit_scaling_law(results):
    """
    用最小二乘拟合: L(N) = a × N^(-α) + b

    返回拟合参数和 R²
    """
    tokens = np.array([r['num_tokens_approx'] for r in results])
    losses = np.array([r['final_val_loss'] for r in results])

    # 使用 log 空间线性回归: log(L - b) ≈ log(a) - α × log(N)
    # 简化：用 scipy curve_fit
    try:
        from scipy.optimize import curve_fit

        def law(N, a, alpha, b):
            return a * N ** (-alpha) + b

        # 初始猜测
        p0 = [10.0, 0.05, losses[-1] * 0.8]
        bounds = ([0, 0.001, 0], [np.inf, 1.0, losses[0]])
        popt, pcov = curve_fit(law, tokens, losses, p0=p0, bounds=bounds, maxfev=10000)
        a, alpha, b = popt

        # R²
        pred = law(tokens, *popt)
        ss_res = np.sum((losses - pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return {'a': a, 'alpha': alpha, 'b': b, 'r2': r2}
    except ImportError:
        Logger("[WARN] scipy not available, skipping curve fit")
        return None


def plot_results(results, fit, save_path):
    """绘制 Scaling Law 曲线"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        Logger("[WARN] matplotlib not available, skipping plot")
        return

    tokens = np.array([r['num_tokens_approx'] for r in results])
    train_losses = np.array([r['final_train_loss'] for r in results])
    val_losses = np.array([r['final_val_loss'] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── 左图: 线性坐标 ──
    ax1.plot(tokens / 1e6, train_losses, 'o-', color='#2196F3', label='Train Loss', markersize=8)
    ax1.plot(tokens / 1e6, val_losses, 's-', color='#FF5722', label='Val Loss', markersize=8)

    if fit:
        N_smooth = np.logspace(np.log10(tokens.min()), np.log10(tokens.max()), 200)
        pred_smooth = fit['a'] * N_smooth ** (-fit['alpha']) + fit['b']
        ax1.plot(N_smooth / 1e6, pred_smooth, '--', color='#FF5722', alpha=0.6,
                 label=f"Fit: L = {fit['a']:.2f} N^(-{fit['alpha']:.4f}) + {fit['b']:.4f}")

    ax1.set_xlabel('Training Tokens (M)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Shared MoE v2 — Scaling Law (Linear)', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 右图: log-log ──
    ax2.loglog(tokens, train_losses, 'o-', color='#2196F3', label='Train Loss', markersize=8)
    ax2.loglog(tokens, val_losses, 's-', color='#FF5722', label='Val Loss', markersize=8)

    if fit:
        ax2.loglog(N_smooth, pred_smooth, '--', color='#FF5722', alpha=0.6,
                   label=f"Fit: R²={fit['r2']:.4f}")

    ax2.set_xlabel('Training Tokens', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Shared MoE v2 — Scaling Law (Log-Log)', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    Logger(f"[ScalingLaw] Plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Scaling Law Experiment — Shared MoE v2")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1, help="每个数据量跑几轮（推荐1轮）")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_expert_intermediate_ratio", type=float, default=0.5)
    parser.add_argument("--flash_attn", type=int, default=0, help="Turing 显卡建议 0")
    parser.add_argument("--from_weight", type=str, default="none")
    # 控制要跑的数据比例
    parser.add_argument("--data_fractions", type=str, default="0.05,0.10,0.25,0.50,1.00",
                        help="逗号分隔的数据比例，如 '0.05,0.10,0.25,0.50,1.00'")
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    data_fractions = [float(x.strip()) for x in args.data_fractions.split(',')]
    Logger(f"[ScalingLaw] Data fractions: {data_fractions}")
    Logger(f"[ScalingLaw] Model: Shared MoE v2 | experts={args.num_experts} | top_k={args.num_experts_per_tok} | ratio={args.moe_expert_intermediate_ratio}")
    Logger(f"[ScalingLaw] hidden={args.hidden_size} | layers={args.num_hidden_layers} | epochs={args.epochs}")

    results = []
    for i, frac in enumerate(data_fractions):
        run_seed = args.base_seed + i * 100
        try:
            result = train_one_run(args, frac, run_seed)
            results.append(result)
        except Exception as e:
            Logger(f"[ERROR] frac={frac:.2f} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── 汇总 ──
    Logger(f"\n{'='*60}")
    Logger(f"[ScalingLaw] === 实验结果汇总 ===")
    Logger(f"{'Frac':>6s}  {'Samples':>8s}  {'Steps':>8s}  {'Tokens(M)':>10s}  {'TrainLoss':>10s}  {'ValLoss':>10s}  {'Time(min)':>10s}")
    for r in results:
        Logger(f"{r['data_fraction']:6.2f}  {r['n_samples']:8d}  {r['num_steps']:8d}  {r['num_tokens_approx']/1e6:10.1f}  {r['final_train_loss']:10.4f}  {r['final_val_loss']:10.4f}  {r['train_time_min']:10.1f}")

    # ── 拟合 Scaling Law ──
    fit = fit_scaling_law(results)
    if fit:
        Logger(f"\n[ScalingLaw] 拟合结果: L(N) = {fit['a']:.4f} × N^(-{fit['alpha']:.4f}) + {fit['b']:.4f}")
        Logger(f"[ScalingLaw] R² = {fit['r2']:.4f}")

    # ── 保存 ──
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = f"{args.save_dir}/scaling_law_results.json"
    with open(results_path, 'w') as f:
        json.dump({'results': results, 'fit': fit, 'config': {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_hidden_layers,
            'num_experts': args.num_experts,
            'num_experts_per_tok': args.num_experts_per_tok,
            'moe_expert_intermediate_ratio': args.moe_expert_intermediate_ratio,
            'use_shared_ffn': True,
            'moe_type': 'v2',
        }}, f, indent=2, ensure_ascii=False)
    Logger(f"[ScalingLaw] Results saved to: {results_path}")

    # ── 画图 ──
    plot_path = f"{args.save_dir}/scaling_law_plot.png"
    plot_results(results, fit, plot_path)

    Logger(f"\n[ScalingLaw] ✅ 实验完成!")
    Logger(f"[ScalingLaw] 图表: {plot_path}")
    Logger(f"[ScalingLaw] 数据: {results_path}")


if __name__ == "__main__":
    main()
