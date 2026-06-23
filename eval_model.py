"""
MiniMind 模型综合性能评估
===========================
评测维度:
  1. 验证集困惑度 (PPL)
  2. 生成质量 (重复率 / 长度 / 多样性)
  3. 参数统计 (权重范数 / 层间分布)
  4. 注意力统计 (熵 / 长程关注距离)
  5. MoE 专家均衡度
  6. MoH Q头均衡度
  7. 推理速度基准 (tokens/s)

用法:
    python eval_model.py --weight pretrain --num_samples 200
    python eval_model.py --weight full_sft --use_moe 1 --moe_type v2 --num_experts 12 --num_experts_per_tok 2
    python eval_model.py --weight pretrain --use_moh 1 --num_attention_heads 12 --moh_shared_heads 6 --moh_routed_head 2
"""

import os, sys, time, math, json, argparse, warnings
import torch, torch.nn.functional as F
from collections import defaultdict
from contextlib import nullcontext
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════════
def init_model(args):
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        moe_type=args.moe_type,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_expert_intermediate_ratio=args.moe_expert_intermediate_ratio,
        use_moh=bool(args.use_moh),
        moh_shared_heads=args.moh_shared_heads,
        moh_routed_head=args.moh_routed_head,
        num_attention_heads=args.num_attention_heads,
        inference_rope_scaling=args.inference_rope_scaling,
    )
    moe_suffix = '_moe' if args.use_moe else ''
    project_root = os.path.dirname(os.path.abspath(__file__))
    ckp = os.path.join(project_root, args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth')
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    model = model.half().eval().to(args.device)
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 困惑度 (Perplexity)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_perplexity(model, tokenizer, data_path, max_length, device, n_samples=200, batch_size=1):
    """在 pretrain 验证集子集上计算困惑度"""
    from dataset.lm_dataset import PretrainDataset

    print(f"\n{'═' * 60}")
    print(f"[1/7] 验证集困惑度 (Perplexity)")

    ds = PretrainDataset(data_path, tokenizer, max_length=max_length)
    indices = torch.randperm(len(ds))[:n_samples].tolist()

    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            input_ids, labels = ds[idx]
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(dtype=model.dtype) if device.type == 'cuda' else nullcontext():
                out = model(input_ids, labels=labels)

            loss = out.loss  # logits_loss only (no aux_loss in eval)
            total_loss += loss.item() * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{n_samples}] ppl: {math.exp(total_loss / max(total_tokens, 1)):.2f}")

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)

    print(f"  ─────────────────────────────")
    print(f"  样本数:    {n_samples}")
    print(f"  Avg Loss:  {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  (越低越好, 接近1=过拟合, 远大于训练loss=欠拟合)")
    return {"n_samples": n_samples, "avg_loss": avg_loss, "perplexity": ppl}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 生成质量
# ═══════════════════════════════════════════════════════════════════════════════
def compute_generation_quality(model, tokenizer, device, max_new_tokens=256, temperature=0.85, top_p=0.95):
    """自动评测: 重复率/长度/生成速度"""
    print(f"\n{'═' * 60}")
    print(f"[2/7] 生成质量")

    prompts = [
        "请介绍一下人工智能的发展历史",
        "用Python写一个冒泡排序",
        "解释一下量子力学的基本原理",
        "推荐几种健康的早餐",
        "什么是机器学习中的过拟合",
        "比较一下CPU和GPU的区别",
        "如何学习一门新的编程语言",
        "描述一下太阳系的结构",
    ]

    metrics = {
        "speed": [],
        "output_len": [],
        "rep_1gram": [],  # 1-gram 重复率
        "rep_2gram": [],  # 2-gram 重复率
        "rep_3gram": [],
        "diversity": [],  # unique/total token ratio
    }

    for i, prompt in enumerate(prompts):
        inputs = tokenizer.bos_token + prompt
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(device)

        st = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=model.dtype) if device.type == 'cuda' else nullcontext():
                generated = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                )
        elapsed = time.time() - st

        output_ids = generated[0][len(inputs["input_ids"][0]):]
        output_tokens = output_ids.tolist()
        n_tokens = len(output_tokens)
        speed = n_tokens / elapsed if elapsed > 0 else 0

        # 重复率: 统计重复 n-gram 比例
        def repetition_rate(tokens, n):
            if len(tokens) < n: return 0.0
            ngrams = [tuple(tokens[j:j+n]) for j in range(len(tokens) - n + 1)]
            if len(ngrams) <= 1: return 0.0
            unique = len(set(ngrams))
            return 1.0 - unique / len(ngrams)

        # 多样性
        vocab_diversity = len(set(output_tokens)) / max(len(output_tokens), 1)

        metrics["speed"].append(speed)
        metrics["output_len"].append(n_tokens)
        metrics["rep_1gram"].append(repetition_rate(output_tokens, 1))
        metrics["rep_2gram"].append(repetition_rate(output_tokens, 2))
        metrics["rep_3gram"].append(repetition_rate(output_tokens, 3))
        metrics["diversity"].append(vocab_diversity)

        resp = tokenizer.decode(output_ids, skip_special_tokens=True)[:60]
        print(f"  [{i+1}/{len(prompts)}] {speed:.1f} tok/s, len={n_tokens}, rep2={metrics['rep_2gram'][-1]:.3f} → {resp}...")

    # 汇总
    def avg(lst): return sum(lst) / max(len(lst), 1)

    print(f"  ─────────────────────────────")
    print(f"  平均速度:     {avg(metrics['speed']):.1f} tok/s")
    print(f"  平均生成长度: {avg(metrics['output_len']):.0f} tokens")
    print(f"  1-gram 重复率: {avg(metrics['rep_1gram']):.4f}  (越高越重复)")
    print(f"  2-gram 重复率: {avg(metrics['rep_2gram']):.4f}")
    print(f"  3-gram 重复率: {avg(metrics['rep_3gram']):.4f}")
    print(f"  词汇多样性:   {avg(metrics['diversity']):.4f}  (越高越多样)")

    return {k: avg(v) for k, v in metrics.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 参数统计
# ═══════════════════════════════════════════════════════════════════════════════
def compute_param_stats(model):
    """统计各层权重范数及其分布"""
    print(f"\n{'═' * 60}")
    print(f"[3/7] 参数统计")

    stats = defaultdict(list)
    layer_norms = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            w_norm = param.data.float().norm(2).item()
            w_mean = param.data.float().mean().item()
            w_std = param.data.float().std().item()
            w_abs_max = param.data.float().abs().max().item()

            # 按类型分组
            if 'self_attn' in name:
                cat = 'attention'
            elif 'mlp' in name:
                cat = 'ffn'
            elif 'embed' in name:
                cat = 'embedding'
            elif 'norm' in name or 'layernorm' in name:
                cat = 'norm'
            else:
                cat = 'other'

            stats[f'{cat}_norm'].append(w_norm)
            stats[f'{cat}_std'].append(w_std)
            stats[f'{cat}_max'].append(w_abs_max)

            # 提取层号
            if 'layers.' in name:
                layer_id = int(name.split('layers.')[1].split('.')[0])
                while len(layer_norms) <= layer_id: layer_norms.append(defaultdict(list))
                layer_norms[layer_id][cat].append(w_norm)

    # 按类型汇总
    for cat in ['attention', 'ffn', 'embedding', 'norm']:
        key = f'{cat}_norm'
        if key in stats and stats[key]:
            print(f"  {cat:12s}: L2norm avg={sum(stats[key])/len(stats[key]):.2f}, "
                  f"max={max(stats[key]):.2f}, std avg={sum(stats[f'{cat}_std'])/len(stats[f'{cat}_std']):.4f}")

    # 跨层趋势
    if layer_norms:
        print(f"\n  ── 跨层 L2 范数趋势 ──")
        for li, ln in enumerate(layer_norms):
            parts = []
            for cat in ['attention', 'ffn']:
                if cat in ln:
                    parts.append(f"{cat}={sum(ln[cat])/len(ln[cat]):.1f}")
            print(f"  L{li:02d}: {' | '.join(parts)}")

    # 存在 NaN/Inf ?
    has_nan = any(torch.any(torch.isnan(p.data)) for p in model.parameters())
    has_inf = any(torch.any(torch.isinf(p.data)) for p in model.parameters())
    if has_nan or has_inf:
        print(f"\n  ⚠️  NaN: {has_nan}, Inf: {has_inf}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 注意力统计
# ═══════════════════════════════════════════════════════════════════════════════
def compute_attention_stats(model, tokenizer, device, seq_len=128):
    """hook 各层 attention weights, 计算熵 + 平均关注距离"""
    print(f"\n{'═' * 60}")
    print(f"[4/7] 注意力统计")

    # 准备一个中等长度的输入
    sample_text = (
        "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支，"
        "旨在开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
        "从1956年达特茅斯会议正式提出人工智能概念以来，AI经历了多次起伏。"
        "近年来，随着深度学习技术的突破，AI在图像识别、自然语言处理、"
        "语音识别等领域取得了令人瞩目的成果。"
    )
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
    input_ids = inputs["input_ids"]
    actual_len = input_ids.shape[1]

    # 注册 hook 捕获 attention weights
    attention_data = []

    def make_hook(layer_id):
        def hook(module, input, output):
            # output shape: (B, H, T, D) — 我们需要 QK^T
            # SDPA 不返回 weights，用手动计算
            pass
        return hook

    # 用 custom attention 模式 (禁用 flash, 强制算 weights)
    original_flash = None
    first_attn = None
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'flash'):
            if original_flash is None:
                original_flash = layer.self_attn.flash
            layer.self_attn.flash = False
        if first_attn is None:
            first_attn = layer.self_attn

    if first_attn is None:
        print("  ⚠️ 未找到 attention 模块")
        return {}

    # 手动计算 attention (取第一层采样)
    from model.model_minimind import apply_rotary_pos_emb, repeat_kv, RMSNorm
    x = model.model.embed_tokens(input_ids)
    freqs_cos = model.model.freqs_cos[:actual_len]
    freqs_sin = model.model.freqs_sin[:actual_len]

    layer_stats = defaultdict(dict)

    for layer_id, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        residual = x
        normed_x = layer.input_layernorm(x)

        # 投影
        xq = attn.q_proj(normed_x).view(1, actual_len, attn.n_local_heads if hasattr(attn, 'n_local_heads') else attn.n_heads, attn.head_dim)
        xk = attn.k_proj(normed_x).view(1, actual_len, attn.n_local_kv_heads if hasattr(attn, 'n_local_kv_heads') else attn.n_kv_heads, attn.head_dim)
        xv = attn.v_proj(normed_x).view(1, actual_len, attn.n_local_kv_heads if hasattr(attn, 'n_local_kv_heads') else attn.n_kv_heads, attn.head_dim)

        # RoPE
        cos = freqs_cos[:actual_len].unsqueeze(0).unsqueeze(0)
        sin = freqs_sin[:actual_len].unsqueeze(0).unsqueeze(0)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # QK^T
        n_kv = attn.n_local_kv_heads if hasattr(attn, 'n_local_kv_heads') else attn.n_kv_heads
        n_rep = attn.n_rep if hasattr(attn, 'n_rep') else (attn.n_heads // attn.n_kv_heads)
        xk_rep = repeat_kv(xk, n_rep)  # (1, T, H, D)
        xv_rep = repeat_kv(xv, n_rep)

        scale = 1.0 / math.sqrt(attn.head_dim)
        scores = torch.matmul(xq.transpose(1, 2), xk_rep.transpose(1, 2).transpose(-2, -1)) * scale  # (1, H, T, T)
        # causal mask
        causal_mask = torch.triu(torch.ones(actual_len, actual_len, device=scores.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1).float()

        # 熵: H = -sum(p * log(p))
        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean().item()

        # 平均关注距离 (最后 token 的 attention distribution)
        last_attn = attn_weights[0, :, -1, :]  # (H, T)
        positions = torch.arange(actual_len, device=last_attn.device).float()
        avg_distance = (last_attn * positions.unsqueeze(0)).sum(dim=-1).mean().item()  # 平均关注到多远

        layer_stats[layer_id] = {"entropy": entropy, "avg_distance": avg_distance, "seq_len": actual_len}

        # Forward 一层给下一层
        attn_out = torch.matmul(attn_weights.half(), xv_rep.transpose(1, 2))
        attn_out = attn_out.transpose(1, 2).reshape(1, actual_len, -1)
        x = residual + attn.resid_dropout(attn.o_proj(attn_out))
        x = x + layer.mlp(layer.post_attention_layernorm(x))

    # 恢复 flash
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'flash'):
            layer.self_attn.flash = original_flash

    # 汇总
    all_entropy = [v["entropy"] for v in layer_stats.values()]
    all_dist = [v["avg_distance"] for v in layer_stats.values()]

    print(f"  注意力熵 (越高越分散): avg={sum(all_entropy)/len(all_entropy):.4f}, "
          f"range=[{min(all_entropy):.4f}, {max(all_entropy):.4f}]")
    print(f"  关注距离 (越大越远):   avg={sum(all_dist)/len(all_dist):.1f}, "
          f"range=[{min(all_dist):.1f}, {max(all_dist):.1f}]")
    print(f"  序列长度: {actual_len}")

    print(f"\n  ── 逐层详情 ──")
    for lid in sorted(layer_stats.keys()):
        s = layer_stats[lid]
        print(f"  L{lid:02d}: entropy={s['entropy']:.4f}, avg_dist={s['avg_distance']:.1f}/{s['seq_len']}")

    return layer_stats


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MoE 专家均衡度
# ═══════════════════════════════════════════════════════════════════════════════
def compute_moe_stats(model, tokenizer, device, data_path, max_length, n_samples=50):
    """在验证数据上统计 MoE 专家利用率"""
    print(f"\n{'═' * 60}")
    print(f"[5/7] MoE 专家均衡度")

    if not model.config.use_moe:
        print("  (未启用 MoE, 跳过)")
        return None

    if not hasattr(model, 'reset_moe_stats'):
        print("  (非 V2 MoE, 无统计接口)")
        return None

    from dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset(data_path, tokenizer, max_length=max_length)
    indices = torch.randperm(len(ds))[:n_samples].tolist()

    model.reset_moe_stats()
    model.train()  # 需 training mode 触发统计

    with torch.no_grad():
        for i, idx in enumerate(indices):
            input_ids, labels = ds[idx]
            input_ids = input_ids.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(dtype=model.dtype) if device.type == 'cuda' else nullcontext():
                _ = model(input_ids, labels=None)

    moe_stats = model.get_moe_stats()
    model.eval()

    if not moe_stats:
        print("  (无 MoE V2 层)")
        return None

    num_experts = len(next(iter(moe_stats.values())))
    header = f"  Expert | " + " | ".join([f"L{lid:02d}  " for lid in sorted(moe_stats.keys())]) + " |  Avg  "
    print(header)
    print("  " + "-" * (len(header) - 2))
    avg_util = torch.stack(list(moe_stats.values())).mean(dim=0)
    for ei in range(num_experts):
        row = f"  E{ei:02d}    | "
        for lid in sorted(moe_stats.keys()):
            row += f"{moe_stats[lid][ei].item()*100:4.1f}% | "
        row += f"{avg_util[ei].item()*100:4.1f}%"
        print(row)

    # 均衡度: 理想是 1/num_experts
    ideal = 1.0 / num_experts
    deviation = (avg_util - ideal).abs().mean().item()
    print(f"  理想分布: {ideal*100:.1f}%/expert, 平均偏离: {deviation*100:.2f}%  (越低越均衡)")

    return {"per_layer": moe_stats, "avg_util": avg_util, "deviation": deviation}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MoH Q头均衡度
# ═══════════════════════════════════════════════════════════════════════════════
def compute_moh_stats(model, tokenizer, device, data_path, max_length, n_samples=50):
    """在验证数据上统计 MoH Q头利用率"""
    print(f"\n{'═' * 60}")
    print(f"[6/7] MoH Q头均衡度")

    if not model.config.use_moh:
        print("  (未启用 MoH, 跳过)")
        return None

    if not hasattr(model, 'reset_moh_stats'):
        print("  (无 MoH 统计接口)")
        return None

    from dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset(data_path, tokenizer, max_length=max_length)
    indices = torch.randperm(len(ds))[:n_samples].tolist()

    model.reset_moh_stats()
    model.train()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            input_ids, labels = ds[idx]
            input_ids = input_ids.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(dtype=model.dtype) if device.type == 'cuda' else nullcontext():
                _ = model(input_ids, labels=None)

    moh_stats = model.get_moh_stats()
    model.eval()

    if not moh_stats:
        print("  (无 MoH 层)")
        return None

    num_experts = len(next(iter(moh_stats.values())))
    header = f"  Expert | " + " | ".join([f"L{lid:02d}  " for lid in sorted(moh_stats.keys())]) + " |  Avg  "
    print(header)
    print("  " + "-" * (len(header) - 2))
    avg_util = torch.stack(list(moh_stats.values())).mean(dim=0)
    for ei in range(num_experts):
        row = f"  E{ei:02d}    | "
        for lid in sorted(moh_stats.keys()):
            row += f"{moh_stats[lid][ei].item()*100:4.1f}% | "
        row += f"{avg_util[ei].item()*100:4.1f}%"
        print(row)

    ideal = 1.0 / num_experts
    deviation = (avg_util - ideal).abs().mean().item()
    print(f"  理想分布: {ideal*100:.1f}%/expert, 平均偏离: {deviation*100:.2f}%")

    return {"per_layer": moh_stats, "avg_util": avg_util, "deviation": deviation}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 推理速度基准
# ═══════════════════════════════════════════════════════════════════════════════
def benchmark_speed(model, tokenizer, device, batch_sizes=[1], seq_lens=[64, 128, 256, 512]):
    """多配置吞吐量基准"""
    print(f"\n{'═' * 60}")
    print(f"[7/7] 推理速度基准")

    for bs in batch_sizes:
        for sl in seq_lens:
            input_ids = torch.randint(1, 5000, (bs, sl), device=device)

            # warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model.generate(input_ids, max_new_tokens=1, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)

            # 测 prefill + decode
            torch.cuda.synchronize() if device.type == 'cuda' else None
            st = time.time()
            n_runs = 5
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=model.dtype) if device.type == 'cuda' else nullcontext():
                    for _ in range(n_runs):
                        _ = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed = time.time() - st
            total_tokens = bs * 32 * n_runs
            speed = total_tokens / elapsed
            print(f"  bs={bs:2d}, seq={sl:3d} → {speed:.1f} tok/s")

    # 纯 prefill benchmark
    print(f"\n  ── Prefill 吞吐 (1 token decode) ──")
    for sl in [128, 256, 512, 1024]:
        input_ids = torch.randint(1, 5000, (1, sl), device=device)
        # warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        st = time.time()
        n_runs = 20
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=model.dtype) if device.type == 'cuda' else nullcontext():
                for _ in range(n_runs):
                    _ = model(input_ids)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - st
        prefill_speed = sl * n_runs / elapsed
        print(f"  seq={sl:4d}: {prefill_speed:.1f} tok/s (prefill)")


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind 模型综合性能评估")
    # 模型
    parser.add_argument('--load_from', default='model', type=str, help="model=原生权重")
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='pretrain', type=str, help="权重前缀")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # MoE
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--moe_type', default='v1', type=str, choices=['v1', 'v2'])
    parser.add_argument('--num_experts', default=4, type=int)
    parser.add_argument('--num_experts_per_tok', default=1, type=int)
    parser.add_argument('--moe_expert_intermediate_ratio', default=0.5, type=float)
    # MoH
    parser.add_argument('--use_moh', default=0, type=int, choices=[0, 1])
    parser.add_argument('--moh_shared_heads', default=4, type=int)
    parser.add_argument('--moh_routed_head', default=1, type=int)
    parser.add_argument('--num_attention_heads', default=8, type=int)
    # 评估
    parser.add_argument('--num_samples', default=200, type=int, help="PPL 评估样本数")
    parser.add_argument('--data_path', default=None, type=str, help="验证数据路径(默认=pretrain数据)")
    parser.add_argument('--max_seq_len', default=340, type=int, help="最大评估序列长度")
    parser.add_argument('--skip_attention', action='store_true', help="跳过注意力统计(较慢)")
    parser.add_argument('--skip_speed', action='store_true', help="跳过速度基准")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true')
    args = parser.parse_args()

    if args.data_path is None:
        # 从项目根目录解析
        project_root = os.path.dirname(os.path.abspath(__file__))
        args.data_path = os.path.join(project_root, 'dataset', 'pretrain_t2t_mini.jsonl')

    print("╔" + "═" * 58 + "╗")
    print(f"║  MiniMind 模型综合评估")
    print(f"║  Weight: {args.weight} | Hidden: {args.hidden_size} | Layers: {args.num_hidden_layers}")
    if args.use_moe:
        print(f"║  MoE: {args.moe_type} | Experts: {args.num_experts} | Top-K: {args.num_experts_per_tok}")
    if args.use_moh:
        print(f"║  MoH: {args.num_attention_heads} heads | Shared: {args.moh_shared_heads} | Top-K: {args.moh_routed_head}")
    print("╚" + "═" * 58 + "╝")

    model, tokenizer = init_model(args)
    device = torch.device(args.device)

    results = {}

    # 1. 困惑度
    results['ppl'] = compute_perplexity(
        model, tokenizer, args.data_path, args.max_seq_len, device, args.num_samples
    )

    # 2. 生成质量
    results['gen'] = compute_generation_quality(model, tokenizer, device)

    # 3. 参数统计
    results['params'] = compute_param_stats(model)

    # 4. 注意力统计
    if not args.skip_attention:
        results['attn'] = compute_attention_stats(model, tokenizer, device, seq_len=args.max_seq_len)

    # 5. MoE 统计
    results['moe'] = compute_moe_stats(model, tokenizer, device, args.data_path, args.max_seq_len, n_samples=min(args.num_samples, 50))

    # 6. MoH 统计
    results['moh'] = compute_moh_stats(model, tokenizer, device, args.data_path, args.max_seq_len, n_samples=min(args.num_samples, 50))

    # 7. 速度基准
    if not args.skip_speed and device.type == 'cuda':
        benchmark_speed(model, tokenizer, device)

    # ═══ 综合小结 ═══
    print(f"\n{'═' * 60}")
    print(f"综合小结")
    print(f"{'═' * 60}")
    print(f"  Perplexity:  {results['ppl']['perplexity']:.2f}")
    print(f"  生成速度:    {results['gen']['speed']:.1f} tok/s")
    print(f"  2-gram 重复:  {results['gen']['rep_2gram']:.4f}")
    print(f"  词汇多样性:  {results['gen']['diversity']:.4f}")
    if results.get('attn'):
        all_ent = [v["entropy"] for v in results['attn'].values()]
        print(f"  注意力熵:    {sum(all_ent)/len(all_ent):.4f}")
    if results.get('moe') and results['moe'] is not None:
        print(f"  MoE 均衡偏离: {results['moe']['deviation']*100:.2f}%")
    if results.get('moh') and results['moh'] is not None:
        print(f"  MoH 均衡偏离: {results['moh']['deviation']*100:.2f}%")
    print()
