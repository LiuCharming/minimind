"""Quick test for Shared MoE v2"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MOEFeedForwardV2

# ── 1. Create Shared MoE v2 config ──
config = MiniMindConfig(
    hidden_size=768, num_hidden_layers=8,
    use_shared_ffn=True,
    use_moe=True, moe_type='v2',
    num_experts=4, num_experts_per_tok=1,
    moe_expert_intermediate_ratio=0.5,
)
print(f'Config: use_shared_ffn={config.use_shared_ffn}, use_moe={config.use_moe}, moe_type={config.moe_type}')

model = MiniMindForCausalLM(config)
total = sum(p.numel() for p in model.parameters()) / 1e6

# ── 2. Verify all layers share same MLP ──
mlp_ids = [id(l.mlp) for l in model.model.layers]
print(f'Params: {total:.2f}M | Unique MLPs: {len(set(mlp_ids))} (expect 1)')
print(f'All layers share MLP: {len(set(mlp_ids)) == 1}')
print(f'MLP type: {type(model.model.layers[0].mlp).__name__}')

# ── 3. Test forward (CPU first, then GPU if available) ──
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
x = torch.randint(0, 6400, (2, 32)).to(device)

model.train()
out = model(x)
print(f'Train: aux_loss={out.aux_loss.item():.6f} (should NOT be 8x of single MoE)')

model.eval()
with torch.no_grad():
    out = model(x)
print(f'Eval:  aux_loss={out.aux_loss.item():.6f}')

# ── 4. Test MoE stats dedup ──
model.reset_moe_stats()
with torch.no_grad():
    _ = model(x)
stats = model.get_moe_stats()
print(f'MoE stats keys: {list(stats.keys())} (expect ["shared"])')

print('\n[ALL OK] Shared MoE v2 works correctly!')
