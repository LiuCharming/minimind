"""快速验证 MiniMind + flash_attention_triton 在 2080 Ti 上的运行情况"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
print(f"1. PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Compute Capability: {torch.cuda.get_device_capability()}")

# 导入 flash_attention_triton
from flash_attention_triton import flash_attention_v2
print("2. flash_attention_triton imported OK")

# 导入 MiniMind
sys.path.insert(0, 'F:/minimind')
from model.model_minimind import MiniMindConfig, Attention, _FLASH_TRITON_AVAILABLE
print(f"3. MiniMind model imported OK, _FLASH_TRITON_AVAILABLE={_FLASH_TRITON_AVAILABLE}")

# 创建 Attention 模块，转为 float16 (模拟 autocast)
config = MiniMindConfig(
    hidden_size=768,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=96,
    dropout=0.0,
    flash_attn=True,
)
attn = Attention(config).cuda().half().eval()
print(f"4. Attention created: use_triton_flash={attn.use_triton_flash}, flash={attn.flash}")

# 测试前向传播 (prefill, 无 KV cache → 走 Triton FlashAttention)
bsz, seq_len = 2, 64
x = torch.randn(bsz, seq_len, 768, device='cuda', dtype=torch.float16)
cos = torch.randn(seq_len, 96, device='cuda', dtype=torch.float16)
sin = torch.randn(seq_len, 96, device='cuda', dtype=torch.float16)

with torch.no_grad():
    output, past_kv = attn(x, (cos, sin))
print(f"5. Forward (prefill): input={list(x.shape)} -> output={list(output.shape)}, dtype={output.dtype}")

# 测试推理 (有 KV cache → 走 SDPA，因为 flash_attention_v2 不支持关闭 causal)
past_k = torch.randn(bsz, 32, 4, 96, device='cuda', dtype=torch.float16)  # 32 个历史 token
past_v = torch.randn(bsz, 32, 4, 96, device='cuda', dtype=torch.float16)
x_one = x[:, :1, :]  # 单 token 输入
cos_one = cos[:1]
sin_one = sin[:1]
with torch.no_grad():
    output2, past_kv2 = attn(x_one, (cos_one, sin_one), past_key_value=(past_k, past_v))
print(f"6. Forward (inference w/ KV cache): input={list(x_one.shape)} -> output={list(output2.shape)}")

# 测试训练模式 (prefill + backward 通过 Triton)
attn.train().half()
x_train = torch.randn(bsz, seq_len, 768, device='cuda', dtype=torch.float16, requires_grad=True)
output_train, _ = attn(x_train, (cos, sin))
loss = output_train.sum()
loss.backward()
print(f"7. Backward OK: grad norm={x_train.grad.norm().item():.4f}")

# 测试完整模型
print("\n8. Testing full MiniMindForCausalLM...")
from model.model_minimind import MiniMindForCausalLM
model = MiniMindForCausalLM(config).cuda().half().eval()
input_ids = torch.randint(0, 6400, (2, 64), device='cuda')
with torch.no_grad():
    out = model(input_ids)
print(f"   Full model forward OK: logits shape={list(out.logits.shape)}")

# 测试 generate
print("9. Testing generate...")
gen_ids = model.generate(input_ids=input_ids, max_new_tokens=8)
print(f"   Generate OK: shape={list(gen_ids.shape)}")

print("\n✅ 全部测试通过！MiniMind + Triton FlashAttentionV2 在 RTX 2080 Ti 上完美运行！")
