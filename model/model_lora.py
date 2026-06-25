import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16, target_modules=None):
    """
    对模型应用 LoRA 低秩适配器。

    Args:
        model: MiniMindForCausalLM 模型
        rank: LoRA 秩 (默认 16)
        target_modules: 要适配的模块名列表 (默认: ['q_proj', 'o_proj'])
                       设为 None 或 ['all'] 则适配所有 in==out 的 Linear 层
                       设为 ['attention'] 则只适配 self_attn 下的 q_proj + o_proj
    """
    if target_modules is None:
        target_modules = ['attention']
    if 'attention' in target_modules:
        target_modules = [m for m in target_modules if m != 'attention'] + ['q_proj', 'o_proj']

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features != module.out_features:
            continue

        # 过滤目标模块
        module_basename = name.split('.')[-1]  # 取最后一层名称 (q_proj, o_proj, gate_up_proj 等)
        if 'all' not in target_modules and module_basename not in target_modules:
            continue

        lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
        setattr(module, "lora", lora)
        original_forward = module.forward

        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
