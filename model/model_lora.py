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


def _is_routed_expert(module_path: str) -> bool:
    """判断是否属于 routed expert (experts.experts.N.xxx)"""
    import re
    return bool(re.search(r'\.experts\.experts\.\d+', module_path))


def _is_router_gate(module_path: str) -> bool:
    """判断是否是 MoE 路由器的 gate.wg (Linear(768, n_experts))"""
    import re
    return bool(re.search(r'\.gate\.wg$', module_path))


def apply_lora(model, rank=16, target_modules=None):
    """
    对模型应用 LoRA 低秩适配器。

    Args:
        model: MiniMindForCausalLM 模型
        rank: LoRA 秩 (默认 16)
        target_modules: 适配策略 (默认: ['attention'])
            ['attention']    — 仅 self_attn 的 q_proj + o_proj
            ['moe']          — attention + router gate (适配路由决策)
            ['all']          — 所有 in==out 的 Linear (含 routed experts)
            ['q_proj', ...]  — 自定义目标模块名列表
    """
    if target_modules is None:
        target_modules = ['attention']

    mode = target_modules[0] if isinstance(target_modules, list) and len(target_modules) == 1 else None
    skip_routed = False
    include_router = False

    if mode == 'attention':
        target_names = {'q_proj', 'o_proj'}
    elif mode == 'moe':
        target_names = {'q_proj', 'o_proj', 'wg'}
        skip_routed = True       # 跳过 routed expert 的 wg
        include_router = True    # 但保留 router gate.wg
    elif mode == 'all':
        target_names = {'__all__'}
    else:
        target_names = set(target_modules)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        module_basename = name.split('.')[-1]

        # 按名称过滤
        if '__all__' not in target_names and module_basename not in target_names:
            continue

        # MoE 路由适配: gate.wg (768→4, 非方阵) 纳入; routed expert wg 跳过
        is_router_wg = include_router and _is_router_gate(name)
        is_routed = skip_routed and _is_routed_expert(name)

        if is_routed:
            continue

        # 非路由模块要求方阵; 路由 gate.wg 允许非方阵
        is_square = module.in_features == module.out_features
        if not is_square and not is_router_wg:
            continue

        # 路由器 gate.wg 的 rank 裁剪到 min(rank, out_features)
        router_rank = min(rank, module.out_features) if is_router_wg else rank

        lora = LoRA(module.in_features, module.out_features, rank=router_rank).to(model.device)
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
