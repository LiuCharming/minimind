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


def _get_expert_parent(model: nn.Module, module_path: str):
    """根据路径获取 Linear 所属的 expert 模块 (experts.experts.N 中的容器)"""
    import re
    match = re.search(r'\.(experts\.experts\.(\d+))', module_path)
    if not match:
        return None
    idx = int(match.group(2))
    prefix_parts = module_path[:match.start()].split('.')
    obj = model
    for p in prefix_parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    # obj 现在是 MOELayer, obj.experts 是 Experts 容器,
    # obj.experts.experts 是 nn.ModuleList
    return obj.experts.experts[idx]


def _is_special_expert(model: nn.Module, module_path: str) -> bool:
    """
    检查该 Linear 是否属于特殊 expert (Constant/Copy/Zero) 内部。
    这些专家不应该加 LoRA — 只有 FFN 专家 (_SwiGLUExpert / _FusedSwiGLUExpert) 可以。
    """
    from model.moe import ConstantExpert, CopyExpert, ZeroExpert
    expert = _get_expert_parent(model, module_path)
    return isinstance(expert, (ConstantExpert, CopyExpert, ZeroExpert))


def _is_ffn_expert(model: nn.Module, module_path: str) -> bool:
    """检查该 Linear 是否属于 FFN 专家 (_SwiGLUExpert / _FusedSwiGLUExpert)"""
    from model.moe import _SwiGLUExpert, _FusedSwiGLUExpert
    expert = _get_expert_parent(model, module_path)
    return isinstance(expert, (_SwiGLUExpert, _FusedSwiGLUExpert))


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
    is_moe_mode = (mode == 'moe')
    is_all_mode = (mode == 'all')

    if mode == 'attention':
        target_names = {'q_proj', 'o_proj'}
    elif is_moe_mode:
        # MoE 策略: attention + router gate.wg + FFN expert 层
        target_names = {'q_proj', 'o_proj',          # attention
                        'gate_up_proj', 'gate_proj', 'up_proj', 'down_proj',  # FFN experts
                        'wg'}                         # router gate.wg + FFN gate
    elif is_all_mode:
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

        # ── MoE 相关过滤 ──
        # 特殊 expert (Constant/Copy/Zero) → 跳过
        if is_moe_mode and _is_special_expert(model, name):
            continue

        # Router gate.wg: 允许非方阵 (768→n_experts)
        is_router_wg = is_moe_mode and _is_router_gate(name)
        # FFN expert 层 (gate_up_proj 等): 允许非方阵
        is_ffn_layer = is_moe_mode and _is_ffn_expert(model, name)

        # 方阵检查
        is_square = module.in_features == module.out_features
        if not is_square and not (is_router_wg or is_ffn_layer):
            continue

        # 路由器的 rank 裁剪到 min(rank, out_features)
        cur_rank = min(rank, module.out_features) if is_router_wg else rank

        lora = LoRA(module.in_features, module.out_features, rank=cur_rank).to(model.device)
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
