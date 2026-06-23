import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
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
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--moe_type', default='v1', type=str, choices=['v1', 'v2'], help="MoE类型（v1=原始, v2=独立MoEBlock）")
    parser.add_argument('--num_experts', default=4, type=int, help="专家数量")
    parser.add_argument('--num_experts_per_tok', default=1, type=int, help="每个token激活的专家数")
    parser.add_argument('--moe_expert_intermediate_ratio', default=0.5, type=float, help="V2专家FFN宽度比例 (1.0=全尺寸, 0.5=半宽)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--show_moe_stats', default=0, type=int, choices=[0, 1], help="显示MoE V2各层专家利用率分布")
    parser.add_argument('--use_moh', default=0, type=int, choices=[0, 1], help="是否使用MoH(Mixture-of-Heads)注意力")
    parser.add_argument('--moh_shared_heads', default=4, type=int, help="MoH始终激活的Q头数")
    parser.add_argument('--moh_routed_head', default=1, type=int, help="MoH每个token激活的专家Q头数(top-k)")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="Q头总数（需为KV头数的整数倍）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # MoE V2 专家均衡统计
        if args.show_moe_stats:
            model.reset_moe_stats()

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

        if args.show_moe_stats:
            moe_stats = model.get_moe_stats()
            if moe_stats:
                print('━' * 55)
                print(f'[MoE Stats] 各层专家平均利用率 (共 {gen_tokens} tokens)')
                num_experts = len(next(iter(moe_stats.values())))
                # 表头
                header = f'  Expert | ' + ' | '.join([f'L{lid:02d}  ' for lid in sorted(moe_stats.keys())]) + ' |  Avg  '
                print(header)
                print('-' * len(header))
                # 每行一个专家
                avg_util = torch.stack(list(moe_stats.values())).mean(dim=0)
                for ei in range(num_experts):
                    row = f'  E{ei:02d}    | '
                    for lid in sorted(moe_stats.keys()):
                        row += f'{moe_stats[lid][ei].item()*100:4.1f}% | '
                    row += f'{avg_util[ei].item()*100:4.1f}%'
                    print(row)
                print('━' * 55)
            else:
                print('[MoE Stats] 无 MoE V2 层，跳过统计')

            # MoH 统计
            if args.use_moh:
                moh_stats = model.get_moh_stats()
                if moh_stats:
                    print(f'\n[MoH Stats] 各层Q头专家平均利用率 (共 {gen_tokens} tokens)')
                    num_experts = len(next(iter(moh_stats.values())))
                    header = f'  Expert | ' + ' | '.join([f'L{lid:02d}  ' for lid in sorted(moh_stats.keys())]) + ' |  Avg  '
                    print(header)
                    print('-' * len(header))
                    avg_util = torch.stack(list(moh_stats.values())).mean(dim=0)
                    for ei in range(num_experts):
                        row = f'  E{ei:02d}    | '
                        for lid in sorted(moh_stats.keys()):
                            row += f'{moh_stats[lid][ei].item()*100:4.1f}% | '
                        row += f'{avg_util[ei].item()*100:4.1f}%'
                        print(row)
                    print('━' * 55)
                else:
                    print('[MoH Stats] 无 MoH 层，跳过统计')

if __name__ == "__main__":
    main()