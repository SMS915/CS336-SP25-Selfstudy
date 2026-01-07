import wandb

# 1. 设置你的 WandB 项目路径
ENTITY = "sms915wut-liu-wuhan-university-of-technology"
PROJECT = "CS336-TransformerLM-Training"

# 2. 定义默认值字典 (根据你的代码提取)
# 这些是如果 config 中不存在时，我们要补填的值
default_values = {
    "post_norm": False,
    "no_norm": False,
    "gated_ffn": True,
    "activation": "silu",
    "Weight_Tying": False,  # 注意：你的代码里取的是 config['Weight_Tying']
    "n_kv_heads": None,     # 对应 config.get('n_kv_heads', None)
    "pos_emb_type": "rope",
    "layer_norm": True,    # 对应 config.get('layer_norm', False)
    "bias": False,
    "gated_attn": False
}

target_values = {
    # "post_norm": False,
    # "no_norm": False,
    # "gated_ffn": False,
    # "activation": "silu",
    # "Weight_Tying": False,  # 注意：你的代码里取的是 config
    # "n_kv_heads": None,     # 对应 config.get('n_kv_heads', None)
    "pos_emb_type": "sinusoidal",
    "layer_norm": True,    # 对应 config.get('layer_norm', False)
    "num_params": 16_024_832,
    # "bias": False,
    # "gated_attn": False
}

def fix_runs():
    api = wandb.Api()
    # 获取该项目下所有的 runs
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    print(f"正在扫描项目 {ENTITY}/{PROJECT} 中的 {len(runs)} 个 Run...")

    count = 0
    for run in runs:
        updated = False
        
        # 遍历我们需要检查的每一个默认参数
        for key, default_val in default_values.items():
            # 如果 run.config 中完全没有这个 key
            if key not in run.config:
                # 补全默认值
                run.config[key] = default_val
                updated = True
                print(f"  [修复] Run {run.name}: 缺少 '{key}' -> 补全为 {default_val}")
        
        # 只有当 config 发生变化时才提交更新，节省网络请求
        if updated:
            run.update()
            count += 1
            print(f"  --> Run {run.name} 更新完成。\n")
    
    print(f"扫描结束。共修复了 {count} 个 Run。")

if __name__ == "__main__":
    # 建议先在网页上找一个 Run ID 测试一下，确认无误后再放开跑所有
    # 单个测试写法: 
    # runs = []
    run_name = "sms915wut-liu-wuhan-university-of-technology/CS336-TransformerLM-Training/nkuod2az"
    # run = wandb.Api().run(f"{ENTITY}/{PROJECT}/4gours2n")
    run = wandb.Api().run(run_name)

    updated = False
    for key, target_val in target_values.items():
            # 如果 run.config 中完全没有这个 key
            if key in run.config and run.config[key] != target_val:
                # 补全默认值
                run.config[key] = target_val
                updated = True
                print(f"  [修复] Run {run.name}:  '{key}' -> 修正为 {target_val}")
            elif key not in run.config:
                run.config[key] = target_val
                updated = True
                print(f"[补全] Run {run.name}:  '{key}' -> 补充为 {target_val}")
        
    # 只有当 config 发生变化时才提交更新，节省网络请求
    if updated:
        run.update()
        print(f"  --> Run {run.name} 更新完成。\n")
    
    # fix_runs()