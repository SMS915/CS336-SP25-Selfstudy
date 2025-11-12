import fasttext
import argparse
import os
import yaml
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    default_config_path = SCRIPT_DIR / "config.yaml"
    parser = argparse.ArgumentParser(description="训练质量分类器, 可用yaml配置文件指定参数")
    parser.add_argument('--config', type=str, default=default_config_path, help='配置文件路径')

    parser.add_argument("--input_file", type=str, help="覆盖配置文件中的训练文件路径。")
    parser.add_argument("--output_dir", type=str, help="覆盖配置文件中的模型输出目录。")
    parser.add_argument("--epoch", type=int, help="覆盖配置文件中的训练轮数。")
    
    return parser.parse_args()


def train_classifier(config, args):
    print(config)

    relative_train_file = args.input_file or config['data']['train_file']
    relative_output_dir = args.output_dir or config['model']['output_dir']

    output_dir = str((SCRIPT_DIR / relative_output_dir).resolve())
    train_file = str((SCRIPT_DIR / relative_train_file).resolve())
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            num_samples = sum(1 for line in f)
        print(f"检测到训练文件共有 {num_samples} 个样本。")
    except FileNotFoundError:
        print(f"错误：输入文件 {train_file} 不存在！")
        return
    
    training_state = {'final_loss': None} # 使用字典来存储状态，以便在函数内部修改

    def training_progress_callback(progress, loss, ws, lr, an):
        """
        这个函数会在fastText训练的每个进度点被调用。
        我们只关心训练结束时的最后一次调用。
        """
        # progress 是一个0到1的浮点数
        training_state['final_loss'] = loss
    
    train_params = config.get('training', {})
    # 如果命令行指定了epoch，就用它
    if args.epoch is not None:
        train_params['epoch'] = args.epoch
    
    for key, value in train_params.items():
        print(f"训练参数: {key} = {value}")

    print("--- 开始训练质量分类器 ---")
    print(f"最终配置:")
    print(f"  - 训练数据: {train_file}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 训练参数: {train_params}")
    print("-" * 30)

    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 **kwargs 将配置字典解包传入
    model = fasttext.train_supervised(
        input=train_file,
        **train_params
    )

    print("\n--- 训练完成！ ---")

    # --- 保存模型 ---
    exp_name = config.get('experiment', {}).get('name', 'model')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch = train_params.get('epoch', 'def') # 'def' for default
    lr = train_params.get('lr', 'def')
    ngrams = train_params.get('wordNgrams', 'def')

    model_name = (
        f"{exp_name}_"
        f"samples{num_samples}_"
        f"epoch{epoch}_"
        f"lr{lr}_"
        f"ngrams{ngrams}_"
        f"{timestamp}.bin"
    )
    model_path = os.path.join(output_dir, model_name)

    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")

def classify_quality(model_path: str, text: str) -> tuple[str, float]:
    model = fasttext.load_model(model_path)
    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    predictions = model.predict(clean_text)
    label = predictions[0][0].replace('__label__', '')
    score = predictions[1][0].item()
    if(label == 'high_quality'):
        label = 'cc'
    else:
        label = 'wiki'
    return label, score # 返回预测的标签

if __name__ == "__main__":
    args = parse_arguments()
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"错误：配置文件 {args.config} 未找到！")
        exit(1)

    train_classifier(config, args)