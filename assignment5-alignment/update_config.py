import os
import yaml
import glob

def update_yaml_configs():
    # 目标目录和匹配规则
    config_dir = "configs/eval"
    target_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    
    # 查找和替换的路径
    old_path = "data/MATH/validation.jsonl"
    new_path = "data/MATH/test_split.jsonl"
    
    count = 0
    print(f"开始扫描 {config_dir} 下的 YAML 文件...")

    for file_path in target_files:
        try:
            # 1. 读取 YAML 内容
            with open(file_path, 'r', encoding='utf-8') as f:
                # 使用 safe_load 避免执行任意代码
                content = yaml.safe_load(f)
            
            # 如果文件为空或者不是字典格式，跳过
            if not isinstance(content, dict):
                continue

            # 2. 检查并修改字段
            if content.get("example_path") == old_path:
                content["example_path"] = new_path
                
                # 3. 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    # default_flow_style=False 保证输出是整齐的块格式
                    # sort_keys=False 尽量保持原有的键排序
                    yaml.dump(content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                
                print(f"✅ 已更新: {os.path.basename(file_path)}")
                count += 1
                
        except Exception as e:
            print(f"❌ 处理 {file_path} 时出错: {e}")

    print(f"\n任务完成！共修改了 {count} 个文件。")

if __name__ == "__main__":
    update_yaml_configs()