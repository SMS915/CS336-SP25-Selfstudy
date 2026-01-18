import os
import json
import glob
import pandas as pd
import numpy as np

def prepare_bespoke_data_local():
    local_data_dir = "data/Bespoke-Stratos-17k/data"
    print(f"正在读取本地 Parquet 文件: {local_data_dir} ...")
    
    parquet_files = glob.glob(os.path.join(local_data_dir, "train-*.parquet"))
    if not parquet_files:
        print("错误: 未找到 Parquet 文件。")
        return

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    
    output_path = "data/CoT/Bespoke-Stratos-17k-formatted.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    debug_printed = False # 用于只打印第一条数据的结构，帮助调试

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            messages = row.get("conversations")
            
            # 1. 基础判空
            if messages is None: 
                continue
            if hasattr(messages, '__len__') and len(messages) == 0:
                continue
            
            # 如果 messages 是 numpy array，先转为 list
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            
            # 如果 messages 的第一个元素还是 list (即 [[dict, dict]])，说明是嵌套的，需要取出来
            if len(messages) > 0 and isinstance(messages[0], list):
                messages = messages[0]
            
            # --- 调试打印 (只打印第一条) ---
            if not debug_printed:
                print(f"\n[DEBUG] 第一条数据的 messages 类型: {type(messages)}")
                print(f"[DEBUG] 第一条数据的内容 (前100字符): {str(messages)[:100]}")
                debug_printed = True
            # -----------------------------

            prompt = ""
            raw_response = ""
            
            # 3. 遍历提取
            for msg in messages:
                # 再次防御：确保 msg 是字典
                if not isinstance(msg, dict):
                    continue
                    
                # 兼容从截图看出的 'from'/'value' 键名
                role = msg.get("from") or msg.get("role")
                content = msg.get("value") or msg.get("content")
                
                if role == "user":
                    prompt = content
                elif role == "assistant":
                    raw_response = content
            
            if not prompt or not raw_response:
                # 如果只有思考没有内容，跳过
                continue

            # 4. 标签替换
            response = raw_response.replace("<|begin_of_thought|>\n\n", "<think>")
            response = response.replace("\n\n<|end_of_thought|>\n\n", "</think>")
            response = response.replace("<|begin_of_solution|>\n\n", " <answer>")
            response = response.replace("\n\n<|end_of_solution|>", "</answer>")
            
            response = response.strip()
            
            # 5. 写入
            entry = {
                "prompt": prompt,
                "response": response
            }
            f.write(json.dumps(entry) + "\n")
            count += 1
            
    print(f"\nSFT 数据准备完毕！")
    print(f"共转换有效数据: {count} 条")
    print(f"文件已保存至: {output_path}")
    
    # 预览结果
    if count > 0:
        print("\n最终数据预览:")
        with open(output_path, "r") as f:
            first_line = json.loads(f.readline())
            print(first_line["prompt"])
            print(first_line["response"][:200] + "...")
    else:
        print("警告：依然没有提取到数据，请检查上方的 [DEBUG] 信息。")

def check_sft_data(output_path : str):
     with open(output_path, "r") as f:
            first_line = json.loads(f.readline())
            print("Prompt: ", end='  ')
            print(first_line["prompt"][-100:])
            print('Response: ', end="  ")
            print(first_line["response"][:200] + "...")
            f.close()
        

if __name__ == "__main__":
    prepare_bespoke_data_local()
    # output_path = "data/MATH/sft.jsonl"
    # check_sft_data(output_path)