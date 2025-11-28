import cProfile
import pstats
import os
from cs336_basics.FastBPE import BPETrainer 
from cs336_basics.BPE import train_bpe_run

def run_training_for_profile():
    # 1. 设置参数
    # 使用一个足够大的语料，才能看出性能瓶颈
    # 如果没有 corpus.en，换成你项目中实际使用的测试文件路径
    input_path = "tests/fixtures/corpus.en" 
    vocab_size = 500  # 可以适当调大，比如 1000 或 2000，让合并循环跑久一点
    special_tokens = ["<|endoftext|>"]
    
    # 2. 初始化 Trainer
    trainer = BPETrainer()
    
    # 3. 运行训练
    # 确保这里调用的是你最新的、单进程优化过的逻辑
    # 如果你想测多进程，确保 train 方法里开启了多进程
    # vocab, merges = trainer.train(
    #     input_path=input_path,
    #     vocab_size=vocab_size,
    #     special_tokens=special_tokens
    # )

    vocab, merges = train_bpe_run(input_path, vocab_size, special_tokens)

if __name__ == "__main__":
    # 创建 Profile 对象
    profiler = cProfile.Profile()
    
    print("开始性能分析...")
    # 开启分析
    profiler.enable()
    
    # --- 运行你的代码 ---
    run_training_for_profile()
    # ------------------
    
    # 停止分析
    profiler.disable()
    print("分析结束。")

    # --- 输出分析结果 ---
    # 将结果保存到文件，以便用可视化工具查看
    profiler.dump_stats("bpe_stats.prof")
    
    # 也可以直接在控制台打印前 20 行最耗时的函数
    stats = pstats.Stats(profiler).sort_stats('tottime') # 按自身运行时间排序
    print("\n=== Top 20 functions by internal time (tottime) ===")
    stats.print_stats(20)
    
    stats = pstats.Stats(profiler).sort_stats('cumtime') # 按累积运行时间排序
    print("\n=== Top 20 functions by cumulative time (cumtime) ===")
    stats.print_stats(20)