import cProfile
import pstats
import os
from cs336_basics.bpe_fast import BPETokenizer as FastTokenizer
from cs336_basics.bpe_baseline import BPETokenizer as BaseTokenizer
from cs336_basics.bpe_baseline import train_bpe_run

def run_training_for_profile():
    # 1. 设置参数
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]



    # bpe_fast 训练逻辑
    # trainer = FastTokenizer.train(
    #     input_path=input_path,
    #     vocab_size=vocab_size,
    #     special_tokens=special_tokens
    # )
    # vocab, merges = trainer.get_vocab(), trainer.get_merges()


    # bpe_baseline 训练逻辑
    vocab, merges = train_bpe_run(input_path, vocab_size, special_tokens)


    print("实际训练出的vocab size: ", len(vocab))
    print("实际训练出的merges size: ", len(merges))

if __name__ == "__main__":
    # 创建 Profile 对象
    profiler = cProfile.Profile()
    
    print("开始性能分析...")
    # 开启分析
    profiler.enable()
    
    # --- 运行代码 ---
    run_training_for_profile()
    # ------------------
    
    # 停止分析
    profiler.disable()
    print("分析结束。")

    # --- 输出分析结果 ---
    # 将结果保存到文件，以便用可视化工具查看
    profiler.dump_stats("bpe_stats.prof")
    
    # 在控制台打印前 20 行最耗时的函数
    stats = pstats.Stats(profiler).sort_stats('tottime') # 按自身运行时间排序
    print("\n=== Top 20 functions by internal time (tottime) ===")
    stats.print_stats(20)
    
    stats = pstats.Stats(profiler).sort_stats('cumtime') # 按累积运行时间排序
    print("\n=== Top 20 functions by cumulative time (cumtime) ===")
    stats.print_stats(20)