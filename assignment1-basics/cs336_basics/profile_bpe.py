import cProfile
import pstats
import os
from cs336_basics.BPE.bpe_fast import BPETokenizer as FastTokenizer
from cs336_basics.BPE.bpe_naive import BPETokenizer as NaiveTokenizer
from cs336_basics.BPE.bpe_naive import train_bpe_run


def run_training_for_profile():
    # 设置参数
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
    profiler = cProfile.Profile()
    
    print("开始性能分析...")
    profiler.enable()

    run_training_for_profile()
    # 停止分析
    profiler.disable()
    print("分析结束。")

    # 将结果保存到文件，以便用可视化工具查看
    profiler.dump_stats("bpe_stats.prof")
    
    # 在控制台打印前 20 行最耗时的函数
    stats = pstats.Stats(profiler).sort_stats('tottime')
    print("\n=== Top 20 functions by internal time (tottime) ===")
    stats.print_stats(20)
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    print("\n=== Top 20 functions by cumulative time (cumtime) ===")
    stats.print_stats(20)