import torch
import numpy as np
import os
from torch.utils.data import IterableDataset, DataLoader

class BinaryTokenDataset(IterableDataset):
    def __init__(self, data_path, context_length, num_samples=None, seed=42):
        self.data_path = data_path
        self.context_length = context_length
        self.seed = seed
        # 标记数据集大小
        self.num_samples = num_samples 
        
        # 如果文件小于 10GB 且内存足够，直接加载进 RAM，否则使用 mmap
        file_size = os.path.getsize(data_path)
        if file_size < 10 * 1024**3:  # < 10GB
            print(f"正在将{data_path}加载进内存...")
            with open(data_path, 'rb') as f:
                self.data = np.frombuffer(f.read(), dtype=np.uint16)
            self.use_mmap = False
        else:
            print(f"对{data_path}使用mmap模式")
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
            self.use_mmap = True
            
        self.num_tokens = len(self.data)

    def __iter__(self):
        # 识别当前是哪个 worker 进程
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单进程模式
            rng = np.random.default_rng(self.seed)
        else:
            # 多进程模式：每个 worker 使用不同的随机种子
            rng = np.random.default_rng(self.seed + worker_info.id)

        max_start_index = self.num_tokens - (self.context_length + 1)

        while True:
            # 随机采样一个位置
            idx = rng.integers(0, max_start_index)
            
            # 切片
            chunk = self.data[idx : idx + self.context_length + 1]
            chunk = chunk.astype(np.int32)
            
            yield chunk[:-1], chunk[1:]

    def __len__(self):
        return self.num_samples if self.num_samples else 1000000
    
def create_dataloader(config, is_train=True):
    dataset_path = config['train_data_path'] if is_train else config['val_data_path']
    
    # 估算每个epoch的步数
    estimated_samples = config['max_steps'] * config['batch_size'] if is_train else config['eval_steps'] * config['batch_size']

    dataset = BinaryTokenDataset(
        data_path=dataset_path,
        context_length=config['context_length'],
        num_samples=estimated_samples,
        seed=config['seed']
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,      # 内部已经随机,不需要 shuffle=True，
        drop_last=True,     # 丢弃最后不足一个batch的数据
        # 多进程加载
        num_workers=8,      
        # 允许 CPU 内存直接 DMA 传输到 GPU 显存
        pin_memory=True,    
        # 每个 worker 提前准备 2 个 batch
        prefetch_factor=2,  
        # 持久化 worker
        persistent_workers=True 
    )
    return loader