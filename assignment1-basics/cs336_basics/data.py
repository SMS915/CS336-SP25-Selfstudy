import mmap
import numpy as np
import torch
import os


class DataLoader:
    """
    一个用于大规模语言模型训练的单进程（非分布式）数据加载器。

    通过内存映射（mmap）流式处理大型二进制token文件，以保持低内存占用。
    在每个epoch开始时，它会从数据集中的随机位置开始采样连续的批次，
    既保证了训练的随机性，又利用了文本的局部连续性。

    """
    def __init__(self,
                 dataset_path: str,
                 context_length: int,
                 batch_size: int,
                 seed: int = 42,
                 device: str = 'cpu',
                 token_dtype: np.dtype = np.uint16):
        """
        初始化 DataLoader。

        Args:
            dataset_path (str): 预处理过的二进制token文件的路径。
            context_length (int): 模型的上下文长度。
            batch_size (int): 每个批次包含的样本数量。
            seed (int): 用于可复现采样的随机种子。
            device (str): 最终张量应放置的设备（'cpu', 'cuda', etc.）。
            token_dtype (np.dtype): 数据文件中token的Numpy数据类型（如 np.uint16）。
        """
        self.dataset_path = dataset_path
        self.context_length = context_length
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.token_dtype = token_dtype

        # --- 在初始化时就映射文件并获取token ID ---
        # 这种方式假设在DataLoader的生命周期内，文件不会改变
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件未找到: {dataset_path}")

        with open(self.dataset_path, "rb") as f:
            # 使用 mmap 进行内存映射
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # 从内存映射中创建Numpy数组，这是一个视图，不会消耗大量RAM
            self.token_ids = np.frombuffer(mm, dtype=self.token_dtype)

        self.num_tokens = len(self.token_ids)
        print(f"成功加载数据集: {dataset_path}, 共 {self.num_tokens} 个token。")

    def _generator(self, epoch: int):
        """一个生成器函数，无限产出批次。"""
        # 为当前 epoch 设置独立的随机种子，确保每个epoch的采样都不同且可复现
        rng = np.random.default_rng(self.seed + epoch)

        # 无限循环，持续产出批次
        while True:
            # 1. 随机选择 batch_size 个起始点
            # 确保切片不会越界
            max_start_index = self.num_tokens - (self.context_length + 1)
            start_indices = rng.integers(0, max_start_index, size=self.batch_size)

            # 2. 从这些起始点切片并堆叠
            # 使用列表推导式高效地构建输入和目标序列
            # x[i : i + L]
            inputs_np = np.stack([self.token_ids[i: i + self.context_length] for i in start_indices])
            # y[i] = x[i+1]
            targets_np = np.stack([self.token_ids[i + 1: i + self.context_length + 1] for i in start_indices])

            # 3. 转换为 PyTorch 张量并移动到指定设备
            # 将numpy数组转换为torch张量时，指定为long类型（int64），这是嵌入层和损失函数期望的
            inputs = torch.from_numpy(inputs_np.astype(np.int64)).to(self.device)
            targets = torch.from_numpy(targets_np.astype(np.int64)).to(self.device)

            yield inputs, targets

    def __iter__(self, epoch: int = 0):
        """在每个epoch开始时被调用，返回一个新的生成器。"""
        yield from self._generator(epoch)

def get_batch(x: np.array, batch_size: int, context_length: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = np.random.randint(0, x.shape[0] - context_length, batch_size)
    inputs = np.stack([x[i : i + context_length] for i in idxs])
    outputs = np.stack([x[i + 1 : i + context_length + 1] for i in idxs])
    return torch.tensor(inputs).to(device), torch.tensor(outputs).to(device)