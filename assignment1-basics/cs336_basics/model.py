from math import sqrt, log
from typing import Optional

import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat
from jaxtyping import Float, Bool, Int
from typing import Literal

from cs336_basics.utils import Softmax


class Linear(nn.Module):
    """
    一个偏置项可选的线性变换层。
    Attributes:
        in_features (int): 每个输入样本的大小。
        out_features (int): 每个输出样本的大小。
        weight (nn.Parameter): 模块的可学习权重，形状为 (out_features, in_features)。
        bias (nn.Parameter, optional): 模块的可学习偏置项(可选)，形状为(out_features, )。
    """
    def __init__(self, in_features : int, out_features : int, bias: bool = False, device : torch.device | None = None, dtype : torch.dtype | None = None):
        """
        Args:
            in_features (int): 输入特征的数量。
            out_features (int): 输出特征的数量。
            bias (bool): 是否使用偏置项
            device (torch.device | None, optional): 目标设备。默认为 None。
            dtype (torch.dtype | None, optional): 数据类型。默认为 None。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype, requires_grad=True))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean = 0, std = std, a = -3 * std, b = 3 * std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        对输入张量 x 应用线性变换 (y = x * W^T)。
        Args:
            x (torch.Tensor): 输入张量，形状为 (..., in_features)。

        Returns:
            torch.Tensor: 输出张量，形状为 (..., out_features)。
        """
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

class Embedding(nn.Module):
    """
    实现一个嵌入层，将 token ID 映射到连续的向量表示。

    Attributes:
        num_embeddings (int): 词汇表的大小。
        embedding_dim (int): 每个嵌入向量的维度。
        embed_matrix (nn.Parameter): 嵌入层的可学习权重。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,weight_tying: bool = False, device=None, dtype=None):
        """
        初始化 Embedding 模块。

       Args:
           num_embeddings (int): 词汇表的大小。
           embedding_dim (int): 每个嵌入向量的维度。
           weight_tying (bool): 模型是否启用嵌入层和输出层权重绑定，默认为 False。
           device (torch.device | None, optional): 目标设备。默认为 None。
           dtype (torch.dtype | None, optional): 数据类型。默认为 None。
       """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype


        self.embed_matrix = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device = self.device, dtype = self.dtype, requires_grad=True))

        if not weight_tying:
            nn.init.trunc_normal_(self.embed_matrix, mean = 0, std = 1, a = -3, b = 3)
        else:
            # 权重绑定情况下，对数值稳定性要求严格，需要使用和正常线性层一样的 Xavier 初始化
            std = (2 / (self.embedding_dim + self.num_embeddings)) ** 0.5
            nn.init.trunc_normal_(self.embed_matrix, mean = 0, std = std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        """
        根据 token ID 检索嵌入向量。

        Args:
            token_ids (torch.Tensor): 包含整数 token ID 的输入张量。
                形状: (...)。

        Returns:
            torch.Tensor: 对应的嵌入向量张量。
                形状: (..., embedding_dim)。
        """
        return self.embed_matrix[token_ids]

class RMSNorm(nn.Module):
    """
    实现均方根层归一化 (Root Mean Square Layer Normalization)。
    这是 LayerNorm 的一个简化版本，仅使用一个可学习的缩放参数 weight。

    Attributes:
        d_model (int): 模型的维度。
        eps (float): 用于数值稳定性的 epsilon 值。
        weight (nn.Parameter): 可学习的缩放参数。
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        初始化 RMSNorm 模块。

        Args:
            d_model (int): 模型的维度或特征数。
            eps (float, optional): 用于数值稳定性的 epsilon 值。默认为 1e-5。
            device (torch.device | None, optional): 目标设备。默认为 None。
            dtype (torch.dtype | None, optional): 数据类型。默认为 None。
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model).to(self.device), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用 RMSNorm。

        Args:
            x (torch.Tensor): 输入张量，形状为 `(..., d_model)`。

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入张量相同。
        """
        input_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim = -1, keepdim=True) + self.eps)
        x_norm = x / rms

        return (self.weight * x_norm).to(input_type)

class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络模块，遵循 LLaMA 等现代Transformer架构。
    公式: FFN(x) = W2( SiLU(x @ W1) ⊙ (x @ W3) )
    """
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model (int): 输入的特征维度
            d_ff (int): 隐藏层维度
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.ffn = GatedFeedForward(d_model=self.d_model, d_ff=self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

def get_activation_fn(name: str):
    """
       根据名称获取对应的 PyTorch 激活函数。
       Args:
           name (str): 激活函数的名称 (不区分大小写)。
               支持: 'relu', 'gelu', 'silu', 'identity'。

       Returns:
           Callable: 对应的 PyTorch 函数接口 (如 F.relu)。

       Raises:
           NotImplementedError: 如果传入了不支持的激活函数名称。
       """
    name = name.lower()
    if name == 'relu': return F.relu
    elif name == 'gelu': return F.gelu
    elif name == 'silu': return F.silu
    elif name == 'identity': return lambda x: x
    else: raise NotImplementedError(f"未知的激活函数: {name}")

class StandardFeedForward(nn.Module):
    """
   标准的前馈神经网络 (FFN) 模块。
   这是最经典的 Transformer MLP 结构，由两个线性层及中间的激活函数组成。
   公式: FFN(x) = Act(x @ W1) @ W2

   Attributes:
       d_model (int): 输入和输出的特征维度。
       d_ff (int): 隐藏层 (中间层) 的维度。
       W1 (Linear): 上投影层 (d_model -> d_ff)。
       W2 (Linear): 下投影层 (d_ff -> d_model)。
       act_fn (Callable): 激活函数。
   """
    def __init__(self, d_model: int, d_ff: int = None, activation: str = 'silu', bias: bool = False):
        """
        初始化 StandardFeedForward 模块。

        Args:
            d_model (int): 模型的维度。
            d_ff (int, optional): 前馈网络的隐藏维度。
                如果为 None，默认设置为 4 * d_model。
            activation (str, optional): 激活函数名称。默认为 'silu'。
            bias (bool, optional): 线性层是否使用偏置项。默认为 False。
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        self.W1 = Linear(self.d_model, self.d_ff, bias=bias)
        self.W2 = Linear(self.d_ff, self.d_model, bias=bias)
        self.act_fn = get_activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.act_fn(self.W1(x)))

class GatedFeedForward(nn.Module):
    """
    门控前馈神经网络 (Gated FFN) 模块。
    常用于 LLaMA、PaLM 等现代大模型中 (如 SwiGLU 结构)。
    它包含三个线性层：两个用于上投影 (一个经过激活函数，一个不经过)，一个用于下投影。
    公式: FFN(x) = (Act(x @ W1) * (x @ W3)) @ W2

    Attributes:
        d_model (int): 输入和输出的特征维度。
        d_ff (int): 隐藏层 (中间层) 的维度。
        W1 (Linear): 门控投影层，d_model -> d_ff。
        W2 (Linear): 下投影层，d_ff -> d_model。
        W3 (Linear): 上投影层，d_model -> d_ff。
        act_fn (Callable): 激活函数 (通常为 SiLU)。
    """
    def __init__(self, d_model: int, d_ff: int = None, activation: str = "silu", bias: bool = False):
        """
        初始化 GatedFeedForward 模块。

        Args:
            d_model (int): 模型的维度。
            d_ff (int, optional): 前馈网络的隐藏维度。
                如果为 None，则根据 LLaMA 论文中的公式计算：
                d_ff = 2/3 * 4 * d_model，并向上取整到 64 的倍数以优化硬件效率。
            activation (str, optional): 激活函数名称。默认为 "silu"。
            bias (bool, optional): 线性层是否使用偏置项。默认为 False。
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = 64 * ((round(self.d_model * 8 / 3) + 63) // 64) if d_ff is None else d_ff
        self.W1 = Linear(self.d_model, self.d_ff, bias=bias)
        self.W2 = Linear(self.d_ff, self.d_model, bias=bias)
        self.W3 = Linear(self.d_model, self.d_ff, bias=bias)
        self.act_fn = get_activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.act_fn(self.W1(x)) * self.W3(x))


class RoPE(nn.Module):
    """
    RoPE (旋转位置编码) 模块。
    通过旋转输入向量的成对维度来注入位置信息，实现了用绝对位置编码相对位置注意力的能力。
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        """
        Args:
            d_k (int): 输入向量的维度 (必须是偶数)。
            max_seq_len (int): 模型支持的最大序列长度。
            theta (float): RoPE的基数，对应旋转公式中的 base，计算 freq = 1.0 / theta ** (2i / d)。默认为10000.0
        """
        super().__init__()
        self.theta = theta if theta else 10000.0
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        k_vec = torch.arange(0, self.d_k, 2) / self.d_k
        freq = 1.0 / self.theta ** k_vec

        rotation_matrix = torch.outer(torch.arange(0, max_seq_len), freq) # Shape: (seq_len, d_k/2)
        cos_table = torch.cos(rotation_matrix)
        sin_table = torch.sin(rotation_matrix)
        self.register_buffer('cos_table', cos_table, persistent=False)
        self.register_buffer('sin_table', sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量。Shape: (batch_size, d_h, seq_len, d_k) OR (batch_size, seq_len, d_k)
            token_positions (torch.Tensor): 每个token的绝对位置。Shape: (batch_size, seq_len)
        Returns:
            torch.Tensor: 应用RoPE后的张量。Shape: (batch_size, seq_len, d_k)
        """
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        # 1. 如果 x 是 4 维 (B, H, S, D)，说明包含 Head 维度。
        # 2. 如果 token_positions 是 2 维 (B, S)，得到的 cos 是 (B, S, D)。
        #    此时必须插入 Head 维度变成 (B, 1, S, D) 才能与 x (B, H, S, D) 正确广播。
        # 3. 如果 token_positions 是 1 维 (S,)，得到的 cos 是 (S, D)。
        #    PyTorch 会自动右对齐广播: (S, D) -> (1, 1, S, D)。
        if x.ndim == 4 and token_positions.ndim == 2:
            cos = cos.unsqueeze(1) # (B, S, D) -> (B, 1, S, D)
            sin = sin.unsqueeze(1)

        # 执行旋转
        # 将输入向量的特征维度 d_k 分解为两两一组的切片。
        # 根据 RoPE 旋转公式，需要分离偶数索引分量和奇数索引分量。
        # even_x: 包含所有偶数索引位置的值 [x_0, x_2, ..., x_{d-2}]
        # odd_x:  包含所有奇数索引位置的值 [x_1, x_3, ..., x_{d-1}]
        even_x, odd_x = rearrange(x, '... (d_half odd_even) -> odd_even ... d_half', odd_even=2) # (B, S, D/2) or (B, H, S, D/2)
        
        x1_rot = even_x * cos - odd_x * sin
        x2_rot = even_x * sin + odd_x * cos
        
        return einx.rearrange('... d_half, ... d_half -> ... (d_half (1 + 1))', x1_rot, x2_rot).contiguous()
    
class SinusoidalPositionalEncoding(nn.Module):
    """
    原版 Transformer 使用实现经典的绝对正余弦位置编码 (Sinusoidal Positional Encoding)
    该模块生成一个固定的、不可学习的矩阵，并将其加到输入嵌入中。

    对于位置 pos 和维度索引 i (0 <= i < d_model/2):
        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    Attributes:
        pe (torch.Tensor): 预计算的位置编码矩阵 (Buffer)。
            形状: (max_seq_len, d_model)。
    """
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0, device = None):
        """
        初始化位置编码模块。

        Args:
            d_k (int): 模型的嵌入维度 (d_model)。
            max_seq_len (int): 预计算的最大序列长度。
            device (torch.device | None, optional): 目标设备。
        """
        super().__init__()

        # 创建一个足够长的 PE 矩阵
        pe = torch.zeros(max_seq_len, d_k)
        # 生成位置索引向量
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 在对数空间计算分母项 (div_term)
        div_term = torch.exp(torch.arange(0, d_k, 2).float() * (-log(theta) / d_k))

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
       将位置编码加到输入张量上。
       Args:
           x (torch.Tensor): 输入嵌入张量。
               形状: (batch_size, seq_len, d_model)。

       Returns:
           torch.Tensor: 加入位置信息后的张量。
               形状: (batch_size, seq_len, d_model)。

       Raises:
           ValueError: 如果输入序列长度超过了初始化的 max_seq_len。
       """

        # x: [batch_size, seq_len, d_model]

        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"输入序列长度{seq_len}大于最大序列长度{self.pe.size(0)}")

        # 截取对应长度的位置编码并相加
        return x + self.pe[:seq_len, :].unsqueeze(0)

def ScaledDotProductAttention(Q: Float[torch.Tensor, "batch_size num_q q_seq_len d_q"],
                              K: Float[torch.Tensor, "batch_size num_k k_seq_len d_k"],
                              V: Float[torch.Tensor, "batch_size num_v v_seq_len d_v"],
                              mask: Bool[torch.Tensor, "batch_size q_seq_len k_seq_len"] = None)\
                              -> Float[torch.Tensor, "batch_size q_seq_len d_v"]:
    """
   计算缩放点积注意力。
   公式: Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

   Args:
       Q: 查询张量。Shape: (*batch, query_seq_len, d_q)
       K: 键张量。Shape: (*batch, key_seq_len, d_k)
       V: 值张量。Shape: (*batch, value_seq_len, d_v)
       mask: 可选的布尔掩码。Shape: (*batch, query_seq_len, key_seq_len)

   Returns:
       注意力输出。Shape: (*batch, query_seq_len, d_v)
   """
    is_gqa = False
    if Q.ndim == 4:
        num_heads_q, num_heads_k = Q.shape[1], K.shape[1]
        if num_heads_q != num_heads_k:
            # Q 的头数多于 K/V 的头数 (例如 Q=32, K=8, n_rep=4)
            # 需要将 Q 的 Heads 维度拆解为 (Kv_Heads, Group_Size)。
            # 同时将 K, V 的 Group 维度广播为 1，以便后续通过广播机制进行计算。
            # 变换后: Q -> (B, H_kv, n_rep, L, D), K/V -> (B, H_kv, 1, L, D)
            is_gqa = True
            n_rep = num_heads_q // num_heads_k
            Q = rearrange(Q, 'b (h_kv n_rep) l d -> b h_kv n_rep l d', n_rep=n_rep)
            K = rearrange(K, 'b h_kv l d -> b h_kv 1 l d', n_rep=n_rep)
            V = rearrange(V, 'b h_kv l d -> b h_kv 1 l d', n_rep=n_rep)

    d_k = K.shape[-1]

    # einsum 自动处理广播。
    # 标准 MHA: b h q d, b h k d -> b h q k
    # GQA: b h_kv n_rep q d, b h_kv 1 k d -> b h_kv n_rep q k (K 的维度 1 自动广播匹配 n_rep)
    attn_scores = einsum(Q, K, "b ... q d, b ... k d -> b ... q k") / sqrt(d_k) # 计算缩放后的注意力分数
    if mask is not None: # 应用掩码
        attn_scores = torch.where(mask, attn_scores, float('-inf'))
    attn_weights = Softmax(attn_scores, -1) # 计算注意力权重
    output =  einsum(attn_weights, V, "b ... q k, b ... k d -> b ... q d")

    if is_gqa:
        # 计算完成后，将 (Kv_Heads, Group_Size) 重新合并回 (Total_Heads)
        output = rearrange(output, "b h_kv n_rep l d -> b (h_kv n_rep) l d")

    return output


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention) 模块。

    可选择性地集成旋转位置编码 (RoPE)，分组查询注意力 (GQA) 和门控注意力 (gated attention)，并支持手写和内置flash-attention加速的不同SDPA切换。

    Attributes:
        d_model (int): 输入和输出的嵌入维度。
        num_heads (int): Query (查询) 的注意力头数量。
        num_kv_heads (int): Key (键) 和 Value (值) 的注意力头数量, 用于GQA。
        head_dim (int): 每个注意力头的维度 (d_model // num_heads)。
        num_rep (int): 每个 KV 头对应的 Q 头数量 (num_heads // num_kv_heads)。
        q_proj (Linear): Query投影层。
        k_proj (Linear): Key投影层。
        v_proj (Linear): Value投影层。
        out_proj (Linear): 输出投影层。
        rope (RoPE | None): 旋转位置编码模块实例。
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None, 
                 num_kv_heads: int = None, gated_attn: bool = False, flash_attn = False, bias: bool = False):
        """
        初始化多头自注意力模块。

        Args:
            d_model (int): 模型的总维度 (必须能被 num_heads 整除)。
            num_heads (int): Query 的注意力头数量。
            max_seq_len (int, optional): RoPE 预计算所需的最大序列长度。默认为 None。
            theta (float, optional): RoPE 的基准频率。如果不为 None 且 max_seq_len 存在，则启用 RoPE。
            num_kv_heads (int, optional): Key/Value 的头数量。
                - 如果为 None，默认等于 num_heads (标准 MHA)。
                - 如果小于 num_heads，则启用 GQA。
                - 如果为1，则特化为 MQA。
            gated_attn (bool, optional): 是否启用门控注意力机制。默认为 False。
            flash_attn (bool, optional): 是否使用内置的多头自注意力加速
            bias (bool, optional): 线性层是否包含偏置项，默认为 False。
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # GQA设置: 如果没有指定 kv_heads, 则默认为 num_heads, 即标准的 MHA
        # 否则，若使用较小的 num_kv_heads，则为 GQA，当 num_kv_heads 为 1，则特化为 MQA。
        self.num_kv_heads = self.num_heads if num_kv_heads is None else num_kv_heads
        self.gated_attn = gated_attn

        # 确保Query 头的数量能被 KV头的数量整除
        assert self.num_heads % self.num_kv_heads == 0, "Query 头数必须为 KV 头数的整数倍"

        # 每个 KV 头对应的 Q 头的数量
        self.num_rep = self.num_heads // self.num_kv_heads

        # 单个头的维度
        self.head_dim = self.d_model // self.num_heads
        self.d_k = self.d_v = self.head_dim

        # Query 投影, (B, L, D)
        self.q_proj = Linear(self.d_model, self.d_model, bias=bias)

        # Key/Value 投影, (B, L, D) -> (B, L, d_head * num_kv_heads) 使用 GQA 的情况下输出维度可能小于d_model
        self.k_proj = Linear(self.d_model, self.head_dim * self.num_kv_heads, bias=bias)
        self.v_proj = Linear(self.d_model, self.head_dim * self.num_kv_heads, bias=bias)

        # 输出投影，将多头结果融合回 d_model
        self.out_proj = Linear(self.d_model, self.d_model, bias=bias)

        # 若提供了参数，则初始化 rope
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, self.max_seq_len)
        else:
            self.rope = None

        # 注意力门控机制
        if self.gated_attn:
            # Gating 投影层 (对应论文中 W_theta)
            self.gate_proj = nn.Linear(self.d_model, self.d_model, True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)

        # 使用内置加速实现还是自身实现
        self.flash_attn = flash_attn

    def forward(self, x : torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        对输入张量应用多头自注意力前向计算。

        Args:
            x (torch.Tensor): 输入张量。
                形状: `(batch_size, sequence_length, d_model)`。
            token_positions (torch.Tensor, optional): 每个 token 的绝对位置索引，用于 rope。
                形状: `(batch_size, sequence_length)`。

        Returns:
            torch.Tensor: 输出张量。
                形状: `(batch_size, sequence_length, d_model)`。

        符号对应:
            B: batch_size,
            L: seq_len,
            H: num_heads,
            D: d_model,
            H_kv: num_kv_heads,
            D_h: head_dim
        """
        query = self.q_proj(x) # Shape: (B, L, H * D_h)
        key = self.k_proj(x)   # Shape: (B, L, H_kv * D_h)
        value = self.v_proj(x)
        context_length = query.shape[1]
        queries_multi_head = rearrange(query, "b l (h d) -> b h l d", h=self.num_heads)

        # GQA情况下，reshape按照kv_head的维度进行调整, 进行分组投影
        keys_multi_head = rearrange(key, "b l (h_kv d) -> b h_kv l d", h_kv=self.num_kv_heads) # H_kv 而不是 H
        values_multi_head = rearrange(value, "b l (h_kv d) -> b h_kv l d", h_kv=self.num_kv_heads)

        # 应用位置旋转编码
        if self.rope is not None:
            if token_positions is None:
                # 若未提供位置索引，则生成默认的索引[0, 1, 2, ..., S-1]
                token_positions = torch.arange(context_length, device=x.device)

                # 显式广播
                # b = query.shape[0]
                # token_positions = torch.arange(context_length, device=x.device).unsqueeze(0).expand(b, -1)
                # token_positions = repeat(torch.arange(context_length, device=x.device), 's -> b s', b=x.shape[0])  # einops 版本

            # RoPE旋转 query 和 key
            # RoPE不旋转V的原因，一方面利用到位置相对关系的查询操作只存在于Q, K之间
            # 另一方面，因为旋转会破坏语义信息，不希望 V 承载的语义信息因为位置而改变
            queries_multi_head = self.rope(queries_multi_head, token_positions)
            keys_multi_head = self.rope(keys_multi_head, token_positions)

        if not self.flash_attn:
            # 创建因果掩码，保证位置 i 只能看到 0...i，实际上就是一个对角线为 0 的下三角矩阵。
            mask = torch.tril(torch.ones(context_length, context_length, device=x.device, dtype=torch.bool), diagonal=0)
            # mask = mask.view(1, 1, context_length, context_length) # 显式广播

            # 使用手动实现的 SDPA，需要传入mask信息
            out = ScaledDotProductAttention(queries_multi_head, keys_multi_head, values_multi_head, mask)
        else:
            out = F.scaled_dot_product_attention(
                queries_multi_head,
                keys_multi_head,
                values_multi_head,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True # 自动处理因果掩码
            )

        # 拼接多头并输出
        out = rearrange(out, "b h s d -> b s (h d)", h=self.num_heads, d=self.d_v)

        # 门控机制 (若启用)
        if self.gated_attn:
            gate_score = torch.sigmoid(self.gate_proj(x))
            out = gate_score * out

        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """
    Transformer 解码器层 (Decoder Block)，现代大语言模型的核心构建单元.
    包含一个多头自注意力子层和一个可定制是否采用门控和使用不同激活函数的前馈网络子层
    默认采用 Pre-Norm结构，即归一化在残差连接内部。
    相比原版的Post-Norm结构，Pre-Norm提供了更好的训练稳定性，使得深层网络更易收敛.


    Attributes:
        norm1 (nn.Module): 对注意力的归一化层
        attn (MultiHeadSelfAttention): 多头自注意力模块（支持GQA, RoPE, Flash-Attention）
        norm2 (nn.Module): 对前馈网络的归一化层
        ffn (nn.Module): 前馈网络模块(Gated 或 Standard, 并支持不同的激活函数)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, theta: float = None,
                 max_seq_len: int = None, gated_ffn: bool = True, activation: str = 'silu',
                 post_norm = False, no_norm: bool = False, num_kv_heads: int = None, gated_attn: bool = False,
                 layer_norm: bool = False,bias: bool = False, flash_attn: bool = False):
        """
        初始化 TransformerBlock。

        Args:
            d_model (int): 模型的维度。
            num_heads (int): 注意力头的数量。
            d_ff (int, optional): 前馈网络的隐藏维度。默认为 None。
            theta (float, optional): 用于 RoPE 的基准频率。默认为 None。
            max_seq_len (int, optional): RoPE 预计算的最大序列长度。默认为 None。
            gated_ffn (bool, optional): 是否使用门控 FFN(如 SwiGLU), 默认为True
                - True: 使用GatedFeedForward (3个线性层)
                - False: 使用StandardFeedForward (2个线性层)
            activation (str, optional): 激活函数类型 ('silu', 'relu', 'gelu'等)
            post_norm (bool, optional): 是否使用Post-Norm结构, 即原版Transformer风格, 默认为False
            no_norm (bool, optional): 是否移除所有归一化层 (仅用于实验), 默认为False
            num_kv_heads (int, optional): KV头数, 用于控制 GQA/MQA
                - None: 默认为 num_heads, 即标准 MHA
                - < num_heads: GQA
                - 1: MQA
            gated_attn (bool, optional)


        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.post_norm = post_norm
        self.gated_ffn = gated_ffn
        self.activation = activation
        self.num_kv_heads = num_kv_heads
        self.gated_attn = gated_attn
        self.layer_norm = layer_norm
        self.bias = bias
        self.flash_attn = flash_attn

        if no_norm:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        elif not self.layer_norm:
            self.norm1 = RMSNorm(self.d_model)
            self.norm2 = RMSNorm(self.d_model)
        else:
            self.norm1 = nn.LayerNorm(self.d_model)
            self.norm2 = nn.LayerNorm(self.d_model)
            

        self.attn = MultiHeadSelfAttention(d_model=self.d_model, num_heads=self.num_heads, 
                                           max_seq_len=self.max_seq_len, theta=self.theta,
                                           num_kv_heads=self.num_kv_heads, gated_attn=self.gated_attn, 
                                           bias=self.bias, flash_attn=self.flash_attn)
        if self.gated_ffn:
            self.ffn = GatedFeedForward(d_model=self.d_model, d_ff=self.d_ff, activation=self.activation, bias=bias)
        else:
            self.ffn = StandardFeedForward(d_model=self.d_model, d_ff=self.d_ff, activation=self.activation, bias=bias)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"], token_positions: Float[torch.Tensor, "batch_size seq_len"] = None) -> torch.Tensor:
        """
        执行 TransformerBlock 的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
                形状: `(batch_size, seq_len, d_model)`。
            token_positions (torch.Tensor, optional): 每个 token 的绝对位置。
                形状: `(batch_size, seq_len)`。

        Returns:
            torch.Tensor: 输出张量。
                形状: `(batch_size, seq_len, d_model)`。
        """
        if self.post_norm:
            x = self.norm1(x + self.attn(x, token_positions))
            x = self.norm2(x + self.ffn(x))
        else:    
            x = x + self.attn(self.norm1(x), token_positions)
            x = x + self.ffn(self.norm2(x))

        return x

class TransformerLM(nn.Module):
    """
    一个完整全功能的、仅解码器（Decoder-Only）的 Transformer 语言模型。
    设计为一个较通用的架构框架，通过调整配置参数，可以重现多种主流大模型的结构
    e.g
    原始Transformer风格:  pos_emb_type = 'sinusoidal', activation = 'relu', gated_ffn = True, layer_norm = True, post_norm = True
    GPT-2 风格:           pos_emb_type = 'learned' , activation = 'gelu', gated_ffn = False, layer_norm = True, post_norm = False
    LLaMA 风格:           pos_emb_type = 'rope', activation = 'silu', gated_ffn = True, layer_norm = False, post_norm = False

    Attributes:
        vocab_size (int): 词汇表的大小。
        context_length (int): 模型的最大上下文长度。
        d_model (int): 模型的内部维度。
        embed (Embedding): 词嵌入层。
        pos_embed (nn.Module, optional): 绝对位置编码模块 (如果 pos_emb_type 不为 'rope')。
        blocks (nn.ModuleList): 包含 num_layers 个 TransformerBlock 的列表。
        norm_final (nn.Module): 最后的归一化层 (通常为 RMSNorm 或 LayerNorm)。
        lm_head (Linear): 输出线性层，将隐状态映射回词表概率 logits。
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, theta: float = None, d_ff: int = None, tie_weights: bool = False,
                 post_norm: bool = False, no_norm: bool = False, num_kv_heads: int = None,
                 gated_attn: bool = False, gated_ffn: bool = True, activation: str = 'silu',
                 flash_attn = False, pos_emb_type: Literal['rope', 'sinusoidal', 'learned', 'removed'] = 'rope', layer_norm: bool = False,
                 bias: bool = False):
        """
        初始化 TransformerLM 模型。

        Args:
            vocab_size (int): 词汇表的大小。
            context_length (int): 模型的最大上下文长度。
            d_model (int): 模型的维度。
            num_layers (int): TransformerBlock 的层数
            num_heads (int): Query 注意力头的数量。
            theta (float): 位置编码的基准频率。若 pos_emb_type='rope'，则传给rope_theta， 否则若为pos_emb_type = 'sinusoidal'，则传给正余弦位置编码，默认为10000.0。
            rope_theta (float): RoPE的基准频率
            d_ff (int, optional): 前馈网络的隐藏维度。默认为 None 下，SwiGLU 模式下为 2/3 * 4d，标准模式下为 4d。
            tie_weights (bool, optional): 是否绑定输入嵌入层和输出 LM Head 的权重，绑定权重可以显著减少参数量，默认为 False。
            post_norm (bool, optional): 是否使用 Post-Norm 结构，默认为 False
            no_norm (bool, optional): 是否移除所有 Norm 层 (仅用于消融实验)。
            num_kv_heads (int, optional): Key/Value 的头数量，用于启用 GQA (Grouped Query Attention) 或 MQA，默认为 None
            gated_attn (bool, optional): 是否启用门控注意力，默认为 False。
            gated_ffn (bool, optional): 是否启用 SwiGLU 结构的 FFN (LLaMA 风格)。默认为 True。
            activation (str, optional): 激活函数名称 (如 'silu', 'gelu', 'relu')。
            flash_attn (bool, optional): 是否启用 Flash Attention 算子加速。
            pos_emb_type (str, optional): 位置编码类型选择。
                - 'rope': 旋转位置编码 (现代 LLM 首选)。
                - 'learned': 可学习的绝对位置嵌入 (GPT-2 风格)。
                - 'sinusoidal': 固定正余弦绝对位置编码 (原版 Transformer)。
                - 'removed': 不使用显式位置编码 (依赖因果掩码泄露位置信息)。
            layer_norm (bool, optional): 归一化层类型。
                - True: nn.LayerNorm。
                - False: RMSNorm
            bias (bool, optional): 线性层是否使用偏置项。默认为 False。
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.theta = theta
        self.rope_theta = None
        self.d_ff = d_ff
        self.context_length = context_length
        self.post_norm = post_norm
        self.no_norm = no_norm
        self.gated_attn = gated_attn
        self.gated_ffn = gated_ffn
        self.activation = activation
        self.pos_emb_type = pos_emb_type
        self.layer_norm = layer_norm
        self.bias = bias
        self.flash_attn = flash_attn

        # 词嵌入层
        self.embed = Embedding(self.vocab_size, self.d_model, weight_tying=tie_weights)

        # 位置编码初始化
        if pos_emb_type == 'rope':
            self.rope_theta = theta
        elif self.pos_emb_type == 'learned':
            self.pos_embed = nn.Embedding(context_length, d_model)
            nn.init.trunc_normal_(self.pos_embed.weight, mean=0.0, std=0.02, a=-0.04, b=0.04) # 控制方差
        elif self.pos_emb_type == 'removed':
            self.pos_embed = None
        elif self.pos_emb_type == 'sinusoidal':
            self.pos_embed = SinusoidalPositionalEncoding(self.d_model, self.context_length, self.theta)
        else:
            raise ValueError(f"未知的位置编码类型{pos_emb_type}")

        # 构建 Transformer 层堆叠
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(d_model = d_model, num_heads = self.num_heads, d_ff = self.d_ff,
                                                theta=self.rope_theta, max_seq_len=self.context_length,
                                                gated_ffn=self.gated_ffn, activation=self.activation,
                                                post_norm=self.post_norm,
                                                num_kv_heads=self.num_kv_heads, gated_attn=self.gated_attn, 
                                                layer_norm = self.layer_norm,
                                                bias = self.bias, flash_attn=self.flash_attn))

        # 最终归一化层
        # Pre-Norm 架构中，Transformer 块的输出需要再过一次 Norm 才能进 LM Head
        if not self.no_norm:
            if not self.post_norm:
                if layer_norm:
                    self.norm_final = nn.LayerNorm(self.d_model, bias=True)
                else:
                    self.norm_final = RMSNorm(self.d_model)
            else:
                # 因为最后一层的输出已经是 Norm 过的，所以Post-Norm 架构通常不需要 Final Norm
                self.norm_final = nn.Identity()
        else:
            self.norm_final = nn.Identity()

        # 语言模型头 (LM Head)
        self.lm_head = Linear(d_model, vocab_size)

        if tie_weights:
            self.lm_head.weight = self.embed.embed_matrix
            print("启用权重绑定")

    def forward(self, tokens: Int[torch.Tensor, "batch_size seq_len"], token_positions: Optional[Int[torch.Tensor, "batch_size seq_len"]] = None) -> torch.Tensor:
        """
        执行模型的前向传播。

        Args:
            tokens (torch.Tensor): 输入的 token ID 张量。
                形状: `(batch_size, seq_len)`。
            token_positions (torch.Tensor, optional): 每个 token 的绝对位置。
                形状: `(batch_size, seq_len)`。

        Returns:
            torch.Tensor: 模型输出的 logits 张量。
                形状: `(batch_size, seq_len, vocab_size)`。
        """
        x = self.embed(tokens)  # batch_size seq_len d_model
        if self.pos_emb_type == 'learned':
            # 1. 可学习绝对位置编码
            assert self.pos_embed is not None
            batch_size, seq_len = tokens.shape
            pos = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
            x = x + self.pos_embed(pos)
            # 不需要 RoPE 的 token_positions
            for block in self.blocks:
                x = block(x, token_positions=None) 

        elif self.pos_emb_type == 'sinusoidal':
            # 2. 固定正余弦位置编码
            assert self.pos_embed is not None
            x = self.pos_embed(x)
            for block in self.blocks:
                x = block(x, token_positions=None)

        elif self.pos_emb_type == 'removed':
            # 3. 不添加位置编码，验证模型直接学习位置关系的假设
            for block in self.blocks:
                x = block(x, token_positions=None)
        else:
            # 4. 旋转位置编码，用绝对位置编码相对位置信息
            if token_positions is None:
                batch_size, seq_len = tokens.shape
                token_positions = repeat(torch.arange(seq_len, device=tokens.device), 's -> b s', b=batch_size)
            
            for block in self.blocks:
                x = block(x, token_positions)

        x = self.norm_final(x)
        x = self.lm_head(x) # batch_size seq_len vocab_size
        return x

    def count_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        return total_params

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

