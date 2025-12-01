from math import sqrt
from typing import Optional

import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat
from jaxtyping import Float, Bool, Int
from typing import Literal

from cs336_basics.utils import Softmax
from torch.backends.cuda import SDPBackend


class Linear(nn.Module):
    """
    一个不带偏置项的线性变换层。
    Attributes:
        in_features (int): 每个输入样本的大小。
        out_features (int): 每个输出样本的大小。
        W (nn.Parameter): 模块的可学习权重，形状为 (out_features, in_features)。
    """
    def __init__(self, in_features : int, out_features : int, device : torch.device | None = None, dtype : torch.dtype | None = None):
        """
        Args:
           in_features (int): 输入特征的数量。
            out_features (int): 输出特征的数量。
            device (torch.device | None, optional): 目标设备。默认为 None。
            dtype (torch.dtype | None, optional): 数据类型。默认为 None。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype, requires_grad=True))

        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean = 0, std = std, a = -3 * std, b = 3 * std)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        对输入张量 x 应用线性变换 (y = x * W^T)。
        Args:
            x (torch.Tensor): 输入张量，形状为 `(..., in_features)`。

        Returns:
            torch.Tensor: 输出张量，形状为 `(..., out_features)`。
        """
        return x @ self.W.T

class Embedding(nn.Module):
    """
    实现一个嵌入层，将 token ID 映射到连续的向量表示。

    Attributes:
        num_embeddings (int): 词汇表的大小。
        embedding_dim (int): 每个嵌入向量的维度。
        embed_matrix (nn.Parameter): 嵌入层的可学习权重。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None, weight_tying = False):
        """
        初始化 Embedding 模块。

       Args:
           num_embeddings (int): 词汇表的大小。
           embedding_dim (int): 每个嵌入向量的维度。
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
            std = (2 / (self.embedding_dim + self.num_embeddings)) ** 0.5
            nn.init.trunc_normal_(self.embed_matrix, mean = 0, std = std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        """
        根据 token ID 检索嵌入向量。

        Args:
            token_ids (torch.Tensor): 包含整数 token ID 的输入张量。
                形状: `(...)`。

        Returns:
            torch.Tensor: 对应的嵌入向量张量。
                形状: `(..., embedding_dim)`。
        """
        return self.embed_matrix[token_ids]

class RMSNorm(nn.Module):
    """
    实现均方根层归一化 (Root Mean Square Layer Normalization)。
    这是 LayerNorm 的一个简化版本，仅使用一个可学习的缩放参数 gamma。

    Attributes:
        d_model (int): 模型的维度。
        eps (float): 用于数值稳定性的 epsilon 值。
        gamma (nn.Parameter): 可学习的缩放参数。
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

        self.gamma = nn.Parameter(torch.ones(d_model).to(self.device), requires_grad=True)

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

        return (self.gamma * x_norm).to(input_type)

class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络模块，遵循 LLaMA 等现代Transformer架构。
    公式: FFN(x) = W2( SiLU(x @ W1) ⊙ (x @ W3) )
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.ffn = FFN(d_model, d_ff, gated=True, activation='silu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ffn(x)

def get_activation_fn(name: str):
    name = name.lower()
    if name == 'relu':
        return F.relu
    elif name == 'gelu':
        return F.gelu
    elif name == 'silu' or name == 'swish':
        return F.silu
    elif name == 'identity':
        return lambda x: x
    else:
        raise NotImplementedError(f"未实现的激活函数: {name}")

class StandardFeedForward(nn.Module):
    """
    使用SiLU的前馈网络模块， 用于对比试验
    公式: FFN(x) = W2(SiLU(x @ W1))
    """

    def __init__(self, d_model: int, d_ff: int = None, activation: str = 'silu'):
        """
        Args:
            d_model (int): 输入和输出维度。
            d_ff (int, optional): 隐藏层维度。若为None，则自动计算并对齐到64的倍数。
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        self.W1 = Linear(self.d_model, self.d_ff)
        self.W2 = Linear(self.d_ff, self.d_model)
        self.act_fn = get_activation_fn(activation)
        self.act_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量。Shape: (..., d_model)
        Returns:
            torch.Tensor: 输出张量。Shape: (..., d_model)
        """
        return self.W2(self.act_fn(self.W1(x)))

class GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, activation: str = "silu"):
        self.d_ff = 64 * ((round(self.d_model * 8 / 3) + 63) // 64) if d_ff is None else d_ff
        self.W1 = Linear(self.d_model, self.d_ff)
        self.W2 = Linear(self.d_ff, self.d_model)
        self.W3 = Linear(self.d_model, self.d_ff)

        self.act_fn = get_activation_fn(activation)
        self.act_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.act_fn(self.W1(x)) * self.W3(x))


class FFN(nn.Module):
    def __init__(self, d_model: int, gated: bool, activation: str, d_ff: int = None):
        self.d_model = d_model
        self.d_ff = d_ff
        self.gated = gated
        self.activation = activation

        if self.gated:
            self.ffn = GatedFeedForward(self.d_model, self.d_ff, self.activation)
        else:
            self.ffn = StandardFeedForward(self.d_model, self.d_ff, self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


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
            theta (float): RoPE的基数，默认为10000.0。
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
            x (torch.Tensor): 输入张量。Shape: (batch_size, d_h, seq_len, d_k)
            token_positions (torch.Tensor): 每个token的绝对位置。Shape: (batch_size, seq_len)
        Returns:
            torch.Tensor: 应用RoPE后的张量。Shape: (batch_size, seq_len, d_k)
        """
        cos, sin = self.cos_table[token_positions].unsqueeze(1), self.sin_table[token_positions].unsqueeze(1) # (batch_size, 1, seq_len, d_half)
        # 按照最后一个维度两两奇偶拆分， 并将最后一个维度移到最前面
        even_x, odd_x = rearrange(x, '... (d_half odd_even) -> odd_even ... d_half', odd_even=2) # Shape: (batch_size, d_h, seq_len, d_half)
        x1_rot = even_x * cos - odd_x * sin
        x2_rot = even_x * sin + odd_x * cos
        return einx.rearrange('... d_half, ... d_half -> ... (d_half (1 + 1))', x1_rot, x2_rot).contiguous()

def ScaledDotProductAttention(Q: Float[torch.Tensor, "batch_size, num_q, q_seq_len, d_q"],
                              K: Float[torch.Tensor, "batch_size, num_k, k_seq_len, d_k"],
                              V: Float[torch.Tensor, "batch_size, num_v, v_seq_len, d_v"],
                              mask: Bool[torch.Tensor, "batch_size q_seq_len k_seq_len"] = None)\
                              -> Float[torch.Tensor, "*batch_size q_seq_len d_v"]:
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
    num_heads_q, num_heads_k = Q.shape[1], K.shape[1]
    if num_heads_q != num_heads_k:
        # GQA情况, Q 头与 K 头的数量不一致, 需要调整形状
        n_rep = num_heads_q // num_heads_k
        Q = rearrange(Q, 'b (h_kv n_rep) l d -> b h_kv n_rep l d', n_rep=n_rep)
        K = rearrange(K, 'b h_kv l d -> b h_kv 1 l d', n_rep=n_rep)
        V = rearrange(V, 'b h_kv l d -> b h_kv 1 l d', n_rep=n_rep)

    d_k = K.shape[-1]
    attn_scores = einsum(Q, K, "b ... q d, b ... k d -> b ... q k") / sqrt(d_k) # 计算缩放后的注意力分数
    if mask is not None: # 应用掩码
        attn_scores = torch.where(mask, attn_scores, float('-inf'))
    attn_weights = Softmax(attn_scores, -1) # 计算注意力权重
    output =  einsum(attn_weights, V, "b ... q k, b ... k d -> b ... q d")

    if num_heads_q != num_heads_k:
        # 如果是GQA情况，还原回去
        output = rearrange(output, "b h_kv n_rep l d -> b (h_kv n_rep) l d")

    return output


class MultiHeadSelfAttention(nn.Module):
    """
    实现多头自注意力机制，可选择性地集成旋转位置编码 (RoPE)，分组查询注意力 (GQA) 和门控注意力 (gated attention) 。

    Attributes:
        d_model (int): 模型的总嵌入维度。
        num_heads (int): Query (查询) 的注意力头数量。
        num_kv_heads (int): Key (键) 和 Value (值) 的注意力头数量。
        head_dim (int): 每个注意力头的维度 (d_model // num_heads)。
        rope (RoPE | None): 旋转位置编码模块实例。
        gated_attn (bool): 是否启用门控机制。
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None, num_kv_heads: int = None, gated_attn: bool = False, flash_attn = False):
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
            gated_attn (bool, optional): 是否启用门控注意力机制。默认为 False。
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # GQA设置: 如果没有指定kv_heads, 则默认为num_heads, 即标准的MHA
        self.num_kv_heads = self.num_heads if num_kv_heads is None else num_kv_heads
        self.gated_attn = gated_attn

        # 确保Query 头的数量能被 KV头的数量整除
        assert self.num_heads % self.num_kv_heads == 0
        # 每个 KV 头对应的 Q 头的数量
        self.num_rep = self.num_heads // self.num_kv_heads

        # 单个头的维度
        self.head_dim = self.d_model // self.num_heads
        self.d_k = self.d_v = self.head_dim
        # Query 投影, (B, L, D)
        self.q_proj = Linear(self.d_model, self.d_model)

        # Key/Value 投影, (B, L, D) -> (B, L, d_head * num_kv_heads) 注意，使用GQA的情况下可能小于d_model
        self.k_proj = Linear(self.d_model, self.head_dim * self.num_kv_heads)
        self.v_proj = Linear(self.d_model, self.head_dim * self.num_kv_heads)

        self.out_proj = Linear(self.d_model, self.d_model)

        # rope初始化
        self.rope = RoPE(theta, self.d_k, self.max_seq_len) if theta is not None and max_seq_len is not None else None

        # 注意力门控机制
        if self.gated_attn:
            # Gating 投影层 (对应论文中 W_theta)
            self.gate_proj = nn.Linear(self.d_model, self.d_model, True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)

        self.flash_attn = flash_attn

    def forward(self, x : torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        对输入张量应用多头自注意力。

        Args:
            x (torch.Tensor): 输入张量。
                形状: `(batch_size, sequence_length, d_model)`。
            token_positions (torch.Tensor, optional): 每个 token 的绝对位置。
                形状: `(batch_size, sequence_length)`。

        Returns:
            torch.Tensor: 输出张量。
                形状: `(batch_size, sequence_length, d_model)`。

        B: batch_size, L: seq_len, H: num_heads,
        D: d_model, H_kv: num_kv_heads, D_h: head_dim
        """
        query = self.q_proj(x) # Shape: (B, L, H * D_h)
        key = self.k_proj(x)   # Shape: (B, L, H_kv * D_h)
        value = self.v_proj(x)
        context_length = query.shape[1]
        queries_multi_head = rearrange(query, "b l (h d) -> b h l d", h=self.num_heads)
        # GQA情况下，reshape按照kv_head的维度进行调整, 进行分组投影
        keys_multi_head = rearrange(key, "b l (h_kv d) -> b h_kv l d", h=self.num_kv_heads) # H_kv 而不是 H
        values_multi_head = rearrange(value, "b l (h_kv d) -> b h_kv l d", h=self.num_kv_heads)

        if self.rope is not None:
            if token_positions is None:
                # 若未提供位置索引，则生成默认的索引[0, 1, 2, ..., S-1]
                # b = query.shape[0]
                # token_positions = torch.arange(context_length, device=x.device).unsqueeze(0).expand(b, -1)
                token_positions = repeat(torch.arange(context_length, device=x.device), 's -> b s', b=x.shape[0])  # einops 版本
            queries_multi_head = self.rope(queries_multi_head, token_positions)
            keys_multi_head = self.rope(keys_multi_head, token_positions)

        if not self.flash_attn:
            mask = torch.tril(torch.ones(context_length, context_length, device=x.device, dtype=torch.bool), diagonal=0)
            mask = mask.view(1, 1, 1, context_length, context_length)
            out = ScaledDotProductAttention(queries_multi_head, keys_multi_head, values_multi_head, mask)
        else:
            out = F.scaled_dot_product_attention(
                queries_multi_head,
                keys_multi_head,
                values_multi_head,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )

        out = rearrange(out, "b h s d -> b s (h d)", h=self.num_heads, d=self.d_v)
        if self.gated_attn:
            gate_score = torch.sigmoid(self.gate_proj(x))
            out = gate_score * out

        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """
    实现一个采用 pre-norm 结构的 Transformer 解码器模块。

    包含一个多头自注意力子层和一个 SwiGLU 前馈网络子层，每个子层
    都有残差连接。

    Attributes:
        d_model (int): 模型的维度。
        num_heads (int): 注意力头的数量。
        attn (MultiHeadSelfAttention): 多头自注意力模块。
        ffn (SwiGLU): SwiGLU 前馈网络模块。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None,theta: float = None,
                 max_seq_len: int = None, gated_ffn: bool = True, activation: str = 'silu',
                 post_norm = False, no_norm: bool = False, num_kv_heads: int = None, gated_attn: bool = False,
                 flash_attn: bool = False):
        """
        初始化 TransformerBlock。

        Args:
            d_model (int): 模型的维度。
            num_heads (int): 注意力头的数量。
            d_ff (int, optional): 前馈网络的隐藏维度。默认为 None。
            theta (float, optional): 用于 RoPE 的基准频率。默认为 None。
            max_seq_len (int, optional): RoPE 所需的最大序列长度。默认为 None。
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
        self.flash_attn = flash_attn

        if not no_norm:
            self.norm1 = RMSNorm(self.d_model)
            self.norm2 = RMSNorm(self.d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, self.max_seq_len, self.theta, self.num_kv_heads, self.gated_attn, self.flash_attn)

        self.ffn = FFN(self.d_model, self.d_ff, self.gated_ffn, self.activation)

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
    一个完整的、仅解码器（Decoder-Only）的 Transformer 语言模型。

    Attributes:
        vocab_size (int): 词汇表的大小。
        context_length (int): 模型的最大上下文长度。
        d_model (int): 模型的内部维度。
        num_layers (int): TransformerBlock 的层数。
        num_heads (int): 注意力头的数量。
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, rope_theta: float, d_ff: int = None, tie_weights: bool = False,
                 post_norm: bool = False, no_norm: bool = False, num_kv_heads: int = None,
                 gated_attn: bool = False, gated_ffn: bool = True, activation: str = 'silu',
                 flash_attn = False):
        """
        初始化 TransformerLM 模型。

        Args:
            vocab_size (int): 词汇表的大小。
            context_length (int): 模型的最大上下文长度。
            d_model (int): 模型的维度。
            num_layers (int): TransformerBlock 的数量。
            num_heads (int): 注意力头的数量。
            rope_theta (float): RoPE 的基准频率。
            d_ff (int, optional): 前馈网络的隐藏维度。默认为 None。
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.post_norm = post_norm
        self.no_norm = no_norm
        self.gated_attn = gated_attn
        self.gated_ffn = gated_ffn
        self.activation = activation
        self.flash_attn = flash_attn

        self.embed = Embedding(self.vocab_size, self.d_model, weight_tying=tie_weights)
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(d_model = d_model, num_heads = self.num_heads, d_ff = self.d_ff,
                                                theta=self.rope_theta, max_seq_len=self.context_length,
                                                gated_ffn=self.gated_ffn, activation=self.activation,
                                                post_norm=self.post_norm, no_norm=self.no_norm,
                                                num_kv_heads=self.num_kv_heads, gated_attn=self.gated_attn, flash_attn=self.flash_attn))
        if not self.no_norm:
            if not self.post_norm:
                self.norm_final = RMSNorm(self.d_model)
            else:
                self.norm_final = nn.Identity()
        else:
            self.norm_final = nn.Identity()
        
        self.lm_head = Linear(d_model, vocab_size)

        if tie_weights:
            self.lm_head.W = self.embed.embed_matrix
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
        if token_positions is None:
            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
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

