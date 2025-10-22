import torch
from math import sqrt
from einops import einsum
from cs336_basics.Softmax import Softmax
from jaxtyping import Float, Bool

def ScaledDotProductAttention(Q: Float[torch.Tensor, "*batch_size query_seq_len d_k"],
                              K: Float[torch.Tensor, "*batch_size key_seq_len d_k"],
                              V: Float[torch.Tensor, "*batch_size value_seq_len d_v"],
                              mask: Bool[torch.Tensor, "*batch_size query_seq_len key_seq_len"] = None)\
                              -> Float[torch.Tensor, "*batch_size query_seq_len d_v"]:
    """
   计算缩放点积注意力。
   公式: Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

   Args:
       q: 查询张量。Shape: (*batch, query_seq_len, d_k)
       k: 键张量。Shape: (*batch, key_seq_len, d_k)
       v: 值张量。Shape: (*batch, value_seq_len, d_v)
       mask: 可选的布尔掩码。Shape: (*batch, query_seq_len, key_seq_len)

   Returns:
       注意力输出。Shape: (*batch, query_seq_len, d_v)
   """
    d_k = K.shape[-1]
    attn_scores = einsum(Q, K, "batch_size ... query d_k, batch_size ... key d_k -> batch_size ... query key") / sqrt(d_k) # 计算缩放后的注意力分数
    if mask is not None: # 应用掩码
        attn_scores = torch.where(mask, attn_scores, float('-inf'))
    attn_weights = Softmax(attn_scores, -1) # 计算注意力权重
    return einsum(attn_weights, V, "batch_size ... query key, batch_size ... key d_v -> batch_size ... query d_v")
