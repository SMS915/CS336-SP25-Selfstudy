import math
import torch
import triton
import triton.language as tl
from typing import Tuple
from jaxtyping import Float, Int, Bool


class flash_attention_torch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q:Float[torch.Tensor, "b h s_q d_q"], 
        K:Float[torch.Tensor, "b h s_k d_k"], 
        V:Float[torch.Tensor, "b h s_v d_v"],
        is_causal: Bool = True
    )->Float[torch.Tensor, "b h s_q d_v"]:
        input_3d = False
        if Q.ndim == 3:
            input_3d = True
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
        B_q = B_k = 16
        B, H, S_q, D_q = Q.shape
        
        Q_blocks = torch.split(Q, B_q, 2)
        K_blocks = torch.split(K, B_k, 2)
        V_blocks = torch.split(V, B_k, 2)

        O = torch.zeros(B, H, S_q, V.shape[-1], device=Q.device)
        L = torch.zeros(B, H, S_q, 1, device=Q.device, dtype=torch.float32)
        q_index = 0

        for Q_i in Q_blocks:
            q_size = Q_i.shape[2]
            m_i = -torch.inf * torch.ones(B, H, q_size, 1, dtype=torch.float32)       # B, H, B_q, 1
            L_i = torch.zeros(B, H, q_size, 1, dtype=torch.float32)                   # B, H, B_q, 1
            O_i = torch.zeros(B, H, q_size, D_q, dtype=Q.dtype)                       # B, H, B_q, D_q  

            k_index = 0
            for K_j, V_j in zip(K_blocks, V_blocks):                                  # B, H, B_k, d_k
                k_size = K_j.shape[2]
                prev_m_i = m_i
                prev_l_i = L_i
                S_ij = Q_i @ K_j.transpose(2, 3)/ math.sqrt(D_q)                      # B, H, B_q, B_k = (B, H, B_q, D_q) x (B, H, D_k(D_q), B_k)
                if is_causal:
                    q_abs = (q_index + torch.arange(q_size, device=Q.device)).view(q_size, 1)
                    k_abs = (k_index + torch.arange(k_size, device=K.device)).view(1, k_size)
                    future_mask = k_abs > q_abs
                    S_ij[future_mask] = -torch.inf
                m_curr = torch.max(S_ij, dim=-1, keepdim=True).values                 # B, H, B_q, 1
                m_i = torch.maximum(prev_m_i, m_curr)                                 # B, H, B_q, 1
                exp_max_diff = torch.exp(prev_m_i - m_i)                              # B, H, B_q, 1

                P_ij = torch.exp(S_ij - m_i)                                          # B, H, B_q, B_k
                L_i = exp_max_diff * prev_l_i + torch.sum(P_ij, dim=-1, keepdim=True) # B, H, B_q, 1
                O_i = exp_max_diff * O_i + (P_ij.to(V_j.dtype) @ V_j)
                k_index += k_size

            O_i_final = O_i * pow(L_i, -1)
            L_i_final = m_i + torch.log(L_i)

            O[:, :, q_index : q_index + q_size, :] = O_i_final
            L[:, :, q_index : q_index + q_size, :] = L_i_final
            q_index += q_size

        if input_3d:
            # 移除虚拟的 head 维度
            O = O.squeeze(1)
            L = L.squeeze(1)
        L = L.squeeze(-1)
        ctx.save_for_backward(Q, K, V, O, L)
        return O
            
@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr
    ):
    
    pass
    



