import math
import torch
import triton
import triton.language as tl
from typing import Tuple
from jaxtyping import Float, Int, Bool
import setproctitle

setproctitle.setproctitle("python")

class flash_attention_torch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q:Float[torch.Tensor, "b h s_q d_q"], 
        K:Float[torch.Tensor, "b h s_k d_k"], 
        V:Float[torch.Tensor, "b h s_v d_v"],
        is_causal: Bool = True
    )->Float[torch.Tensor, "b h s_q d_v"]:
        device = Q.device
        acc_dtype = Q.dtype
        compute_type = torch.float32
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

        O = torch.zeros(B, H, S_q, V.shape[-1], device=device, dtype=acc_dtype)
        L = torch.zeros(B, H, S_q, 1, device=device, dtype=acc_dtype)
        q_index = 0

        for Q_i in Q_blocks:
            q_size = Q_i.shape[2]
            m_i = -torch.inf * torch.ones(B, H, q_size, 1, dtype=compute_type, device=device)       # B, H, B_q, 1
            L_i = torch.zeros(B, H, q_size, 1, dtype=acc_dtype, device=device)                   # B, H, B_q, 1
            O_i = torch.zeros(B, H, q_size, D_q, dtype=acc_dtype, device=device)                       # B, H, B_q, D_q  

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
                    S_ij = S_ij.masked_fill(future_mask[None, None, :, :], -torch.inf)
                m_curr = torch.max(S_ij, dim=-1, keepdim=True).values                 # B, H, B_q, 1
                m_i = torch.maximum(prev_m_i, m_curr).to(m_curr.device)                                 # B, H, B_q, 1
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
            Q = Q.squeeze(1)
            K = K.squeeze(1)
            V = V.squeeze(1)
            O = O.squeeze(1)
            L = L.squeeze(1)
        L = L.squeeze(-1)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.input_3d = input_3d
        return O
    
    @staticmethod
    def backward(ctx, dO:torch.Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        device = Q.device
        acc_type = Q.dtype
        compute_type = torch.float32
        input_3d = ctx.input_3d
        if input_3d:
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
            O = O.unsqueeze(1)
            dO = dO.unsqueeze(1)
            L = L.unsqueeze(1)
        B_q, H, S_q, D_q = Q.shape
        _, _, S_k, _ = K.shape
        B_q = 64
        B_k = 64

        T_q = S_q // B_q
        T_k = S_k // B_k
        D = torch.sum(O * dO, dim=-1).to(compute_type)               # (B, S_q)
        dQ = torch.zeros_like(Q, device=Q.device).to(compute_type)   # (B, S_q, D_q)
        dK = torch.zeros_like(K, device=K.device).to(compute_type)   # (B, S_k, D_q)
        dV = torch.zeros_like(V, device=V.device).to(compute_type)   # (B, S_k, D_q)

        for i in range(T_q):
            Q_i =   Q[:, :, i * B_q:(i+1) * B_q, :].to(compute_type)    # (B, B_q, D_q)
            dO_i = dO[:, :, i * B_q:(i+1) * B_q, :].to(compute_type)    # (B, B_q, D_q)
            L_i =   L[:, :, i * B_q:(i+1) * B_q].to(compute_type)       # (B, B_q)
            D_i =   D[:, :, i * B_q:(i+1) * B_q].to(compute_type)       # (B, B_q)
            for j in range(T_k):
                K_j = K[:, :, j * B_k:(j+1) * B_k, :].to(compute_type)  # (B, B_k, D_q)
                V_j = V[:, :, j * B_k:(j+1) * B_k, :].to(compute_type)  # (B, B_k, D_q)

                S_ij = Q_i @ K_j.transpose(-2, -1) / (D_q**0.5) # (B, B_q, B_k)
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))      # (B, B_q, B_k)


                dV_j = P_ij.transpose(-2, -1) @ dO_i            # (B, B_k, D_q)
                dV[:, :, j * B_k:(j+1) * B_k, :] += dV_j          

                dP_ij = dO_i @ V_j.transpose(-2, -1)            # (B, B_q, B_k)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))      # (B, B_q, B_k)

                dQ_ij = dS_ij @ K_j / (D_q**0.5)                # (B, B_q, D_q)
                dQ[:, :, i * B_q:(i+1) * B_q, :] += dQ_ij         

                dK_j = dS_ij.transpose(-2, -1) @ Q_i / (D_q**0.5)   # (B, B_k, D_q)
                dK[:, :, j * B_k:(j+1) * B_k, :] += dK_j

        if input_3d:
            dQ = dQ.squeeze(1)
            dK = dK.squeeze(1)
            dV = dV.squeeze(1)

        return dQ.to(acc_type), dK.to(acc_type), dV.to(acc_type), None
                


            
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
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr
    ):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + stride_qb * batch_index,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + stride_kb * batch_index,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + stride_vb * batch_index,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + stride_ob * batch_index,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_str = tl.make_block_ptr(
        L_ptr + stride_lb * batch_index,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, stride_lb),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0)
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')    # (Q_TILE_SIZE, D)
    O_i_run = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L_i_run = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    m_i_run = tl.full((Q_TILE_SIZE, 1), -float('inf'), dtype=tl.float32)
    log2_e = 1.44269504

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (K_TILE_SIZE, D)

        S_ij = tl.dot(Q_i, K_j.T) * scale   # (Q_TILE_SIZE, K_TILE_SIZE)
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        q_mask = q_idx < N_QUERIES
        k_mask = k_idx < N_KEYS
        boundary_mask = q_mask[:, None] & k_mask[None, :]
        if is_causal: 
            causal_mask = q_idx[:, None] < k_idx[None, :]
            valid_mask = boundary_mask & causal_mask
        else:
            valid_mask = boundary_mask
        S_ij = tl.where(valid_mask, -1e6, S_ij)


        m_i_new = tl.maximum(m_i_run, tl.max(S_ij, axis=1, keep_dims=True)) 
        P_ij = tl.exp2(log2_e * (S_ij - m_i_new))    # (K_TILE_SIZE, D)
        L_i_new = tl.exp2(log2_e * (m_i_run - m_i_new)) * L_i_run + tl.sum(P_ij, axis=1, keep_dims=True)    

        P_ij_cast = P_ij.to(V_block_ptr.type.element_ty)
        O_i_new = tl.exp2(log2_e * (m_i_run - m_i_new)) * O_i_run + tl.dot(P_ij_cast, V_j) # (Q_TILE_SIZE, D)

        O_i_run = O_i_new
        L_i_run = L_i_new
        m_i_run = m_i_new
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i_run / L_i_run
    L_i = m_i_run + tl.log(L_i_run)

    tl.store(O_block_ptr, O_i)
    tl.store(L_block_str, L_i)

class flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        Q:torch.Tensor, 
        K:torch.Tensor, 
        V:torch.Tensor, 
        is_causal=False
    ):
        batch_size = Q.shape[0]

        B_q = Q.shape[1]
        B_k = K.shape[1]
        d = Q.shape[2]

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        scale = 1 / d**0.5
        grid = (B_q // Q_TILE_SIZE, batch_size)

        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, B_q, device=Q.device)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L, 
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            B_q,
            B_k,
            scale,
            d,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
        )

        ctx.save_for_backward(Q, K, V, L)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(ctx):
        raise NotImplementedError




