"""
使用 triton.language.make_tensor_descriptor 而非 TensorDescriptor
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.amp import autocast

import triton
from _flash_attention_kernel_optimized_v2 import (
    flash_attention_forward_kernel, 
    flash_attention_dQ_kernel, 
    flash_attention_dKV_kernel
)
from typing import Optional

# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

def flash_attention_forward(Q, K, V, is_causal):
    """
    负责分配 O/LSE 并 launch forward kernel
    """
    B, H, S_q, D = Q.shape
    _, _, S_k, _ = K.shape

    device = Q.device
    dtype = Q.dtype
    O = torch.empty((B, H, S_q, D), dtype=dtype, device=device)
    LSE = torch.empty((B, H, S_q), dtype=torch.float32, device=device)
    
    triton.set_allocator(alloc_fn)

    grid = lambda META: (triton.cdiv(S_q, META['BLOCK_M']), B * H)
    flash_attention_forward_kernel[grid](
        Q, K, V, O, LSE, 
        LSE.stride(0), LSE.stride(1), LSE.stride(2),                              
        1 / (D ** 0.5), 
        B, H, S_q, S_k, D,
        is_causal=is_causal,
    )
    return O, LSE

def flash_attention_backward(Q, K, V, O, dO, LSE, is_causal):
    """
    负责分配 dQ/dK/dV 并 launch backward kernels
    """
    B, H, S_q, D = Q.shape
    _, _, S_k, _ = K.shape

    device = Q.device
    dtype = Q.dtype
    dQ = torch.empty((B, H, S_q, D), dtype=dtype, device=device)
    dK = torch.empty((B, H, S_k, D), dtype=dtype, device=device)
    dV = torch.empty((B, H, S_k, D), dtype=dtype, device=device)
    delta = torch.empty((B, H, S_q), dtype=torch.float32, device=device)
    
    triton.set_allocator(alloc_fn)

    # 计算 dQ
    grid_Q = lambda META: (triton.cdiv(S_q, META['BLOCK_M']), B * H)
    flash_attention_dQ_kernel[grid_Q](
        Q, K, V, dO, O, LSE, dQ, delta, 
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        delta.stride(0), delta.stride(1), delta.stride(2),    
        1 / (D ** 0.5),
        B, H, S_q, S_k, D,
        is_causal=is_causal,
    )

    # 计算 dK/V
    grid_KV = lambda META: (triton.cdiv(S_k, META['BLOCK_N']), B * H)
    flash_attention_dKV_kernel[grid_KV](
        Q, K, V, dO, LSE, dK, dV, delta,
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        delta.stride(0), delta.stride(1), delta.stride(2), 
        1 / (D ** 0.5),
        B, H, S_q, S_k, D,
        is_causal=is_causal, 
    )

    return dQ, dK, dV

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool):
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.dtype in (torch.float16, torch.bfloat16)
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
        assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
        # 1. 确保输入连续
        Q_ = Q.contiguous()
        K_ = K.contiguous()
        V_ = V.contiguous()
        # 2. 调用 Forward Kernel
        # 注意：这里调用的是你封装好的 Python 启动函数
        O, LSE = flash_attention_forward(Q_, K_, V_, is_causal)
        # 3. 保存用于 Backward 的 Tensor
        ctx.save_for_backward(Q_, K_, V_, O, LSE)
        # 4. 保存非 Tensor 参数
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        # 1. 取出保存的 Tensor
        Q, K, V, O, LSE = ctx.saved_tensors
        # 2. 确保输入连续
        dO_ = dO.contiguous()
        # 3. 调用 Backward Kernel
        # 注意：这里调用的也是封装好的 Python 启动函数
        dQ, dK, dV = flash_attention_backward(
            Q, K, V, O, dO_, LSE, ctx.is_causal
        )        
        # 4. 返回梯度
        # 返回值的顺序必须与 forward 的参数顺序完全一致
        # Q, K, V 对应的梯度分别是 dQ, dK, dV
        # is_causal 不是 Tensor，不需要梯度，返回 None
        return dQ, dK, dV, None

# 为了方便用户使用，我们在外部再包一层简单的 API   
def flash_attention(Q, K, V, is_causal=False):
    return FlashAttentionFunction.apply(Q, K, V, is_causal)

def compare_with_sdpa(Q, K, V, is_causal):
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)

    # reference
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION): # 显式启用 FlashAttention 后端
        with autocast(device_type="cuda", dtype=torch.float16):
            O_ref = F.scaled_dot_product_attention(
                Q_ref, 
                K_ref, 
                V_ref,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            )

    dO = torch.randn_like(O_ref)

    O_ref.backward(dO)
    dQ_ref, dK_ref, dV_ref = Q_ref.grad, K_ref.grad, V_ref.grad

    # My FlashAttention
    Q_ = Q.detach().clone().requires_grad_(True)
    K_ = K.detach().clone().requires_grad_(True)
    V_ = V.detach().clone().requires_grad_(True)

    O = flash_attention(Q_, K_, V_, is_causal=is_causal)
    O.backward(dO)
    dQ, dK, dV = Q_.grad, K_.grad, V_.grad

    # verify results
    from _verify_func import verify_results
    print(f"{"=" * 30} O test {"=" * 30}")
    verify_results(O_ref, O)
    print(f"{"=" * 30} dQ test {"=" * 30}")
    verify_results(dQ_ref, dQ)
    print(f"{"=" * 30} dK test {"=" * 30}")
    verify_results(dK_ref, dK)
    print(f"{"=" * 30} dV test {"=" * 30}")
    verify_results(dV_ref, dV)

if __name__ == "__main__":
    DEVICE = torch.device(torch.cuda.current_device())
    B = 4
    H = 8
    S_q = 256
    S_k = 256
    D = 64

    Q = torch.randn((B, H, S_q, D), dtype=torch.float16, device=DEVICE)
    K = torch.randn((B, H, S_k, D), dtype=torch.float16, device=DEVICE)
    V = torch.randn((B, H, S_k, D), dtype=torch.float16, device=DEVICE)

    compare_with_sdpa(Q, K, V, is_causal=True)