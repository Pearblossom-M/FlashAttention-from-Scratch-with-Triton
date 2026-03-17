# 从零实现 FlashAttention（Phase 4）：反向传播 —— Recomputation 的智慧

目标：

- 掌握 FlashAttention backward 的数学原理和 recomputation 策略
- 实现 dQ 和 dKV 两个 backward kernel
- 将 FlashAttention 集成到 PyTorch 的 autograd 系统中

说明：

- 本章专注于**算法正确性**，性能优化留到 Phase 5

------

## 4.1 Backward 的数学基础

**目标**：理解 dQ, dK, dV 的梯度公式

### 4.1.1 矩阵梯度基础

如果你还没有矩阵微积分基础，那么非常建议先看完这一节（简单但够用）。如果你已经学过矩阵梯度，则可以直接跳过。

在推导 Attention 的梯度时，我们会用到以下几个**基础公式**：

**公式 1：矩阵乘法的梯度**

如果 `C = A @ B`（矩阵乘法），并且已知 `dC`（即损失对 C 的梯度：$\frac{\partial L}{\partial C}$），那么：

```python
dA = dC @ B^T
dB = A^T @ dC
```

* `dA`、`dB` 分别表示损失对 A、B 的梯度：$\frac{\partial L}{\partial A}$ 和 $\frac{\partial L}{\partial B}$

这个公式可以用微分法 + Frobenius 内积推导，但记住结论就够用了。其中：

- `A` 的形状：`[M, K]`，`B` 的形状：`[K, N]`
- `C` 的形状：`[M, N]`，`dC` 的形状：`[M, N]`
- `dA = dC @ B^T` 的形状：`[M, N] @ [N, K] = [M, K]`，与 `A` 形状相同✅
- `dB = A^T @ dC` 的形状：`[K, M] @ [M, N] = [K, N]`，与 `B` 形状相同✅

**公式 2：element-wise 运算的梯度**

如果 `C = A ⊙ B`（element-wise 乘法，即对应位置相乘），那么：

```python
dA = dC ⊙ B # A, B, C shape: [M, N]
dB = dC ⊙ A
```

不同位置之间完全解耦，每个 `C_ij` 只与 `A_ij` 和 `B_ij` 有关，所以 `dA_ij` =  $\frac{\partial L}{\partial A_{ij}}$ = $\frac{\partial L}{\partial C_{ij}}$$\frac{\partial C_{ij}}{\partial A_{ij}}$ = `dC_ij * B_ij`。

**公式 3：缩放的梯度**

如果 `B = αA`（标量乘法，α 为常数标量），那么：

```python
dA = dB * α
```

* 这是一个标量线性函数，因此$\frac{\partial B_{ij}}{\partial A_{ij}}$ = α
* `dA` = $\frac{\partial L}{\partial A}$ = $\frac{\partial L}{\partial B}$$\frac{\partial B}{\partial A}$ = `dB * α`

**公式 4：reduction**

如果 $B_i = \sum_j A_{ij}$ （对 j 维度做 reduction）， 那么：

```python
dA_ij = dB_i # A shape: [M, N], B shape: [M,]
```

$\frac{\partial B_{i}}{\partial A_{ij}}$ = 1，所以：`dA_ij` = $\frac{\partial L}{\partial A_{ij}}$ = $\frac{\partial L}{\partial B_{i}}$ $\frac{\partial B_{i}}{\partial A_{ij}}$ = `dB_i`

**公式 5：broadcast**

如果 $B_{ij} = A_i$， 那么：

```python
dA_i = sum_j dB_ij # A shape: [M,], B shape: [M, N]
```

$\frac{\partial B_{ij}}{\partial A_{i}}$ = 1，所以：`dA_i` = $\sum_j \frac{\partial L}{\partial A_{i}}$ = $\sum_j \frac{\partial L}{\partial B_{ij}}$ $\frac{\partial B_{ij}}{\partial A_{i}}$ = $\sum_j$ `dB_ij`

> 一个变量在 forward 中被用几次，它在 backward 中的梯度就要累加几次。或者说：当一个变量通过多条路径影响损失时，梯度就是这些路径贡献的和。

### 4.1.2 标准 Attention 的反向传播

先回顾一下标准 Attention 的 forward 公式：

```python
S = Q @ K^T / sqrt(D) 
P = softmax(S)  # row-wise
O = P @ V
```

标准 Attention 的梯度公式：

```python
dV = P^T @ dO
dP = dO @ V^T
delta = row_sum(dO ⊙ O) # 优化：避免显式计算 P ⊙ dP
dS = P ⊙ (dP - delta[:, None])
dQ = dS @ K / sqrt(D)
dK = dS^T @ Q / sqrt(D)
```

> 详细的推导过程放在了本章最后的**附录**中。

可以发现：

- 梯度求解依赖 `P` 矩阵
- 标准实现中，`P` 在 forward 时保存了下来，但在 FlashAttention 中，`P` 并未保存

那如何在没有 `P` 的情况下计算梯度呢？这就要用到我们在 Phase 3 的 3.3 节中提到的 **Recomputation** 了。



------

## 4.2 Recomputation 策略

**目标**：理解如何用 `Q`、`K`、`LSE` 重建 `P`

### 4.2.1 从 LSE 重建 P

回顾 forward 时的计算：

```python
# 计算 score 矩阵
S_ij = (Q_i · K_j) / sqrt(D)
# 计算统计量
m_i = max_j(S_ij)
l_i = sum_j(exp(S_ij - m_i))
# 计算概率矩阵
P_ij = exp(S_ij - m_i) / l_i
```

问题描述：

* 已知：`Q`、`K`、`LSE` ，其中 `LSE_i = m_i + log(l_i)`，求：`P_ij`

**重建公式：**

* **步骤一**：重算 `S_ij = (Q_i · K_j) / sqrt(D)`
* **步骤二**：用 `S_ij` 和 `LSE` 还原 `P_ij`，`P_ij = exp(S_ij - LSE_i)`

证明：

```python
LSE_i = m_i + log(l_i)
P_ij = exp(S_ij - m_i) / l_i
= exp(S_ij - m_i) / exp(log(l_i))
= exp(S_ij - m_i - log(l_i))
= exp(S_ij - (m_i + log(l_i)))
= exp(S_ij - LSE_i)
```

> 这就是为什么我们可以在 Phase 3 的 3.3 节存一个合并的值 $LSE$ ，而不用分别存 $m$ 和 $l$ 的原因

通过 Recomputation 重建 `P` 后，就可以用于计算梯度了。

### 4.2.2 为什么使用 Recomputation？

这里大家可能会有一个疑问，为什么一定要**使用 Recomputation 重建 `P`** 呢？

主要原因有两点：

1. FlashAttention 的核心原理就是不存储完整的 `P` 矩阵，从而减少显存占用，降低访存开销，提高计算吞吐量
2. 使用 Recomputation 重建 `P` 实际上**更快**！

这里对比一下两种策略的访存开销：

|          | 策略 A —— 存 P                 | 策略 B —— Recomputation                                  |
| -------- | ------------------------------ | -------------------------------------------------------- |
| Forward  | 写 `P` 到 HBM —— $O(N^2)$ 访存 | 写 `LSE` 到 HBM —— $O(N)$ 访存                           |
| Backward | 从 HBM 读 `P` —— $O(N^2)$ 访存 | 从 HBM 读 `LSE`，重算 `P` —— $O(N)$ 访存 + $O(N^2)$ 计算 |
| 总访存   | $O(N^2)$                       | $O(N)$                                                   |

由于 GPU 的计算速度 >> 访存速度，使得 $O(N^2)$ 的计算代价 < $O(N^2)$ 的访存代价，所以 **Recomputation 反而更快**！

> Recomputation 的本质就是用计算换空间，幸运的是，这里计算很快，所以实际上不仅换了空间，也换到了时间 —— Recomputation = 血赚不亏！



------

## 4.3 Backward Kernel 的设计

**目标**：理解为什么需要两个 kernel，以及它们的分块策略

### 4.3.1 为什么需要两个 Kernel？

梯度公式中：

- dQ 和 dO 的形状相同：`[B, H, S_q, D]`
- dK, dV 和 K, V 的形状相同：`[B, H, S_k, D]`

Forward 的并行策略：

- 每个 program 负责一个 Q_block（`[BLOCK_M, D]`）
- 扫描所有 K/V blocks 来计算这个 Q_block 的输出

Backward 如果也想用同样的策略：

- 计算 dQ 可以，因为 dQ 和 Q 形状相同
- 但计算 dK/dV 就不行了：
  - 每个 K/V_block 会被多个 Q_blocks 用到
  - 所以需要**累加所有 Q_blocks 对这个 K/V_block 梯度的贡献**，这和 forward 的逻辑不同
- 虽然可以通过**原子加（`tl.atomic_add`）**实现一个 kernel 在计算 dK/dV 的同时，累加dQ，但是，对于 Triton 而言，使用 `tl.atomic_add` 会让编译器难以进行极致优化，通常性能低于双 kernel 的方案。

> 如果使用 CUDA，则可以使用单 kernel 的方案，因为 CUDA 可以通过更底层的优化（比如手写 PTX）更好的设计 Pipeline，掩盖原子加操作带来的延迟。

**结论**：需要两个 kernel

- **dQ kernel**：和 forward 类似，每个 program 负责一个 Q_block
- **dKV kernel**：每个 program 负责一个 K/V_block，扫描所有 Q_blocks 并累加

### 4.3.2 dQ Kernel 的设计

输入：

- Q, K, V：形状 `[B, H, S_q/S_k, D]`
- dO：形状 `[B, H, S_q, D]`
- LSE：形状 `[B, H, S_q]`

输出：

- dQ：形状 `[B, H, S_q, D]`

算法流程：

```python
for each Q_block:
    dQ_block = 0
    delta_block = row_sum(dO_block ⊙ O_block)  # 预计算, Q_block 确定, dO_block 和 O_block 就确定, 因此不需要在内循环计算

    for each K/V_block:
        # 重建 P_block
        S_block = Q_block @ K_block.T / sqrt(D)
        P_block = exp(S_block - LSE[:, None])
        
        # 计算 dP_block
        dP_block = dO_block @ V_block.T
        
        # 计算 dS_block
        dS_block = P_block ⊙ (dP_block - delta_block[:, None])
        
        # 累加到 dQ
        dQ_block += dS_block @ K_block / sqrt(D)
    
    write back dQ_block
```

循环内部类似 forward，但计算的是梯度

### 4.3.3 dKV Kernel 的设计

输入：

- Q, K, V, dO, LSE

输出：

- dK, dV：形状 `[B, H, S_k, D]`

算法流程：

```python
for each K/V_block:
    dK_block = 0
    dV_block = 0
 
    for each Q_block:
        # 重建 P_block (和 dQ kernel 一样)
        S_block = Q_block @ K_block.T / sqrt(D)
        P_block = exp(S_block - LSE[:, None])
        
        # 计算 dV
        dV_block += P_block.T @ dO_block
        
        # 计算 delta_block
        delta_block = row_sum(dO_block ⊙ O_block)
        
        # 计算 dK (需要 dS)
        dP_block = dO_block @ V_block.T
        dS_block = P_block ⊙ (dP_block - delta_block[:, None])
        dK_block += dS_block.T @ Q_block / sqrt(D)
    
    write back dK_block, dV_block
```

**注意**：

- 需要累加所有 Q_blocks 对 K/V 梯度的贡献
- dK 和 dV 可以在同一个 kernel 里算（共享 P 的重建）



------

## 4.4 dQ Kernel 实现

经过前面几章的学习，相信大家已经对 triton 比较熟悉了，所以之后就不再使用 python 模拟，而是直接给出 triton 的代码实现。

### 4.4.1 Triton 实现

```python
import triton
import triton.language as tl

@triton.jit
def flash_attention_dQ_kernel(
    # -------------------- 指针 ------------------------
    Q, K, V, dO, O, LSE,  # 输入指针    
    dQ,                   # 输出指针

    # -------------------- stride ----------------------
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,

    # -------------------- 缩放因子 --------------------
    scale, # 1 / sqrt(D)

    # -------------------- 维度参数 --------------------
    B, H, S_q, S_k, 
    D: tl.constexpr,

    # -------------------- 配置参数 --------------------
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

    # -------------------- Flag参数 --------------------
    is_causal: tl.constexpr = False,
):
    """
    FlashAttention dQ kernel
    每个 program 负责计算一个 Q_block 的梯度
    """
    # 获取当前 program 的索引
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    batch_idx = pid_1 // H
    head_idx = pid_1 % H
    
    # 计算基地址偏移
    Q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    K_base = K + batch_idx * stride_kb + head_idx * stride_kh
    V_base = V + batch_idx * stride_vb + head_idx * stride_vh
    dO_base = dO + batch_idx * stride_dob + head_idx * stride_doh
    O_base = O + batch_idx * stride_ob + head_idx * stride_oh
    LSE_base = LSE + batch_idx * stride_lse_b + head_idx * stride_lse_h
    dQ_base = dQ + batch_idx * stride_dqb + head_idx * stride_dqh
    
    # 计算 Q_block 的起始位置
    q_block_start = pid_0 * BLOCK_M

    # 行列索引
    Sq_offs = q_block_start + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)

    # mask：处理最后一个 Q_block
    mask_Sq = Sq_offs < S_q
   
    # load Q_block
    q_ptrs = Q_base + Sq_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    Q_block = tl.load(q_ptrs, mask=mask_Sq[:, None], other=0.0) # [BLOCK_M, D]
    
    # load dO_block 和 O_block
    do_ptrs = dO_base + Sq_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
    dO_block = tl.load(do_ptrs, mask=mask_Sq[:, None], other=0.0) # [BLOCK_M, D]
    o_ptrs = O_base + Sq_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    O_block = tl.load(o_ptrs, mask=mask_Sq[:, None], other=0.0) # [BLOCK_M, D]
    
    # load LSE_block
    lse_ptrs = LSE_base + Sq_offs * stride_lse_s
    LSE_block = tl.load(lse_ptrs, mask=mask_Sq, other=0.0) # [BLOCK_M,]
    
    # 预计算 delta = sum(dO ⊙ O, dim=-1)
    delta_block = tl.sum(dO_block.to(tl.float32) * O_block.to(tl.float32), 
                         axis=-1)  # [BLOCK_M,]
    
    # 初始化 dQ_block 累加器
    dQ_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    LOG2_E = 1.44269504 # log2(e), 用于tl.exp 到 tl.exp2 的转化

    # Causal mask 的循环边界
    loop_end = q_block_start + BLOCK_M if is_causal else S_k   
    for start_s in range(0, loop_end, BLOCK_N):
        Sk_offs = start_s + tl.arange(0, BLOCK_N)
        mask_Sk = Sk_offs < S_k
        
        # 加载 K_block 和 V_block
        k_ptrs = K_base + Sk_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        K_block = tl.load(k_ptrs, mask=mask_Sk[:, None], other=0.0) # [BLOCK_N, D]
        v_ptrs = V_base + Sk_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        V_block = tl.load(v_ptrs, mask=mask_Sk[:, None], other=0.0) # [BLOCK_N, D]
        
        # 1. 计算 S_block = Q_block @ K_block.T * scale
        S_block = tl.dot(Q_block, tl.trans(K_block)) * scale  # [BLOCK_M, BLOCK_N]
        S_block = tl.where(mask_Sk[None, :], S_block, float('-inf')) # [BLOCK_M, BLOCK_N]
        # 处理 padding 行, 也可以不加这一句, 因为 padding 的那几行并不会影响到结果
        S_block = tl.where(mask_Sq[:, None], S_block, float('-inf')) 

        # 2. 应用 causal mask
        if is_causal:
            q_idx_min = q_block_start
            k_idx_max = start_s + BLOCK_N - 1
            if not (q_idx_min >= k_idx_max):
                causal_mask = Sq_offs[:, None] >= Sk_offs[None, :] # [BLOCK_M, BLOCK_N]
                S_block = tl.where(causal_mask, S_block, float('-inf')) # [BLOCK_M, BLOCK_N]
        
        # 3. 重建 P_block = exp(S_block - LSE_block)
        P_block = tl.exp2((S_block - LSE_block[:, None]) * LOG2_E)  # [BLOCK_M, BLOCK_N]
        
        # 4. 计算 dP_block = dO @ V^T
        dP_block = tl.dot(dO_block, tl.trans(V_block))  # [BLOCK_M, BLOCK_N]
        
        # 5. 计算 dS_block = P_block ⊙ (dP_block - delta_block)
        dS_block = P_block * (dP_block - delta_block[:, None])  # [BLOCK_M, BLOCK_N]
        
        # 6. 累加 dQ = dS @ K * scale
        dQ_acc += tl.dot(dS_block.to(tl.float16), K_block) * scale  # [BLOCK_M, D]
    
    # write back to dQ_block
    dq_ptrs = dQ_base + Sq_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, dQ_acc, mask=mask_Sq[:, None])
```



### 4.4.2 正确性验证

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.amp import autocast

def test_dQ_kernel(B, H, S_q, S_k, D, BLOCK_M, BLOCK_N, is_causal, device):
    torch.manual_seed(42)

    Q = torch.randn((B, H, S_q, D), dtype=torch.float16, device=device)
    K = torch.randn((B, H, S_k, D), dtype=torch.float16, device=device)
    V = torch.randn((B, H, S_k, D), dtype=torch.float16, device=device)
    dO = torch.randn((B, H, S_q, D), dtype=torch.float16, device=device)
    O = torch.empty((B, H, S_q, D), dtype=torch.float16, device=device)
    dQ = torch.empty((B, H, S_q, D), dtype=torch.float16, device=device)
    LSE = torch.empty((B, H, S_q), dtype=torch.float32, device=device)

    # 使用 PyTorch 计算参考结果
    Q_ref = Q.detach().clone().requires_grad_(True)  # fp16
    K_ref = K.detach().clone().requires_grad_(True)  # fp16
    V_ref = V.detach().clone().requires_grad_(True)  # fp16

    # Forward (使用 PyTorch 的 scaled_dot_product_attention)
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

    # Backward
    O_ref.backward(dO)
    dQ_ref = Q_ref.grad # 参考结果
    
    # 使用我们在 Phase 3 中完成的前向算子, 先 forward 获得 O 和 LSE
    from _fwd_LogSumExp import flash_attention_forward_kernel # _fwd_LogSumExp：存放前向算子的文件名, 大家需要换成自己的文件路径
    grid = (triton.cdiv(S_q, BLOCK_M), B * H)
    flash_attention_forward_kernel[grid](
        Q, K, V, O,
        LSE, 
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        1 / (D ** 0.5),
        B, H, S_q, S_k, D, 
        BLOCK_M, BLOCK_N,       
        is_causal,
        num_warps=4,
        num_stages=3,
    )
    
    # 计算 dQ
    flash_attention_dQ_kernel[grid](
        Q, K, V, dO, O, LSE, dQ,        
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        1 / (D ** 0.5),
        B, H, S_q, S_k, D,
        BLOCK_M, BLOCK_N, 
        is_causal, 
        num_warps=4,
        num_stages=3,
    )

    return dQ_ref, dQ
```

校验函数依然使用 Phase 2 的 2.2.3 所展示的 `verify_results`，测试结果如下：

`is_causal = True` 时：

* 当 B = 32，H = 8，S_q = 128，S_k = 128，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* 当 B = 32，H = 8，S_q = 500，S_k = 500，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* ……
* 当 B = 32，H = 8，S_q = 1024，S_k = 1024，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过

`is_causal = False` 时：

* 当 B = 32，H = 8，S_q = 128，S_k = 128，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* 当 B = 32，H = 8，S_q = 500，S_k = 500，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* ……
* 当 B = 32，H = 8，S_q = 1024，S_k = 1024，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过

------

## 4.5 dKV Kernel 实现

### 4.5.1 Triton 实现

```python
import triton
import triton.language as tl

@triton.jit
def flash_attention_dKV_kernel(
    # -------------------- 指针 ------------------------
    Q, K, V, dO, O, LSE,  # 输入指针    
    dK, dV,               # 输出指针

    # -------------------- stride ----------------------
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,

    # -------------------- 缩放因子 --------------------
    scale, # 1 / sqrt(D)

    # -------------------- 维度参数 --------------------
    B, H, S_q, S_k, 
    D: tl.constexpr,

    # -------------------- 配置参数 --------------------
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

    # -------------------- Flag参数 --------------------
    is_causal: tl.constexpr = False,
):
    """
    FlashAttention dKV kernel
    每个 program 负责计算一个 K/V_block 的梯度
    """
    # 获取当前 program 的索引
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    batch_idx = pid_1 // H
    head_idx = pid_1 % H
    
    # 计算基地址偏移
    Q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    K_base = K + batch_idx * stride_kb + head_idx * stride_kh
    V_base = V + batch_idx * stride_vb + head_idx * stride_vh
    dO_base = dO + batch_idx * stride_dob + head_idx * stride_doh
    O_base = O + batch_idx * stride_ob + head_idx * stride_oh
    LSE_base = LSE + batch_idx * stride_lse_b + head_idx * stride_lse_h
    dK_base = dK + batch_idx * stride_dkb + head_idx * stride_dkh
    dV_base = dV + batch_idx * stride_dvb + head_idx * stride_dvh
    
    # 计算 K/V_block 的起始位置
    kv_block_start = pid_0 * BLOCK_N

    # 行列索引
    Sk_offs = kv_block_start + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    # mask：处理最后一个 K/V_block
    mask_Sk = Sk_offs < S_k

    # load K/V_block
    k_ptrs = K_base + Sk_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
    K_block = tl.load(k_ptrs, mask=mask_Sk[:, None], other=0.0) # [BLOCK_N, D]
    v_ptrs = V_base + Sk_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
    V_block = tl.load(v_ptrs, mask=mask_Sk[:, None], other=0.0) # [BLOCK_N, D]
    
    # 初始化 dK_block, dV_block 累加器
    dK_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dV_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    
    LOG2_E = 1.44269504 # log2(e), 用于tl.exp 到 tl.exp2 的转化
    
    # Causal mask 的循环边界
    loop_start = kv_block_start if is_causal else 0
    for start_s in range(loop_start, S_q, BLOCK_M):
        Sq_offs = start_s + tl.arange(0, BLOCK_M)
        mask_Sq = Sq_offs < S_q
        
        # load Q_block, dO_block, O_block, LSE_block
        q_ptrs = Q_base + Sq_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
        Q_block = tl.load(q_ptrs, mask=mask_Sq[:, None], other=0.0) # [BLOCK_M, D]
        do_ptrs = dO_base + Sq_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
        dO_block = tl.load(do_ptrs, mask=mask_Sq[:, None], other=0.0) # [BLOCK_M, D]
        o_ptrs = O_base + Sq_offs[:, None] * stride_os + d_offs[None, :] * stride_od
        O_block = tl.load(o_ptrs, mask=mask_Sq[:, None], other=0.0) # [BLOCK_M, D]
        lse_ptrs = LSE_base + Sq_offs * stride_lse_s
        LSE_block = tl.load(lse_ptrs, mask=mask_Sq, other=0.0) # [BLOCK_M,]
        
        # 1. 计算 S_block = Q_block @ K_block.T * scale
        S_block = tl.dot(Q_block, tl.trans(K_block)) * scale  # [BLOCK_M, BLOCK_N]
        S_block = tl.where(mask_Sk[None, :], S_block, float('-inf')) # [BLOCK_M, BLOCK_N]
        # 处理 padding 行, 不加这一行会导致 P_block 的 padding 行=1, 而不是0, 从而影响 dV_acc
        S_block = tl.where(mask_Sq[:, None], S_block, float('-inf')) 
        
        # 2. 应用 causal mask
        if is_causal:
            q_idx_min = start_s
            k_idx_max = kv_block_start + BLOCK_N - 1
            if not (q_idx_min >= k_idx_max):
                causal_mask = Sq_offs[:, None] >= Sk_offs[None, :] # [BLOCK_M, BLOCK_N]
                S_block = tl.where(causal_mask, S_block, float('-inf')) # [BLOCK_M, BLOCK_N]
        
        # 3. 重建 P_block = exp(S_block - LSE_block)
        P_block = tl.exp2((S_block - LSE_block[:, None]) * LOG2_E)  # [BLOCK_M, BLOCK_N]

        # 4. 累加 dV = P^T @ dO
        dV_acc += tl.dot(tl.trans(P_block).to(tl.float16), dO_block)   # [BLOCK_N, D]
        
        # 5. 计算 dP_block = dO @ V^T
        dP_block = tl.dot(dO_block, tl.trans(V_block))  # [BLOCK_M, BLOCK_N]
        
        # 6. 计算 delta = sum(dO ⊙ O, dim=-1)
        delta_block = tl.sum(dO_block.to(tl.float32) * O_block.to(tl.float32), 
                            axis=-1)  # [BLOCK_M,]
    
        # 7. 计算 dS_block = P_block ⊙ (dP_block - delta_block)
        dS_block = P_block * (dP_block - delta_block[:, None])  # [BLOCK_M, BLOCK_N]
        
        # 8. 累加 dK = dS^T @ Q * scale
        dK_acc += tl.dot(tl.trans(dS_block).to(tl.float16), Q_block) * scale  # [BLOCK_N, D]
    
    # write back to dK/dV_block
    dk_ptrs = dK_base + Sk_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd
    tl.store(dk_ptrs, dK_acc, mask=mask_Sk[:, None])
    dv_ptrs = dV_base + Sk_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd
    tl.store(dv_ptrs, dV_acc, mask=mask_Sk[:, None])
```



### 4.5.2 正确性验证

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.amp import autocast

def test_dKV_kernel(B, H, S_q, S_k, D, BLOCK_M, BLOCK_N, is_causal, device):
    torch.manual_seed(42)

    Q = torch.randn((B, H, S_q, D), dtype=torch.float16, device=device)
    K = torch.randn((B, H, S_k, D), dtype=torch.float16, device=device)
    V = torch.randn((B, H, S_k, D), dtype=torch.float16, device=device)
    dO = torch.randn((B, H, S_q, D), dtype=torch.float16, device=device)
    O = torch.empty((B, H, S_q, D), dtype=torch.float16, device=device)
    dK = torch.empty((B, H, S_k, D), dtype=torch.float16, device=device)
    dV = torch.empty((B, H, S_k, D), dtype=torch.float16, device=device)
    LSE = torch.empty((B, H, S_q), dtype=torch.float32, device=device)

    # 使用 PyTorch 计算参考结果
    Q_ref = Q.detach().clone().requires_grad_(True)  # fp16
    K_ref = K.detach().clone().requires_grad_(True)  # fp16
    V_ref = V.detach().clone().requires_grad_(True)  # fp16

    # Forward (使用 PyTorch 的 scaled_dot_product_attention)
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

    # Backward
    O_ref.backward(dO)
    dK_ref = K_ref.grad
    dV_ref = V_ref.grad
    
    # 使用我们的 kernel, 先 forward 获得 O 和 LSE
    from _fwd_LogSumExp import flash_attention_forward_kernel # _fwd_LogSumExp：存放前向算子的文件名, 大家需要换成自己的文件路径
    grid_fwd = (triton.cdiv(S_q, BLOCK_M), B * H)
    flash_attention_forward_kernel[grid_fwd](
        Q, K, V, O,
        LSE, 
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        1 / (D ** 0.5),
        B, H, S_q, S_k, D, 
        BLOCK_M, BLOCK_N,       
        is_causal,
        num_warps=4,
        num_stages=3,
    )
    
    # 计算 dK/V
    grid_bwd = (triton.cdiv(S_k, BLOCK_N), B * H)
    flash_attention_dKV_kernel[grid_bwd](
        Q, K, V, dO, O, LSE, dK, dV,      
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        1 / (D ** 0.5),
        B, H, S_q, S_k, D,
        BLOCK_M, BLOCK_N, 
        is_causal, 
        num_warps=4,
        num_stages=3,
    )

    return dK_ref, dV_ref, dK, dV
```

校验函数同上，测试结果如下：

`is_causal = True` 时：

* 当 B = 32，H = 8，S_q = 128，S_k = 128，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* 当 B = 32，H = 8，S_q = 500，S_k = 500，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* ……
* 当 B = 32，H = 8，S_q = 1024，S_k = 1024，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过

`is_causal = False` 时：

* 当 B = 32，H = 8，S_q = 128，S_k = 128，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* 当 B = 32，H = 8，S_q = 500，S_k = 500，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过
* ……
* 当 B = 32，H = 8，S_q = 1024，S_k = 1024，D = 64，BLOCK_M = 64，BLOCK_N = 64：测试通过



------

## 4.6 PyTorch Autograd 集成

**目标**：把我们写好的 forward / backward kernels 包装成一个 PyTorch 算子，使其可以参与真正的模型训练

### 4.6.1 理解 autograd.Function 的工作流程

**Forward kernel 返回什么？**

- `O`：注意力输出，形状 `[B, H, S_q, D]`
- `LSE`：每一行 softmax 的统计量（LogSumExp），形状 `[B, H, S_q]`

**Backward kernel 需要什么？**

- `Q, K, V`：用于重算 `S` 与 `P`
- `O, dO`：用于计算 `delta = row_sum(dO ⊙ O)`
- `LSE`：用于重建 `P = exp(S - LSE[:, None])`

**ctx 里存什么？**

```
ctx.save_for_backward(Q, K, V, O, LSE)
ctx.is_causal = is_causal
```

`dO` 由外部传入，不使用 `ctx` 存储。

### 4.6.2 准备 Python 封装

为了让 `torch.autograd.Function` 更简洁，需要准备两个 Python 包装函数：

- `flash_attention_forward(Q, K, V, is_causal)`：负责分配 `O/LSE` 并 launch forward kernel
- `flash_attention_backward(Q, K, V, O, dO, LSE, is_causal)`：负责分配 `dQ/dK/dV` 并 launch backward kernels

```python
def flash_attention_forward(Q, K, V, is_causal):
    """
    负责分配 O/LSE 并 launch forward kernel
    """
    B, H, S_q, D = Q.shape
    _, _, S_k, _ = K.shape
    BLOCK_M = 64
    BLOCK_N = 64

    device = Q.device
    dtype = Q.dtype
    O = torch.empty((B, H, S_q, D), dtype=dtype, device=device)
    LSE = torch.empty((B, H, S_q), dtype=torch.float32, device=device)

    grid = (triton.cdiv(S_q, BLOCK_M), B * H)
    flash_attention_forward_kernel[grid](
        Q, K, V, O, LSE,                     
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),  
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),  
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),  
        O.stride(0), O.stride(1), O.stride(2), O.stride(3), 
        LSE.stride(0), LSE.stride(1), LSE.stride(2),         
        1 / (D ** 0.5), 
        B, H, S_q, S_k, D,
        BLOCK_M, BLOCK_N,
        is_causal,
        num_warps=4,
        num_stages=3,
    )
    return O, LSE

def flash_attention_backward(Q, K, V, O, dO, LSE, is_causal):
    """
    负责分配 dQ/dK/dV 并 launch backward kernels
    """
    B, H, S_q, D = Q.shape
    _, _, S_k, _ = K.shape
    BLOCK_M = 64
    BLOCK_N = 64

    device = Q.device
    dtype = Q.dtype
    dQ = torch.empty((B, H, S_q, D), dtype=dtype, device=device)
    dK = torch.empty((B, H, S_k, D), dtype=dtype, device=device)
    dV = torch.empty((B, H, S_k, D), dtype=dtype, device=device)

    # 计算 dQ
    grid_Q = (triton.cdiv(S_q, BLOCK_M), B * H)
    flash_attention_dQ_kernel[grid_Q](
        Q, K, V, dO, O, LSE, dQ,        
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        1 / (D ** 0.5),
        B, H, S_q, S_k, D,
        BLOCK_M, BLOCK_N, 
        is_causal, 
        num_warps=4,
        num_stages=3,
    )

    # 计算 dK/V
    grid_KV = (triton.cdiv(S_k, BLOCK_N), B * H)
    flash_attention_dKV_kernel[grid_KV](
        Q, K, V, dO, O, LSE, dK, dV,      
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        1 / (D ** 0.5),
        B, H, S_q, S_k, D,
        BLOCK_M, BLOCK_N, 
        is_causal, 
        num_warps=4,
        num_stages=3,
    )

    return dQ, dK, dV
```

### 4.6.3 定义 torch.autograd.Function

```python
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
```

为了方便使用，我们在外部再包一层简单的 API：

```python
def flash_attention(Q, K, V, is_causal=False):
    return FlashAttentionFunction.apply(Q, K, V, is_causal)
```

### 4.6.4 测试梯度正确性

```python
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
    from _verify_func import verify_results # Phase 2 的 2.2.3 中的 verify_results 函数
    print(f"{"=" * 30} dQ test {"=" * 30}")
    verify_results(dQ_ref, dQ)
    print(f"{"=" * 30} dK test {"=" * 30}")
    verify_results(dK_ref, dK)
    print(f"{"=" * 30} dV test {"=" * 30}")
    verify_results(dV_ref, dV)

DEVICE = torch.device(torch.cuda.current_device())
B = 32
H = 8
S_q = 1024
S_k = 1024
D = 64

Q = torch.randn((B, H, S_q, D), dtype=torch.float16, device=DEVICE)
K = torch.randn((B, H, S_k, D), dtype=torch.float16, device=DEVICE)
V = torch.randn((B, H, S_k, D), dtype=torch.float16, device=DEVICE)

compare_with_sdpa(Q, K, V, is_causal=True) # ✅ Test Passed!
```

> 由于 FlashAttention 工作在 fp16/bf16 精度并包含指数运算，数值微分的 `gradcheck` 在中等规模下容易误报。因此我们采用与工业界一致的方式：使用 PyTorch 原生实现作为数值参考进行正确性校验。

------

## 4.7 小结

### 4.7.1 回顾 recomputation 的智慧

在本章中，我们攻克了 FlashAttention 最具挑战性的部分——反向传播。我们也深刻体会到了算法设计中的权衡之美：

- **空间换时间？不，是计算换空间！也换时间！**：通常我们认为重计算（Recomputation）会拖慢速度，但在 Memory Wall（内存墙）日益严峻的今天，利用高速的 Tensor Core 重新计算 $S$ 和 $P$ 矩阵，竟然比从 HBM 中读取它们还要快。
- **统计量 (LSE) 的妙用**：我们不需要保存巨大的 $O(N^2)$ 的 $P$ 矩阵，只需要保存 $O(N)$ 大小的 $LSE$ 统计量，就能完美还原出 Softmax 的结果。这是 FlashAttention 能够节省显存，加速计算的关键。

### 4.7.2 我们现在站在哪里？

到这里为止，我们已经**从零实现**了一个具备以下能力的 **FlashAttention 算子**！👏👏👏

* **正确性**：通过了 PyTorch 原生算子的数值对齐测试。

* **功能性**：支持 Batch、Multi-Head、Causal Mask 和反向传播，并接入了 `autograd` 系统。

### 4.7.3 Phase 5 预告：极致性能优化

虽然我们的算子已经“能用”了，但在“好用”之前，还有一段路要走。目前的实现中，我们还存在许多性能瓶颈：

- kernel 内大量地址计算引入额外的指令与寄存器占用，怎么解决？
- 流水线（Pipeline）是否重叠？有没有气泡？
- Block 大小选择是否最优？
- ……

在接下来的 **Phase 5** 中，我们将不再关注新的功能，而是带上“性能显微镜”，深入 GPU 架构的微观世界。我们将学习 Autotune、TensorDescriptor、计算流程优化等高级技巧，让我们的 FlashAttention 真正飞起来！

准备好压榨 GPU 的最后一滴算力了吗？让我们进入 Phase 5 吧！

---

## 4.8 附录

**目标**：详细推导标准 Attention 的 backward 梯度公式

> 这部分内容较为数学化，如果你只关心结论，可以跳过。但如果你想深入理解，建议仔细阅读。

### A. 符号定义

- $Q \in \mathbb{R}^{S_q \times D}$：Query 矩阵
- $K,V \in \mathbb{R}^{S_k \times D}$：Key / Value 矩阵  
- $S \in \mathbb{R}^{S_q \times S_k}$：Score 矩阵  
- $P \in \mathbb{R}^{S_q \times S_k}$：注意力权重矩阵
- $O \in \mathbb{R}^{S_q \times D}$：Attention 输出
- $dO \in \mathbb{R}^{S_q \times D}$：Attention 输出的梯度（已知）

我们的目标是求：

- $dQ \in \mathbb{R}^{S_q \times D}$
- $dK,dV \in \mathbb{R}^{S_k \times D}$

### B. dV 和 dP 的推导

```python
O = P @ V
```

根据**矩阵乘法的梯度公式**（见 4.1.1 节）：

```python
dV = P^T @ dO
dP = dO @ V^T
```

**验证维度**：

- `P` 的形状：`[S_q, S_k]`，`V` 的形状：`[S_k, D]`，`dO` 的形状：`[S_q, D]`
- `dV = P^T @ dO` 的形状：`[S_k, S_q] @ [S_q, D] = [S_k, D]` ✅
- `dP = dO @ V^T` 的形状：`[S_q, D] @ [D, S_k] = [S_q, S_k]` ✅

### C. dS 的推导（核心部分）

#### C.1 Row-wise Softmax 的定义

对于矩阵 S 的第 i 行，softmax 的定义是：

```python
P_ij = exp(S_ij) / sum_k(exp(S_ik))
```

为了数值稳定性，实际计算时会减去最大值：

```python
m_i = max_k(S_ik)
P_ij = exp(S_ij - m_i) / sum_k(exp(S_ik - m_i))
```

注意：该变换并不会改变函数值，由于梯度只依赖于函数本身，而函数本身并未改变，所以在推导梯度时，我们可以忽略这个技巧，直接用原始定义。

#### C.2 先算 Softmax 的 Jacobian 矩阵

对于第 i 行，softmax 是一个从 $\mathbb{R^{S_k}}$ 到 $\mathbb{R^{S_k}}$ 的函数：

```python
P_i = softmax(S_i)
```

其中 `S_i = [S_i1, S_i2, ...]`（共 S_k 个元素）, `P_i = [P_i1, P_i2, ...]`（共 S_k 个元素）。

**问题**：求 `∂P_ik / ∂S_ij`（P 的第 k 个元素对 S 的第 j 个元素的偏导）。

**情况 1**：`k = j`（对角线元素）

```python
P_ij = exp(S_ij) / l_i    # 其中 l_i = sum_k(exp(S_ik))

∂P_ij / ∂S_ij = ∂/∂S_ij [exp(S_ij) / l_i]
              = [exp(S_ij) * l_i - exp(S_ij) * exp(S_ij)] / l_i^2 
              = exp(S_ij) / l_i * [1 - exp(S_ij) / l_i]
              = P_ij * (1 - P_ij)
```

**情况 2**：`k ≠ j`（非对角线元素）

```python
P_ik = exp(S_ik) / l_i

∂P_ik / ∂S_ij = ∂/∂S_ij [exp(S_ik) / l_i]
              = [0 - exp(S_ik) * exp(S_ij)] / l_i^2    # exp(S_ik) 不依赖 S_ij
              = -exp(S_ik) / l_i * exp(S_ij) / l_i
              = -P_ik * P_ij
```

**总结**：

```python
∂P_ik / ∂S_ij = {
    P_ij * (1 - P_ij),  if k = j
    -P_ik * P_ij,       if k ≠ j
}
```

#### C.3 应用链式法则

现在我们要计算 `dS_ij`（损失 L 对 S_ij 的梯度）。

根据链式法则：

```python
dS_ij = ∂L/∂S_ij
      = sum_k( ∂L/∂P_ik * ∂P_ik/∂S_ij )    # 对所有受 S_ij 影响的 P_ik 求和
```

将 C.2 的结果代入：

```python
dS_ij = dP_ij * P_ij * (1 - P_ij) + sum_{k≠j}( dP_ik * (-P_ik * P_ij) )
      = dP_ij * P_ij * (1 - P_ij) - P_ij * sum_{k≠j}( dP_ik * P_ik )
      = dP_ij * P_ij - dP_ij * P_ij^2 - P_ij * sum_{k≠j}( dP_ik * P_ik )
      = dP_ij * P_ij - P_ij * ( dP_ij * P_ij + sum_{k≠j}( dP_ik * P_ik ) )
      = dP_ij * P_ij - P_ij * sum_k( dP_ik * P_ik )
```

**定义** `delta_i = sum_k( dP_ik * P_ik )`，则：

```python
dS_ij = P_ij * (dP_ij - delta_i)
```

#### C.4 矩阵形式

写成矩阵形式：

```python
delta = row_sum(dP ⊙ P)         # delta_i = sum_k(dP_ik * P_ik)
dS = P ⊙ (dP - delta[:, None])  # element-wise 操作
```

### D. 优化 `delta` 计算方式

如 C 所示：如果直接按照原始定义实现，就意味着 `delta` 需要在 K/V loop 中计算，这会导致反复计算，增加额外负担。

然而，通过数学变换可以发现，`delta` 实际上只依赖于 forward 的输出 `O` 与上游梯度 `dO`，完全可以在进入 `K/V` loop 之前，单独、一次性的计算出 `delta`。

```python
delta_i = sum_j( dP_ij * P_ij )
    	= sum_j( sum_k(dO_ik * V_jk) * P_ij )    # dP = dO @ V^T
    	= sum_k( dO_ik * sum_j(V_jk * P_ij) )    # 交换求和顺序
    	= sum_k( dO_ik * sum_j(P_ij * V_jk) )    # 乘法交换律
   	 	= sum_k( dO_ik * O_ik )                  # O = P @ V
```

因此：

```python
delta = row_sum(dO ⊙ O)
```

### E. dQ 和 dK 的推导

#### E.1 分解计算

```python
S = Q @ K^T / sqrt(D)
```

这可以分解为两步：

```python
S_temp = Q @ K^T
S = S_temp / sqrt(D)
```

#### E.2 dQ 的推导

**步骤 1**：计算 `dS_temp`

```python
S = S_temp / sqrt(D)
```

根据缩放的梯度公式：

```python
dS_temp = dS / sqrt(D)
```

**步骤 2**：计算 `dQ`

```python
S_temp = Q @ K^T
```

根据矩阵乘法的梯度公式：

```python
dQ = dS_temp @ K
   = (dS / sqrt(D)) @ K
   = dS @ K / sqrt(D)
```

#### E.3 dK 的推导

同理：

```python
dK = dS_temp^T @ Q
   = (dS / sqrt(D))^T @ Q
   = dS^T @ Q / sqrt(D)
```

### F. 完整公式总结

```python
# Forward
S = Q @ K^T / sqrt(D)
P = softmax(S)           # row-wise
O = P @ V

# Backward
# 步骤 1：计算 dV
dV = P^T @ dO

# 步骤 2：计算 delta
delta = row_sum(dO ⊙ O)

# 步骤 3：计算 dS (需要重建 P)
dP = dO @ V^T
dS = P ⊙ (dP - delta[:, None])

# 步骤 4：计算 dQ 和 dK
dQ = dS @ K / sqrt(D)
dK = dS^T @ Q / sqrt(D)
```

这就是标准 Attention 的完整梯度公式！希望这个详细的推导能帮助你理解 Attention 反向传播的数学原理！
