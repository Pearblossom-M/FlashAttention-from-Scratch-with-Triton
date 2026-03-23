"""
使用 triton.language.make_tensor_descriptor 而非 TensorDescriptor
"""
import triton
import triton.language as tl

# ==============================================================================
# Forward Pass
# ==============================================================================
@triton.autotune(
    configs = [       
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    ],
    key=['S_q', 'S_k', 'D', 'is_causal'],  # 根据这些参数选择最优配置
)

@triton.jit
def flash_attention_forward_kernel(
    # -------------------- pointer --------------------
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, 

    # -------------------- stride --------------------
    stride_lb, stride_lh, stride_ls, # LSE 在三个维度上的 stride

    # -------------------- 缩放因子 --------------------
    scale, # 1 / sqrt(D)

    # -------------------- 维度信息 --------------------
    # Q: [B, H, S_q, D],  K/V: [B, H, S_k, D],  O: [B, H, S_q, D]
    B: tl.constexpr,       # batch size
    H: tl.constexpr,       # 注意力头数
    S_q: tl.constexpr,     # 序列长度 (Q 和 O 的行数)
    S_k: tl.constexpr,     # 序列长度 (K/V 的行数)
    D: tl.constexpr,       # head_dim

    # -------------------- 配置参数 ---------------------
    BLOCK_M: tl.constexpr, # Q_block 的行数
    BLOCK_N: tl.constexpr, # 流式扫描 K/V 的列块大小 (沿 N 维)
    
    # -------------------- Flag参数 --------------------
    is_causal: tl.constexpr = False,
):
    """
    FlashAttention forward kernel
    """
    pid_0 = tl.program_id(0) 
    pid_1 = tl.program_id(1)
    batch_idx = pid_1 // H
    head_idx = pid_1 % H

    Sq_offs = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M,]

    # mask：处理最后一个 Q_block, 因为可能越界, 用于 score 
    # mask_Sq = Sq_offs < S_q
    
    # 创建 TensorDescriptor
    desc_q = tl.make_tensor_descriptor(q_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    desc_k = tl.make_tensor_descriptor(k_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    desc_v = tl.make_tensor_descriptor(v_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    desc_o = tl.make_tensor_descriptor(o_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    # tl.make_tensor_descriptor 目前仅支持 2-5 维张量, 如下写法错误
    # desc_lse = tl.make_tensor_descriptor(lse_ptr, shape=[B*H*S_q], strides=[1], block_shape=[BLOCK_M])

    # load Q block
    qo_offsets_y = batch_idx * H * S_q + head_idx * S_q + pid_0 * BLOCK_M
    Q_block = desc_q.load([qo_offsets_y, 0])

    # 初始化统计量
    m = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    o = tl.zeros([BLOCK_M, D], tl.float32)

    LOG2_E = 1.44269504 # log2(e), 用于tl.exp 到 tl.exp2 的转化

    # 逐块处理 K/V block
    loop_end = (pid_0 + 1) * BLOCK_M if is_causal else S_k # causal 模式可以提前截断循环
    for start_s in range(0, loop_end, BLOCK_N):
        # load K/V block
        kv_offsets_y = batch_idx * H * S_k + head_idx * S_k + start_s
        K_block = desc_k.load([kv_offsets_y, 0]) # [BLOCK_N, D]
        V_block = desc_v.load([kv_offsets_y, 0]) # [BLOCK_N, D]

        Sk_offs = start_s + tl.arange(0, BLOCK_N)
        mask_Sk = Sk_offs < S_k
    
        # 计算 score, [BLOCK_M, D] @ [D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        S_block = tl.dot(Q_block, tl.trans(K_block)) * scale  # [BLOCK_M, BLOCK_N]
        S_block = tl.where(mask_Sk[None, :], S_block, float('-inf')) # [BLOCK_M, BLOCK_N]

        # 当前 block 全部能被看见的判定条件：Q block的最小索引 >= K block的最大索引
        # 需要 causal mask 的判定条件：        
        if is_causal:
            q_idx_min = pid_0 * BLOCK_M
            k_idx_max = start_s + BLOCK_N - 1
            if not (q_idx_min >= k_idx_max):
                causal_mask = Sq_offs[:, None] >= Sk_offs[None, :] # [BLOCK_M, BLOCK_N]
                S_block = tl.where(causal_mask, S_block, float('-inf')) # [BLOCK_M, BLOCK_N]

        # 更新统计量
        m_new = tl.maximum(m, tl.max(S_block, axis=1)) # [BLOCK_M,]

        correction = tl.exp2((m - m_new) * LOG2_E) # [BLOCK_M,]
        numerator = tl.exp2((S_block - m_new[:, None]) * LOG2_E) # [BLOCK_M, BLOCK_N]

        l = l * correction + tl.sum(numerator, axis=1) # [BLOCK_M,]
        # Tensor Core 的矩阵乘法要求输入为半精度(或 TF32)
        # 因此这里将 numerator 转回 fp16, 使 tl.dot 有机会使用 Tensor Core
        # 而累加仍然在 fp32 的 o 中完成, 以保证数值稳定性
        o = o * correction[:, None] + tl.dot(numerator.to(tl.float16), V_block) # [BLOCK_M, D]

        m = m_new
    
    # 最终归一化
    o_final = o / l[:, None] 

    # write back to O_ptr
    desc_o.store([qo_offsets_y, 0], o_final) # [BLOCK_M, D]

    # 计算 LogSumExp
    lse = m + tl.log(l) # [BLOCK_M,]
    
    # write back to lse_ptr
    l_ptrs = (
        lse_ptr
        + batch_idx * stride_lb
        + head_idx * stride_lh 
        + Sq_offs * stride_ls
    )
    # mask：处理最后一个 LSE_block, 因为可能越界
    mask_Sq = Sq_offs < S_q
    tl.store(l_ptrs, lse, mask=mask_Sq)

# ==============================================================================
# Backward Pass
# ==============================================================================
@triton.autotune(
    configs = [       
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    ],
    key=['S_q', 'S_k', 'D', 'is_causal'],  # 根据这些参数选择最优配置
)

@triton.jit
def flash_attention_dQ_kernel(
    # -------------------- pointer --------------------
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, lse_ptr, dq_ptr, delta_ptr,

    # -------------------- stride --------------------
    stride_lb, stride_lh, stride_ls, # LSE 在三个维度上的 stride
    stride_delta_b, stride_delta_h, stride_delta_s, # delta 在三个维度上的 stride

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

    # 计算 Q_block 的起始位置
    q_block_start = pid_0 * BLOCK_M

    # 行索引
    Sq_offs = q_block_start + tl.arange(0, BLOCK_M)

    # mask：处理最后一个 Q_block
    mask_Sq = Sq_offs < S_q

    # 创建 TensorDescriptor
    desc_q = tl.make_tensor_descriptor(q_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    desc_k = tl.make_tensor_descriptor(k_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    desc_v = tl.make_tensor_descriptor(v_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    desc_do = tl.make_tensor_descriptor(do_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    desc_o = tl.make_tensor_descriptor(o_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    # desc_lse = tl.make_tensor_descriptor(lse_ptr, shape=[B*H*S_q], strides=[1], block_shape=[BLOCK_M])
    desc_dq = tl.make_tensor_descriptor(dq_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    # desc_delta = tl.make_tensor_descriptor(delta_ptr, shape=[B*H*S_q], strides=[1], block_shape=[BLOCK_M])
   
    # load Q_block, dO_block, O_block, LSE_block
    qo_offsets_y = batch_idx * H * S_q + head_idx * S_q + q_block_start
    Q_block = desc_q.load([qo_offsets_y, 0])   # [BLOCK_M, D]
    dO_block = desc_do.load([qo_offsets_y, 0]) # [BLOCK_M, D]
    O_block = desc_o.load([qo_offsets_y, 0])   # [BLOCK_M, D]
    # LSE_block = desc_lse.load([qo_offsets_y])  # [BLOCK_M,]
    lse_ptrs = lse_ptr + batch_idx * stride_lb + head_idx * stride_lh + Sq_offs * stride_ls
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
        kv_offsets_y = batch_idx * H * S_k + head_idx * S_k + start_s
        K_block = desc_k.load([kv_offsets_y, 0]) # [BLOCK_N, D]
        V_block = desc_v.load([kv_offsets_y, 0]) # [BLOCK_N, D]
        
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
    desc_dq.store([qo_offsets_y, 0], dQ_acc.to(tl.float16))
    # save delta
    # desc_delta.store([qo_offsets_y], delta_block)
    delta_ptrs = (
        delta_ptr
        + batch_idx * stride_delta_b
        + head_idx * stride_delta_h 
        + Sq_offs * stride_delta_s
    )
    tl.store(delta_ptrs, delta_block, mask=mask_Sq)


@triton.autotune(
    configs = [       
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    ],
    key=['S_q', 'S_k', 'D', 'is_causal'],  # 根据这些参数选择最优配置
)

@triton.jit
def flash_attention_dKV_kernel(
    # -------------------- pointer --------------------
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, dk_ptr, dv_ptr, delta_ptr,

    # -------------------- stride --------------------
    stride_lb, stride_lh, stride_ls, # LSE 在三个维度上的 stride
    stride_delta_b, stride_delta_h, stride_delta_s, # delta 在三个维度上的 stride

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
    
    # 计算 K/V_block 的起始位置
    kv_block_start = pid_0 * BLOCK_N

    # 行索引
    Sk_offs = kv_block_start + tl.arange(0, BLOCK_N)

    # mask：处理最后一个 K/V_block
    mask_Sk = Sk_offs < S_k

    # 创建 TensorDescriptor
    desc_q = tl.make_tensor_descriptor(q_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    desc_k = tl.make_tensor_descriptor(k_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    desc_v = tl.make_tensor_descriptor(v_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    desc_do = tl.make_tensor_descriptor(do_ptr, shape=[B*H*S_q, D], strides=[D, 1], block_shape=[BLOCK_M, D])
    # desc_lse = tl.make_tensor_descriptor(lse_ptr, shape=[B*H*S_q], strides=[1], block_shape=[BLOCK_M])
    desc_dk = tl.make_tensor_descriptor(dk_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    desc_dv = tl.make_tensor_descriptor(dv_ptr, shape=[B*H*S_k, D], strides=[D, 1], block_shape=[BLOCK_N, D])
    # desc_delta = tl.make_tensor_descriptor(delta_ptr, shape=[B*H*S_q], strides=[1], block_shape=[BLOCK_M])

    # load K/V_block
    kv_offsets_y = batch_idx * H * S_k + head_idx * S_k + kv_block_start
    K_block = desc_k.load([kv_offsets_y, 0]) # [BLOCK_N, D]
    V_block = desc_v.load([kv_offsets_y, 0]) # [BLOCK_N, D]
    
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
        qo_offsets_y = batch_idx * H * S_q + head_idx * S_q + start_s
        Q_block = desc_q.load([qo_offsets_y, 0]) # [BLOCK_M, D]        
        dO_block = desc_do.load([qo_offsets_y, 0]) # [BLOCK_M, D]           
        # LSE_block = desc_lse.load([qo_offsets_y]) # [BLOCK_M,]
        lse_ptrs = lse_ptr + batch_idx * stride_lb + head_idx * stride_lh + Sq_offs * stride_ls
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
        
        # 6. 从 dQ kernel 保存的 delta 直接加载
        # delta_block = desc_delta.load([qo_offsets_y])  # [BLOCK_M,]
        delta_ptrs = delta_ptr + batch_idx * stride_delta_b + head_idx * stride_delta_h + Sq_offs * stride_delta_s
        delta_block = tl.load(delta_ptrs, mask=mask_Sq, other=0.0) # [BLOCK_M,]
    
        # 7. 计算 dS_block = P_block ⊙ (dP_block - delta_block)
        dS_block = P_block * (dP_block - delta_block[:, None])  # [BLOCK_M, BLOCK_N]
        
        # 8. 累加 dK = dS^T @ Q * scale
        dK_acc += tl.dot(tl.trans(dS_block).to(tl.float16), Q_block) * scale  # [BLOCK_N, D]
    
    # write back to dK/dV_block
    desc_dk.store([kv_offsets_y, 0], dK_acc.to(tl.float16))
    desc_dv.store([kv_offsets_y, 0], dV_acc.to(tl.float16))