# 从零实现 FlashAttention（Phase 0）：Attention 的计算模型与性能瓶颈

目标：建立统一的 mental model，理解 Attention 的计算瓶颈和 FlashAttention 的解决思路。这一阶段完全不涉及代码实现。

------

## 0.1 Attention 的“真实计算图”

### 0.1.1 标准 Attention 的计算流程

标准 Attention 的计算可以分解为三个清晰的步骤：

```python
step 1: S = Q @ K.transpose(-2, -1) / sqrt(d) # Score matrix, shape: [B, H, N, N]
step 2: P = softmax(S) # Attention probabilities/weights, shape: [B, H, N, N]
step 3: O = P @ V      # Output, shape: [B, H, N, d]
```

其中：

- `Q, K, V` 的 shape 都是 `[B, H, N, d]`（B 是batch size，H 是注意力头数，N 是序列长度，d 是 head dimension）
- `S` 和 `P` 的 shape 都是 `[B, H, N, N]`

>标准 Softmax 算法为了计算某一行的概率值，必须先知道这一行**所有的** $S_{i,j}$，这种全局依赖迫使我们必须把整个 $S$ 存下来。

### 0.1.2 关键认知1：中间矩阵 S 和 P 是内存杀手

让我们算一笔账：

- 假设序列长度 N = 2048（常见的上下文长度）
- Batch size = 32，8 个 attention heads
- Score matrix S 的大小：`32 × 8 × 2048 × 2048 = 1GB` 个元素
- 如果用 FP16 存储：`1GB × 2 bytes = 2GB`

当序列长度增加到 8192 时，这个数字会变成 **32GB**，这还只是**单层 Attention Layer** 的中间矩阵 S 的存储，还不考虑反向传播，实际显存占用只会更高！这也就解释了为什么早期的 Transformer 架构模型难以处理长序列任务——内存压力太大！

### 0.1.3 关键认知2：**硬件瓶颈（Memory-Bound）**

GPU 的计算速度（FLOPS）极快，但从显存（HBM）读取数据的速度相对极慢。标准 Attention 频繁地将 $N^2$ 级的大中间矩阵在显存和芯片之间搬来搬去，时间全浪费在**搬运**上了。从标准 Attention 的计算流程可以发现，**核心问题**在于中间矩阵 S 和 P 太大，无法放在 GPU 的**片上 SRAM**（相比于 HBM，SRAM 的访存速度极快，但容量很小）中，所以必须频繁地从 HBM（慢）中搬运数据，内存访问就成为了瓶颈。这太“憋屈”了，空有一身武功（算力）却发挥不出！

> 这里的 SRAM 指的是GPU SM 内部的寄存器文件、shared memory、L1 cache 等低延迟存储的统称

要解决这个问题，大家或许会想到，既然整个 S 和 P 矩阵放不进 SRAM，那能不能使用 GPU 编程中常用的 **tiling** 思想，将矩阵分块后，加载到 SRAM 中呢？决定这个思路是否可行的关键就在下一节。

```python
标准 Attention 的计算流程：
GPU SRAM (fast)  ←→  HBM (slow)
    
    读取 Q, K        [从 HBM 读取]
       ↓
    计算 S           [写回 HBM] ← 第一次写入
       ↓
    读取 S           [从 HBM 读取]
       ↓
    计算 P           [写回 HBM] ← 第二次写入
       ↓
    读取 P, V        [从 HBM 读取]
       ↓
    计算 O           [写回 HBM]
```



------

## 0.2 FlashAttention 的核心思想

### 0.2.1 一句话总结

> **Softmax 可以在线（online）计算，不需要看见整个 score 矩阵，从而消除对大中间矩阵的依赖，而且这是数值上完全等价的计算方式。**

### 0.2.2 什么是"在线计算"？

**传统方式**（需要看到所有数据）：

```python
计算 softmax([3, 1, 2]):
1. 先扫描一遍找最大值: max = 3
2. 计算分母: sum = exp(3-3) + exp(1-3) + exp(2-3) = 1 + 0.135 + 0.368 = 1.503
3. 计算每个位置的权重: weights = [exp(3-3)/1.503, exp(1-3)/1.503, exp(2-3)/1.503] = [0.665, 0.090, 0.245]
4. 用权重加权求和: output = weights · values
```

**在线方式**（数据流式到达）：

```python
数据流: 3 → 1 → 2
对应的 values: v_1 → v_2 → v_3
 ↓
初始化 m_old = -∞, l_old = 0, o_old = 0向量 (o 始终维护的是"未归一化"的加权和，最后一步才除以 l 完成归一化)
 ↓
看到 3:  
	计算：m_new = max(m_old, 3) = 3, 
		 l_new = exp(m_old - m_new) × l_old + exp(3 - 3) = 1, 
		 o_new = exp(m_old - m_new) × o_old + exp(3 - 3) × V_1 = V_1, (当前的临时输出)
	更新：m_old = m_new = 3, l_old = l_new = 1, o_old = o_new = V_1
 ↓		
看到 1:  
	计算：m_new = max(m_old, 1) = 3, 
		 l_new = exp(m_old - m_new) × l_old + exp(1 - 3) = 1.135, 
		 o_new = exp(m_old - m_new) × o_old + exp(1 - 3) × V_2 = V_1 + 0.135 × V_2, (修正并累加)
	更新：m_old = 3, l_old = 1.135, o_old = V_1 + 0.135 × V_2
 ↓		
看到 2:  
	计算：m_new = max(m_old, 2) = 3, 
		 l_new = exp(m_old - m_new) × l_old + exp(2 - 3) = 1.503, 
		 o_new = exp(m_old - m_new) × o_old + exp(2 - 3) × V_3 = V_1 + 0.135 × V_2 + 0.368 × V_3, (再次修正并累加)
	更新：m_old = 3, l_old = 1.503, o_old = V_1 + 0.135 × V_2 + 0.368 × V_3
 ↓
最终归一化: output = o_old / l_old = (V_1 + 0.135 × V_2 + 0.368 × V_3) / 1.503
                                 = 0.665 × V_1 + 0.090 × V_2 + 0.245 × V_3
 ↓ 
验证：与传统 softmax 结果完全一致
传统方法: 
  	权重 = softmax([3,1,2]) = [0.665, 0.090, 0.245]
  	输出 = 0.665 × V_1 + 0.090 × V_2 + 0.245 × V_3
 ↓  	
# 这几个式子暂时看不懂不要紧，可以先看下一节的数学推导
```

关键区别：**在线方式不需要存储所有数据，可以边读边算**，这与 **tiling** 思想天然契合。

### 0.2.3 Online Softmax 的数学推导

假设我们已经处理了一部分数据，维护了三个统计量（这三个量是 **per-row** 的）：

- `m_old`：目前见过的最大值（**局部最大值**）
- `l_old`：目前的归一化分母（在 `m_old` 基础上计算的**局部累加和**）
- `o_old`：目前累积的输出向量（**这是关键！**）

现在来了新的一批数据 `score_block` 和对应的 `V_block`，如何更新？

#### 步骤 1: 更新最大值

```
m_new = max(m_old, max(score_block))
```

这个很直观，新的最大值要么是旧的最大值，要么在新数据块中。

#### 步骤 2: 更新归一化分母

```
l_new = exp(m_old - m_new) × l_old + Σ exp(score_block - m_new)
```

#### 步骤 3: 更新输出（这一步容易被忽略！）

```
o_new = exp(m_old - m_new) × o_old + (exp(score_block - m_new) @ V_block)
```

#### **步骤 4: 最后一步**

所有块处理完后，`output = o / l` 得到归一化的最终输出

#### 为什么需要修正？

当最大值从 `m_old` 变成 `m_new` 时：

- 所有基于 `m_old` 计算的 `exp(score - m_old)` 都需要重新校准
- 修正系数就是 `exp(m_old - m_new)`

**推导过程**：

旧的分母 `l_old` 基于 `m_old` 计算：

```
l_old = Σ exp(old_scores - m_old)
```

当基准变成 m_new 时，需要修正：

```
Σ exp(old_scores - m_new) 
= Σ exp(old_scores - m_old + m_old - m_new))
= Σ [exp(old_scores - m_old) × exp(m_old - m_new)]
= exp(m_old - m_new) × Σ exp(old_scores - m_old)
= exp(m_old - m_new) × l_old
```

修正后的 `l_old` 再累加新数据就得到新的分母 `l_new`：

```
l_new = exp(m_old - m_new) × l_old + Σ exp(score_block - m_new)
```

输出的推导完全类似，只是多了与 V 的矩阵乘法。

### 0.2.4 为什么这解决了内存问题？

* 每一行的 softmax 独立计算
* 处理每一行时，可以将其分成小块（blocks）
* 对每个 block 计算局部结果，用 online 公式更新统计量
* **不需要存储整个中间矩阵**，只需要存储：
  * 当前的 m 和 l（每行两个标量）
  * 当前正在处理的小块数据

### 0.2.5 直观对比

| **维度**     | **Standard Attention**             | **FlashAttention**                                           |
| ------------ | ---------------------------------- | ------------------------------------------------------------ |
| **显存占用** | $O(N^2)$ —— 存储完整的 S 和 P 矩阵 | $O(N \cdot d)$ —— 只存统计量和小块数据                       |
| **访存模式** | 频繁读写 HBM                       | 绝大部分在 SRAM 完成                                         |
| **计算方式** | 先算完 S，再算完 P，再算 O         | 分块流式计算，边算边丢弃中间结果                             |
| **本质**     | 算法优先                           | **硬件感知 (Hardware-aware)**                                |
| **代价**     | 显存占用高，计算速度慢             | 前向传播中几乎不增加 FLOPs，<br />反向传播通过 **recomputation** 用额外计算换取显存节省 |

关于 **Recomputation**：为了省显存，FlashAttention 在反向传播时不读取前向传播的 P 矩阵（所以不用存储 P），而是根据已有的 $Q, K, V$ 重新算一遍。这听起来疯了，但在 GPU 上反而更快，这一点我们会在 Phase 4 中详细讲解。



------

## 0.3 小结

### 0.3.1 为什么标准 Attention 慢？

总结三个核心问题：

#### 问题 1: 内存占用过大

- `O(N²)` 的 S 和 P 矩阵
- 长序列下直接 OOM（Out of Memory）

#### 问题 2: 内存访存延迟

- GPU 计算速度 >> 内存读写速度
- 频繁从 HBM 搬运数据
- 大部分时间在等待数据，而不是计算

#### 问题 3: 无法利用计算和访存的重叠

- 必须等 S 计算完成，完全写入 HBM 才能读取计算 P
- 必须等 P 计算完成，完全写入 HBM 才能读取计算 O
- 三个阶段**串行执行**，无法有效 pipeline

### 0.3.2 FlashAttention 如何解决？

#### 解决方案 1: 分块（Tiling）

- 将 Q, K, V 分成小块
- 每次只在 SRAM 中处理一个块
- 用 online 算法融合多个块的结果

#### 解决方案 2: Kernel Fusion

- 不存储整个 S 和 P，在 SRAM 中完成所有计算，减少 HBM 访问次数
- 计算流程：读取块 → 计算 S → 计算 P → 更新 O → 丢弃中间结果

#### 解决方案 3: Recomputation

- 前向传播不存 S 和 P
- 反向传播时重新计算需要的值
- Trade-off：用额外的计算换内存节省和减少访存消耗

------

## 注意事项

在进入后续阶段之前，请确保你理解了：

✅ 标准 Attention 的计算流程

✅ 标准 Attention 的瓶颈是 **内存占用**，**访存延迟 **和 **串行执行**

✅ **Online Softmax** 的几个核心公式及其意义

✅ FlashAttention 的本质：**用分块和在线算法避免存储大矩阵**

后续阶段（Phase 1~5），我们将用 Triton 逐步实现这些想法。

> 如果你现在能在脑中想象出一个按 block 扫描 K/V，并维护 (m, l, o)的循环，那么你已经理解了 FlashAttention 的本质👍。