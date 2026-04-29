# Triton Kernel 编写流程速记

这份文档面向这个作业目录里的 Triton 代码，尤其是 [cs336_systems/FlashAttention.py](../cs336_systems/FlashAttention.py)。目标不是把 Triton API 全讲完，而是总结一个“从想法到能跑、到能测、到能调优”的常用流程。

## 一句话概括

写 Triton kernel 通常不是先上来就写 `tl.load` / `tl.store`，而是按下面顺序走：

1. 先写清楚 PyTorch 参考实现，明确输入输出、shape、stride、数值稳定性。
2. 决定一个 Triton program instance 负责哪一块数据。
3. 把这块数据的地址算出来，用 `tl.load` / `tl.store` 加 mask 读写。
4. 先把正确性跑通，再做 tile、`num_warps`、`num_stages`、`autotune` 等调优。

## 推荐开发顺序

下面这个顺序适合大多数 Triton 算子，尤其适合第一次把一个 PyTorch 版本翻译成 Triton 版本。

### 1. 先写 reference implementation

先用 PyTorch 写一个慢但正确的版本，作用有两个：

- 它定义了“这个 kernel 到底该算什么”。
- 它是后面做 `torch.testing.assert_close(...)` 的基准。

在这个作业里，你已经有一个很好的起点：`flash_attention_torch.forward(...)`。

建议在开始写 Triton 版本前先明确下面这些事：

- 输入输出 shape 是什么。
- 支持不支持非 contiguous tensor；如果支持，就必须显式传 stride。
- 哪些中间量必须用 `float32` 累加。
- 边界条件是什么：尾块、不规则长度、causal mask、空块。

### 2. 想清楚并行映射

Triton 是 SPMD 模型。一个 kernel 启动后，会有很多 program instance 并行执行；每个 instance 负责一块数据。最关键的问题是：

- 一个 program 负责 1 个向量块，还是 1 个矩阵 tile？
- launch grid 是 1D、2D 还是 3D？
- `pid=tl.program_id(axis)` 如何映射到 batch/head/row/block？

常见思路：

- 向量算子：一个 program 负责一个一维 block。
- 矩阵乘：一个 program 负责一个 `BLOCK_M x BLOCK_N` 输出 tile。
- Attention：常见做法是一个 program 负责一个 `(batch, head, q_block)`，再在 `k_block` 上循环。

对 FlashAttention 来说，通常先把并行粒度定成：

- `axis 0`: query block
- `axis 1`: batch * head

这样心智负担最小，和在线 softmax 的写法也比较对齐。

### 3. 设计 kernel 接口

Triton kernel 一般长这样：

```python
@triton.jit
def kernel(
    x_ptr, y_ptr, out_ptr,
    stride_x0, stride_y0, stride_out0,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    ...
```

接口设计时，一般区分三类参数：

- 指针参数：输入/输出 tensor。
- 运行时参数：shape、stride、scale、是否 causal 等。
- 编译期元参数：tile 大小、`BLOCK_SIZE`、`HEAD_DIM` 等，用 `tl.constexpr` 标出来。

经验上：

- 只要可能遇到非连续布局，就把 stride 显式传进去。
- tile 大小、head dim 这类会影响代码生成的值，优先做成 `tl.constexpr`。
- wrapper 里做输入检查，kernel 里只做必要的边界保护。

### 4. 先写最小可运行 kernel

最小 kernel 的骨架通常就是下面四步：

```python
@triton.jit
def add_like_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)
```

这里最重要的是三件事：

- 用 `tl.program_id(...)` 找到“我是哪一个 program”。
- 用 `tl.arange(...)` 生成本 program 负责的局部索引。
- 用 `mask` 保护越界读写。

如果是二维/三维问题，就把 `offsets` 换成更复杂的 pointer arithmetic。矩阵乘和 attention 的难点基本都在这里。

### 5. 把地址计算写对

很多 Triton kernel 真正难的不是算子公式，而是地址计算。

二维 row-major tensor 的地址可以理解成：

```text
X[i, j] = X_ptr + i * stride_xi + j * stride_xj
```

所以写 kernel 时一般会经历这个过程：

1. 先算出 tile 对应的逻辑坐标。
2. 再把逻辑坐标转成实际地址。
3. 最后 `tl.load` / `tl.store`。

如果 kernel 是 block 化的，常见写法是：

- 用 `offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`
- 用 `offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)`
- 通过广播拼出二维地址

像 FlashAttention 这种 kernel，通常会显式传入 `stride_q*`, `stride_k*`, `stride_v*`, `stride_o*`，然后在 kernel 里拼出每个 tile 的指针。

### 6. 先保证数值正确，再考虑性能

第一版只追求正确：

- 不规则长度必须加 mask。
- softmax / attention 累加建议用 `float32`。
- 和 PyTorch reference 做 close check。
- 测试规则形状和非规则形状。

建议的验证顺序：

1. 小 shape，方便肉眼检查。
2. 非整除 tile 的 shape，专门测尾块。
3. 随机输入，多轮 `assert_close`。
4. 如果有 backward，再测梯度或直接对比 reference backward。

在这个作业里，最直接的验证入口就是 [tests/test_attention.py](../tests/test_attention.py)。

### 7. 再做 wrapper

实际可用的 Triton kernel，通常都要有一个 Python wrapper，负责：

- 检查输入 shape / dtype / device。
- 分配输出 tensor。
- 定义 launch grid。
- 传入 stride、shape、meta-parameters。

典型模式：

```python
def op(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
```

对于 FlashAttention，这一层通常还要负责：

- 把 3D 输入升成 4D 或把 4D 输入按 `(B, H, S, D)` 解释清楚。
- 分配 `O`、`L` 这类输出/中间缓存。
- 按需保存 backward 需要的中间量到 `ctx`。

### 8. 性能调优通常调什么

第一版 kernel 跑通以后，再看性能。常见杠杆有这些：

- tile 大小：比如 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`。
- `num_warps`：影响一个 program 用多少 warps 协作。
- `num_stages`：影响软件流水。
- 数据复用：尽量让加载的数据在寄存器/SRAM 里多用几次。
- kernel fusion：把本来要多次读写 HBM 的步骤合并掉。
- `triton.autotune`：让不同配置按 shape 自动选最优。

如果问题是 matmul 类/attention 类，官方教程里很典型的优化点包括：

- block 化计算
- 多维 pointer arithmetic
- program re-ordering
- 自动调参

如果你已经知道不同 shape 下最优 tile 不一样，就不要硬写一个常数，直接考虑 `triton.autotune(...)` + `triton.Config(...)`。

### 9. 调试顺序

Triton kernel 出错时，建议按这个顺序排查：

1. 先缩小 shape，让每个 program 处理的块很小。
2. 检查 pointer arithmetic 和 mask。
3. 用 Triton 自带 debug 工具看中间值。
4. 用 interpreter 在 CPU 上单步。
5. 正确性稳定后再 benchmark。

常用工具：

- 编译期调试：`tl.static_print`、`tl.static_assert`
- 运行时调试：`tl.device_print`、`tl.device_assert`
- interpreter：设置 `TRITON_INTERPRET=1`
- NVIDIA 内存/竞争检查：`compute-sanitizer`

一个很实用的原则是：先把“地址有没有算错”查清，再查“公式有没有写错”。

### 10. 写 FlashAttention kernel 时的落地建议

结合这个仓库里的 [cs336_systems/FlashAttention.py](../cs336_systems/FlashAttention.py)，推荐的实现顺序是：

1. 先只做 forward。
2. 先不做最复杂的优化，先让一个 program 处理一个 `q_block`。
3. 在 `k_block` 循环中维护 online softmax 需要的 `m_i`、`l_i`、`acc`。
4. 中间累加用 `float32`。
5. 先支持最常见的 `(B, H, S, D)` contiguous 输入。
6. 跑通 `tests/test_attention.py` 里的 forward 测试。
7. 再补 backward，最后再谈 autotune 或更激进的 tiling。

如果要把当前文件里的骨架补全，通常会先决定这些元参数：

- `Q_TILE_SIZE`
- `K_TILE_SIZE`
- `D`
- `num_warps`
- `num_stages`

然后在 kernel 中做这些事：

1. 根据 `pid` 定位当前 `(batch, head, q_tile)`。
2. 载入 `Q` tile。
3. 循环载入 `K/V` tile。
4. 更新分块 softmax 的 `m_i`、`l_i`、`acc`。
5. 把最终 `O` 和 log-sum-exp 之类的辅助量写回去。

## 常见坑

- 没有给尾块加 mask，导致越界读写。
- 默认假设 contiguous，但 wrapper 没有传 stride。
- softmax/attention 累加直接用 fp16，数值容易炸。
- 只测了整齐 shape，没有测非整除 tile。
- benchmark 前没有先验证正确性。
- autotune 过早引入，结果 bug 和性能问题混在一起，难排查。

## 一个实用模板

可以把写 kernel 的日常流程压缩成这个 checklist：

1. 写 PyTorch reference。
2. 确定一个 program 负责的数据块。
3. 设计 kernel 参数：ptr / stride / shape / constexpr。
4. 写最小可运行 kernel。
5. 用小输入和不规则输入测正确性。
6. 包一层 Python wrapper。
7. 加 benchmark。
8. 再做 autotune 和更激进优化。

## 推荐阅读顺序

Triton 官方教程页明确建议按顺序看教程。对这份作业最有帮助的顺序通常是：

1. Vector Addition：先熟悉 `@triton.jit`、`tl.program_id`、`tl.arange`、`mask`
2. Fused Softmax：看 reduction、mask、数值稳定 softmax
3. Matrix Multiplication：看 block 化、二维指针、autotune
4. Layer Normalization：看 backward kernel 的组织方式
5. Fused Attention：看和 FlashAttention 最接近的完整例子

## 参考资料

- Triton 教程总览: <https://triton-lang.org/main/getting-started/tutorials/>
- Vector Addition: <https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html>
- Fused Softmax: <https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
- Matrix Multiplication: <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
- Layer Normalization: <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>
- Fused Attention: <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- Triton Python API 总览: <https://triton-lang.org/main/python-api/triton.html>
- `triton.language` API: <https://triton-lang.org/main/python-api/triton.language.html>
- Debugging Triton: <https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>
