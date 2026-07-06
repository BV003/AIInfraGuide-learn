# Triton

Triton 是 OpenAI 开源的 GPU 编程语言，使用 Python 语法编写 GPU kernel，编译器自动处理内存合并、共享内存管理、Warp 调度等底层细节。

Triton 的优势：
Python 语法，学习曲线平缓
编译器自动处理内存合并、共享内存 Tiling、Warp 调度
性能可以达到手写 CUDA 的 80-95%
FlashAttention 的原始实现就使用了 Triton

```
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算当前 Block 处理的元素范围
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载、计算、存储
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# 调用
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

# 使用
x = torch.randn(1000000, device='cuda')
y = torch.randn(1000000, device='cuda')
result = add(x, y)

```