# CUDA

## 核函数
核函数是在 GPU 上执行的函数，使用 __global__ 关键字声明。调用时通过 <<<gridDim, blockDim>>> 语法指定并行规模

CUDA 有三种函数修饰符：

修饰符	执行位置	调用方	说明
__global__	GPU	CPU（或 GPU）	核函数，启动 GPU 并行执行
__device__	GPU	GPU	设备函数，只能被核函数或其他设备函数调用
__host__	CPU	CPU	普通 CPU 函数（默认，可省略）

## Grid / Block / Thread 三层线程层级

CUDA 将线程组织为三层结构，这是理解并行编程的关键。可以把它比喻为”学校 / 班级 / 学生”——Grid 是整个学校，Block 是一个班级，Thread 是班里的每个学生，每个学生独立做自己那份作业，但同一个班级的学生可以通过”黑板”（Shared Memory）互相交流。

## 存储层次总览

寄存器 (Registers)           ← 最快，每线程私有
  │  ~20 TB/s, ~0 cycle
  ▼
共享内存 (Shared Memory)     ← 可编程的片上缓存，Block 内共享
  │  ~20 TB/s, ~20-30 cycles
  ▼
L1 / L2 Cache               ← 自动管理的硬件缓存
  │  L2: ~12 TB/s, ~200 cycles
  ▼
全局内存 (Global Memory/HBM) ← "显存"，所有线程可访问
  │  ~3.35 TB/s (H100), ~400-600 cycles
  ▼
主机内存 (Host/CPU Memory)   ← 需要通过 PCIe 传输
     ~64 GB/s (PCIe 5.0)

SM = Streaming Multiprocessor，中文叫流式多处理器，是 NVIDIA GPU 最基础的并行计算单元，相当于 GPU 里的 “CPU 核心”，但专门跑并行线程。

Warp 是 NVIDIA SM 硬件调度、执行线程的最小单位，固定包含 32 个线程。
GPU 硬件不会单独调度 1 个线程运行，而是一次性打包 32 条线程组成一个 warp 同步执行。（SIMT（Single Instruction, Multiple Threads，单指令多线程））

共享内存被分为 32 个 Bank，每个 Bank 宽度为 4 字节。同一 Warp 内的不同线程如果访问同一 Bank 的不同地址，就会产生 Bank Conflict，访问变为串行。

Occupancy = 实际活跃 Warp 数 / SM 最大 Warp 数。先保证没有寄存器溢出，再考虑 Occupancy。

## Vecadd.cu

First cuda file to run.
Add two vectors.

```
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 核函数：向量加法
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20;  // 1M 元素
    size_t bytes = N * sizeof(float);

    // 分配主机内存并初始化
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Host → Device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 计算 Grid 和 Block 维度
    // Block大小指定为256
    // N为线程总数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动 Kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Device → Host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // 验证
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at index %d: %f\n", i, h_c[i]);
            break;
        }
    }
    printf("Vector addition completed successfully!\n");

    // 释放
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

## Reduce（并行归约）

Reduce 是将一组数据聚合为一个值（如求和、求最大值）的操作。它是理解 CUDA 并行思维的最佳入门。

```

// 树形归约：步长从 blockDim/2 开始缩小，保证低编号线程连续工作
__global__ void reduce_base(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存，都是在GPU上
    // 由于向上取整，我们可以去掉一些没有用到的线程
    if (gid < n) {
        // 下标合法，读取全局内存数据存入共享内存
        smem[tid] = input[gid];
    } else {
        // 下标越界，填充0，求和不影响结果
        smem[tid] = 0.0f;
    }
    __syncthreads();

    // 步长从大到小：保证同一 Warp 内线程要么全部工作，要么全部空闲
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 每个 Block 的结果写回全局内存
    if (tid == 0) output[blockIdx.x] = smem[0];
}
```
优化 1：展开最后一个 Warp，Warp 内线程天然 SIMT 锁步执行，不需要 __syncthreads()。直接展开这几轮循环可以省去多余的同步屏障开销。

优化 2：每线程处理多个元素，在加载的时候就相加两次

## GEMM

同一个block内部的线程都是并行执行，block之间的可能会有先后执行，但是从逻辑上都是并行
分为逻辑世界和真实执行世界

```
// C = A * B，A: MxK, B: KxN, C: MxN
__global__ void gemmNaive(float* A, float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 不同的row,col作为并行的基本线程单元
    // 每个过程都需要从全局内存中去取A，B单元的值
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

优化 1：Shared Memory Tiling（分块）
```

#define TILE_SIZE 32

__global__ void gemmTiled(float* A, float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // 沿 K 维度分块迭代
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载 Tile 到共享内存
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // 加载A的tile到共享内存As
        // 从全局内存到共享内存
        float a_val = 0.0f;
        if (row < M && aCol < K) {
            a_val = A[row * K + aCol];
        }
        As[threadIdx.y][threadIdx.x] = a_val;

        // 加载B的tile到共享内存Bs
        float b_val = 0.0f;
        if (bRow < K && col < N) {
            b_val = B[bRow * N + col];
        }
        Bs[threadIdx.y][threadIdx.x] = b_val;

        //同一个block内所有的线程同步
        __syncthreads();

        // 使用共享内存计算（无全局内存访问）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Softmax

### 朴素实现（两趟）

```
__global__ void softmaxNaive(float* input, float* output, int N) {
    // 假设一个 Block 处理一行
    __shared__ float smem[256];
    int tid = threadIdx.x;

    // 1. 求 max（Reduce 操作）
    // we only get one block
    float maxVal = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        maxVal = fmaxf(maxVal, input[i]);
    }
    smem[tid] = maxVal;
    __syncthreads();
    
    // Typical tree reduce for max in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads(); // Sync after each layer of the tree
    }
    float globalMax = smem[0];
    __syncthreads();



    // 2. 求 exp 之和（Reduce 操作）
    float sumExp = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sumExp += expf(input[i] - globalMax);
    }
    smem[tid] = sumExp;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] = smem[tid] + smem[tid + s];
        }
        __syncthreads();
    }

    float globalSum = smem[0];
    __syncthreads();

    // 3. 计算 softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - globalMax) / globalSum;
    }
}

```

### Online Softmax
NVIDIA 提出的 Online Normalizer Calculation 方法，可以在一趟遍历中同时维护 max 和 sum，减少一次全局内存读取。

```
// 核心思想：在线更新 max 和 sum
float m = -FLT_MAX;  // 当前 max
float d = 0.0f;       // 当前 sum(exp(x - m))

for (int i = tid; i < N; i += blockDim.x) {
    float x = input[i];
    float m_new = fmaxf(m, x);
    // 关键：旧的 sum 需要用校正因子调整
    d = d * expf(m - m_new) + expf(x - m_new);
    m = m_new;
}
// 最终：softmax(x_i) = exp(x_i - m) / d

```

### 算子融合

```
未融合：
kernel1: A = input + bias     → 写回 HBM
kernel2: B = ReLU(A)          → 读 A 从 HBM，写 B 回 HBM
kernel3: output = LayerNorm(B) → 读 B 从 HBM

融合后：
fused_kernel: output = LayerNorm(ReLU(input + bias))
  → 只读一次 input，中间结果在寄存器/共享内存中流转
```

## Attention 算子

标准 Attention 的计算公式,Attention logits matrix = QK⊤

The size is 512GB.

这远超单卡 80GB 显存。即使显存够用，反复在 HBM 和 SRAM 之间搬运这个巨大矩阵也会严重拖慢速度。

### FlashAttention

FlashAttention 的核心思想：通过 Tiling（分块）避免在 HBM 中存储完整的 QK^T 矩阵，将所有中间计算保持在 Shared Memory 中。


FlashAttention：
  对 Q 分块 → 每块与 K, V 的所有块做 Attention → 用 Online Softmax 拼接结果
  ↑ 中间矩阵只在 SRAM 中，不写回 HBM

Tiling：将 Q、K、V 分成小块，每块能装进 Shared Memory
Online Softmax：在分块计算中正确维护 softmax 的全局 max 和 sum
重计算（Recomputation）：反向传播时不存储中间 Attention 矩阵，而是重新计算（用计算换显存）

## Triton

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

## 性能分析工具

Nsight Compute（Kernel 级分析）
Nsight Systems（系统级分析）
编译器输出
