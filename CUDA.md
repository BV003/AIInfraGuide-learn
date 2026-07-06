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