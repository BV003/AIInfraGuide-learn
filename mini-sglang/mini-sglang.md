# mini-sglang

## Project Architecture Overview

```
mini-sglang/
├── README.md
├── LICENSE
├── pyproject.toml
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
│
├── assets/
│   └── logo.png
│
├── docs/
│   ├── features.md              # Feature descriptions & CLI args
│   └── structures.md            # System architecture & code organization
│
├── benchmark/
│   ├── offline/
│   │   ├── bench.py
│   │   └── bench_wildchat.py
│   └── online/
│       ├── bench_qwen.py
│       └── bench_simple.py
│
├── tests/
│   ├── core/
│   │   ├── test_cache_allocate.py
│   │   └── test_scheduler.py
│   ├── kernel/
│   │   ├── test_comm.py
│   │   ├── test_index.py
│   │   ├── test_store.py
│   │   └── test_tensor.py
│   └── misc/
│       └── test_serialize.py
│
└── python/minisgl/              # ← ALL source code (~5000 lines)
    │
    ├── __main__.py              # Entry point: `python -m minisgl ...`
    ├── core.py                  # Req, Batch, Context, SamplingParams
    ├── env.py                   # Environment variable config
    │
    ├── server/                  # ── API & process launcher ──
    │   ├── args.py              # CLI argument parsing (ServerArgs)
    │   ├── launch.py            # Spawns all subprocesses
    │   └── api_server.py        # FastAPI server + interactive shell
    │
    ├── message/                 # ── ZMQ message protocol ──
    │   ├── backend.py           # Scheduler ↔ Tokenizer/Detokenizer msgs
    │   ├── frontend.py          # API Server ↔ Tokenizer msgs
    │   ├── tokenizer.py         # Tokenize/Detokenize message types
    │   └── utils.py             # ZMQ async queue wrappers
    │
    ├── tokenizer/               # ── Tokenization workers ──
    │   ├── server.py            # Main worker loop
    │   ├── tokenize.py          # Text → token IDs
    │   └── detokenize.py        # Token IDs → text
    │
    ├── scheduler/               # ── Core scheduling logic ──
    │   ├── config.py            # SchedulerConfig dataclass
    │   ├── scheduler.py         # Main Scheduler class + event loop
    │   ├── prefill.py           # PrefillManager (chunked prefill)
    │   ├── decode.py            # DecodeManager (batching decode reqs)
    │   ├── cache.py             # CacheManager (wraps KV cache)
    │   ├── table.py             # TableManager (page table slots)
    │   ├── io.py                # SchedulerIOMixin (ZMQ send/recv)
    │   └── utils.py
    │
    ├── engine/                  # ── GPU execution ──
    │   ├── config.py            # EngineConfig dataclass
    │   ├── engine.py            # Engine class (model load, forward, sample)
    │   ├── graph.py             # GraphRunner (CUDA Graph capture/replay)
    │   └── sample.py            # Sampler (top-k, top-p, temperature)
    │
    ├── models/                  # ── Model architectures ──
    │   ├── base.py              # BaseLLM (Transformer forward loop)
    │   ├── config.py            # ModelConfig dataclass
    │   ├── llama.py             # Llama 3
    │   ├── qwen2.py             # Qwen 2.5
    │   ├── qwen3.py             # Qwen 3 (dense)
    │   ├── qwen3_moe.py         # Qwen 3 (MoE)
    │   ├── mistral.py           # Mistral
    │   ├── weight.py            # HuggingFace weight loading & sharding
    │   ├── register.py          # Model registry
    │   └── utils.py
    │
    ├── layers/                  # ── Layer building blocks ──
    │   ├── base.py              # BaseLayer (TP-aware)
    │   ├── linear.py            # ColumnParallelLinear, RowParallelLinear
    │   ├── attention.py         # AttentionLayer
    │   ├── embedding.py         # VocabParallelEmbedding
    │   ├── norm.py              # RMSNorm
    │   ├── rotary.py            # RoPE (Rotary Position Embedding)
    │   ├── activation.py        # SiLU, etc.
    │   └── moe.py              # MoE layer (router + experts)
    │
    ├── attention/               # ── Attention kernel backends ──
    │   ├── base.py              # BaseAttnBackend interface
    │   ├── fa.py                # FlashAttention backend
    │   ├── fi.py                # FlashInfer backend
    │   ├── trtllm.py            # TensorRT-LLM backend
    │   └── utils.py
    │
    ├── kvcache/                 # ── KV cache management ──
    │   ├── base.py              # BaseKVCachePool, BaseCacheHandle
    │   ├── mha_pool.py          # MHAKVCache (physical memory pool)
    │   ├── naive_cache.py       # NaiveCacheManager (FIFO)
    │   └── radix_cache.py       # RadixCacheManager (prefix sharing)
    │
    ├── moe/                     # ── Mixture of Experts backends ──
    │   ├── base.py
    │   └── fused.py             # Fused MoE kernel
    │
    ├── distributed/             # ── Tensor Parallelism ──
    │   ├── info.py              # DistributedInfo dataclass
    │   └── impl.py              # all_reduce, all_gather wrappers
    │
    ├── kernel/                  # ── Custom CUDA kernels (TVM-FFI) ──
    │   ├── __main__.py
    │   ├── index.py             # Indexing kernel
    │   ├── store.py             # KV store kernel
    │   ├── tensor.py            # Tensor ops kernel
    │   ├── radix.py             # Radix tree kernel
    │   ├── moe_impl.py          # MoE kernel implementation
    │   ├── pynccl.py            # PyNCCL (custom NCCL wrapper)
    │   ├── utils.py
    │   └── triton/
    │       └── fused_moe.py     # Triton fused MoE kernel
    │
    ├── llm/                     # ── Python API ──
    │   └── llm.py               # LLM class (programmatic interface)
    │
    ├── utils/                   # ── Utilities ──
    │   ├── arch.py              # GPU architecture detection
    │   ├── hf.py                # HuggingFace model download
    │   ├── logger.py            # Logging setup (rank-aware)
    │   ├── misc.py              # div_even, etc.
    │   ├── mp.py                # Multiprocessing helpers
    │   ├── registry.py          # Generic registry pattern
    │   └── torch_utils.py       # torch_dtype, etc.
    │
    └── benchmark/               # ── Internal benchmark utils ──
        ├── client.py
        └── perf.py
```

Dependency order:
```
utils → distributed → kernel → layers → models → attention/kvcache/moe
                                                    ↓
server → message → tokenizer → scheduler → engine → (uses all above)
                                                    ↓
                                              llm (public API)
```

### Module Dependency Map

```
┌─────────────────────────────────────────────────┐
│                  APPLICATION LAYER              │
│                                                 │
│   server/     message/     tokenizer/    llm/   │
│   (launch,    (ZMQ msg     (text↔token   (LLM   │
│    API)        protocol)    workers)      API)  │
└───────────────┬─────────────────────────────────┘
                │ uses
                ▼
┌─────────────────────────────────────────────────┐
│                SCHEDULING LAYER                 │
│                                                 │
│   scheduler/                                    │
│   (prefill, decode, cache, table, io)           │
└───────────────┬─────────────────────────────────┘
                │ drives
                ▼
┌─────────────────────────────────────────────────┐
│                 EXECUTION LAYER                 │
│                                                 │
│   engine/        core.py                        │
│   (Engine,       (Req, Batch, Context)          │
│    GraphRunner,                                 │
│    Sampler)                                     │
└───────┬──────────────────┬──────────────────────┘
        │ uses             │ uses
        ▼                  ▼
┌───────────────┐  ┌──────────────────────────────┐
│  MODEL LAYER  │  │       INFRASTRUCTURE         │
│               │  │                              │
│  models/      │  │  kvcache/   attention/       │
│  (llama,      │  │  (radix,    (fa, fi, trtllm) │
│   qwen3,      │  │   naive)                     │
│   qwen3_moe)  │  │                              │
│       │       │  │  moe/       distributed/     │
│       ▼       │  │  (fused)    (tp all-reduce)  │
│  layers/      │  │                              │
│  (linear,     │  │  kernel/                     │
│   attention,  │  │  (custom CUDA, pynccl)       │
│   norm, rope, │  │                              │
│   embedding,  │  └──────────────────────────────┘
│   moe)        │              ▲
└───────────────┘              │ depends on
        │                      │
        └──────────┬───────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│                 UTILITY LAYER                   │
│                                                 │
│   utils/   env.py   distributed/                │
│   (logger,  (env      (info)                    │
│    hf,      vars)                               │
│    arch,                                        │
│    registry)                                    │
└─────────────────────────────────────────────────┘
```

## Core Data Structures

### SamplingParams — what the user wants 

Temperature, top-k, top-p, max_tokens, ignore_eos. Also has is_greedy the helper.


### 2. Req — one request's state

Lifecycle methods: complete_one() shifts the window forward by 1 token after each decode step.append_host() grows the CPU-side token array.

### 3. Batch — a group of Reqs processed together

Two phases: "prefill" or "decode". Fields (input_ids, positions, out_loc, attn_metadata) are set later by the scheduler and attention backend — not by the constructor. 

### 4. Context — global state
Singleton holding everything the model layers need during forward: page_table, kv_cache, attn_backend, moe_backend, and the current Batch. The forward_batch() context manager ensures only one batch is active at a time. 

## The Engine — Model Loading & Execution

Files: 
`engine/config.py`, 
`engine/engine.py`, 
`engine/graph.py`,
`engine/sample.py`

The Engine runs on each GPU (TP rank). One Engine per Scheduler process. It owns the model, KV cache, attention backend, and handles forward execution.

### Initialization Pipeline (Engine.__init__)

```
1. TP setup          set_tp_info(rank, size), torch.cuda.set_device
2. Communication      init_process_group (nccl or gloo+pynccl)
3. Model creation     create_model() on meta device → load HF weights → shard across TP
4. KV cache           allocate MHAKVCachePool (determined by available GPU memory)
5. Page table         tensor [max_running_req+1, max_seq_len] mapped to KV pages
6. Attention backend  FlashAttention / FlashInfer / TensorRT-LLM
7. MoE backend        fused MoE kernel (if model is MoE)
8. Sampler            top-k, top-p, temperature
9. CUDA Graph capture replay-able decode graphs for batch sizes [1,2,4,8,...,256]
```

### Forward & Sampling

```python
def forward_batch(self, batch, args) -> ForwardOutput:
    # 1. Try CUDA Graph replay (decode), or fall back to eager forward (prefill)
    if self.graph_runner.can_use_cuda_graph(batch):
        logits = self.graph_runner.replay(batch)      # fast path
    else:
        logits = self.model.forward()                   # eager path

    # 2. Mark each request as one token completed
    for req in batch.reqs:
        req.complete_one()

    # 3. Sample next token, async copy to CPU
    next_tokens_gpu = self.sampler.sample(logits, args)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
    copy_done = torch.cuda.Event(); copy_done.record()
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done)
```

### CUDA Graph (GraphRunner)

Why: each decode step has identical compute graph but different data. Without CUDA Graph, the CPU must launch dozens of tiny CUDA kernels per step → CPU overhead. CUDA Graph captures the whole decode forward pass into one replay-able unit.

针对解码这种「计算流程固定、仅数据变化」的场景，CUDA Graph 一次性录下整套 GPU 运算流程，后续循环直接回放，省去 CPU 反复启动内核带来的巨大性能损耗，提升推理速度、降低单 token 延迟。

- Captures graphs at staggered batch sizes: `[1, 2, 4, 8, 16, ..., max_bs]`
- `pad_batch()`: pads real requests with dummy requests to nearest captured batch size
- `replay()`: copies input data into pre-allocated buffers, then `graph.replay()`
- Only used for decode phase (prefill has variable-length inputs → can't graph)

### Sampler

- greedy (argmax): when all requests have temperature=0
- FlashInfer sampling: softmax → optional top-k → optional top-p → sample from probs
- All sampling runs on GPU

## The Scheduler — Request Lifecycle Management

Files: 
`scheduler/scheduler.py`, 
`scheduler/prefill.py`, 
`scheduler/decode.py`, 
`scheduler/cache.py`, 
`scheduler/table.py`, 
`scheduler/io.py`

The Scheduler is the brain. It runs on each GPU, manages the Engine beneath it, and decides what to run next.

### Architecture: Scheduler = 5 Managers + Event Loop

```
┌───────────────────────────────────────────┐
│              SCHEDULER                    │
│                                           │
│  ┌────────────────┐  ┌───────────────────┐│
│  │ PrefillManager │  │  DecodeManager    ││
│  │ (chunked       │  │  (running reqs)   ││
│  │  prefill)      │  │                   ││
│  └──────┬─────────┘  └────────────┬──────┘│
│         │ schedule_next_batch()   │       │
│         └──────────┬──────────────┘       │
│                    ▼                      │
│  ┌────────────────────────────────────┐   │
│  │  CacheManager                      │   │
│  │  (KV cache alloc/free,prefix match)│   │
│  └────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ TableManager │  │ SchedulerIOMixin │   │
│  │ (page slots) │  │ (ZMQ recv/send)  │   │
│  └──────────────┘  └──────────────────┘   │
└───────────────────────────────────────────┘
```

### Main Event Loop

Two modes:

Normal loop (sequential):
```
receive_msgs → schedule_batch → forward → process_results → repeat
```

Overlap loop (pipelined):
```
        CPU stream              │        GPU (Engine) stream
────────────────────────────────┼────────────────────────────
Step 1: receive_msgs            │
Step 2: schedule_batch          │
        └── launch forward ─────│──→  forward (async)
Step 3: process LAST results    │     ...compute...
        receive_msgs            │
        schedule_batch          │
        └── launch forward ─────│──→  forward (async)
Step 4: process LAST results    │     ...compute...
        ...                     │
```
The key insight: while GPU computes batch N, CPU prepares batch N+1 on a separate CUDA stream.

### PrefillManager — Chunked Prefill

Why chunk? A single request with 100K tokens would eat all GPU memory. Chunked prefill splits long prompts into smaller pieces processed across multiple steps.

```
Request: "Hello world, how are..." (5000 tokens, budget=2000)

Step 1: prefill "Hello world, how" (2000 tokens) → ChunkedReq
Step 2: prefill " are you today? I" (2000 tokens) → ChunkedReq
Step 3: prefill " was wondering if..." (1000 tokens) → Req (last chunk → can decode)
Step 4: decode → produce output tokens one by one
```

`ChunkedReq` is a subclass of `Req` that:
- Cannot be sampled (`can_decode=False`)
- Won't be added to DecodeManager
- Carries the same `cache_handle` and `table_idx` across chunks

### DecodeManager — Running Requests

Keeps a `Set[Req]` of requests currently decoding. Simple but careful about:
- `filter_reqs()`: after a batch completes, remove finished reqs, add new ones ready for decode
- `inflight_tokens`: estimates reserved memory to prevent PrefillManager from over-allocating

### TableManager — Page Table Slots

Each request gets a `table_idx` (row in the page table tensor). TableManager is a simple free-list allocator:
- `allocate()` → pop a slot, `free(slot)` → push it back
- `token_pool[table_idx, :]` stores the actual token IDs on GPU

### CacheManager — Bridge Between KV Cache & Scheduler

Wraps the prefix cache (Naive/Radix). Key operations:
- `match_req()`: look up shared prefixes before allocating
- `allocate_paged()`: assign physical KV cache pages to reqs' page table entries
- `cache_req()`: after prefill, insert computed KV into prefix cache; evict if needed
- `lazy_free_region()`: context manager that defers page freeing for correctness during overlap scheduling

### I/O (SchedulerIOMixin) — ZMQ Communication

- **Single GPU**: direct ZMQ PULL from tokenizer, PUSH to detokenizer
- **Multi GPU**: TP Rank 0 receives from ZMQ, broadcasts to other ranks via PUB/SUB, then synchronizes count via `broadcast()` on CPU group
- Only Rank 0 sends results back to detokenizer

## KV Cache Management

## Attention Backends

## Model Implementations (Layers & Architectures) 

## Tensor Parallelism & Distributed Communication

## Overlap Scheduling 

## Serving API & Tokenizer