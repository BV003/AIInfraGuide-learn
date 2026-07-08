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

Overlap(重叠) loop (pipelined):
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
- `inflight_tokens`(在运行中的): estimates reserved memory to prevent PrefillManager from over-allocating

### TableManager — Page Table Slots

Each request gets a `table_idx` (row in the page table tensor). TableManager(表管理器) is a simple free-list allocator:
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

Three layers:

1. GPU Buffer (`kvcache/mha_pool.py`) — raw tensor `(2, num_layers, num_pages, page_size, kv_heads, head_dim)`. `[0]`=K, `[1]`=V. `store_kv()` copies attention output into the pool.

2. Prefix Cache (`kvcache/radix_cache.py` or `naive_cache.py`) — radix tree matches token prefixes and shares cached K/V across requests. Naive variant always returns miss. Key ops: `match_prefix`, `insert_prefix`, `lock_handle`, `evict`. Caching is page-aligned.

3. CacheManager (`scheduler/cache.py`) — scheduler-facing coordinator. `allocate_paged()` assigns free pages (evicts if needed). `cache_req()` inserts completed prefill tokens into prefix cache, frees duplicate pages. `lazy_free_region()` batches frees for efficiency.

Page table: `page_table[req_id, token_pos]` maps to physical KV index. Shape `(max_running_reqs, max_seq_len)`. Backends divide by `page_size` to get logical page indices.

Request lifecycle:
- Prefill: `match_req()` → get `cached_len` → copy cached page-table entries → `allocate_paged()` for new tokens → attention runs → `cache_req()` inserts into prefix cache.
- Decode: extend 1 token/step → allocate 1 slot → store KV → `complete_one()`.
- Completion: `cache_req(req, finished=True)` frees pages + releases handle; `TableManager.free()` returns slot.

Design notes: page-aligned (page_size 16/32/64), reference-counted handles prevent eviction of in-use cache, LRU eviction via min-heap over leaf nodes, chunked prefill shares `cache_handle`/`table_idx` across splits, KV cache sharded by TP rank.

## Attention Backends

Purpose: compute attention without recomputing past K/V from scratch. Each backend reads from the paged KV cache, only computes attention for new tokens, then stores the new K/V back into the cache. This is how we avoid O(n²) cost over the full sequence.

All three backends can handle both prefill and decode. The interface is the same (`BaseAttnBackend`). They differ in which GPU kernel library they call and which hardware they optimize for.

1. FlashAttention (`attention/fa.py`) — calls `sgl_kernel.flash_attn.flash_attn_with_kvcache`. Supports page_size > 1.

2. FlashInfer (`attention/fi.py`) — uses `flashinfer.BatchPrefillWithPagedKVCacheWrapper` / `BatchDecodeWithPagedKVCacheWrapper`. Requires page_size=1. Runs an async planning step (`wrapper.plan()`) before each forward.

3. TensorRT-LLM (`attention/trtllm.py`) — uses `flashinfer.trtllm_batch_{context,decode}_with_kv_cache`. For Blackwell (B200).

Hybrid mode: combine separate prefill and decode backends. Example: `"fa,fi"` = FA prefill + FlashInfer decode. This is the default on H100 because FA handles variable-length prefill better, while FlashInfer decode integrates cleaner with CUDA graphs.

Common flow per layer: `prepare_metadata(batch)` → scheduler computes cu_seqlens, cache_seqlens, page_table from requests. `forward(q, k, v, layer_id, batch)` → stores K/V into cache, then runs paged attention kernel.

CUDA graph support: all three implement `init_capture_graph` / `prepare_for_capture` / `prepare_for_replay` for decode. Pre-allocated buffers hold seq_lens and page_table; during replay real data is copied in.

Auto-selection: B200 → trtllm, H100 → fa+fi hybrid, others → fi.

## Model Implementations (Layers & Architectures)

All models share the same structure: `ForCausalLM` wraps a `Model` (embedding + decoder layers + final norm) + `lm_head`. The forward pass reads `input_ids` from the global context batch, runs through layers, produces logits.

### Building blocks (`layers/`)

Decoder layer pattern (same across Llama, Qwen2, Qwen3):
1. input_layernorm (RMSNormFused — fused add + norm, saves a kernel launch)
2. self_attn: QKV projection → Q/K norm (optional) → RoPE → AttentionLayer → O projection
3. post_attention_layernorm
4. mlp: gate+up projection → activation (silu/gelu) → down projection

Residual is streamed through layers as a separate tensor: each `RMSNormFused.forward(x, residual)` does `x = x + residual` then norms, and returns `(normed_x, old_x_as_new_residual)`. This avoids materializing the full residual at every layer.

Linear layers are TP-aware (tensor-parallel sharded):
- `LinearQKVMerged`: fused QKV in one matrix. Output dim split across TP ranks (column-parallel).
- `LinearColParallelMerged`: gate+up fused. Output dim split across TP ranks.
- `LinearRowParallel`: down_proj. Input dim split → each rank computes partial → all-reduce.
- `LinearOProj`: same as row-parallel but for attention output.
- `LinearReplicated`: full copy on every rank (used for MoE router gate).
- `VocabParallelEmbedding`: vocab sharded across ranks. Lookup returns partial embeddings → all-reduce. `ParallelLMHead` extends it with all-gather at the end (prefill: only last positions, decode: all).

AttentionLayer (`layers/attention.py`): splits fused QKV, optionally applies Q/K norm, applies RoPE to Q and K, then calls `ctx.attn_backend.forward(q, k, v, layer_id, batch)` — the backend handles KV cache read/write and paged attention.

MoE layer (`layers/moe.py`): `gate_up_proj` has shape `(num_experts, 2*intermediate, hidden)` and `down_proj` has shape `(num_experts, hidden, intermediate)`. Forward delegates to `moe_backend.forward()` (fused kernel). If TP > 1, all-reduce after.

### Model registry (`models/register.py`)

Maps HF architecture names to implementations: `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `MistralForCausalLM`.

### ModelConfig (`models/config.py`)

Parsed from HuggingFace `config.json`. Canonical field names hide differences between model families (e.g., `rope_theta` location differs between Llama and Mistral). The `is_moe` property checks if `model_type` contains "moe".

### Weight loading (`models/weight.py`)

Streams from safetensors files. For each tensor:
1. Shard by TP rank (dim-0 for Q/K/V/gate/up, dim-1 for O/down, vocab range for embeddings)
2. Fuse Q+K+V into `qkv_proj` and gate+up into `gate_up_proj` (single matrix multiply instead of three/two)
3. For MoE: stack per-expert weights into a single `(num_experts, ...)` tensor

### Model differences

| Model | Attn Bias | Q/K Norm | MLP Type |
|--------|-----------|----------|----------|
| Llama 3 | No | No | Gated (silu) |
| Qwen 2.5 | Yes | No | Gated (silu) |
| Qwen 3 (dense) | No | Yes | Gated (silu) |
| Qwen 3 (MoE) | No | Yes | MoE (silu) |
| Mistral | No | No | Gated (silu) |

Despite these differences, all use the same `RopeAttn` and `GatedMLP`/`MoEMLP` helpers from `models/utils.py` — just parameterized differently.

## Tensor Parallelism & Distributed Communication

TP splits model weights across multiple GPUs on the same node. Each GPU holds a shard of every weight matrix and computes a partial result, then communicates to combine.

### How TP is set up

`launch.py` spawns one process per GPU (passed via `--tp-size`). Each process gets `DistributedInfo(rank=i, size=N)`. Processes join a process group via `init_process_group`. Rank 0 also spawns tokenizer/detokenizer workers.

`set_tp_info(rank, size)` stores a global singleton. Every layer reads it via `get_tp_info()` to know its shard dimensions.

### Communication (`distributed/impl.py`)

Two backends, both wrapped by `DistributedCommunicator`:

1. `TorchDistributedImpl` (default) — uses `torch.distributed.all_reduce` / `all_gather_into_tensor`. Simple, works everywhere.

2. `PyNCCLDistributedImpl` (enabled by default via `--use-pynccl`) — custom NCCL wrapper in `kernel/pynccl.py`. Lower latency than torch's default because it bypasses the Python NCCL binding layer.

`enable_pynccl_distributed()` pushes PyNCCL onto a plugin stack. `DistributedCommunicator` delegates to the last plugin. If TP=1, no plugin is pushed and communication is a no-op.

### Where communication happens

| Layer | Operation | Why |
|--------|-----------|-----|
| LinearRowParallel (down_proj) | all-reduce | Each rank computes partial output → sum |
| LinearOProj (attn O) | all-reduce | Same pattern: partial O → sum |
| VocabParallelEmbedding | all-reduce | Each rank looks up its vocab shard → sum embeddings |
| ParallelLMHead | all-gather | Each rank computes logits for its vocab shard → gather full vocab |
| MoELayer | all-reduce | Each rank processes its expert shard → sum |

Column-parallel layers (QKV, gate_up) and replicated layers (MoE router) don't need communication because each rank's output is already correct for the next column-parallel layer.

## Overlap Scheduling

Two CUDA streams: `self.stream` (scheduler) and `self.engine.stream` (engine). The idea: while GPU computes batch N, CPU prepares batch N+1 and processes results from batch N-1.

`normal_loop`: receive → schedule → forward → wait → process results → repeat. Simple, one batch at a time.

`overlap_loop(last_data)`: each iteration does three things concurrently:
1. Receive new messages, process into queues.
2. `_schedule_next_batch()` → `_prepare_batch()` builds ForwardInput (page table, KV allocation, attention metadata) on the scheduler stream. Then `engine.stream.wait_stream(self.stream)` syncs, and `_forward()` launches model forward + sampling on the engine stream (async). Returns `ongoing_data` for next iteration.
3. `_process_last_data(last_data)` — processes the *previous* iteration's results (copy tokens to CPU, update Req state, free finished, cache prefixes). GPU meanwhile runs the new forward.

Two correctness tricks:
- `lazy_free_region()`: defers page freeing until after `_process_last_data` finishes. If we freed immediately, the running forward might read pages we just freed.
- `finished_reqs` set: prevents double-free when the same request finishes in two consecutive overlap iterations.

Disabled via `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` (falls back to `normal_loop`).

## Serving API & Tokenizer

Process architecture:

```
FastAPI Server (rank0) ──ZMQ──→ Tokenizer/Detokenizer ──ZMQ──→ Scheduler (TP0..N)
      ↑                                        ↑                      │
      └──────────────ZMQ───────────────────────┘                      │
                                                                      │
                          ←─────────────ZMQ──────────────←────────────┘
```

API to token: FastAPI receives request → sends `TokenizeMsg` via ZMQ PUSH to tokenizer → tokenizer encodes text to token IDs → sends `UserMsg` to scheduler → scheduler runs inference → returns `DetokenizeMsg` back to detokenizer → detokenizer decodes to text → sends `UserReply` back to frontend → SSE streamed to client.

### API Server (`server/api_server.py`)

FastAPI with three endpoints: `/generate` (simple), `/v1/chat/completions` (OpenAI-compatible, streaming + non-streaming), `/v1/models`. All return `StreamingResponse` with SSE for streaming.

`FrontendManager`: tracks each request by `uid`, maintains `ack_map[uid]` for collecting incremental replies and `event_map[uid]` for async notification. Each request: allocate uid → send TokenizeMsg → async generator yields chunks from wait_for_ack → stream to client.

Client disconnect detection via `request.is_disconnected()` — sends `AbortMsg` to cancel.

Interactive shell mode (`--shell-mode`): TUI using `prompt_toolkit`, maintains chat history, calls the same backend.

### Tokenizer/Detokenizer (`tokenizer/server.py`)

Single worker process that does both tokenization and detokenization. Loop: pull batch from ZMQ → classify messages (DetokenizeMsg / TokenizeMsg / AbortMsg) → process → push results back.

TokenizeManager: calls `tokenizer.encode()` or `tokenizer.apply_chat_template()` for chat messages.

DetokenizeManager: per-uid state tracking. Incremental decoding handles the fact that a single new token might not form a complete character. Uses `batch_decode()` and tracks `decoded_ids` / `read_offset` / `sent_offset` to only output complete, printable text. Special handling for CJK characters (single chars are always safe) and incomplete UTF-8 (`�`).

### Message Protocol (`message/`)

Three message families, all ZMQ-serialized via a custom JSON-based protocol (supports nested dataclasses and 1D tensors as numpy byte buffers):

- Tokenizer messages: `TokenizeMsg` (text → tokens), `DetokenizeMsg` (token → text), `AbortMsg`
- Backend messages: `UserMsg` (input_ids + sampling params), `AbortBackendMsg`, `ExitMsg`
- Frontend messages: `UserReply` (incremental text + finished flag)

Each family has a `Batch*` variant for batching multiple messages in one ZMQ frame.

ZMQ pattern: PUSH/PULL for point-to-point. Async wrappers (`ZmqAsyncPushQueue` / `ZmqAsyncPullQueue`) for the FastAPI frontend, sync wrappers for the tokenizer and scheduler. For multi-TP, PUB/SUB for broadcasting from rank 0 to other ranks.
