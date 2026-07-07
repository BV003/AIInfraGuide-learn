# Inference

推理是大模型从训练走向落地的关键环节。本章建立推理优化的基础概念和分析框架。

自回归生成过程详解 LLM 推理的两阶段：Prefill（一次性处理完整 Prompt，Compute Bound）和 Decode（逐 Token 生成，Memory Bound），以及为什么 Decode 阶段效率远低于 Prefill。

KV Cache是推理优化的核心概念：缓存已计算的 Key 和 Value 避免重复计算，包括 KV Cache 的生命周期（分配 → 填充 → 使用 → 释放）、显存计算公式和碎片化问题。

关键性能指标覆盖 TTFT（首 Token 延迟）、TPOT（每 Token 延迟）、Throughput（吞吐量）、P50/P95/P99 尾延迟和 Goodput（满足 SLO 的有效吞吐）。

推理链路拆解走通 Tokenize → Prefill → Decode Loop → Sampling → Detokenize 的完整流程，分析每个环节的计算特性与潜在瓶颈。

## The core technology of reasoning engine

## 主流推理引擎

## 量化

## PD 分离

## 性能分析和benchmark

## Reference