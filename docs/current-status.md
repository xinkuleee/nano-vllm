# 当前实现与审查状态

本文描述当前功能分支，而 `phase1-summary.md` 和
`resume-and-interview.md` 中的部分边界描述的是历史 Phase 1 提交。

## 已实现

### 推理运行时

- 显式 prefill/decode 调度契约和不可变 worker 输入快照。
- chunked prefill、paged KV-cache、prefix cache、引用计数和 recompute
  preemption。
- 逻辑 KV-cache 管理与物理 GPU cache 分离。
- 模型注册表、AttentionBackend、作用域化 AttentionMetadata。
- 单 GPU eager、CUDA Graph decode，以及原有 tensor parallel 控制路径。

### GPU 基线工具

- macOS/CPU 使用 `uv` 的轻量控制面环境。
- WSL2/Linux NVIDIA 环境检查。
- eager/CUDA Graph、attention backend 和 Mini-MoE token parity 测试入口。
- TTFT、TPOT、吞吐、显存、逻辑 KV-cache 利用率和抢占次数 JSON 报告。

### Attention

- 默认路径继续使用第三方 `flash-attn`。
- 自写 Triton contiguous causal prefill，支持 packed variable-length batch、
  chunked query suffix、MHA/GQA、FP16/BF16 和 head dimension 不超过 128。
- 自写 Triton KV-cache scatter write。
- paged prefix-cache prefill 和 paged decode 仍由第三方 `flash-attn` 提供。

### Mini-MoE

- 将指定 Qwen3 dense MLP 层转换成 checkpoint-compatible Mini-MoE。
- top-k router、归一化 gate、负载统计。
- `dense_masked` 静态基线和 `sparse_reference` PyTorch 正确性参考。
- Triton assignment histogram、exclusive prefix sum、destination reservation、
  token permutation、gate/up grouped GEMM + fused SwiGLU、down grouped GEMM、
  weighted unpermute。
- expert packed-weight cache、失效处理、dense checkpoint 克隆和 native
  Mini-MoE checkpoint 名称保留。
- 当前 Triton sparse 路径仅支持 inference、eager、单 GPU、无 bias expert。

## 本轮全项目审查修复

- 将关键配置和 SamplingParams 的 `assert` 改成运行时输入校验。
- 拒绝未知 engine 参数、超出上下文长度的请求和不匹配的 sampling 列表。
- 对永远无法装入 KV cache 的请求给出确定错误，不再走到模糊断言。
- CUDA Graph capture 覆盖非标准 `max_num_seqs`，并在 replay 前清理旧的
  block-table 尾部。
- tensor parallel 的物理 KV-cache 容量采用所有 rank 的最小值。
- shared-memory worker 命令增加容量和方法名防护。
- checkpoint packed-name 映射改为完整 path component 匹配，避免
  `v_proj` 等字符串误匹配已经 packed 的参数名。
- 支持完整 runtime-format tensor 直接加载，同时保留 full source tensor 的
  TP sharding loader。
- native Mini-MoE checkpoint 的 `experts.*` 和 `router.*` 不再被错误地再次
  克隆。
- 缺少 safetensors checkpoint 时立即报错。

## 当前验证结果

- macOS 默认环境：105 passed，1 skipped，17 GPU tests deselected。
- 显式 GPU suite：17 skipped，因为本机没有 NVIDIA/PyTorch GPU 环境。
- `compileall`、`git diff --check` 和 `uv lock --check` 通过。
- 另用可用的 CPU PyTorch 环境验证了 Mini-MoE packed-weight cache 更新。

## 尚未验证或未实现

- NVIDIA 上 Triton JIT、FP16/BF16 数值 parity、端到端 token parity。
- FlashAttention 和 grouped GEMM 的真实性能、occupancy 和 Tensor Core 利用率。
- CUDA Graph 修复的真实 GPU replay。
- 多 GPU NCCL/TP 运行和共享 cache-capacity 行为。
- AMD/ROCm、expert parallel、all-to-all、capacity/drop policy。
- Mini-MoE 训练、load-balancing loss 和训练后专家分化。

在 NVIDIA 验证完成前，只能声明“实现并通过本地静态/控制面审查”，不能
声明性能提升或生产可用。
