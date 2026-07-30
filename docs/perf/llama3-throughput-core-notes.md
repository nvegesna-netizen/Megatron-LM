<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# Llama 3 8B / 70B dense training — Megatron-Core notes

Working notes for a Llama-3 throughput investigation, produced on branch
`perf/llama3-throughput-core` at `157c023f2` (`origin/main`, 2026-07-28).

These are **notes, not conclusions**. Claims are tagged `OBSERVED` (read from a file or command
output), `INFERRED`, or `HYPOTHESIS`. No training run backs anything here.

Downstream consumer context (Megatron-Bridge / JET-LLM specifics, ranked experiment plan) lives
outside this repo at `JET-LLM/docs/performance/llama3-silicon/megatron-core-analysis.md`.

Model shape assumed throughout: dense GPT, GQA (`num_query_groups=8`), SwiGLU, RMSNorm, RoPE,
`seq_length=8192`, `add_bias_linear=False`, untied embeddings, vocab 128256. No MoE, no MLA, no
Mamba.

---

## 1. What Megatron-Core ships for this shape

### 1.1 CUDA graphs

Three orthogonal flags since `e41b37002` ("Refactor CUDA graph API: decompose `cuda_graph_scope` into
`full_iteration` impl, inference scope, and per-layer capture modules", #4292). `OBSERVED`,
`docs/user-guide/features/cuda_graph.md:22-38`:

| Flag | Field | Values | Default |
|---|---|---|---|
| `--cuda-graph-impl` | `megatron/core/transformer/transformer_config.py:1032` | `none` / `local` / `transformer_engine` / `full_iteration` | `none` |
| `--cuda-graph-modules` | `transformer_config.py:1047` | `attn` `mlp` `moe` `moe_router` `moe_preprocess` `mamba`, empty = whole layer | `"full"` → `[]` |
| `--inference-cuda-graph-scope` | `transformer_config.py:1065` | `none` / `layer` / `block`, non-`none` only under `local` | derived |

`cuda_graph_scope` is deprecated and migrated in `__post_init__`
(`transformer_config.py:1086-1092`; migration table `cuda_graph.md:208-215`).
`--cuda-graph-impl local --cuda-graph-scope full_iteration` → `--cuda-graph-impl full_iteration`.

Implementations:
- `local` — `megatron/core/transformer/cuda_graphs.py:1763` `CudaGraphManager`, per-layer, one
  process-wide `graph_pool_handle` (`:1766-1767, 1837-1838`). Runners are reused across microbatches
  only when PP=1 (`:1836`); PP>1 allocates a runner per microbatch (`:1917-1930`).
  `cuda_graph_use_single_mempool` is **not read** on this path (`INFERRED` — grep finds only the
  stale comment at `:1766`; the field is consumed by `full_cuda_graph.py:44-52`, i.e. by
  `full_iteration`). The docstring at `transformer_config.py:1012-1016` is therefore stale.
- `transformer_engine` — `cuda_graphs.py:2086` `TECudaGraphHelper` → TE `make_graphed_callables()`
  (`:2686-2688`); TE owns the pool; `_reuse_graph_input_output_buffers=True` on TE ≥ 2.7.0 (`:2566-2601`).
- `full_iteration` — `megatron/core/full_cuda_graph.py:138` `FullCudaGraphWrapper`, installed at
  `megatron/training/training.py:3617-3622` (train) and `:4203-4208` (eval); shares a capture stream
  and, by default, a mempool with the optimizer graph (`full_cuda_graph.py:21-52`).

Constraints that bite a dense Llama-3 (`OBSERVED`, all asserts):

| Constraint | Site |
|---|---|
| Requires `use_te_rng_tracker=True` | `megatron/training/models/gpt.py:216-220`; `cuda_graphs.py:1821-1825`; auto-enabled at `arguments.py:1873-1886` |
| `full_iteration` requires `--no-check-for-nan-in-loss-and-grad` | `arguments.py:1195-1198` |
| `full_iteration` requires empty `--cuda-graph-modules` | `transformer_config.py:2461-2463`; `arguments.py:1888-1890` |
| `--recompute-granularity full` requires `full_iteration` | `transformer_config.py:2521-2526` |
| CPU offloading requires `full_iteration` | `transformer_config.py:2467-2468` |
| `transformer_engine` impl + `expandable_segments:True` requires `NCCL_GRAPH_REGISTER=0` | `arguments.py:1880-1886`; `cuda_graphs.py:2098-2109` |
| Static shapes only | `cuda_graphs.py:1647-1710` `get_mismatch_errors`, then `:1895-1900` |
| Graphed `attn` + `attention_dropout != 0` + `core_attn` recompute rejected | `transformer_config.py:2546-2552` |
| Hybrid CP rejected | `arguments.py:1406` |

Sequence parallel and fp8/fp4 are **not** rejected (`cuda_graphs.py:2680-2684`, `:975-988`).

Recent correctness work, both very new (`OBSERVED`):
- `4b18b260f` (2026-07-27) — five memory bugs: alias-chain-aware buffer metadata
  (`cuda_graphs.py:97, 109, 1553-1566`), a `save_for_backward` observer fixing forward→backward
  buffer lifetimes (`:230, 494-509, 1079-1135`), a multi-consumer input-copy fix requiring
  `input_use_count == 1` (`:773-786`), a `PackedSeqParams.seq_idx` rebuild corruption during capture
  (`:293-322`), and removal of the last-layer output clone.
- `bacd3404c` (2026-07-28) — DDP init must run on `get_shared_capture_stream()` when
  `cuda_graph_impl == "full_iteration"`, else `AccumulateGrad` nodes reference a non-capturing stream
  (`training.py:1856-1863`, `models/dist_utils.py:297-303`).

`INFERRED` — any CUDA-graph measurement taken on a build older than these two commits should be
treated as correctness-suspect, not merely slower or faster.

### 1.2 FP8 / FP4

Fields are on `TransformerConfig`, not `ModelParallelConfig` (`OBSERVED`, grep):
`fp8` (`transformer_config.py:566`, `--fp8-format e4m3|hybrid`),
`fp8_recipe` (`:573`, `tensorwise|delayed|mxfp8|blockwise|custom`, default **`delayed`**),
`fp8_param` (`:581`, CLI `--fp8-param-gather`, mapped at `megatron/training/argument_utils.py:312`),
`fp8_output_proj` (`:614`, MXFP8-only LM-head projection),
`fp8_dot_product_attention` (`:619`),
`first_last_layers_bf16` + `num_layers_at_start/end_in_bf16` (`:628-637`),
`fp8_amax_history_len` (`:600`, default **1**),
`fp4` / `fp4_recipe` / `fp4_param` (`:652-671`).

Recipe dispatch `megatron/core/fp8_utils.py:714-772`; enum `megatron/core/enums.py:12-26`.
Version floors: `tensorwise` TE ≥ 2.2.0.dev0, `blockwise` TE ≥ 2.3.0.dev0, `custom` TE ≥ 2.9.0.dev0,
FP4 TE ≥ 2.7.0.dev0 (`arguments.py:1073-1074`); TE < 2.1.0 → `delayed` only (`fp8_utils.py:760-766`).

Structural note (`OBSERVED`, `megatron/core/transformer/transformer_block.py:600-611`): **delayed
scaling wraps the whole block in one `fp8_autocast`**, every other recipe uses a per-layer inner
context, with an in-code comment that the per-layer enter/exit is free. `INFERRED` — this is a
structural reason current-scaling / mxfp8 can interact with CUDA graphs differently from delayed.

GEMM alignment: 32 for mxfp8, 16 otherwise (`fp8_utils.py:318-323`); 128 for FP4
(`fp4_utils.py:176-205`).

Constraints:

| Constraint | Site |
|---|---|
| `first_last_layers_bf16` rejected with `delayed` | `transformer_config.py:1400-1403`; `fp8_utils.py:828-830` |
| `--fp8-param-gather` requires dist-opt / FSDP2 / Megatron-FSDP | `arguments.py:1060-1062` |
| torch-FSDP2 + TE ≥ 2.0 silently disables `fp8_param_gather` | `arguments.py:1017-1023` |
| mxfp8 without `reuse_grad_buf_for_mxfp8_param_ag` warns about large extra memory | `optimizer_config.py:413-422` |
| `reuse_grad_buf_for_mxfp8_param_ag` ⊥ `overlap_param_gather_with_optimizer_step` | `optimizer_config.py:424-428` |
| fp8 + `moe_act`/`layernorm` recompute rejected under `delayed`; needs TE ≥ 2.6.0dev0 | `transformer_config.py:1821-1834` |
| fp8 + `first_last_layers_bf16` + TP overlap needs a second BF16 UB pool | `megatron/training/initialize.py:226-232` |
| FP4 ⊥ FP8 | `arguments.py:1065-1066`; `transformer_config.py:1454-1455` |

**No runtime device-capability assert exists for `mxfp8` or `nvfp4`** (`OBSERVED`, grep across
`megatron/`). The Blackwell-only statements are docstrings (`transformer_config.py:578, 660`) and
test skips (`tests/unit_tests/determinism/correctness/test_fp8_determinism.py:23, 30-39`).
`INFERRED` — enforcement is delegated to TransformerEngine, so a wrong-arch recipe surfaces as a TE
error rather than a clean MCore message. `HYPOTHESIS` — worth an upstream issue proposing an
explicit `get_device_arch_version() >= 10` guard with an actionable message.

### 1.3 TP comm overlap (userbuffers)

All flags on `megatron/core/model_parallel_config.py`: `tp_comm_overlap` (`:265`, `False`),
`tp_comm_bulk_wgrad` (`:271`, `True`), `tp_comm_bulk_dgrad` (`:276`, `True`),
`tp_comm_overlap_ag` (`:281`, `True`), `tp_comm_overlap_rs` (`:286`, `True`),
`tp_comm_overlap_rs_dgrad` (`:291`, **`False`**), the deprecated `split`/`atomic` quartet
(`:296-314`), `tp_comm_overlap_disable_qkv` (`:330`), `tp_comm_overlap_disable_fc1` (`:335`),
`tp_comm_bootstrap_backend` (`:340`).

`tp_comm_overlap_disable_qkv`, `tp_comm_overlap_disable_fc1`, `tp_comm_atomic_ag` and
`tp_comm_atomic_rs` have **no CLI flag** — they are in the `ArgumentGroupFactory` exclude list
(`arguments.py:2225-2231, 2263-2265`) and are settable only programmatically.
`tp_comm_overlap_cfg` is CLI-only (`arguments.py:2743-2744`), not a config field.

Init: `megatron/training/initialize.py:187-260`, gated at `:158-159`;
`input_shape = [(seq_length * micro_batch_size) // context_parallel_size, hidden_size]` (`:207-218`).
Requires `--sequence-parallel` (`arguments.py:1401-1402`), hence TP>1 (SP is force-disabled at TP=1,
`arguments.py:1393-1400`). **No NVLink/topology assert exists in this repo** — that lives in TE's
`initialize_ub`.

Buffer names are exactly `qkv` / `proj` / `fc1` / `fc2`
(`megatron/core/extensions/transformer_engine.py:922-932`). An unrecognised name **silently sets
`config.tp_comm_overlap = False`** with only a warning — `INFERRED`, a silent-perf-loss footgun
worth a louder failure mode upstream.

Only one UB config YAML ships in-tree, and it is a TP=2 FP8 test file:
`tests/functional_tests/test_cases/gpt/gpt3_weekly_mcore_tp2_pp2_current_scaling_native_fp8_tp_pp_sp_tp_overlap/tp_comm_overlap_cfg.yaml`
(`ring_exchange` for `qkv_fprop`/`fc1_fprop`/`fc2_fprop`/`fc2_dgrad`/`proj_dgrad`; `pipeline`
`num_sm:4 num_splits:4` for `proj_fprop`; `bulk` `num_sm:2` for the dgrad/wgrad pairs).
**There is no Llama-3-shaped UB config in this repo.**

No Blackwell-specific TP-overlap path exists. The only arch conditionals near comm are
`arguments.py:1413-1416` (the `CUDA_DEVICE_MAX_CONNECTIONS=1` requirement applies only to
`get_device_arch_version() < 10`) and Blackwell high-priority NCCL stream groups, which apply only
under FSDP or GTP (`arguments.py:1441-1448, 1484-1505`).

Incompatible with determinism (`megatron/training/determinism.py:24`;
`docs/user-guide/deterministic-training.md:44`), batch-invariant GEMM
(`batch_invariant_kernels.py:961-966`), and ModelOpt distillation
(`megatron/post_training/model_builder.py:481-483`). **No CUDA-graph or fp8 incompatibility assert
exists** (`OBSERVED`, grep).

### 1.4 Distributed optimizer / DDP / `nccl_ub`

`megatron/core/distributed/distributed_data_parallel_config.py`: `use_distributed_optimizer` (`:29`),
`overlap_grad_reduce` (`:18`), `overlap_param_gather` (`:21`), `align_param_gather` (`:24`, CLI
default `True` via `--no-align-param-gather`), `bucket_size` (`:48`), `num_buckets` (`:53`, mutually
exclusive with `bucket_size` at `:269-271`), `pad_buckets_for_high_nccl_busbw` (`:58`),
`average_in_collective` (`:74`), `data_parallel_sharding_strategy` (`:100`, dataclass default
`'no_shard'` but **CLI default `'optim_grads_params'`** at `arguments.py:3023-3025`),
`nccl_ub` (`:118`, CLI `--use-nccl-ub`), `fsdp_double_buffer` (`:135`),
`disable_symmetric_registration` (`:168`), `fsdp_manual_registration` (`:174`).

`megatron/core/optimizer/optimizer_config.py`: `use_precision_aware_optimizer` (`:187`),
`overlap_param_gather_with_optimizer_step` (`:337`), `optimizer_cpu_offload` (`:344`),
`optimizer_cuda_graph` (`:386`), `main_grads_dtype` / `main_params_dtype` / `exp_avg_dtype` /
`exp_avg_sq_dtype` (`:197-207`).

Default bucket size (`megatron/core/distributed/distributed_data_parallel.py:57-72`):
`max(40000000, 1000000 * dp_size)`, set to `None` when `overlap_grad_reduce` is off; bucketing is
disabled on non-first PP stages (`:96-106`). Layout at
`megatron/core/distributed/param_and_grad_buffer.py:946-994`; 64-element param alignment and
DP-size bucket-end padding at `:1449-1461`.

`overlap_param_gather_with_optimizer_step` requires interleaved VPP (`arguments.py:1037-1038`) plus
distributed optimizer (`:1033`) and `--overlap-param-gather` (`:1035`), and is incompatible with
dist checkpointing (`:1039`).

`nccl_ub` (`--use-nccl-ub`, `dest='nccl_ub'`, `arguments.py:3010-3013`) works on the plain DDP
`_ParamAndGradBuffer` path, not only Megatron-FSDP (`param_and_grad_buffer.py:1129-1152`).
`megatron/core/nccl_allocator.py:153-156` **unconditionally sets `NCCL_NVLS_ENABLE=1`** and
`TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0`. Symmetric (window) registration needs torch ≥
2.9.0a0 with a soft fallback (`:116-122`). **Hard incompatibility:** `nccl_ub` +
`PYTORCH_CUDA_ALLOC_CONF` containing `expandable_segments:True` raises `ValueError` unless torch ≥
2.11.0a0 (`distributed_data_parallel_config.py:256-261`).

The SM-cost table motivating it (`distributed_data_parallel_config.py:126-132`): NVL → 4 SMs AG /
5 SMs RS; NVL+IB without SHARP → 16/16; with SHARP → 6/6; IB without SHARP → 1/4; with SHARP → 1/1.
`INFERRED` — on GB200 NVL72 this frees ~12 SMs from DP collectives.

`ddp_num_buckets_for_weight_grad_compute` **does not exist** in this repo (`OBSERVED`, grep).

### 1.5 Activation recompute

`megatron/core/transformer/transformer_config.py`:
`recompute_granularity` (`:518`, `full|selective`), `recompute_method` (`:529`, `uniform|block`),
`recompute_num_layers` (`:537`), `distribute_saved_activations` (`:543`),
`recompute_modules` (`:546`, defaults to `["core_attn"]` at `:1763-1764`; allowed set at `:1767-1780`
is `core_attn`, `moe_act`, `layernorm`, `mla_up_proj`, `mlp`, `moe`, `shared_experts`,
`gdn_norm_out`).

The only shorthand argparse flag is `--recompute-activations` (`arguments.py:2732-2734` →
`recompute_granularity='selective'` at `:589-591`); everything else is auto-generated from the
dataclass by `ArgumentGroupFactory` (`arguments.py:2290-2291`).

**`recompute_granularity_per_stage` does not exist here** (`OBSERVED`, grep). Nearest knob:
`num_microbatches_with_partial_activation_checkpoints` (`model_parallel_config.py:239-245`).

The repo's only cost guidance is qualitative:
- `transformer_config.py:519-526` — selective-recompute rationale, citing arXiv 2205.05198.
- `transformer_config.py:1804-1810` — *"For fused attention, you have no need to set 'core_attn' to
  recompute. Please check that the core_attn recompute is really needed."*
- `docs/user-guide/features/fine_grained_activation_offloading.md:126-131` — recompute lightweight
  modules (`layernorm`, activations), offload heavy ones (`core_attn`, `expert_fc1`).
- `docs/user-guide/features/context_parallel.md:36` — TP+CP can outperform full recompute.
- `megatron/core/transformer/moe/README.md:237` — selective recompute rated "Low" performance impact.

**There is no measured recompute-cost number anywhere in the repo** (`OBSERVED`).

Constraints: CPU offloading ⊥ any recompute (`transformer_config.py:1719-1722`);
`distribute_saved_activations` ⊥ `sequence_parallel` (`:1756-1760`) and requires TP>1 + `full` + a
method (`arguments.py:1371-1379`); `full` requires `cuda_graph_impl == 'full_iteration'` when graphs
are on (`:2521-2526`); fp8 + `moe_act`/`layernorm` recompute rejected under `delayed`, TE ≥ 2.6.0dev0
otherwise (`:1821-1833`).

**Upstream bug worth a one-line PR (`OBSERVED`):** the `moe_layer_recompute` deprecation message at
`transformer_config.py:1838` instructs users to pass `--recompute-modules moe_layer`, but
`moe_layer` is not in the allowed set; the code appends `"moe"` (`:1845-1846`). Following the message
fails the assert at `:1779`.

### 1.6 Fusions

Dataclass defaults are `False`, but the training CLI flips most on via `store_false` flags
(`megatron/training/argument_utils.py:301-303`). `INFERRED` — **any consumer that constructs
`TransformerConfig` directly instead of going through `megatron/training/arguments.py` gets the
dataclass defaults**, i.e. most fusions off. That is the single biggest source of "why does my
throughput differ from the upstream reference".

| Fusion | Field | Dataclass | CLI effective | Note for dense Llama-3 |
|---|---|---|---|---|
| `apply_rope_fusion` | `transformer_config.py:499` | `False` | **True** (`--no-rope-fusion`) | forced off if not RoPE (`arguments.py:1573`) |
| `bias_activation_fusion` | `:482` | `False` | **True** via `--no-bias-swiglu-fusion` | consumer `megatron/core/transformer/mlp.py:276`; ⊥ `use_te_activation_func` (`:2114-2144`) |
| `bias_dropout_fusion` | `:496` | `False` | **True** | `transformer_layer.py:684, 726, 980` |
| `masked_softmax_fusion` | `:485` | `False` | **True** | **inert under `--transformer-impl transformer_engine`**; only consumer `dot_product_attention.py:104` |
| `persist_layer_norm` | `:488` | `False` | **True** | `fused_layer_norm.py:66-68` asserts `normalization == "LayerNorm"` → **dead for RMSNorm** |
| `gradient_accumulation_fusion` | `model_parallel_config.py:250` | `False` | **True** | needs APEX `fused_weight_gradient_mlp_cuda` |
| `cross_entropy_loss_fusion` / `_fusion_impl` | `model_parallel_config.py:320` / `:325` | `False` / `native` | same | `te` impl rejected — see §2 |
| `use_transformer_engine_op_fuser` | `:511` | `False` | `False` | **dense-only** (`gpt_layer_specs.py:503-511`); TE ≥ 1.13 |
| `fused_residual_rmsnorm` | `:508` | `False` | `False` | **RMSNorm-only** (`:2147-2151`); TE ≥ 1.13 |
| `fused_single_qkv_rope` | `:505` | `False` | `False` | **no CLI flag** (`arguments.py:2251`); ⊥ gated attention (`:2189-2191`) |
| `use_grouped_gemm_for_dense_mlp` | `:822-826` | `False` | `False` | needs op fuser + SwiGLU; TE ≥ 2.14; SM100+; MXFP8 |
| `defer_embedding_wgrad_compute` | `model_parallel_config.py:410` | `False` | `False` | needs PP>1 **and** `gradient_accumulation_fusion` (`:545-558`); ⊥ TE MXFP8 output proj (`extensions/transformer_engine.py:1562`) |

Dense MLP selection (`OBSERVED`, `megatron/core/models/gpt/gpt_layer_specs.py:534-540`):

```
use_grouped_gemm_for_dense_mlp and use_te_op_fuser  ->  TEFusedMLPWithGroupedLinear
use_te_op_fuser                                     ->  TEFusedMLP
otherwise                                           ->  MLP
```

`TEFusedMLP` (`megatron/core/extensions/transformer_engine.py:2722`, builder `:2732-2853`) collapses
`Norm → BasicLinear(FC1) → [Bias] → activation → BasicLinear(FC2) → ReduceScatter|AllReduce → [Bias]`
into one `te.pytorch.ops.Sequential`, forwarding `userbuffers_options` when `tp_comm_overlap` is on.
`TEFusedMLPWithGroupedLinear` (`:3036-3065`) targets `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8` on
SM100+ and requires TE ≥ 2.14.0, `add_bias_linear=False`, and `F.silu` + `gated_linear_unit` —
**Llama-3 satisfies all three**, and this path appears to be untested by anything Llama-shaped.

`fused_residual_rmsnorm` is a **two-level opt-in** (`extensions/transformer_engine.py:800-808,
824-840`): the config flag alone is not sufficient; the build site must also pass `has_residual=True`
to `TENorm`. Verify the `TEFusedResidualRMSNorm` class is actually selected before attributing any
delta to the flag.

**cuDNN LayerNorm has no MCore plumbing** (`OBSERVED`, grep for `NVTE_NORM_FWD_USE_CUDNN` /
`use_cudnn_norm` in `megatron/` returns nothing). It is a pure TE env var, set in this repo only in
the GB200 dense release test configs (§3).

---

## 2. The TE cross-entropy fusion situation

`OBSERVED`, and worth flagging because the guard now lives in only one of the two entry paths:

- `168cb15d7` ("Disable TE cross entropy loss fusion", #5115) added a hard assert in both
  `model_parallel_config.py` and `arguments.py`, reason: *"Transformer Engine cross entropy loss
  fusion is disabled due to stability issues."*
- `b57449928` ("Move TE cross entropy guard to training args", #5162) **softened the core-side check
  to a `warnings.warn(UserWarning)`** (`model_parallel_config.py:536-543`) and left the hard
  rejection only in `megatron/training/arguments.py:1642-1647`.

`INFERRED` — a downstream consumer that builds `TransformerConfig` directly (Megatron-Bridge, NeMo,
any custom trainer) never reaches `arguments.py`, so it can and does run
`cross_entropy_loss_fusion=True` + `cross_entropy_fusion_impl='te'` with only a `UserWarning`.
Upstream's own GB200 dense release config uses `native`
(`tests/functional_tests/test_cases/gpt/gpt3_15b_8t_release_gb200/model_config.yaml:31-32`).

`HYPOTHESIS` — if the stability issue is real for library users too, the core-side check should be
an error, or the field default should be changed. If it is not, the `arguments.py` assert is
over-broad. Either way the two paths should agree. Candidate upstream issue.

---

## 3. Dense-GPT reference configurations and measured values in this repo

### 3.1 Llama-3 8B scripts

`examples/llama/train_llama3_8b_h100_fp8.sh` — exact Llama-3-8B shape (`:57-77`), `TP=1 CP=1 PP=1`,
`MBS=1 GBS=128`, bf16 base + optional FP8 (`--fp8-format hybrid --fp8-amax-history-len 1024
--fp8-amax-compute-algo max --fp8-param-gather`, `:106-112`), dist-opt with both overlaps
(`:126-128`), `--cross-entropy-loss-fusion`, `--manual-gc`, `--empty-unused-memory-level 1`
(`:97-100`). No TP overlap, no CUDA graphs. `CUDA_DEVICE_MAX_CONNECTIONS=1` at `:3`.

`OBSERVED` inconsistency: the script uses `CP_SIZE=1` (`:34`) while `examples/llama/README.md:80,107`
and `docs/user-guide/parallelism-guide.md:149` state CP=2, and the comment at `:120` says
"Always enable sequence parallelism with TP_SIZE=2" against `TP_SIZE=1`. The two in-repo config
tables also disagree on the 70B pipeline size: `examples/llama/README.md:108` says PP=8,
`docs/user-guide/parallelism-guide.md:150` says PP=4. Candidate doc-fix PR.

`examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh` — same shape, and notably the more
aggressive config: `unset CUDA_DEVICE_MAX_CONNECTIONS` (`:113`),
`--data-parallel-sharding-strategy optim_grads_params` (`:117`), `--use-nccl-ub` (`:122`),
`--fsdp-double-buffer` (`:123`), `--fsdp-manual-registration` (`:124`), with
`# --cuda-graph-impl full_iteration` (`:137`) and `# --num-distributed-optimizer-instances 2`
(`:128`) left commented out.

### 3.2 The FP8 perf table is not measured here

`examples/llama/README.md:105-109` (8B 13812 tok/s/GPU @ 800 TFLOP/s/GPU; 70B 1621 @ 780; 405B
315 @ 834). `README.md:101` says the configuration "follows those defined in NeMo Framework's
performance scripts" and `:121` redirects to the NeMo perf summary. `INFERRED` — these numbers are
imported from NeMo, carry no hardware/date/container attribution, are not regenerated by any CI job
here, and the 8B row's topology (CP=2) does not match the script it accompanies. Treat as
unverified.

### 3.3 Dense-GPT golden values that *are* measured here

`tests/functional_tests/test_cases/gpt/gpt3_7b_{tp4_pp1,tp1_pp4}_memory_speed/` — Llama-3-8B geometry
(32 layers, hidden 4096, 32 heads, GQA 8, seq 8192, vocab 131073; `ffn_hidden_size` left at the 4×h
default rather than Llama-3's 14336). Deterministic-mode regression tests, so not perf-tuned
(`NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0`, `NCCL_ALGO: Ring`, `--no-gradient-accumulation-fusion`).

Steady-state `iteration-time`:

| Test | GPUs | MBS | dgx_h100 | dgx_gb200 | dgx_a100 |
|---|---|---|---|---|---|
| `gpt3_7b_tp4_pp1_memory_speed` | 4 | 2 | 1.115 s | 0.852 s | 2.58 s |
| `gpt3_7b_tp1_pp4_memory_speed` | 4 | 1 | 1.150 s | 0.785 s | 2.94 s |

`INFERRED` — GB200 is 1.31× (TP4/PP1) to 1.46× (TP1/PP4) faster than H100 on an identical,
deterministic, untuned bf16 dense Llama-3-8B-shaped workload.

`tests/functional_tests/test_cases/gpt/gpt3_15b_8t_release*/` — 15B dense GPT (32L, hidden 6144,
48 heads, GQA 8, seq 4096, squared-relu, bf16), with a batch-size ramp
(`--step-batch-size-schedule "0:384 200B:768 400B:1152"`), so only same-step comparisons are valid.
The `_sm` suffix means a **shorter run** (`--train-samples 4882812`, `--exit-interval 13000`), not
symmetric memory.

| Test case | Golden | Config | @ step 100 | @ last | peak mem |
|---|---|---|---|---|---|
| `gpt3_15b_8t_release` | `dev_dgx_h100` | TP8/PP1 | 1.5915 s | 1.4405 s @50800 | 28.16 GB |
| `gpt3_15b_8t_release_sm` | `dev_dgx_h100` | TP8/PP1 | 1.5847 s | 1.5955 s @12700 | 28.16 GB |
| `gpt3_15b_8t_release_sm` | `dev_dgx_gb200` | TP8/PP1 | **1.1374 s** | 0.9194 s @12700 | 28.11 GB |
| `gpt3_15b_8t_release_sm_gb200` | `dev_dgx_gb200` | TP4/PP2/VP8 + TP overlap | **1.7101 s** | 1.2196 s @13000 | **42.64 GB** |

Two observations:

1. `INFERRED` — on GB200 the plain TP8/PP1 config is **1.50× faster** than the GB200-specific
   TP4/PP2/VP8 + `--tp-comm-overlap` config at the same step, with 34% less peak memory. On an NVL72
   domain TP8 is entirely intra-NVLink; trading TP width for pipeline depth buys bubbles and VPP
   activation memory in exchange for a collective that was already cheap.
2. **Data-quality caveat.** `gpt3_15b_8t_release_sm_gb200/golden_values_dev_dgx_h100.json` reports
   3.5554 s @100 / 3.38807 s @12700 — byte-identical to that test's `lts_dgx_a100` series
   (3.55899 / 3.39109). That file is stale or mis-copied. `gpt3_15b_8t_release_gb200` has **no**
   golden file at all. Release golden values here are not uniformly reliable as cross-platform
   measurements.

### 3.4 H100 vs GB200 environment recipe, per upstream's own dense configs

Diff of `gpt3_15b_8t_release/model_config.yaml` vs `gpt3_15b_8t_release_gb200/model_config.yaml`
(`OBSERVED`):

| | H100 variant | GB200 variant |
|---|---|---|
| Env | `NCCL_IB_SL=1`, `NCCL_IB_TIMEOUT=19`, `NVTE_FWD/BWD_LAYERNORM_SM_MARGIN=16`, `NCCL_P2P_NET_CHUNKSIZE=2097152` | `NVTE_NORM_FWD_USE_CUDNN=1`, `NVTE_NORM_BWD_USE_CUDNN=1`, `NVTE_FUSED_ATTN=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `USE_MNNVL=1` |
| Parallelism | TP=8 PP=1 | TP=4 PP=2 VP=8 layers/stage |
| TP overlap | absent | `--tp-comm-overlap: true`, no `--tp-comm-overlap-cfg` |
| Cross-entropy | absent | `--cross-entropy-loss-fusion` + `--cross-entropy-fusion-impl native` |
| RoPE fusion | `--no-rope-fusion: true  # TODO: We can remove this once upgrading to the DEV container` | on |

`USE_MNNVL` is read by nothing in `megatron/` (`OBSERVED`, grep) — `INFERRED` it is a launcher/
container variable.

---

## 4. Known issues and CI coverage gaps affecting this work

`OBSERVED`.

- **No training-throughput CI gate for dense models.** All four `*-perf.yaml` recipes
  (`tests/test_utils/recipes/{h100,gb200}/gpt-perf*.yaml`) drive
  `tests/performance_tests/shell_test_utils/run_perf_test.sh`, an **inference-serving** harness. The
  only training-speed signal is the `iteration-time` metric inside functional/release golden values.
- **Both multi-GPU GB200 perf recipes are `allow_failure: true`** —
  `gb200/gpt-perf-dp4.yaml:48` (issue #5692) and `gb200/hybrid-perf.yaml:49` (issue #5693).
- **Issue #513 (ckpt-resume hang) has degraded six dense-GPT tests, all of them overlap tests.**
  Each carries `TEST_TYPE: regular # Usually ckpt-resume, but as a WAR to #513 set to regular`:
  `gpt3_mcore_te_tp1_pp4_vp1_resume_torch_dist:57`,
  `..._dist_optimizer_overlap_grad_reduce_untied:60`,
  `..._dist_optimizer_overlap_grad_reduce_param_gather:63`, `..._tunable_overlap:58`,
  `..._resume_torch_decoupled_lr:54`, `..._decoupled_lr_1node:54`. Additional commented-out products
  at `h100/gpt.yaml:215, 229, 235, 242, 405`, `gb200/gpt.yaml:183`, and — including the
  attention-CUDA-graph resume case — `h100/moe.yaml:130, 135, 140, 165` /
  `gb200/moe.yaml:122, 127, 132`.
- **The only CP=4 dense test is disabled on both platforms** — `h100/gpt.yaml:138-143` and
  `gb200/gpt.yaml:130-133`, `# Non-deterministic: #487`.
- **Dense FP8 + `tp_comm_overlap` coverage is weekly-only** — `h100/gpt.yaml:516, 521`.
- **Platform gating:** `.gitlab/stages/04.functional-tests.yml:283-292` covers exactly `dgx_a100`,
  `dgx_h100`, `dgx_gb200`. `grep -i "gb300\|b300"` over `megatron/core/` returns nothing. There is
  no B200/GB300/B300 CI target.
- `docs/get-started/releasenotes.md` is 19 lines with no per-release history, no known-issues
  section, and no perf notes. There is no `CHANGELOG.md`.

---

## 5. Candidate upstream contributions identified

None have been filed; listed so the next person does not re-derive them.

1. `transformer_config.py:1838` — the `moe_layer_recompute` deprecation message recommends an
   invalid `--recompute-modules` value (`moe_layer`; the allowed value is `moe`).
2. `transformer_config.py:1012-1016` — `cuda_graph_use_single_mempool`'s docstring describes the
   deprecated `local` + `full_iteration`-scope combination; the field is actually consumed only by
   `full_cuda_graph.py:44-52`.
3. `extensions/transformer_engine.py:922-932` — an unrecognised `tp_comm_buffer_name` silently
   disables `tp_comm_overlap` process-wide with only a warning. A hard error would be safer.
4. `mxfp8` / `nvfp4` have no runtime `get_device_arch_version() >= 10` guard; failures surface from
   inside TransformerEngine.
5. `examples/llama/train_llama3_8b_h100_fp8.sh` (`CP_SIZE=1`, TP=1 with a "TP_SIZE=2" comment)
   disagrees with `examples/llama/README.md:80,107` and
   `docs/user-guide/parallelism-guide.md:149`; the 70B PP size disagrees between the two docs
   (8 vs 4).
6. The TE cross-entropy guard is an error in `megatron/training/arguments.py:1642-1647` but only a
   `UserWarning` in `megatron/core/model_parallel_config.py:536-543`, so library consumers bypass it
   (§2).
7. `docs/user-guide/training-examples.md:143` documents `--fp8-hybrid`, a flag that no longer exists
   (the current spelling is `--fp8-format hybrid`).

---

## 6. Local partial CUDA-graph correctness fixes

Two independent failures from the matched H100 Llama 3 70B local partial-graph probes are fixed in
this worktree.

### 6.1 Dense `[mlp]` construction

`GraphableMegatronModule.__init__` dynamically invokes
`TransformerLayer.create_mcore_cudagraph_manager()` before `TransformerLayer` constructs its MLP.
The dense `[mlp]` branch read `self.is_moe_layer` during that hook even though the attribute was
initialized only after MLP construction. The fix installs a dense `False` default before `super()`,
while preserving the `True` value that `MoETransformerLayer` deliberately sets before entering the
base constructor. The post-construction assignment from `isinstance(self.mlp, MoELayer)` remains
unchanged.

### 6.2 PP/VPP output pseudo-deallocation

`_CudagraphReplayNode.apply()` exposes a directly returned static graph surface as a view. On the
last local graph runner in a pipeline stage, that can turn the otherwise-viewless layer output into
a view rejected by `deallocate_output_tensor()`. The fix conditionally applies MCore's existing
`make_viewless_tensor(..., keep_graph=True)` after replay when both `is_last_layer` and
`deallocate_pipeline_outputs` are true. This aliases the same activation storage without the
activation-sized clone used by an older implementation and retains the replay node in autograd.
Non-last-layer runners and configurations that disable pipeline-output deallocation are unchanged.

Focused regressions cover dense local `[mlp]` construction and replay followed by viewlessness
assertion, pseudo-deallocation, and `custom_backward`. AST parsing, `git diff --check`, and Ruff
checks pass. The production-only diff is byte-identical to the JET `MCORE_PATCH` artifact
`local_partial_cudagraph_pipeline.patch`. GPU unit/distributed execution and the 64-GPU silicon
retries remain pending; neither fix is yet performance-validated.

The broader fact that current dense local `[attn]`/`[mlp]` selection still attaches a whole-layer
manager, rather than implementing true fine-grained dense capture, is intentionally not mixed into
these correctness fixes.

---

## 7. Provenance

Initial analysis began at `157c023f2`; the branch is nine non-code-affecting commits behind
`origin/main` at `c7d186578`. The two source fixes and focused tests above are currently uncommitted.
No training was run, no SLURM job was submitted, and nothing was pushed during this fix pass.
Repository instruction files were read first as required, including the complete
`skills/mcore-testing/SKILL.md` for the test strategy.
