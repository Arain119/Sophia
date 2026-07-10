# Sophia

ML 是一个以 `ml` Python 包交付的语言模型训练、推理、导出、评测和数据工具仓库。当前主线是单一路径 decoder 架构，覆盖 pretrain、SFT、runtime、export、eval 和 release gate。

本 README 是项目接手入口。核心真源见：

- `ml/core/spec/model.py`
- `ml/runtime/model/`
- `ml/tasks/pretrain/`
- `ml/training/pretrain/`
- `ml/training/posttrain/`

## 快速上手

本地接手先跑三件事：

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python3 -m ml.cli.pretrain_check --data_path dataset/pretrain_tokens --device cuda:0
PYTHONPATH=. python3 -m ml.cli.rehearsal --help
```

默认 pretrain 走当前固定 RTX 5090 BF16 recipe：

```bash
PYTHONPATH=. python3 -m ml.cli.train \
  --data_path dataset/pretrain_tokens/train \
  --tokenizer_path ml/modeling/text \
  --output_dir runs/pretrain_rtx5090_bf16
```

正式开训前先跑 GPU readiness probe。这个脚本会在 BF16/Torch/Inductor/Muon 路径上完成 torch inductor 审计、加速能力审计和 BF16 精度 probe。

```bash
bash tools/pretrain_gpu_probe.sh
```

开训前先确认：

- 当前固定目标机器是 `NVIDIA GeForce RTX 5090`，CUDA capability `[12, 0]`。
- `dataset/pretrain_tokens/{train,val,test}/manifest.json` 必须存在。
- tokenizer bundle sha1 必须匹配 token shard manifests。
- 默认 recipe 是 `configs/pretrain/machine_recipes/rtx5090_bf16_seq4096_bs3_acc22.json`。
- release gate 会校验完整 machine signature；非目标 GPU 不应直接复用该 recipe。
- 当前 release 精度栈固定为 BF16 + PyTorch SDPA Flash + torch linears + Liger fused linear CE。
- checkpoint/output/staging 不要放在 `/mnt/h` 这类 Windows mount；本地和正式机都使用 Linux 原生盘。

## 系统总览

当前有效主线：

- 模型：decoder causal LM，代码命名统一为 decoder。
- Attention：exact causal attention，通过 PyTorch `scaled_dot_product_attention` 调用。
- 架构：GQA、RMSNorm、QK-Norm、SwiGLU、tied embeddings、`z-loss`。
- 上下文：训练窗口 `4096`。
- 优化器：Muon + fused AdamW 混合优化器，fp32 master weights。
- 裁剪：release pretrain 采用 AGC，post-train 使用同一 shared clipping 工具链。
- 评测：训练内 loss/report、contamination scan、export/load smoke。
- 训练阶段：pretrain -> export -> SFT -> export。
- 门禁：machine recipe、manifest/tokenizer 指纹、resume、release audit。

推荐接手阅读顺序：

1. 先读本 README 到 `Training Lifecycle`，理解系统边界。
2. 再读 `ml/core/spec/model.py`，理解模型真源。
3. 然后读 `ml/tasks/pretrain/pipeline.py`，理解 pretrain 入口如何拼装 runtime。
4. 再读 `ml/training/pretrain/engine/train_step.py`、`optimizer.py`、`grad_clip.py`，理解每步训练语义。
5. 最后读 `ml/tasks/sft/` 和 `ml/tooling/scripts/release_gate/`。

## Configuration Ownership

项目不允许多处各自定义“看起来相同”的训练语义。接手时优先找真源，不要复制常量。

| 领域 | 真源 |
| --- | --- |
| 模型规格 | `ml/core/spec/model.py` |
| 模型语义/校验 | `ml/core/spec/model.py`, `ml/core/spec/semantics/` |
| runtime 模型 | `ml/runtime/model/` |
| pretrain release defaults | `ml/training/pretrain/release_config.py` |
| canonical pretrain machine recipe | `ml/training/pretrain/release_config.py`, `configs/pretrain/machine_recipes/rtx5090_bf16_seq4096_bs3_acc22.json` |
| pretrain run args | `ml/training/pretrain/run_config.py` |
| pretrain profile | `ml/training/pretrain/profiles.py` |
| machine recipe | `configs/pretrain/machine_recipes/` |
| optimizer | `ml/training/runtime_tools.py`, `ml/training/pretrain/optimizer.py` |
| grad clipping | `ml/training/pretrain/engine/grad_clip.py` |
| scheduler | `ml/training/scheduler.py` |
| posttrain defaults | `ml/training/posttrain/defaults.py` |
| posttrain curriculum | `dataset/posttrain_length_curriculum.json` |
| release audit | `ml/tooling/scripts/release_gate/audit_training_release.py` |

命名原则：

- 命名只保留当前真源和通用表达。
- 不引入非通用别名。
- 不把临时实验名写进核心 registry。

## 默认模型规格

默认规格由 `ModelSpec.default()` 定义，并投影到 `SophiaModelConfig`、`SophiaDecoderConfig` 和 runtime `ModelArgs`。

| 字段 | 默认值 |
| --- | ---: |
| `vocab_size` | 49152 |
| `dim` | 1536 |
| `n_layers` | 30 |
| `n_heads` | 12 |
| `head_dim` | 128 |
| `num_key_value_heads` | 4 |
| `ffn_hidden` | 4096 |
| `use_qk_norm` | `True` |
| `norm_eps` | `1e-6` |
| `max_seq_len` | 4096 |
| `max_batch_size` | 4 |
| `dropout` | `0.0` |
| `rope_head_dim` | 64 |
| `rope_theta` | 500000.0 |
| `original_seq_len` | 0 |
| `rope_factor` | 16.0 |
| `beta_fast` | 32 |
| `beta_slow` | 1 |

RoPE 语义：

- 当前默认采用高 base RoPE。
- `original_seq_len=0` 表示标准训练窗口语义。

## Model Architecture

模型实现分三层：

- `ModelSpec` 是规格真源。
- `SophiaDecoderConfig` 是训练、保存、导出的配置对象。
- `runtime/model/` 是实际 forward、cache、attention、block、norm、logits path。

整体结构：

- decoder causal LM。
- token embedding: `nn.Embedding(vocab_size, dim)`。
- transformer blocks: `n_layers = 30`。
- final norm: RMSNorm。
- output projection: bias-free `RuntimeLinear(dim, vocab_size)`。
- word embeddings tied: output weight 与 token embedding weight 是同一参数。
- dropout: 当前默认 `0.0`。

Transformer block：

- pre-norm attention residual: `x = x + Attention(RMSNorm(x))`。
- pre-norm FFN residual: `x = x + FeedForward(RMSNorm(x))`。
- attention norm 和 FFN norm 都是 RMSNorm。
- release recipe 关闭 gradient checkpointing。

Attention：

- exact causal GQA，不使用近似 attention。
- attention kernel 通过 PyTorch `scaled_dot_product_attention`。
- release pretrain 固化 `attn_backend=flash`。
- `n_heads = 12`。
- `num_key_value_heads = 4`。
- 每个 query head 维度 `head_dim = 128`。
- Q/K/V/O projections 都是 bias-free `RuntimeLinear`。
- QK-Norm 默认开启：Q 和 K reshape 到 head 后各自过 RMSNorm。
- QK-Norm 之后再应用 RoPE。
- KV cache 只用于推理/增量 decode；训练 full path 不依赖 cache。

RoPE：

- rotary dim 是 `rope_head_dim = 64`，只旋转 head 内最后的 rotary 子维。
- `rope_theta = 500000.0`。
- `original_seq_len = 0` 表示标准训练窗口语义。
- release train window 是 `4096`。

FFN / activation：

- FFN 是 SwiGLU。
- 计算式：`silu(gate_proj(x)) * up_proj(x)` 后接 `down_proj`。
- `gate_proj`、`up_proj`、`down_proj` 都是 bias-free `RuntimeLinear`。
- `ffn_hidden = 4096`。
- 激活函数是 SiLU，不是 GELU/ReLU。

Norm：

- 全模型使用 RMSNorm。
- `norm_eps = 1e-6`。
- QK-Norm 也使用 RMSNorm。
- final logits path 是 final RMSNorm 后接 tied output projection。

Loss：

- 默认训练接口使用 torch CE。
- release pretrain 固化 `loss_backend=torch`。
- `z_loss_weight = 1e-4`。
- labels 使用 causal shift 语义，ignore index 为 `-100`。
- `loss_chunk_size=0` 表示 release recipe 不做额外 chunked torch CE path。

Initialization：

- embedding 和常规 linear 权重初始化标准差 `0.02`。
- attention output projection 和 FFN down projection 使用 residual-scaled std: `0.02 / sqrt(2 * n_layers)`。
- bias 如存在会置零；当前主路径 projection 默认 bias-free。

Runtime linear：

- `RuntimeLinear` 是统一线性层入口。
- release pretrain 的 inner linear backend 固化为 `torch`。
- 当前训练路径使用 BF16 autocast + torch linears，optimizer 仍按 fp32 master weights 管理。

## Evaluation Methodology

项目评测覆盖训练质量、工程可恢复性和导出加载状态。

Pretrain 评测：

- train loss：每个 log interval 记录训练目标变化。
- val loss：按 `eval_interval` 在 val split 上评估。
- test loss：用于训练后质量确认，不参与训练调参循环。
- contamination scan：检测评测子集与训练数据污染风险。
- checkpoint/export smoke：确认训练状态和导出模型可加载。

SFT 评测：

- supervised loss：主训练指标。
- eval split loss：验证 SFT 是否过拟合或退化。
- posttrain report：按 split / family / length bucket 汇总。
- export/load smoke：确认最终模型目录可被本地 runtime 加载。

## Pretrain

入口：

- CLI: `ml-train`
- Module: `python -m ml.cli.train`
- 编排：`ml/tasks/pretrain/`
- runtime：`ml/training/pretrain/`

release pretrain 输入：

- token shards 数据目录
- tokenizer bundle
- local machine recipe JSON
- train/val/test manifest

默认 release 语义：

- `target_tokens_per_update = 270336`
- `learning_rate = 1e-4`(BF16 基线;任何调整都必须重新通过目标机 ≥800 步 stability probe 门禁)
- `weight_decay = 0.05`
- `beta1 = 0.9`
- `beta2 = 0.95`
- `adam_eps = 1e-6`
- `warmup_steps = 800`
- `lr_schedule = "wsd"`
- `wsd_stable_ratio = 0.9`
- `min_lr_ratio = 0.1`
- `embedding_lr_scale = 0.25`
- `ema_decay = 0.0`
- `max_grad_norm = 1.0`
- `grad_clip_mode = "agc"`
- `agc_clip = 0.01`
- `agc_eps = 1e-3`
- `agc_exclude_bias_and_norm = True`
- `muon_ns_steps = 4`
- `muon_target_rms = 0.18`

RTX 5090 BF16 recipe：

- recipe: `configs/pretrain/machine_recipes/rtx5090_bf16_seq4096_bs3_acc22.json`
- code default: `canonical_pretrain_machine_recipe_path()`
- selected name: `release_pretrain_rtx5090_bf16_sdpa_flash_liger_inductor`
- device: `NVIDIA GeForce RTX 5090`
- capability: `[12, 0]`
- fixed runtime: PyTorch SDPA Flash attention, torch linears, Liger fused linear CE
- batch: `batch_size=3`, `accumulation_steps=22`, `seq_len=4096`
- update budget: `target_tokens_per_update=270336`
- machine runtime: `step_execution_backend=inductor`
- data loader runtime: `num_workers=0`, `prefetch_factor=0`, `persistent_workers=0`
- shard preload: `shard_preload=1`, `shard_preload_bytes=4194304`
- gradient checkpointing disabled
- checkpoint saves enabled

Recipe 固化边界：

- `ml-train` 默认使用固定 RTX 5090 BF16 recipe。
- release preflight 会校验 recipe machine signature 与当前 machine signature；显存只允许极小浮点容差。
- release preflight 会拒绝非 CUDA 或 machine signature 不匹配的机器。
- 目标机变更时必须重新生成并签入新的 machine recipe，不应复用 5090 测量结果。

Release pretrain 会校验：

- machine recipe 与 machine signature
- canonical CUDA recipe 不变量（仅显式使用 CUDA recipe 时）
- tokenizer 和 manifest 指纹
- 输出目录与 resume 语义
- contamination scan
- 磁盘空间

本地/远程 GPU probe：

```bash
bash tools/pretrain_gpu_probe.sh
```

常用覆盖项：

```bash
SOPHIA_REMOTE_ROOT=/root/autodl-tmp \
SOPHIA_SDPA_TIMED_ITERS=12 \
bash tools/pretrain_gpu_probe.sh
```

RTX 5090 正式训练前硬门槛：

- PyTorch/CUDA 与本地预飞栈一致：`torch==2.8.0+cu128`、CUDA runtime `12.8`。
- 先在目标 RTX 5090 机器上跑 GPU probe 和 release preflight。
- 通过 BF16 stability probe、checkpoint save、`auto` resume 和关键回归测试后，才允许开长跑。
- 正式 checkpoint 和 staging 目录必须在 Linux 原生盘。

RTX 5090 预飞示例：

```bash
bash tools/pretrain_gpu_probe.sh
```

本地启动示例：

```bash
mkdir -p runs
PYTHONPATH=. \
nohup python3 -m ml.cli.train \
  --data_path dataset/pretrain_tokens/train \
  --tokenizer_path ml/modeling/text \
  --output_dir runs/pretrain_rtx5090_bf16 \
  > runs/pretrain_rtx5090_bf16.launch.log 2>&1 &
```

目标 RTX 5090 机启动示例：

```bash
PYTHONPATH=. python3 -m ml.cli.train \
  --data_path dataset/pretrain_tokens/train \
  --tokenizer_path ml/modeling/text \
  --output_dir runs/pretrain_rtx5090_bf16 \
  --machine_recipe_json configs/pretrain/machine_recipes/rtx5090_bf16_seq4096_bs3_acc22.json \
  --decay_data_path dataset/pretrain_decay/train
```

## WSD Decay 段数据切换

release pretrain 支持在 WSD 进入 decay 段时切换到高质量数据子集：

- CLI: `--decay_data_path`(数据集根目录或 train split 目录;留空 = 全程用 `--data_path`)。
- decay 起点由调度器同一公式推导(`warmup + stable`),stage planner 在该步拆分训练阶段并切换 manifest。
- decay 数据用 mix 管线 `--preset pretrain_decay` 构建(1.6B 配额,books/wiki/math 加权),再做块级交织。
- release gate 会把 decay manifest sha1 记入 data signature;resume 时校验指纹一致。

## Post-hoc Checkpoint EMA

训练结束后对保存的 checkpoint 做零训练成本的权重平均：

```bash
PYTHONPATH=. python3 -m ml.tooling.scripts.posthoc_checkpoint_ema \
  --ckpt_dir <run>/checkpoints --last 5 --mode ema --decay 0.7 \
  --output <run>/posthoc_ema.pt
```

输出是标准 checkpoint payload(`{"model": ...}`),可直接走既有 export 链路。训练期 `ema_decay` 保持 `0.0`。

注意:启动日志必须写在 `output_dir` 之外。`output_dir` 必须是空目录(或已被 Sophia 标记的 run 目录);预先往里面写 `train.log` 会被目录检查拒绝。

## Training Lifecycle

完整训练生命周期：

1. 准备 tokenizer bundle。
2. 构建 pretrain token shards。
3. 审计 token shards、机器后端、精度、Inductor 状态。
4. 验证当前 machine recipe。
5. 启动 release pretrain。
6. pretrain 过程中写 checkpoint、metrics、external eval report、contamination report。
7. 导出 pretrain model。
8. 启动 SFT，写 SFT report 和 export。
9. 执行 release audit。

阶段产物：

| 阶段 | 关键产物 |
| --- | --- |
| pretrain | checkpoint、metrics、train log、external eval、contamination scan |
| pretrain export | local model directory、tokenizer bundle、remote-code export files |
| SFT | posttrain report、export |
| release audit | audit JSON、错误原因、缺失 artifact 列表 |

逐阶段细节：

| 阶段 | 输入 | 训练/处理目标 | 评测方式 | 必需产物 | 失败边界 |
| --- | --- | --- | --- | --- | --- |
| tokenizer | tokenizer bundle | 固定词表、special tokens、chat template | bundle sha1 | `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | 改 tokenizer 后未重建 shards |
| pretrain shard build | raw text/jsonl, tokenizer | 离线编码 token shards | manifest/tokenizer fingerprint | train/val/test manifests, shard bin files | tokenizer 指纹不一致、split 缺失 |
| pretrain preflight | shards, canonical CUDA recipe, output dir | 校验机器、数据、resume、磁盘、外部 eval 输入 | release gate | machine recipe artifacts, data signature | 非 CUDA、显存不足、非 canonical recipe、manifest 漂移 |
| pretrain train | token shards, decoder config | causal LM next-token prediction | train loss, val loss, external eval subset, contamination scan | checkpoint, metrics, train log | loss 发散、checkpoint 写入失败、eval split 缺失 |
| pretrain finalize | latest/best checkpoint | 确认可导出状态 | test loss, checkpoint audit | final checkpoint | checkpoint 不可恢复 |
| pretrain export | checkpoint, tokenizer | 生成本地模型目录 | load/export smoke, local eval | `export/`, config, tokenizer, weights | export 缺 tokenizer 或 config 不匹配 |
| SFT train | pretrain export, SFT chat jsonl | 学习格式、语气、指令遵循、对话行为 | supervised loss, eval split loss | posttrain report, checkpoint/export | 数据格式错、长度裁剪错、loss 退化 |
| SFT finalize/export | SFT state | 导出 SFT 模型并写报告 | report completeness, load smoke | `export/`, posttrain report | export 不可加载、report 缺 split 指标 |
| release audit | 全部 artifacts | 验证工程证据链完整 | audit JSON | release audit report | artifact 缺失、证据含义混淆 |

重要边界：

- release audit 只证明工程证据链完整，不等价于模型能力达标。

## Same-Chain Validation

本地同链路验证入口：

- CLI: `ml-pretrain-check`
- Module: `python -m ml.cli.pretrain_check`

用途：

- 不替代 release pretrain。
- 用于快速验证数据读取、模型 forward/backward、优化器、checkpoint、export 是否仍在同一链路内工作。

示例：

```bash
PYTHONPATH=. python3 -m ml.cli.pretrain_check \
  --data_path dataset/pretrain_tokens \
  --device cuda:0
```

## Optimizer

训练优化器统一通过：

- `ml/training/runtime_tools.py`
- `ml/training/pretrain/optimizer.py`

策略：

- 2D transformer matrix weights 使用 Muon。
- embeddings、lm_head、bias、norm 和其它非 Muon 参数使用 fused AdamW。
- 模型参数可为 bf16 compute view。
- 优化器持有 fp32 master weights。
- 每步将 bf16 参数梯度同步到 fp32 masters，更新 fp32 masters，再写回模型参数。

这样做的原因：

- release LR 下单步更新可能小于 bf16 半 ULP。
- 如果直接更新 bf16 参数，小更新可能被舍入掉。
- fp32 master weights 保证 Muon/AdamW 的小幅更新不会消失。

CUDA 要求：

- 训练 optimizer 创建要求 CUDA 可用。
- PyTorch 必须支持 `torch.optim.AdamW(..., fused=True)`。

## Gradient Scaling And Clipping

裁剪实现：

- `ml/training/pretrain/engine/grad_clip.py`
- `ml/core/engine/session_batches.py`

支持模式：

- `agc`: adaptive gradient clipping。
- `norm` / `global_norm`: global norm clipping。
- `hybrid_auto` / `optimizer`: 使用 optimizer 自带 `clip_gradients()`。
- `none`: 不裁剪。

release pretrain pinned policy：

- `PINNED_GRAD_CLIP_MODE = "agc"`
- `PINNED_AGC_CLIP = 0.01`
- `PINNED_AGC_EPS = 1e-3`
- `PINNED_AGC_EXCLUDE_BIAS_AND_NORM = True`

Muon hybrid optimizer 的 `clip_gradients()` 会：

- 对 Muon 参数执行 global norm clip。
- 对 AdamW 参数执行 AGC。
- 返回组合后的 pre-clip grad norm。

## Scheduler And EMA

学习率调度：

- release pretrain 默认 `wsd`。
- warmup: `warmup_steps = 800`
- stable ratio: `wsd_stable_ratio = 0.9`
- decay style: `cosine`
- min LR ratio: `0.1`

EMA：

- release pretrain 当前 `ema_decay = 0.0`，默认不导出 EMA。
- validation profile 可使用非零 EMA。
- EMA 逻辑位于 `ml/training/ema.py`。

## Precision And Runtime Backends

release pretrain 当前精度策略：

- base precision: bf16
- inner linear backend: torch
- attention backend: PyTorch SDPA flash
- loss backend: torch CE
- step execution backend: Inductor

对应配置：

- `RELEASE_PRETRAIN_PRECISION_POLICY`
- `RELEASE_STEP_EXECUTION_POLICY`
- `CANONICAL_PRETRAIN_MACHINE_RECIPE_*`
- CUDA machine recipe JSON

边界：

- `attn_backend=flash` 是 release recipe 固化语义，不应在 release 命令行重复传不同后端。
- `linear_backend=torch` 是当前 RTX 5090 生产路径。
- `loss_backend=torch` 是 release recipe 固化语义。
- release 精度策略必须经过目标机器验证。

## Post-Train

后训练由 SFT 组成：

- SFT：用监督对话数据学习格式、语气、指令遵循和基础对话行为。

共同输入：

- 上一阶段 export 出来的本地模型目录。
- 同一 tokenizer bundle。
- post-train chat jsonl。
- 长度课程配置。

共同产物：

- `export/`：下一阶段或最终 eval 使用的模型目录。
- posttrain report：split、family、length bucket、loss 等指标。

共同边界：

- 后训练不改变 tokenizer。
- 后训练不改变 decoder 架构。

## SFT

入口：

- CLI: `ml-sft`
- Module: `python -m ml.cli.sft`
- 实现：`ml/tasks/sft/`

SFT 从本地 export 读取模型和 tokenizer，不依赖远程模型仓库。

默认 release 参数：

- `curriculum_stage = "sft_core"`
- `target_examples_per_update = 16`
- `learning_rate = 5e-6`
- `weight_decay = 0.1`
- `max_steps = 1000`

长样本阶段由课程文件描述：

- `stage = "sft_long_tail"`
- `selection = "long_examples_only"`
- `crop_policy = "preserve_final_turn"`
- `recommended_share_of_updates = 0.08`

启动示例：

```bash
PYTHONPATH=. python3 -m ml.cli.sft \
  --export_dir /path/to/pretrain_export \
  --train_data /path/to/sft_train.jsonl \
  --eval_data /path/to/sft_eval.jsonl \
  --output_dir /path/to/sft_run
```

SFT finalize 会：

- 导出 `export/` 模型目录。
- 写 posttrain report。

## Export And Eval

训练和本地评估使用 `SophiaDecoder`：

- `ml/modeling/sophia_decoder.py`

它封装：

- runtime Transformer
- training loss
- cache decode API
- `save_pretrained()` / `from_pretrained()`
- export bundle 加载

本地 eval / chat：

- CLI: `ml-eval`
- Module: `python -m ml.cli.eval`

行为：

- 自动从 `out/` 搜索本地 export。
- 支持 KV cache。
- 支持 `ask` / `auto` / `manual` 模式。
- 会按上下文窗口裁剪 prompt 和 generation budget。

## Data

数据面分三类：

- pretrain raw text / jsonl: 原始语料输入，不直接进入训练 loop。
- pretrain token shards: 离线 tokenizer 编码后的训练输入，是 release pretrain 的唯一训练数据形态。
- post-train chat jsonl: SFT 使用的多轮对话样本和报告元数据。

pretrain split 约定：

- `train/manifest.json`
- `val/manifest.json`
- `test/manifest.json`
- 每个 manifest 指向同目录下的 token shard 二进制文件。
- `data_path` 可以传数据集根目录，也可以传 `train/` split 目录；runtime 会解析 sibling val/test。

manifest 必须包含：

- shard 路径和 token 数。
- tokenizer 指纹。
- token dtype。
- EOS token 信息。
- 总 token 数。

release gate 数据校验：

- 训练 manifest sha1。
- val manifest sha1。
- test manifest sha1。
- tokenizer bundle sha1。
- tokenizer path。
- data path / eval path / test path。
- run kind。

污染与外部 eval：

- pretrain 过程会写 contamination report。
- external eval 当前用于 contamination scan。

数据修改边界：

- 改 tokenizer 必须重建 token shards。
- 改 raw data 必须重建 token shards 和 manifest。
- 改 split 路径必须重新签 recipe 或确认 data signature 一致。

## Tokenizer

默认 tokenizer bundle 位于：

- `ml/modeling/text/`

训练要求：

- pretrain token shards 必须绑定 tokenizer 指纹。
- release gate 会校验 manifest/tokenizer 指纹，防止数据与 tokenizer 漂移。
- 当前训练窗口为 4096。
- 更换 tokenizer 等价于改变训练数据编码分布；除非确认重训，否则不要替换。

常见风险：

- token shards 与 tokenizer bundle 指纹不一致。
- 本地 export 漏带 tokenizer bundle。
- post-train 数据与 tokenizer special tokens 不一致。
- 手动改 `tokenizer.json` 但没有重建 shards 和 manifest。

### Pretrain Token Shards

pretrain 使用离线 token shards。

入口：

- `ml-shard`
- `python -m ml.cli.shard_builder`
- `ml/data/token_shards/token_shards_dataset.py`

约定：

- `data_path` 可以是数据集根目录或 `train/` split 目录。
- 训练时要求能解析 train / val / test manifests。
- manifest 绑定 tokenizer 指纹。

示例：

```bash
PYTHONPATH=. python3 -m ml.cli.shard_builder \
  --tokenizer_path ml/modeling/text \
  --out_dir /path/to/token_shards \
  --jsonl /path/to/data.jsonl
```

### Post-Train Data

post-train 使用长度课程和规范化后的 chat 样本读取链。

相关实现：

- `ml/training/posttrain/data.py`
- `ml/training/posttrain/curriculum.py`
- `ml/tooling/scripts/data/production/`

默认课程文件：

- `dataset/posttrain_length_curriculum.json`
- 该文件由当前 `dataset/sft` 生成；训练前用 `PYTHONPATH=. python3 -m ml.tooling.scripts.data.production.write_posttrain_length_curriculum --sft_dir dataset/sft --output dataset/posttrain_length_curriculum.json` 刷新。

post-train 样本关键字段：

- `messages` / `conversations`: 多轮对话内容。
- `metadata`: family、bucket、source、reference 等训练和报告字段。
- `reference_answer`: 参考评测类指标使用。

人格语义：

- post-train 数据不使用 `system` persona；样本从 `user` 开始。
- 人格来自 assistant 回复分布，不来自多套 prompt 前缀。
- 数据使用统一的 Sophia 身份和回答风格口径。

裁剪语义：

- SFT 长样本阶段使用 `crop_policy = "preserve_final_turn"`。
- 本地 eval 会按上下文窗口裁剪 prompt 和 generation budget。
- 当前训练窗口是 4096。

## Checkpoint, Resume And Artifacts

checkpoint 体系覆盖：

- 模型权重。
- optimizer state。
- scheduler / step state。
- RNG / resume 相关元数据。
- machine signature。

resume 边界：

- release pretrain 不允许随意从不匹配 machine recipe 的 checkpoint 恢复。
- resume 会校验 checkpoint machine signature。
- optimizer state 会移动到目标 device。
- async checkpoint 打开时，要等待后台写入完成后再进行手工迁移或删除。

常见 artifact：

| artifact | 作用 |
| --- | --- |
| `run_args.json` | 训练参数记录 |
| `machine_signature.json` | 机器和后端签名 |
| `checkpoint-*` | 可 resume 训练状态 |
| `export/` | 后续 SFT/eval 使用的本地模型目录 |
| posttrain report | SFT split 指标、by-family 指标、loss 细节 |

删除规则：

- 不要删除当前 run 的最新 checkpoint，除非确认 export 和 release audit 已完成。
- 不要删除 machine recipe 和 signature 证据。
- 可以清理训练链路之外的临时产物。

## Metrics And Logging

pretrain 主要指标：

- train loss
- eval loss
- z-loss
- grad norm
- LR
- throughput / tokens per second
- optimizer diagnostics
- contamination safe rate

SFT 主要指标：

- supervised loss
- token count
- split report
- by-family report

日志边界：

- TensorBoard 在 release training 中是必需能力。
- JSON report 是 release audit 的主要输入。
- stdout/stderr 日志用于快速排查，但不应作为唯一证据。

## Tooling And Release Gate

统一工具入口：

- `ml-tool`
- `python -m ml.tooling.cli`

主要工具：

- `audit token-shards`
- `audit resume`
- `audit soak`
- `audit release`
- `pretrain check`
- `posttrain select`
- `posttrain export`
- `posttrain profile`
- `posttrain curriculum`

正式开训前建议：

```bash
PYTHONPATH=. python3 -m ml.tooling.cli pretrain check --device cuda:0
PYTHONPATH=. python3 -m ml.tooling.cli audit token-shards --manifest dataset/pretrain_tokens/train/manifest.json
```

release audit 关注：

- pretrain/SFT artifact 是否齐全。
- machine recipe 和 machine signature 是否匹配。
- release pretrain 是否运行在 canonical CUDA recipe 上。
- checkpoint/resume/soak evidence 是否存在。
- posttrain report 是否存在。

release gate 输入证据：

- `run_args.json`
- `machine_signature.json`
- `machine_recipe.json`
- `machine_runtime.json`
- `update_profile.json`
- checkpoint / resume metadata
- tokenizer bundle 和 manifest 指纹
- external eval report
- contamination report
- posttrain report

验收边界：

- pretrain external eval 只证明 contamination scan 状态。
- release audit 通过不等价于模型质量达标；它证明工程证据链完整。

## Training Readiness Checklist

无 GPU 或训练前可完成：

- `python3 -m ruff check ml tests`
- `python3 -m pytest -q`
- 检查 `git status --short`，确认没有未知关键改动。
- 检查 tokenizer bundle 是否存在。
- 检查 token shards manifest 是否存在。
- 运行 release audit，确认缺口列表可解释。

有 GPU 后第一步：

- 审计 CUDA/PyTorch/Inductor/precision。
- 跑 same-chain validation。
- 验证 canonical CUDA machine recipe。
- 跑短 resume drill。
- 跑短 soak。
- 再进入正式 pretrain。

正式训练前不得跳过：

- tokenizer 与 shards 指纹一致性。
- machine recipe 签名。
- output_dir 清洁或 resume 语义明确。
- 目标机器磁盘空间。
- checkpoint 写入路径和 staging 路径。

训练完成后：

- 确认最后 checkpoint 和 export 都存在。
- 跑 release audit。
- 最终报告必须区分“工程证据完整”和“模型质量达标”。

## Installation

要求：

- Python `>=3.12,<3.13`
- Linux + CUDA 环境
- 先安装匹配机器的 CUDA PyTorch wheel，再安装项目依赖

安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install --upgrade pip --break-system-packages
python3.12 -m pip install -e ".[dev]" --break-system-packages
```

核心依赖见 `pyproject.toml`，包括：

- `Python >=3.12,<3.13`
- `torch==2.8.0` with CUDA 12.8 wheels
- `transformers==5.10.2`
- `accelerate>=1.11.0,<2.0.0`
- `tokenizers>=0.22.1,<0.23.0`
- `datasets>=4.0.0,<5.0.0`
- `pyarrow>=23.0.0,<25.0.0`

## Common Commands

查看帮助：

```bash
PYTHONPATH=. python3 -m ml.cli.train --help
PYTHONPATH=. python3 -m ml.cli.eval --help
PYTHONPATH=. python3 -m ml.cli.sft --help
PYTHONPATH=. python3 -m ml.cli.rehearsal --help
PYTHONPATH=. python3 -m ml.tooling.cli --help
```

本地同链路 rehearsal：

```bash
PYTHONPATH=. python3 -m ml.cli.rehearsal \
  --device cuda:0
```

本地推理：

```bash
PYTHONPATH=. python3 -m ml.cli.eval \
  --export_dir /path/to/export_dir \
  --device cuda:0 \
  --mode ask
```

## Repository Layout

```text
ml/
  cli/           命令行入口
  core/          canonical model / schedule specs、通用训练 session / checkpoint / artifacts
  data/          token shard 数据面
  integrations/  HF 适配层、本地模型目录和 remote-code 导出
  modeling/      SophiaDecoder、config、tokenizer bundle、loss API
  runtime/       self-contained runtime model stack（含本地 eval / inference）
  tasks/         pretrain、sft orchestration
  tooling/       审计、release gate、machine recipe、数据脚本
  training/      优化器、EMA、pretrain/posttrain runtime
tests/
configs/
dataset/
tools/          GPU 健康检查、训练监控与实时观测面板
```

## Verification

最小本地验证：

```bash
pytest -q tests/tests/test_config_canonical.py
pytest -q tests/tests/test_pretrain_smoke.py
pytest -q tests/tests/test_cli_entrypoints.py
pytest -q tests/tests/test_runtime_import_boundary.py
```

完整验证：

```bash
python3 -m ruff check ml tests
python3 -m pytest -q
```

当前工程边界由测试守护：

- config 单一真源
- runtime import boundary
- pretrain smoke
- optimizer 变体
- CLI entrypoints
- pretrain / posttrain release gate
- token shard / resume 语义

## Current Status

当前主线状态：

- 模型主线已统一为 decoder。
- release pretrain 已固化到 canonical CUDA recipe。
- pretrain、validation、SFT、export、eval 都有明确入口。
- machine recipe、release gate、audit 工具已接入主流程。

仍需真实目标机器验证：

- 长时间 GPU soak。
- resume drill。
- 正式 recipe 完整训练稳定性。
- 真实数据规模下的吞吐、显存和收敛稳定性。
