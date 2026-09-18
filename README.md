# Sophia

本项目及公开模型权重按 Apache-2.0 发布，详见 [LICENSE](LICENSE) 和
[模型发布说明](docs/model_release.md)。训练数据不随仓库许可证重新授权；使用和再分发必须遵循每个数据源的 lineage 条款。

Python `>=3.12,<3.13`.

Sophia 是一套从预训练到监督后训练与受限偏好优化的语言模型训练系统。唯一正式 base 目标是在单张 RTX 5090 上，以 BF16、seed 42 和 Muon hybrid，从随机权重训练一个 1,012,630,480 参数的 4K causal LM；后训练只接受该正式 base checkpoint 的内容哈希，不从导出权重重新初始化。完成 SFT 后，可在固定行为验收不合格时运行受限 DPO/GRPO 纠偏，且不改变 base 与 SFT 的证据链。

## 固定训练合同

| 项目 | 固定值 |
| --- | ---: |
| 模型参数 | 1,012,630,480 |
| 非 embedding 参数 | 911,967,184 |
| 训练上下文 | 4,096 tokens |
| 训练 token | 20,000,000,000 |
| tokens/update | 1,310,720 |
| 更新步数 | 15,259 |
| 实际消费 token | 20,000,276,480 |
| seed | 42 |
| 精度 | BF16 compute，FP32 main-grad 与 optimizer master state |
| optimizer | Muon matrix domain + AdamW scalar/vector/embedding domain |
| peak LR | 8.825e-4 |
| weight decay | 0.1 |
| scheduler | 153-step warmup + cosine，终点为 peak LR 的 0.1 |

训练语义由 [muon_recipe.json](configs/pretrain/muon_recipe.json) 固定。目标机 recipe 只能确定 micro-batch、梯度累积、checkpointing、loss chunk 和执行后端，不能改动 optimizer、LR、scheduler、token budget 或 seed。

## 模型

模型是 28 层 dense Hybrid decoder，hidden size 1536，SiTU-GLU hidden size 3968。mixer 采用 `[KDA, KDA, KDA, Gated MLA] x 7`：

- KDA：16 heads x 128，ShortConv kernel 4，decay rank 128，`g_min=-5`，`dt=[1e-3,1e-1]`，floor `1e-4`，`A_log=0`。
- MLA：Q rank 384，KV rank 128，采用 Kimi K3 的 NoPE 设计。
- Attention Residuals：每个四层 KDA/MLA 周期构成一个 residual block，共七个 block。
- FFN：SiTU softcap `4/25`，RMSNorm epsilon `1e-5`，dropout 0。
- tied embedding/output：`65,536 x 1,536 = 100,663,296` 个唯一参数。

模型规范位于 [sophia.json](configs/model/sophia.json)，模型 semantic SHA-256 固定为：

```text
c605697ccdcd92b11095e04fe5a3cbad30d6388ec450568996c45db1afa4ba6a
```

### NoPE 与上下文

MLA 不注入显式位置编码；顺序信息由 causal mask、KDA 的递归衰减和 ShortConv 共同传递。`max_seq_len=4096` 仍是本次数据、训练和显存合同。NoPE 不自动保证长度外推；Kimi K3 的长上下文能力来自 8K 到 64K 再到 1M 的实际训练课程，因此超过 4K 的质量需要训练后单独测量。

## 超参推导

embedding 参数为 `65,536 x 1,536 = 100,663,296`，因此非 embedding 参数为：

```text
1,012,630,480 - 100,663,296 = 911,967,184
```

Moonlight/Muon Table 2 中最接近该规模的是 822M non-embedding、20.76B-token 行；固定迁移其 LR `8.825e-4` 和 batch `160 examples @ 8K`。按 token 数等价换算到 4K：

```text
160 x 8192 = 320 x 4096 = 1,310,720 tokens/update
ceil(20,000,000,000 / 1,310,720) = 15,259 steps
15,259 x 1,310,720 = 20,000,276,480 tokens
```

Kimi K3 使用总更新数 1% 的 linear warmup。对固定 15,259 updates：

```text
ceil(15,259 x 0.01) = 153 warmup steps
```

Muon 对 2D hidden weights 使用五次 Newton-Schulz 和 Moonlight shape LR：

```text
parameter_lr = 8.825e-4 x 0.2 x sqrt(max(fan_out, fan_in))
```

embedding、norm、bias、KDA `A_log/dt_bias` 等非 Muon 参数由 AdamW 更新；norm、bias、`A_log/dt_bias` 不做 weight decay。完整依据和仍属于迁移先验的项目见 [hyperparameter_evidence.md](docs/hyperparameter_evidence.md)。

## 数据合同

正式数据根目录是 `dataset/pretrain`：

```text
dataset/pretrain/
  dataset_manifest.json
  train/manifest.json
  val/manifest.json
  test/manifest.json
  lineage/
```

根 manifest 绑定 tokenizer、三个 split、lineage 和每个 shard 的 bytes/SHA-256，训练入口复验。正式训练只使用一个 train manifest。

训练 iterator 在每个 epoch 对所有完整 `seq_len` 窗口做一次全局 seed-42 permutation。checkpoint 保存窗口排列与 cursor，恢复要求 manifest 与 worker topology 一致。

## 目标机准备

先在目标 RTX 5090 上安装环境并测量机器 recipe：

```bash
export SOPHIA_DATA_PATH=dataset/pretrain
bash tools/prepare_pretrain_target_recipe.sh
```

`prepare_pretrain_target_recipe.sh` 在目标机上实测候选 recipe 并完成从 checkpoint 恢复的验证，产出 `release_pretrain_machine_recipe.json` 与 `muon_probe_audit.json`。

## 一键训练

先做不构造 optimizer、不启动训练的完整 preflight：

```bash
SOPHIA_DATA_PATH=dataset/pretrain \
SOPHIA_MACHINE_RECIPE=<target-prep>/recipe_measurement_muon/release_pretrain_machine_recipe.json \
SOPHIA_MUON_PROBE_AUDIT=<target-prep>/muon_probe_audit.json \
  bash tools/one_click_train.sh --dry-run
```

正式启动使用相同环境变量：

```bash
bash tools/one_click_train.sh
```

脚本依次检查数据资产、回归测试、GPU、签名 recipe 与恢复证据，然后 `nohup` 启动。日志为 `<output_dir>.launch.log`，PID 为 `<output_dir>.pid`。

## 监督后训练

公开推理接口保留完整的 `<think>...</think>` 文本。DPO/GRPO 的偏好样本只允许来自独立 rollout，并绑定起始 SFT 导出模型。

SFT 语料为 `release/corpus/`（88,267 train + 1,500 val），由 teacher 模型按行为脚本逐条生成并经机器校验。训练入口核对 parent 是 step 15,259、seed 42、20,000,276,480 tokens 的 full checkpoint。

训练语义由 [sophia_sft_muon_recipe.json](configs/sft/sophia_sft_muon_recipe.json) 固定：completion-only token-weighted loss、Muon hybrid、3 epochs、seed 42 和 MLA QK-Clip。机器 recipe 在 RTX 5090 上实测选定，工件随仓库提交于 `sft_target_prep/`。执行完整 dry-run：

```bash
SOPHIA_SFT_DATASET_DIR=release/corpus \
SOPHIA_SFT_PARENT_CHECKPOINT=/target/base.pt \
SOPHIA_SFT_OUTPUT_DIR=/target/runs/sft \
  bash tools/one_click_sft.sh --dry-run
```

去掉 `--dry-run` 才会启动正式 SFT。每个 epoch 写带 sha256 的 checkpoint；epoch 选择由 `ops/rank_epochs.py` 的判官排序决定。

## 监控与恢复

```bash
python3 tools/train_monitor_loop.py --run-dir <output_dir> --launch-log <output_dir>.launch.log
python3 tools/live_train_dashboard.py --run-dir <output_dir>
```

完整 checkpoint（model、optimizer、scheduler、RNG、data cursor）每 100 updates 保存一次并保留最近 6 份；release preflight 要求目标机至少 80 GiB 可用空间。

## 验证

```bash
python3 -m ruff check ml tests tools
python3 -m pytest -q
bash -n tools/one_click_train.sh
bash -n tools/prepare_pretrain_target_recipe.sh
bash -n tools/one_click_sft.sh
```

## 范围

仓库保留训练所需的数据构建、pretrain、checkpoint/resume、导出、推理和最终报告工具。机器测量是部署步骤，不是训练超参实验；最终模型固定为 seed 42 的最后一个完整 checkpoint，不做跨 seed 选择、optimizer 对照、消融或 Pareto checkpoint 挑选。
