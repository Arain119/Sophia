# Sophia 超参数证据

更新：2026-08-18

本文件记录正式 seed-42 模型中每个会影响参数、训练轨迹或 GPU 预算的固定值。证据分为四类：公式推导、论文或参考实现迁移、结构/预算约束、目标机测量。迁移值不冒充 Sophia 上的最优性实验；本项目的目标是训练一个固定模型，因此不再安排多 seed、AdamW 对照或消融。

## 训练合同

| 项目 | 值 | 依据 |
| --- | ---: | --- |
| 总参数 | 1,012,630,480 | meta-device 实例化后的 unique parameter count |
| embedding 参数 | 100,663,296 | `65,536 x 1,536`，且 input/output tied |
| non-embedding 参数 | 911,967,184 | 总参数减 embedding 参数 |
| 总 token | 20B | Chinchilla 的约 20 tokens/parameter 预算先验；对本模型为 19.75 tokens/parameter |
| tokens/update | 1,310,720 | Moonlight 822M 行 `160 x 8192` 的 4K token-equivalent batch |
| updates | 15,259 | `ceil(20B / 1,310,720)` |
| peak LR | 8.825e-4 | Moonlight Table 2 中最接近 911.97M non-embedding 参数的 822M 行 |
| warmup | 153 updates | Kimi K3 的 1% linear warmup；`ceil(15,259 x 0.01)` |
| scheduler | cosine to 0.1 peak | Moonlight 正式训练 schedule 迁移；单一路径，无 WSD/linear 分支 |
| weight decay | 0.1 | Moonlight 正式 Muon 训练设置 |
| seed | 42 | 用户固定；不用于跨 seed 统计 |

Moonlight 的 batch 是按 token 数迁移，不按 examples/update 生搬：

```text
160 x 8192 = 1,310,720 = 320 x 4096
```

token contract 允许按 token 数重新分解 `batch_size x accumulation_steps`，但当前唯一签名的 SM120 graph 实现固定为 B1/320；在 graph runner 支持新形状并完成新的目标机测量前，不得提交其它分解。机器测量不得缩放 semantic LR。

## 架构

| 项目 | 值 | 决定依据 |
| --- | ---: | --- |
| vocab | 65,536 | 已冻结 tokenizer bundle 的实际词表；bundle SHA-1 `abfcdb43c161a541e08ffb2cbc3794cd2fdc189f` |
| width/layers/FFN | 1536 / 28 / 3968 | 在约 1B 参数、4K、单卡 32GB 约束下的固定整数 shape；不是能力最优性声明 |
| mixer | `[KDA,KDA,KDA,MLA] x 7` | Kimi Linear 3:1 KDA:MLA 质量/吞吐选择，缩放到 28 层 |
| heads/head dim | 16 / 128 | KDA head dim 128 的参考实现迁移；16 heads 给出 2048 维 KDA Q/K/V 投影 |
| MLA ranks | Q 384 / KV 128 | Kimi MLA 系列的低秩结构迁移，并受约 1B 参数预算约束 |
| KDA ranks | decay 128 / output gate full rank | KDA 参考结构；full-rank output gate 避免额外未验证瓶颈 |
| ShortConv | 4 | Kimi Linear KDA 配置 |
| AttnRes block | 4 layers | 与四层 hybrid 周期严格对齐，产生 7 个 residual blocks；K3 约 8 blocks 的深度尺度被保留到最近可整除结构 |
| SiTU softcap | 4 / 25 | Kimi K3；`4*tanh(g/4)*sigmoid(g)` 与 `25*tanh(u/25)`，乘积绝对上界 100 |
| context | 4096 | 正式数据 packing、显存和训练预算合同 |
| position encoding | NoPE | Kimi K3 的 MLA 设计；顺序信息由 causal mask、KDA recurrence 与 ShortConv 提供 |
| RMSNorm epsilon | 1e-5 | Kimi/LLaMA 类 pre-norm decoder 的稳定数值配置迁移 |
| dropout | 0 | 大规模 decoder 预训练常用设置；20B token 本身提供数据随机性 |
| initializer | 0.02 | GPT/LLaMA/Kimi 类 decoder 的迁移先验；residual output 另按 `0.02/sqrt(2L)` 缩放 |

`1536/28/3968` 的严格含义是“满足模型规模、kernel 对齐和单卡预算的已冻结 shape”，不是通过未发生的实验证明“最优”。在不做消融的前提下，结构对齐比假造一个最优性分数更可靠。

## KDA 数值

| 项目 | 值 | 依据 |
| --- | ---: | --- |
| `g_min` | -5 | Kimi K3 对 BF16 tile 动态范围的推导 |
| `A_log` init | 0 | K3/FLA KDA 初始化 |
| `dt` | log-uniform `[1e-3,1e-1]` | FLA v0.5.2 KDA 实现 |
| `dt` floor | 1e-4 | FLA v0.5.2 KDA 实现 |
| recurrence state | FP32 | 避免 4K 累积递推的 BF16 误差放大 |
| training backend | FLA `chunk_kda` | CUDA 正式路径；reference recurrence 仅用于 CPU/parity 测试 |

FLA 固定为 v0.5.2 commit `9c8e42e762fce087c27b673af4922795d9edb85e`。KDA `A_log` 和 `dt_bias` 明确进入 no-weight-decay AdamW 组。

## Muon

| 项目 | 值 | 依据 |
| --- | ---: | --- |
| matrix domain | hidden 2D weights | Muon 官方参数分类；包含 KDA conv 和 AttnRes query，排除 embedding/output |
| momentum | 0.95 | Moonlight/Muon |
| Nesterov | true | Moonlight/Muon |
| NS iterations | 5 | Moonlight/Muon |
| NS coefficients | 3.4445, -4.7750, 2.0315 | Muon quintic Newton-Schulz iteration |
| shape scale | `0.2*sqrt(max(A,B))` | Moonlight Eq./implementation |
| master state | FP32 | BF16 参数更新的数值稳定与精确 resume |
| momentum state | BF16 | 跟随正式 BF16 Muon 参数路径；避免 FP32 master 改变原 `zeros_like(parameter)` 状态精度 |

Newton-Schulz 的实现固定为：

```text
A = X X^T
X <- aX + (bA + cA^2)X
```

Attention Residual query 是形状 `(1,1536)` 的秩一 Muon 参数。对送入 NS5 的
非零 Nesterov 有效矩阵 `g_eff`，Frobenius 归一化后的唯一奇异值为
`s0 = ||g_eff|| / (||g_eff|| + eps)`；当 `||g_eff|| >> 1e-7` 时
`s0 ~= 1`。令

```text
f(s) = 3.4445 s - 4.7750 s^3 + 2.0315 s^5
```

五次 NS 迭代给出 `f^5(1) = 0.696435...`，因此 Moonlight shape scale
作用后的 query update RMS 约为
`0.2 * 0.696435 * base_lr = 0.139287 * base_lr`，而不是精确的
`0.2 * base_lr`。这是固定 NS 多项式在秩一谱上的结果，不是额外的 query LR
超参；零有效矩阵仍产生零更新，`||g_eff||` 与 `eps` 同量级时应使用上面的 `s0`
公式。更高秩矩阵的实际比例取决于完整奇异值谱，不能只由矩阵 rank 推出。

embedding、output、norm、bias、`A_log` 和 `dt_bias` 不属于 Muon matrix domain，使用 AdamW。Adam betas `0.9/0.95`、epsilon `1e-8` 随 Moonlight hybrid 配置迁移；norm、bias、`A_log`、`dt_bias` 的 weight decay 为 0，其余 AdamW 与 Muon 参数 weight decay 为 0.1。

训练中没有 target-RMS normalization、AGC、global clipping、z-loss、EMA、layerwise LR decay 或 embedding LR scale。这些额外机制没有进入正式推导，因而已从实现和 checkpoint contract 删除。

### FP32 主梯度累积

正式 SM120 路径仍使用 BF16 模型参数和 BF16 compute。每个 micro-batch 的
backward 在 BF16 `p.grad` 中产生单次梯度；随后 graph 内立即将其加入该参数的
持久 FP32 `main_grad` 缓冲，并清空 BF16 `p.grad`。因此 32 次 backward、10 个
graph block 的 320 次累积都在 FP32 中完成，token normalization、global grad
norm 和 Muon/AdamW step 直接读取同一组 FP32 缓冲。FP32 optimizer master weight
是更新副本，和 FP32 `main_grad` 是两个独立的状态。

累积使用 `torch._foreach_add_` 与 `torch._foreach_zero_`，操作和静态缓冲均在
CUDA Graph capture 内，保持正式 B1/T4096/320 合同。该实现消除了此前记录的
BF16 320 次累积舍入路径；本地合成的 `0.021165` L2 差异不再是已接受的训练风险，
只保留为改动动机的证据。新增 4.05 GB（约 3.77 GiB）FP32 主梯度显存，并可能改变吞吐，
所以旧 machine recipe、stability probe 和 resume audit 均不可复用，必须在目标机
按当前代码重新测量和签名。

### K3 指定的 FP32 位置

K3 将 flash attention 输出保留在 FP32，以修正其 biased rounding error。正式 MLA 路径因此在 SDPA 输出与 output gate 相乘时使用 FP32，并在 `o_proj` 前转回模型 dtype。Attention Residual mixer 只在 score 与 softmax 中使用 FP32；加权求和沿用 source dtype，不再把完整 `[B,S,T,H]` values 副本提升到 FP32。

固定 seed 6 的 CPU BF16 小模型 parity 对照中，旧的宽 FP32 mixer/source-dtype MLA gate 与当前精度放置得到：loss `0.0366334542632` 对 `0.0366334542632`（绝对差 0），global grad norm `0.118832573295` 对 `0.118832409382`（相对差 `1.37936e-6`）。该测试固定在 `test_k3_fp32_scope_matches_wide_fp32_reference_bf16_cpu`。

### MLA attention-logit telemetry and QK-Clip

正式训练从 step 0 开启 MLA attention-logit telemetry 与 QK-Clip。SM120 路径用 Triton 融合 causal dot-product/max kernel，直接写入静态 device-side `[7,16]` 最大值缓冲，不物化 logits；参考路径以 512 个 query 为 tile，只计算 causal 下三角。每个 optimizer step 后按 head 对 MLA `q_up.weight`、`k_up.weight` 与 FP32 master 使用 `sqrt(min(1, 100 / S_max))` 缩放。阈值 `100.0` 采用 K2 的固定外部来源；当 `S_max <= 100` 时缩放为逐位恒等映射，不改变健康 head 的轨迹。该保护不是 loss/grad finite 检查的替代品，而是针对 MLA logit 增长的单边约束。


## NoPE 上下文边界

MLA 不再消费绝对位置编码；`start_pos` 仍用于 KDA recurrent state、MLA latent cache 和 causal bias。NoPE 消除了固定 position table 或 RoPE scaling 参数，但不推出未训练长度上的困惑度、检索或生成质量。Kimi K3 的长上下文能力来自 8K 到 64K 再到 1M 的实际训练课程；本项目只训练 4K，因此超过 4K 的质量需要训练后单独测量，不能标记为未经训练验证的 inference-only 能力。

NoPE 还使 MLA 的推理期权重吸收重新可行：`q^T k = (W_k^T q)^T c`。这是附带的推理优化机会，本次训练实现不做权重吸收。

## 与 Kimi K3 的已知偏差

K3 对 Q/K/V 使用 Per-Head Muon，本仓库保持 Moonlight 的全矩阵 Muon。Moonlight 的 peak LR `8.825e-4` 与 `0.2*sqrt(max(A,B))` shape scale 是一套来源；若只把 MLA `k_up/v_up` 切为 per-head，shape scale 会从 `0.2*sqrt(2048)` 变为 `0.2*sqrt(128)`，即约缩到四分之一，而本项目不做消融、无法重新标定。保持全矩阵 Muon 与 Moonlight LR/scale 成套迁移，比混搭两篇论文的优化器语义更自洽。

peak LR 与 tokens/update 来自 Moonlight Table 2 的 822M non-embedding dense 行，不来自 K3。K3 只说明这些量由 scaling-law study 确定，没有公开可迁移公式；因此这是整条超参推导链最弱的一环。warmup 则采用 K3 明示的 1% linear warmup：`ceil(15,259 * 0.01) = 153`，不再使用 Moonlight 33B/5.7T 的 0.58% token 比例。

## 目标机测量

目标 RTX 5090 使用唯一的 `sm120_graph` 实现。当前 runner 的运行时合同是 B1/T4096、accumulation=320，32-micro
CUDA Graph block。此前 commit `8603599` 在目标机测得稳态 step 2..3 为 16,979--17,011 tokens/s、捕获峰值 27.256 GiB、稳态 active allocation 13.932 GiB；该数据先于 NoPE、153-step warmup、收窄 FP32 和重捕获修复，旧 recipe/audit 已失效，只能作为性能基线。
此前 20,346--20,368 tokens/s 属于更早的 NoPE、accumulation=64、262,144 tokens/update 路径，同样不能用于当前训练 ETA。
关闭 telemetry、QK-Clip 的历史单变量测量为稳态 17,343 tokens/s；再将 Liger CE 的 CUDA logits
chunk 从启发式 `M=128` 改为完整 `T=4096` 后为 18,120 tokens/s。两次峰值均为 28.604 GiB、
稳态 active allocation 为 13.932 GiB。该数据只作为性能基线；恢复 telemetry/QK-Clip 后必须重新
测量并签名，不能把这两个关闭状态的吞吐用于正式 ETA。

CE chunk 的隔离 CUDA 基准（4095 tokens、1536 hidden、65536 vocab、BF16、同一 Liger kernel）为：
`M=128/512/1024/2048/4095` 分别 `22.32/13.19/12.55/12.59/11.93 ms`（forward+backward）。
因此完整 4K chunk 已接近该算子的实测上界，不能把 20k tokens/s 当作当前路径的可达保证。
32 是图的执行
分块，不是优化器 batch：正式一次 update 严格 replay 10 个 block，故
`10 * 32 * 1 * 4096 = 1,310,720` tokens，之后才做一次梯度归一化和 Muon step。

签名 recipe、154-step stability report 和 resume report 共同绑定仓库 `ml/**/*.py`
的路径分帧 SHA-256。任何生产 Python 实现变化都会使旧 recipe/audit 失效，必须在
目标机重新测量并完成 warmup-boundary resume drill；JSON bytes hash 与该实现指纹
仍是两个独立概念。

目标机 recipe 不再搜索训练配置，只复验这条固定路径的机器签名、可执行性、
峰值显存与稳态吞吐。当前实现还在 FFN backward 中从已保存的 `z` 重算
activation，以减少长期保存的 BF16 激活；它与保存 activation 的首步 loss/grad
一致，但实现指纹变化仍要求重新签名。测量固定执行 3 个 update：第 1 个包含
TorchInductor/Triton autotune 和 CUDA Graph capture，只记录为 cold start；第 2、3
个 update 的总 token / 总耗时才写入训练预算。这样不会把曾实测 48.12 秒的首次
编译成本误报为 5,448 tokens/s。

正式准备先运行 12-step graph 生命周期探针，在 step 10 eval 后于 step 11 重捕获，并记录该次运行的 CUDA 峰值；重捕获前必须临时 offload 已分配的 optimizer state。正式稳定性探针运行 steps 1..154，覆盖 153-step warmup 后的第一个 update；恢复探针从 `ckpt_step153.pt` 独立执行 step 154。通过条件是执行完成、绑定哈希一致、loss/grad norm/validation loss/吞吐为有限值。不存在人为 loss 上限、下降比例或能力分数阈值。

正式 release preflight 在输出所在文件系统上要求至少 80 GiB 可用空间。该下限由约
10.5 GiB 的单个完整 checkpoint、最近 6 份保留策略以及新 checkpoint 序列化和文件系统
余量共同确定；它是卡时保护，不是训练超参。完整 checkpoint 每 100 个 update 保存一次，
并保留最近 6 份；每个 update 在 optimizer/scheduler 更新后
检查模型参数和 optimizer 状态的有限性；Muon hybrid 同时检查 BF16 模型参数、FP32
main_grad、FP32 master、BF16 Muon momentum 以及 AdamW 的内部状态。发现非有限值会写入
`stability_incident.json` 并在下一次训练
动作前停止，最终 checkpoint 写入完成前也不会导出模型。

## 数据数值的边界

清洗阈值、语言比例和 source mix 属于已经物化并哈希冻结的数据资产，不在训练时动态决策。它们的证据位于 dataset `lineage/`、source audit 和 admission policy 中。训练只读取最终 token shards；任何数据策略变化都必须生成新的 manifest 和 lineage，不能在同一训练轨迹中热切换。

训练 iterator 的随机化单位是完整的 B1/T4096 窗口，而不是 shard。当前 train
manifest 产生 `4,882,460` 个完整窗口；seed 42 排列的首个 320-window update 覆盖
251 个 shard，单个 shard 最多贡献 3 个窗口。val 的首个固定 64-window evaluation
覆盖 43 个 shard，单个 shard 最多贡献 10 个窗口。排列以 CPU int32 tensor 保存，
当前 train cursor 为 19,529,840 bytes；checkpoint 恢复复用同一排列和下一个未读
window index，不改变 manifest、shard bytes 或其哈希。

## 主要来源

- Chinchilla scaling: arXiv:2203.15556
- Moonlight / Muon: arXiv:2502.16982
- Kimi Linear: arXiv:2510.26692
- Kimi K2 training stability / QK-Clip: arXiv:2511.21377
- Kimi K3: arXiv:2607.24653
- flash-linear-attention v0.5.2: commit `9c8e42e762fce087c27b673af4922795d9edb85e`
- NVIDIA/Megatron-LM BF16 gradient accumulation policy: commit `9e96f458b5cd7e6bd0871e368b359660b32a135c`, `megatron/training/arguments.py`
