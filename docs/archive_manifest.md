# 训练机资产归档记录(已并入仓库)

来源:`root@connect.westb.seetacloud.com:/root/autodl-tmp/sophia`(seetacloud 5090 训练实例,已退役)
同步完成:2026-09-14 04:51(并发分块传输 + 监督重试)
校验:4,374/4,374 文件尺寸逐文件一致,0 缺失 0 错位
关键哈希:`runs/pretrain_e3bc2d5_bf16_qkclip/checkpoints/ckpt_step15259.pt`
sha256 = `9f55fba52cae04c9cf9f27f9e44dafd2c3cfaf265c03bab4e21c7ac6f4b55aa0` ✓(与远端 sidecar 一致)

**2026-09-14 二次整理**:全部内容归位到仓库标准路径,删除纯重复品 ~105G
(80G tokenized 数据与 `dataset/pretrain/` 完全同一构建 + 25.3G 冗余 ckpt/导出)。
`archive/` 目录已不存在——以下即资产的当前位置。

## 资产位置(原远端路径 → 仓库路径)

| 仓库路径 | 内容 | 大小 |
|---|---|---|
| `runs/` | checkpoint 链(去重后):pretrain ckpt_step15259(8.9G 基座)+ model.safetensors、sft/ckpt_final、sft_export_e3/、export_dpo_e3/(HF 交付)、rft_e3.pt、grpo/ckpt_round1.pt、全部 metrics | ~22G |
| `dataset/pretrain/` | 20B tokenized 数据(879 shard+lineage)——**仓库原有,与远端 output/ 同一构建**(manifest 逐字节相同,远端副本已删) | ~80G |
| `ops/` | 全部评测证据:rl 池/rollout/pairs、eval_gate 探针 | 1.3G |
| `dataset/` | 数据本体+审计记录(benchmarks、source_audit、acquisition/inventory 清单并入) | ~81G |
| `evidence/` | 评测证据 | 13M |
| `sft_target_prep/` | 机器 recipe 测量工件 | 4M |

## 已去重(删除的重复品,均有等价物)

| 删除项 | 等价物 |
|---|---|
| `output/pretrain_tokens/`(80G) | 仓库 `dataset/pretrain/`(manifest+879 文件逐字节相同) |
| `runs/sft/ckpt_epoch{1,2,3,latest}.pt`(8.9G) | `ckpt_final.pt` 保留;.sha256 留档 |
| `runs/dpo_e3.pt`(2.0G) | `release/sophia/sophia.pt`(sha256 已验) |
| `runs/dpo_e3/model.safetensors`(2.2G) | `runs/export_dpo_e3/model.safetensors` |
| `runs/grpo/export_r1/model.safetensors`(2.2G) | `runs/grpo/ckpt_round1.pt` |
| `sft_target_prep/.../ckpt_step4.pt`(8.7G) | 测量探针工件,.sha256 留档 |
| `project_27b4ec2/` worktree | 退役工具随仓库删除;跟踪文件=本仓库 |

## 刻意未含(可重建或已在别处)

- `venv/`(python 环境,由 requirements.txt 重建)
- `*/.git/`(git 历史 = 本仓库)
- `release/`(发布包 = `release/sophia`,权重已验 sha256)
- `__pycache__/`、`.pytest_cache/`、`.ruff_cache/`

## 定位

训练证据链全在仓库标准位置,均已被 .gitignore(不随 git 分发)。
谱系引用(`release/sophia/lineage.json` 的 `runs/...` 键)原地可解析。
