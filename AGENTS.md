# Agent Guide

本文件面向在此仓库工作的 AI agent(Codex、Claude Code 及其它)。项目介绍、架构与
训练方法见 [README.md](README.md);Claude Code 用户可直接调用 `/train` 技能
(定义于 `.claude/skills/train/SKILL.md`,本节内容与其一致)。

## 项目速览

Sophia 是一个语言模型训练系统:约 0.83B 参数的 decoder causal LM,BF16 精度,
目标机为单张 RTX 5090。训练配置以签名后的 machine recipe 固化,数据以带 tokenizer
指纹的 token shards 组织,启动前有严格的 preflight 门禁。

## 一键训练

```bash
bash tools/fetch_dataset.sh                        # 首次:下载公开数据集(约 71 GB)
bash tools/one_click_train.sh pretrain --dry-run   # 先验证门禁与启动命令
bash tools/one_click_train.sh pretrain             # 预训练(后台长跑)
bash tools/one_click_train.sh sft                  # SFT 后训练
```

训练数据发布于 ModelScope 公开数据集
[Arain119/Sophia-dataset](https://www.modelscope.cn/datasets/Arain119/Sophia-dataset);
`fetch_dataset.sh` 先校验 manifest 的 tokenizer 指纹再下载 shard 大文件,支持
断点续传,完成后逐 shard 核对字节数。硬件请直接使用/租用单张 RTX 5090(32GB),
其它卡型会被 preflight 拒绝。

脚本自动完成:环境检查 → 数据资产校验 → 回归测试子集 → GPU readiness probe →
后台启动训练。启动日志在 `<output_dir>.launch.log`,PID 在 `<output_dir>.pid`。
完整流程为 pretrain → sft 两次调用:预训练的 export 直接落在其 `output_dir` 中,
SFT 阶段默认从该目录读取。

常用覆盖项(环境变量):`SOPHIA_DATA_PATH`、`SOPHIA_DECAY_DATA_PATH`、
`SOPHIA_OUTPUT_DIR`、`SOPHIA_EXPORT_DIR`、`SOPHIA_SFT_TRAIN_DATA`、
`SOPHIA_RESUME=auto`(断点恢复)、`SOPHIA_SKIP_TESTS=1`、`SOPHIA_SKIP_PROBE=1`。
完整说明见脚本头部注释。

启动后每 20~30 分钟检查一次:进程存活、launch log 尾部、loss 与 grad norm 走势。
持续巡检:`python3 tools/train_monitor_loop.py --run-dir <output_dir>`;
实时面板:`python3 tools/live_train_dashboard.py --run-dir <output_dir>`
(前端每 5 秒自动刷新,数据源为 run 目录内持久化的 `metrics.jsonl`)。
从零租显卡到部署开训的完整教程见 `.claude/skills/train/REMOTE_GPU.md`。

## 硬边界

- 机器签名不匹配时 preflight 拒绝启动是预期行为;正确做法是在新目标机上重新测量
  并签入新 recipe,而不是放宽校验或复用旧测量。
- checkpoint 与输出目录必须在 Linux 原生盘,禁止 `/mnt/*` Windows 挂载。
- 核心超参(如学习率)调整必须重新通过目标机 ≥800 步稳定性探针。
- tokenizer、token shards、manifest 指纹三者绑定;不一致时重建数据,不改校验。
- `--overwrite_output_dir` 是破坏性操作,须用户明确同意。

## 验证

```bash
python3 -m ruff check ml tests
python3 -m pytest -q
```
