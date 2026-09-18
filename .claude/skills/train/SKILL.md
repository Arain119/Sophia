---
name: train
description: 一键检查、启动或恢复 Sophia 的随机初始化预训练。
---

# Sophia 训练

正式训练固定为 Muon hybrid、seed 42、4K、20B tokens。

## Pretrain

```bash
export SOPHIA_DATA_PATH=dataset/pretrain
export SOPHIA_MACHINE_RECIPE=<target-prep>/recipe_measurement_muon/release_pretrain_machine_recipe.json
export SOPHIA_MUON_PROBE_AUDIT=<target-prep>/muon_probe_audit.json

bash tools/one_click_train.sh --dry-run
```

只有用户明确批准正式 20B-token 训练后，才运行：

```bash
bash tools/one_click_train.sh
```

恢复设置 `SOPHIA_RESUME=auto`、`latest` 或明确 checkpoint。

## 脚本行为

`tools/one_click_train.sh` 执行环境检查、数据验证、回归子集、GPU readiness、严格 preflight，然后后台启动。日志写入 `<output_dir>.launch.log`，PID 写入 `<output_dir>.pid`。

可覆盖变量：

| 变量 | 用途 |
| --- | --- |
| `SOPHIA_DATA_PATH` | dataset root 或 train split |
| `SOPHIA_TOKENIZER_PATH` | tokenizer bundle |
| `SOPHIA_MACHINE_RECIPE` | 当前目标机签名 recipe |
| `SOPHIA_MUON_PROBE_AUDIT` | pretrain 的 step-90/resume audit |
| `SOPHIA_OUTPUT_DIR` | 训练输出目录 |
| `SOPHIA_RESUME` | `auto`、`latest` 或 checkpoint |
| `SOPHIA_SKIP_TESTS=1` | 跳过已在同一代码副本通过的回归子集 |
| `SOPHIA_SKIP_PROBE=1` | 跳过同机同栈近期已通过的 GPU probe |

## 监控

```bash
python3 tools/train_monitor_loop.py --run-dir <output_dir> --launch-log <output_dir>.launch.log
python3 tools/live_train_dashboard.py --run-dir <output_dir>
```

机器签名、数据指纹、checkpoint hash 或训练状态失败时停止并修复根因。不要通过关闭验证、修改 manifest 或覆盖输出目录继续训练。
