---
name: train
description: 一键启动或恢复 Sophia 训练(预训练与 SFT 后训练)。自动串联环境检查、数据资产校验、回归测试、GPU 探针门禁,后台挂起长跑并持续看护。Use when asked to start, launch, one-click train, resume pretraining, or run SFT for Sophia.
---

# Sophia 一键训练

本技能把预训练与 SFT 后训练的完整门禁链和启动流程收敛为一条命令。执行核心是
`tools/one_click_train.sh`,技能负责在它前后补齐判断与看护。

## 前置条件

手头没有目标机时,先按 [REMOTE_GPU.md](REMOTE_GPU.md) 走完租卡到部署的完整
教程(选卡、镜像、SSH、环境、数据下载、远程观测、收尾取回产物),再回到本节。
**选卡时直接租单张 RTX 5090(32GB)**:发布 recipe 与该卡签名绑定,选其它卡型
会被 preflight 拒绝,不要用其它卡型替代。

启动前确认(脚本也会逐项校验,提前确认可少走弯路):

- Linux + CUDA 环境,Python ≥ 3.12,`torch==2.8.0+cu128` 可导入且
  `torch.cuda.is_available()`。
- 预训练阶段:`dataset/pretrain_tokens/{train,val,test}/manifest.json` 存在,
  tokenizer bundle 位于 `ml/modeling/text`。数据缺失时先走下节「数据获取」。
- SFT 阶段:预训练已完成且 export 落盘(config/tokenizer/weights 在预训练
  `output_dir` 中),`dataset/sft/train.jsonl` 存在。
- 目标机与签名 recipe 匹配(默认 recipe 面向 RTX 5090,CUDA capability 12.0)。
  机器不匹配时 preflight 会拒绝启动,这是预期行为,不是待修的 bug。

## 数据获取(首次训练)

训练数据发布于 ModelScope 公开数据集
[Arain119/Sophia-dataset](https://www.modelscope.cn/datasets/Arain119/Sophia-dataset),
`tools/fetch_dataset.sh` 一条命令完成下载与校验,断点续传,落位 `dataset/`:

```bash
bash tools/fetch_dataset.sh pretrain --manifests-only  # 预检:先验指纹再下大文件
bash tools/fetch_dataset.sh                            # 预训练 shards + SFT(约 71 GB)
bash tools/fetch_dataset.sh sft                        # 仅 SFT 数据(约 22 MB)
```

脚本分两段下载:先取各 split 的 `manifest.json` 并校验 tokenizer 指纹与本地
bundle 一致,通过后才开始 shard 大文件传输;完成后逐 shard 核对字节数。
云 GPU 机上直接运行即可(国内机房到 ModelScope 带宽通常远好于本地上行)。

## 执行步骤

1. 先做 dry-run,确认门禁全部通过、启动命令符合预期:

   ```bash
   bash tools/one_click_train.sh pretrain --dry-run
   ```

2. 正式启动(阶段省略时默认 pretrain):

   ```bash
   bash tools/one_click_train.sh pretrain   # 预训练
   bash tools/one_click_train.sh sft        # SFT 后训练
   ```

   脚本按序执行:环境检查 → 数据资产校验 → 回归测试子集 → GPU readiness
   probe → 后台启动训练。启动日志写在 `output_dir` 之外(`<output_dir>.launch.log`),
   PID 记录于 `<output_dir>.pid`。

3. 完整流程为 pretrain → sft 两次调用;预训练的 export 直接落在其 `output_dir`
   中,SFT 阶段默认从该目录读取(`SOPHIA_EXPORT_DIR` 可覆盖)。

4. 常用覆盖项(全部经环境变量传入):

   | 变量 | 适用阶段 | 含义 | 默认值 |
   | --- | --- | --- | --- |
   | `SOPHIA_OUTPUT_DIR` | 两者 | run 输出目录 | `runs/pretrain_rtx5090_bf16` / `runs/sft_rtx5090_bf16` |
   | `SOPHIA_RESUME` | 两者 | `auto` / `latest` / checkpoint 路径 | 空(全新训练) |
   | `SOPHIA_DATA_PATH` | pretrain | 训练数据(root 或 train split) | `dataset/pretrain_tokens/train` |
   | `SOPHIA_DECAY_DATA_PATH` | pretrain | WSD decay 段数据,缺失时自动退回主数据 | `dataset/pretrain_decay/train` |
   | `SOPHIA_MACHINE_RECIPE` | pretrain | 新签入的目标机 recipe | canonical recipe |
   | `SOPHIA_EXPORT_DIR` | sft | 预训练 export 目录 | `runs/pretrain_rtx5090_bf16` |
   | `SOPHIA_SFT_TRAIN_DATA` | sft | 对话训练数据 | `dataset/sft/train.jsonl` |
   | `SOPHIA_SFT_EVAL_DATA` | sft | 对话验证数据,缺失时自动省略 | `dataset/sft/val.jsonl` |
   | `SOPHIA_SKIP_TESTS=1` | 两者 | 跳过回归子集;适用于本次会话已跑过的场合 | 不跳 |
   | `SOPHIA_SKIP_PROBE=1` | 两者 | 跳过 GPU 探针;适用于同机同栈近期已通过的场合 | 不跳 |

## 启动后看护

训练是长跑(预训练数天量级),启动成功不等于任务结束。启动后:

- 每 20~30 分钟主动检查一次并回报:进程存活(`kill -0 $(cat <output_dir>.pid)`)、
  launch log 尾部、loss 与 grad norm 走势是否平稳。
- 持续巡检用 `python3 tools/train_monitor_loop.py --run-dir <output_dir>
  --launch-log <launch_log>`(默认 20 分钟一轮)。
- 实时面板:`python3 tools/live_train_dashboard.py --run-dir <output_dir>`
  (默认 `127.0.0.1:16006`;远程机器经 SSH 端口转发访问,见 REMOTE_GPU.md §6)。
- loss 突增、grad norm 持续走高、吞吐骤降都要立即报告,不要等训练完。

观测记录的机制:训练循环把每步指标持续追加到 `<output_dir>/metrics.jsonl`
(持久化的权威记录);dashboard 前端每 5 秒自动刷新,服务端每次请求重读
metrics.jsonl、GPU 状态与 checkpoint 列表,页面始终是最新状态;面板无状态,
重启不丢历史。monitor loop 的告警写入 `<output_dir>/dashboard_events.jsonl`,
在面板事件区显示。

## 断点恢复

`SOPHIA_RESUME=auto bash tools/one_click_train.sh <stage>`(`SOPHIA_OUTPUT_DIR`
指向原 run 目录)。恢复时会校验 checkpoint 的机器签名与数据签名,不匹配会被拒绝。

## 失败处置

- **manifest 缺失**:先获取公开数据集(`bash tools/fetch_dataset.sh`),自备语料
  时用 `ml-shard` 构建;不要绕过校验。
- **tokenizer 指纹不一致**(fetch 脚本或 preflight 报 sha1 mismatch):说明数据集
  与本地 tokenizer bundle 版本不匹配。核对数据集发布版本与仓库版本并对齐,
  不要改 manifest、不要换算指纹、不要绕过校验。
- **SFT 找不到 export**:预训练尚未完成或 `SOPHIA_EXPORT_DIR` 指向错误;
  export 落在预训练的 `output_dir` 内。
- **机器签名不匹配**:说明当前 GPU 不是 recipe 目标机。正确做法是在新目标机上
  重新测量并签入新 recipe,不复用旧机器的测量结果、也不放宽校验。
- **探针失败**:修复环境(CUDA/Inductor/精度)后重跑,不要用 `SOPHIA_SKIP_PROBE=1`
  绕过首跑门禁。
- **output_dir 非空**:要么 `SOPHIA_RESUME=auto` 恢复,要么换新目录;
  `--overwrite_output_dir` 是破坏性操作,须用户明确同意后才可使用。
- **启动 30 秒内进程退出**:脚本会打印 launch log 尾部,按 preflight 报错逐项处理。

## 硬边界

- checkpoint、输出与 staging 目录必须在 Linux 原生盘;禁止放在 `/mnt/*` 这类
  Windows 挂载(9p 并发写会损坏或极慢)。
- 学习率等核心超参不得随意调整;任何调整必须重新通过目标机 ≥800 步稳定性探针。
- 不修改 tokenizer、token shards 与 recipe 的绑定关系;指纹不一致时重建数据,
  而不是改校验。
- SFT 不改变 tokenizer 与模型架构;对话数据格式问题在数据侧修复。
