# 远程租卡与部署教程

从零租一张云 GPU 到开始训练的完整流程。以 AutoDL 为例,其它平台(仙宫云、
RunPod、Vast.ai 等)步骤等价,差异只在控制台操作和数据盘路径。

## 1. 选卡与开机

- **卡型**:单张 RTX 5090(32GB 显存,CUDA capability 12.0),**请直接选这张卡,
  不要用其它卡型替代**。签名 recipe、GPU 探针与全部默认流程都以 5090 为准,
  其它卡型会被 preflight 拒绝(需自行重新测量并签入新 recipe,不在本教程覆盖
  范围内)。
- **镜像**:选带 CUDA 12.8 的基础镜像(如 PyTorch 官方镜像系列);驱动需支持
  CUDA 12.8。Python 版本不足 3.12 时后续用 conda 补。
- **数据盘**:建议 ≥ 300 GB。公开数据集约 71 GB(pretrain_tokens 62 GB +
  pretrain_decay 9 GB + SFT 22 MB),每个 checkpoint 数 GB 到十余 GB,训练全程
  会保留多个。AutoDL 的数据盘挂载在 `/root/autodl-tmp`,代码、数据、输出都放
  这里(系统盘小且重置镜像会丢)。
- **计费**:按发布配方吞吐(约 27k token/s),train split 全量 16.2B token 约需
  一周量级,建议包日/包周计费;按量计费务必确认余额充足,余额耗尽实例会被
  强制关机。

## 2. 连接实例

控制台拿到 SSH 端口和密码后:

```bash
ssh -p <port> root@<ip>
```

公网出口不稳时用重试循环:

```bash
until ssh -o ConnectTimeout=10 -p <port> root@<ip>; do sleep 5; done
```

长时间操作一律在 `tmux` 内进行,断连后 `tmux attach` 恢复现场。

## 3. 部署代码与环境

```bash
cd /root/autodl-tmp
git clone https://github.com/Arain119/Sophia.git
cd Sophia

# Python 3.12(镜像自带时跳过 conda 步骤)
conda create -n sophia python=3.12 -y && conda activate sophia

# 先装匹配 CUDA 的 torch,再装项目依赖
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

验证:`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
应输出 `2.8.0+cu128 True`。

## 4. 获取数据

训练数据不入 git,发布于 ModelScope 公开数据集
[Arain119/Sophia-dataset](https://www.modelscope.cn/datasets/Arain119/Sophia-dataset)。
**推荐直接在云机上下载**(国内 GPU 机房到 ModelScope 的带宽通常远好于本地上行),
在 `tmux` 内执行:

```bash
cd /root/autodl-tmp/Sophia
bash tools/fetch_dataset.sh          # 预训练 shards + SFT 数据,约 71 GB,断点续传
```

脚本先下载各 split 的 `manifest.json` 并校验 tokenizer 指纹,通过后才传输 shard
大文件,完成后逐 shard 核对字节数——传输损坏会在这里暴露,不会带病进入训练。
中断后重跑同一命令即可续传。

自备数据(不用公开数据集)时改为 rsync 直传:

```bash
rsync -avP -e "ssh -p <port>" dataset/pretrain_tokens root@<ip>:/root/autodl-tmp/Sophia/dataset/
rsync -avP -e "ssh -p <port>" dataset/pretrain_decay  root@<ip>:/root/autodl-tmp/Sophia/dataset/
rsync -avP -e "ssh -p <port>" dataset/sft             root@<ip>:/root/autodl-tmp/Sophia/dataset/
```

传完在云机上核对(fetch 脚本已内置同等校验,rsync 路径需手动执行):

```bash
PYTHONPATH=. python3 -m ml.tooling.cli audit token-shards \
  --manifest dataset/pretrain_tokens/train/manifest.json
```

manifest 校验通过即代表 shard 完整且 tokenizer 指纹一致。

## 5. 启动训练

```bash
tmux new -s train
cd /root/autodl-tmp/Sophia && conda activate sophia
bash tools/one_click_train.sh pretrain --dry-run   # 门禁预演
bash tools/one_click_train.sh pretrain             # 正式启动
```

训练进程由脚本以 nohup 挂后台,SSH 断开不影响;tmux 只是为了保住门禁阶段的
现场。预训练完成后同样方式跑 `sft` 阶段。

## 6. 远程观测

dashboard 与 TensorBoard 都绑定在云机的 127.0.0.1,本地经 SSH 端口转发访问:

```bash
# 本地终端
ssh -p <port> -N -L 16006:127.0.0.1:16006 root@<ip>   # dashboard
ssh -p <port> -N -L 6006:127.0.0.1:6006 root@<ip>     # tensorboard
```

云机上:

```bash
nohup python3 tools/live_train_dashboard.py --run-dir <output_dir> \
  --launch-log <output_dir>.launch.log > /root/autodl-tmp/dashboard.log 2>&1 &
nohup python3 tools/train_monitor_loop.py --run-dir <output_dir> \
  --launch-log <output_dir>.launch.log > /root/autodl-tmp/monitor.log 2>&1 &
```

浏览器打开 `http://127.0.0.1:16006`。观测记录的更新机制:

- 训练循环持续把每步指标追加到 `<output_dir>/metrics.jsonl`,这是持久化的
  权威记录;
- dashboard 前端每 5 秒自动刷新,服务端每次请求都重读 metrics.jsonl、GPU 状态
  与 checkpoint 列表,因此页面始终反映最新状态;
- dashboard 本身无状态,重启面板不丢历史——历史都在 metrics.jsonl 里;
- `train_monitor_loop.py` 每 20 分钟巡检一次,发现异常(指标停更、checkpoint
  停更、GPU 利用率偏低、吞吐骤降)时把告警写入 `<output_dir>/dashboard_events.jsonl`,
  面板的事件区会显示这些告警。

## 7. 收尾与释放实例

释放实例前必须取回的产物(实例释放后数据盘不可恢复):

```bash
rsync -avP -e "ssh -p <port>" root@<ip>:/root/autodl-tmp/Sophia/<output_dir>/ ./runs_backup/
```

至少包含:最终 checkpoint、export 模型目录、`metrics.jsonl`、`run_args.json`、
`machine_signature.json` 及各评测报告(发布审计的证据链)。确认本地校验可加载
(`ml-eval --export_dir ...`)后再释放实例。
