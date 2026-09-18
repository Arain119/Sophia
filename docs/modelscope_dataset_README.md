# Sophia Pretraining Token Shards

该数据资产用于 Sophia 随机初始化预训练和兼容 checkpoint 的继续预训练。它不是文本 benchmark，也不是 SFT 数据集。

## 结构

```text
pretrain/
  dataset_manifest.json
  train/
    manifest.json
    shard_*.bin
  val/
    manifest.json
    shard_*.bin
  test/
    manifest.json
    shard_*.bin
  lineage/
```

token dtype 为 `int32`。根 manifest 绑定三个 split、tokenizer bundle、lineage，以及每个 shard 的 token count、bytes 和 SHA-256。修改 tokenizer、manifest 或任意 shard 后，原 admission 证据失效。

## 使用

```bash
export SOPHIA_DATA_PATH=/path/to/pretrain
```

训练前验证：

```bash
python3 - <<'PY'
from ml.training.pretrain.data_admission import validate_pretrain_dataset_manifest

validate_pretrain_dataset_manifest(
    data_path="/path/to/pretrain",
    tokenizer_path="ml/modeling/text",
)
print("production_dataset_admission=pass")
PY
```

正式训练只读取一个 train split；没有 decay-data 或运行中动态 mix。不要拆分、重排、手改 manifest 或混用来自另一 tokenizer 的 shards。

## 权利与 lineage

数据由具有各自来源条款的语料构成。根 manifest 和 `lineage/` 是使用合同的一部分；仓库许可证不能替代上游许可证、平台条款或内容权利。使用者必须按 lineage 核对用途和再分发范围。
