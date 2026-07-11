# Dataset Layout

This directory mixes two different classes of files:

1. Release dataset inputs used by the training/post-training code.
2. Local reports, recovery outputs, and temporary build artifacts.

Treat them differently.

## Release Inputs

- Token-shard training datasets are finalized roots containing:
  - `train/manifest.json`
  - `val/manifest.json`
  - `test/manifest.json`
- SFT post-train datasets are the split JSONL files under `dataset/sft/`.

The public release inputs are hosted on ModelScope at
[Arain119/Sophia-dataset](https://www.modelscope.cn/datasets/Arain119/Sophia-dataset)
and land in this directory via `bash tools/fetch_dataset.sh` (staged download:
manifests + tokenizer-fingerprint check first, then the shard payload, then a
per-shard size verification).

## Local Artifacts

- `report.json` and `*_report.json` files are locally generated summaries or
  lightweight state files for maintenance flows. They are useful, but they are not
  source-controlled ground truth.
- `manifest.recovered.json` files are outputs from parquet-pool recovery. The release
  pretrain path reads `manifest.json`, not `manifest.recovered.json`.
- `*.tmp_build.*` directories are staging outputs from interrupted or in-progress
  shard builds and must not be treated as finalized datasets.
- `posttrain_length_curriculum.json` is generated from current SFT reports and
  can be regenerated with:

```bash
python3 -m ml.tooling.scripts.data.production.write_posttrain_length_curriculum \
  --sft_dir dataset/sft \
  --output dataset/posttrain_length_curriculum.json
```

## Operational Rule

When in doubt, trust the public CLI API in `pyproject.toml` and
`ml-tool` / `ml.tooling.cli` over ad hoc files already present under `dataset/`.
