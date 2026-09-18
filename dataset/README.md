# Dataset Layout

```text
dataset/
  pretrain/                  canonical pretraining corpus
  benchmarks/                pinned evaluation and decontamination assets
  source_audit/              source-selection evidence the admission policy binds
  legacy_content_exclusion/  legacy-overlap exclusion evidence
  sft/                       frozen evaluation assets, then the corpus
```

`pretrain/` holds `dataset_manifest.json`, `train/`, `val/`, `test/` and
`lineage/`. The lineage directory is self-contained: it carries a hashed copy of
the admission policy, mix policy, resolved mix plan, acquisition, cleaning and
token-profile reports, and the benchmark signatures the build ran against, so
the corpus can be audited without any file outside this root.

`sft/` fills in two stages. `sft regression-manifest` writes the frozen held-out
token shards first, because it needs only the test split: the post-SFT
perplexity comparison is fixed before there is a model to compare. `dataset-build`
then adds the corpus in the same shape as `pretrain/` -- a manifest, `train/`
and `validation/` splits, and a `lineage/` holding a hashed copy of every input:
each run's request spec, seeds, seed report, distillation report, gate report
and the three verification artifacts. It refuses to run until the formal base
checkpoint and its passing `pretrain final-check` exist.

## Names

Directory names carry no version suffix. Where a file's own generation is part
of what it records, the file keeps it: `source_audit/` holds
`distributed_sample_report_v3.json` because the report is of the third sampling
pass, not a third revision of one document.

The corpus was admitted under `configs/data/pretrain_data_admission_policy.json`,
whose bytes are frozen: `dataset_manifest.json` records its SHA-256 and
`validate_pretrain_dataset_manifest` compares them. It therefore still names the
paths that existed when the corpus was built. They map as:

| recorded in the frozen policy | now |
| --- | --- |
| `dataset/benchmarks_v2/benchmark_prompt_signatures_v2.jsonl` | `dataset/benchmarks/benchmark_prompt_signatures.jsonl` |
| `dataset/legacy_content_exclusion_v2/report.json` | `dataset/legacy_content_exclusion/report.json` |
| `dataset/pretrain_source_audit_v2/...` | `dataset/source_audit/...` |
| `configs/eval/generation_quality_v2_2000_decontamination.jsonl` | `configs/eval/generation_quality_decontamination.jsonl` |

Bytes are unchanged, so every SHA-256 in that policy still identifies the same
file. Nothing resolves those paths at run time -- the policy is read for its
tokenizer hash, its forbidden input roots and its Chinese quality proxy only.

## What lives here and what does not

A file belongs here when a manifest, protocol or policy binds it by hash.
Raw acquisitions, build caches and superseded profiles live in
`out/dataset_build_state/`: the legacy-overlap SQLite index, the raw C-Eval and
CMMLU downloads that `benchmarks/chinese_multiple_choice.jsonl` was built from,
and the source-audit generations that `distributed_sample_report_v3.json`
superseded.

## Operational rule

Release training uses `dataset/pretrain` and must pass the full dataset-manifest
validator. Do not recreate versioned aliases or train from unsigned directories.
