from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import INVENTORY_SCHEMA


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _distributed_indexes(total: int, count: int) -> list[int]:
    if not 0 < count <= total:
        raise ValueError("article_shards must be in [1, available_article_shards]")
    if count == 1:
        return [0]
    return [index * (total - 1) // (count - 1) for index in range(count)]


def select_wikimedia_article_shards(
    *,
    inventory_path: str,
    source_name: str,
    article_shards: int,
    output_path: str,
) -> dict[str, Any]:
    parent_path = Path(inventory_path).expanduser().resolve()
    parent = _load_json(parent_path)
    if (
        str(parent.get("schema")) != INVENTORY_SCHEMA
        or str(parent.get("status")) != "complete"
        or not bool(parent.get("mutable_latest_urls_forbidden"))
    ):
        raise ValueError("unsupported or incomplete Wikimedia parent inventory")
    matches = [
        row
        for row in parent.get("sources", [])
        if isinstance(row, dict) and str(row.get("name") or "") == str(source_name)
    ]
    if len(matches) != 1:
        raise ValueError(f"Wikimedia source must resolve exactly once: {source_name}")
    source = matches[0]
    files = source.get("files")
    if not isinstance(files, list):
        raise ValueError("Wikimedia source has no file list")
    pairs: dict[int, list[dict[str, Any]]] = {}
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("Wikimedia source files must be objects")
        shard_index = int(row.get("shard_index") or 0)
        if shard_index <= 0:
            raise ValueError("Wikimedia source file has invalid shard index")
        pairs.setdefault(shard_index, []).append(row)
    ordered_indexes = sorted(pairs)
    if not ordered_indexes:
        raise ValueError("Wikimedia source has no article shards")
    for shard_index in ordered_indexes:
        roles = {str(row.get("role") or "") for row in pairs[shard_index]}
        if roles != {"articles", "index"} or len(pairs[shard_index]) != 2:
            raise ValueError(f"Wikimedia article shard is not a complete pair: {shard_index}")
    selected_indexes = [
        ordered_indexes[index]
        for index in _distributed_indexes(len(ordered_indexes), int(article_shards))
    ]
    selected_set = set(selected_indexes)
    selected_files = [
        row
        for row in files
        if int(row.get("shard_index") or 0) in selected_set
    ]
    selected_files.sort(key=lambda row: (int(row["shard_index"]), str(row["role"])))
    selected_source = {
        **{key: value for key, value in source.items() if key != "files"},
        "full_article_shard_count": len(ordered_indexes),
        "article_shard_count": len(selected_indexes),
        "selected_article_shard_indexes": selected_indexes,
        "selection": {
            "method": "evenly_spaced_page_id_ordered_article_shards",
            "formula": "floor(i * (full_article_shard_count - 1) / (selected_article_shard_count - 1))",
            "full_article_shard_count": len(ordered_indexes),
            "selected_article_shard_count": len(selected_indexes),
        },
        "selected_file_count": len(selected_files),
        "selected_bytes": sum(int(row["bytes"]) for row in selected_files),
        "files": selected_files,
    }
    report = {
        "schema": INVENTORY_SCHEMA,
        "status": "complete",
        "snapshot_date": str(parent.get("snapshot_date") or ""),
        "mutable_latest_urls_forbidden": True,
        "parent_inventory_path": str(parent_path),
        "parent_inventory_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "selection_purpose": "Token-capped, page-id-distributed English Wikipedia supply candidate; expand only if measured unique supply is insufficient.",
        "source_count": 1,
        "selected_file_count": len(selected_files),
        "selected_bytes": int(selected_source["selected_bytes"]),
        "sources": [selected_source],
    }
    write_json_atomic(
        Path(output_path).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select complete, evenly spaced Wikimedia article/index pairs."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--article-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = select_wikimedia_article_shards(
        inventory_path=str(args.inventory),
        source_name=str(args.source),
        article_shards=int(args.article_shards),
        output_path=str(args.output),
    )
    source = report["sources"][0]
    print(
        f"[DONE] source={source['name']} shards={source['article_shard_count']}/"
        f"{source['full_article_shard_count']} bytes={report['selected_bytes']:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
