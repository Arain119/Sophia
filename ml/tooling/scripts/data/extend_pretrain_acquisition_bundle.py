from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.build_pretrain_acquisition_bundle import (
    ACQUISITION_SCHEMA,
    INVENTORY_SCHEMA,
    _derivative_source,
    _verified_hf_sources,
)


CLEANING_SCHEMA = "sophia_clean_pretrain_corpus_v1"
EXTENSION_SCHEMA = "sophia_append_only_acquisition_bundle_v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_prior_bundle(
    *, prior_acquisition_path: Path, previous_cleaning_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    cleaning = _load_json(previous_cleaning_path)
    if str(cleaning.get("schema")) != CLEANING_SCHEMA:
        raise ValueError("unsupported previous cleaning report")
    if (
        Path(str(cleaning.get("acquisition_report") or "")).resolve()
        != prior_acquisition_path
        or str(cleaning.get("acquisition_report_sha256") or "")
        != _sha256(prior_acquisition_path)
    ):
        raise ValueError("previous cleaning report does not bind prior acquisition")
    acquisition = _load_json(prior_acquisition_path)
    if str(acquisition.get("schema")) != ACQUISITION_SCHEMA:
        raise ValueError("unsupported prior acquisition report")
    inventory_path = Path(str(acquisition.get("inventory_path") or "")).resolve()
    inventory = _load_json(inventory_path)
    inventory_sha256 = _sha256(inventory_path)
    if (
        str(inventory.get("schema")) != INVENTORY_SCHEMA
        or str(acquisition.get("inventory_sha256") or "") != inventory_sha256
        or str(cleaning.get("inventory_sha256") or "") != inventory_sha256
    ):
        raise ValueError("prior acquisition inventory lineage mismatch")
    selected = {
        str(source.get("name") or ""): source
        for source in inventory.get("sources", [])
        if isinstance(source, dict)
    }
    acquired = {
        str(source.get("name") or ""): source
        for source in acquisition.get("sources", [])
        if isinstance(source, dict)
    }
    if not selected or set(selected) != set(acquired):
        raise ValueError("prior acquisition sources do not match inventory")
    for name, source in selected.items():
        expected = {
            str(row.get("path") or ""): row
            for row in source.get("files", [])
            if isinstance(row, dict)
        }
        actual = {
            str(row.get("filename") or ""): row
            for row in acquired[name].get("files", [])
            if isinstance(row, dict)
        }
        if not expected or set(expected) != set(actual):
            raise ValueError(f"prior acquisition file mismatch: {name}")
        for filename, expected_row in expected.items():
            actual_row = actual[filename]
            if (
                int(actual_row.get("bytes") or 0)
                != int(expected_row.get("bytes") or 0)
                or str(actual_row.get("sha256") or "")
                != str(expected_row.get("sha256") or "")
            ):
                raise ValueError(f"prior acquisition file metadata drift: {name}")
    return inventory, acquisition


def extend_bundle(
    *,
    prior_acquisition_report: str,
    previous_cleaning_report: str,
    extension_acquisition_reports: list[str],
    output_root: str,
    inventory_output: str,
    acquisition_output: str,
    extension_report: str,
    derivative_reports: list[str] | None = None,
) -> dict[str, Any]:
    prior_path = Path(prior_acquisition_report).expanduser().resolve()
    cleaning_path = Path(previous_cleaning_report).expanduser().resolve()
    bundle_root = Path(output_root).expanduser().resolve()
    prior_inventory, prior_acquisition = _validate_prior_bundle(
        prior_acquisition_path=prior_path,
        previous_cleaning_path=cleaning_path,
    )
    if not extension_acquisition_reports:
        raise ValueError("at least one extension acquisition report is required")
    added_inventory_sources: list[dict[str, Any]] = []
    added_acquired_sources: list[dict[str, Any]] = []
    extension_inputs: list[dict[str, str]] = []
    for value in extension_acquisition_reports:
        path = Path(value).expanduser().resolve()
        selected, acquired = _verified_hf_sources(path, bundle_root=bundle_root)
        added_inventory_sources.extend(selected)
        added_acquired_sources.extend(acquired)
        extension_inputs.append({"path": str(path), "sha256": _sha256(path)})
    derivative_inputs: list[dict[str, str]] = []
    for value in derivative_reports or []:
        path = Path(value).expanduser().resolve()
        selected, acquired = _derivative_source(path, bundle_root=bundle_root)
        added_inventory_sources.append(selected)
        added_acquired_sources.append(acquired)
        derivative_inputs.append({"path": str(path), "sha256": _sha256(path)})
    prior_inventory_sources = list(prior_inventory.get("sources", []))
    prior_acquired_sources = list(prior_acquisition.get("sources", []))
    prior_names = [str(source.get("name") or "") for source in prior_inventory_sources]
    added_names = [str(source.get("name") or "") for source in added_inventory_sources]
    if (
        not added_names
        or len(prior_names) != len(set(prior_names))
        or len(added_names) != len(set(added_names))
        or set(prior_names) & set(added_names)
    ):
        raise ValueError("append-only acquisition source names overlap or are invalid")
    inventory_path = Path(inventory_output).expanduser().resolve()
    inventory_sources = prior_inventory_sources + added_inventory_sources
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "status": "complete",
        "purpose": "Append-only extension of a previously cleaned pretrain acquisition",
        "prior_acquisition_report": str(prior_path),
        "prior_acquisition_sha256": _sha256(prior_path),
        "previous_cleaning_report": str(cleaning_path),
        "previous_cleaning_report_sha256": _sha256(cleaning_path),
        "extension_acquisition_reports": extension_inputs,
        "derivative_reports": derivative_inputs,
        "sources": inventory_sources,
        "selected_file_count": sum(
            len(source.get("files", [])) for source in inventory_sources
        ),
        "selected_bytes": sum(
            int(row.get("bytes") or 0)
            for source in inventory_sources
            for row in source.get("files", [])
            if isinstance(row, dict)
        ),
    }
    write_json_atomic(
        inventory_path,
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    acquired_sources = prior_acquired_sources + added_acquired_sources
    acquisition_path = Path(acquisition_output).expanduser().resolve()
    acquisition = {
        "schema": ACQUISITION_SCHEMA,
        "status": "complete",
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "output_root": str(bundle_root),
        "derivative_reports": derivative_inputs,
        "sources": acquired_sources,
        "total_bytes": inventory["selected_bytes"],
    }
    write_json_atomic(
        acquisition_path,
        acquisition,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    report = {
        "schema": EXTENSION_SCHEMA,
        "status": "pass",
        "prior_acquisition_report": str(prior_path),
        "prior_acquisition_sha256": _sha256(prior_path),
        "previous_cleaning_report": str(cleaning_path),
        "previous_cleaning_report_sha256": _sha256(cleaning_path),
        "extension_acquisition_reports": extension_inputs,
        "derivative_reports": derivative_inputs,
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "acquisition_path": str(acquisition_path),
        "acquisition_sha256": _sha256(acquisition_path),
        "prior_source_count": len(prior_names),
        "added_source_count": len(added_names),
        "added_sources": added_names,
        "total_source_count": len(inventory_sources),
        "total_file_count": int(inventory["selected_file_count"]),
        "total_bytes": int(inventory["selected_bytes"]),
        "deduplication_order_rule": "prior ordered sources are an exact prefix",
    }
    write_json_atomic(
        Path(extension_report).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extend a cleaned pretrain acquisition with verified new sources."
    )
    parser.add_argument("--prior-acquisition", required=True)
    parser.add_argument("--previous-cleaning-report", required=True)
    parser.add_argument("--extension-acquisition", action="append", default=[])
    parser.add_argument("--humanities-derivative", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--inventory-output", required=True)
    parser.add_argument("--acquisition-output", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = extend_bundle(
        prior_acquisition_report=str(args.prior_acquisition),
        previous_cleaning_report=str(args.previous_cleaning_report),
        extension_acquisition_reports=[
            str(value) for value in args.extension_acquisition
        ],
        output_root=str(args.output_root),
        inventory_output=str(args.inventory_output),
        acquisition_output=str(args.acquisition_output),
        extension_report=str(args.report),
        derivative_reports=[str(value) for value in args.humanities_derivative],
    )
    print(
        f"[DONE] sources={report['total_source_count']} "
        f"files={report['total_file_count']} output={args.acquisition_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
