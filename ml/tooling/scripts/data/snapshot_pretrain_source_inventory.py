from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ml.core.common.io import write_json_atomic


SELECTION_SCHEMA = "sophia_hf_source_selection_v1"
INVENTORY_SCHEMA = "sophia_hf_source_inventory_v2"


def _default_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=12,
        connect=12,
        read=12,
        status=12,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "Sophia-data-inventory/1.0"})
    return session


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_policy_path(selection_path: str, value: object) -> Path:
    raw_value = str(value or "").strip()
    if not raw_value:
        raise ValueError("source selection must declare admission_policy")
    raw_path = Path(raw_value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    repo_root = Path(__file__).resolve().parents[4]
    repo_candidate = (repo_root / raw_path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path(selection_path).resolve().parent / raw_path).resolve()


def _load_admission_policy(
    selection_path: str,
    selection: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    policy_path = _resolve_policy_path(selection_path, selection.get("admission_policy"))
    policy = _load_json(str(policy_path))
    if str(policy.get("schema")) != "sophia_pretrain_data_admission_policy_v1":
        raise ValueError("unsupported pretrain data admission policy")
    if bool(policy.get("allow_legacy_assets_as_training_input")):
        raise ValueError("admission policy must forbid legacy training inputs")
    return policy, policy_path


def _forbidden_repo_ids(policy: dict[str, Any]) -> set[str]:
    return {
        str(value).strip().casefold()
        for value in policy.get("forbidden_upstream_repo_ids", [])
        if str(value).strip()
    }


def _endpoint_url(endpoint: str, url: str) -> str:
    base = urlsplit(str(endpoint).rstrip("/"))
    target = urlsplit(str(url))
    return urlunsplit((base.scheme, base.netloc, target.path, target.query, ""))


def _next_link(response: requests.Response, *, endpoint: str) -> str:
    raw = str(response.headers.get("link") or "")
    for item in raw.split(","):
        url_part, _, attributes = item.partition(">")
        if 'rel="next"' not in attributes:
            continue
        url = url_part.strip().lstrip("<")
        return _endpoint_url(endpoint, url)
    return ""


def _repo_metadata(
    *,
    endpoint: str,
    repo_id: str,
    session: requests.Session,
) -> dict[str, Any]:
    url = f"{str(endpoint).rstrip('/')}/api/datasets/{repo_id}"
    response = session.get(url, timeout=(15, 60))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected repo metadata payload: {repo_id}")
    return payload


def _distributed_indexes(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("distributed selection requires 0 < sample_files <= total")
    if count == 1:
        return [0]
    indexes = [index * (total - 1) // (count - 1) for index in range(count)]
    if len(set(indexes)) != count:
        raise RuntimeError("distributed selection produced duplicate indexes")
    return indexes


def _selected_tree_files(
    *,
    endpoint: str,
    repo_id: str,
    revision: str,
    selector: dict[str, Any],
    session: requests.Session,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    path_prefix = str(selector.get("path_prefix") or "").strip("/")
    pattern = re.compile(str(selector.get("include_regex") or r".*"))
    sampling = str(selector.get("sampling") or "prefix")
    recursive = bool(selector.get("recursive", False))
    max_files = int(selector.get("max_files") or 0)
    expected_files = int(selector.get("expected_files") or 0)
    sample_files = int(selector.get("sample_files") or 0)
    if sampling == "prefix":
        if max_files <= 0 or expected_files or sample_files:
            raise ValueError(
                f"prefix selector requires only max_files > 0: {repo_id}/{path_prefix}"
            )
    elif sampling == "evenly_spaced_full_catalog":
        if max_files or expected_files <= 0 or not 0 < sample_files <= expected_files:
            raise ValueError(
                "full-catalog selector requires expected_files and sample_files: "
                f"{repo_id}/{path_prefix}"
            )
    elif sampling == "full_catalog":
        if max_files or expected_files <= 0 or sample_files:
            raise ValueError(
                "full_catalog selector requires only expected_files > 0: "
                f"{repo_id}/{path_prefix}"
            )
    else:
        raise ValueError(f"unsupported selector sampling mode: {sampling}")
    base = (
        f"{str(endpoint).rstrip('/')}/api/datasets/{repo_id}/tree/{revision}"
        + (f"/{path_prefix}" if path_prefix else "")
    )
    url = base
    params: dict[str, object] | None = {
        "recursive": "true" if recursive else "false",
        "limit": 1_000,
    }
    catalog: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    while url and (sampling != "prefix" or len(catalog) < max_files):
        response = session.get(url, params=params, timeout=(15, 60))
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise RuntimeError(f"unexpected repo tree payload: {repo_id}/{path_prefix}")
        for row in rows:
            if not isinstance(row, dict) or str(row.get("type")) != "file":
                continue
            path = str(row.get("path") or "")
            if not path or path in seen_paths or pattern.search(path) is None:
                continue
            size = int(row.get("size") or 0)
            if size <= 0:
                raise RuntimeError(f"source file has no recorded size: {repo_id}/{path}")
            lfs = row.get("lfs") if isinstance(row.get("lfs"), dict) else {}
            lfs_oid = str(lfs.get("oid") or "").lower()
            expected_sha256 = (
                lfs_oid
                if len(lfs_oid) == 64
                and all(character in "0123456789abcdef" for character in lfs_oid)
                else ""
            )
            catalog.append(
                {
                    "path": path,
                    "bytes": size,
                    "sha256": expected_sha256,
                    "git_oid": str(row.get("oid") or ""),
                }
            )
            seen_paths.add(path)
            if sampling == "prefix" and len(catalog) >= max_files:
                break
        url = _next_link(response, endpoint=endpoint)
        params = None
    catalog.sort(key=lambda row: str(row["path"]))
    if sampling == "prefix" and len(catalog) != max_files:
        raise RuntimeError(
            f"selector resolved {len(catalog)} files, expected {max_files}: "
            f"{repo_id}/{path_prefix}"
        )
    if sampling in {"evenly_spaced_full_catalog", "full_catalog"}:
        if len(catalog) != expected_files:
            raise RuntimeError(
                f"full catalog resolved {len(catalog)} files, expected "
                f"{expected_files}: {repo_id}/{path_prefix}"
            )
    if sampling == "full_catalog":
        evidence = {
            "path_prefix": path_prefix,
            "sampling": sampling,
            "recursive": recursive,
            "catalog_complete": True,
            "catalog_file_count": len(catalog),
            "catalog_bytes": sum(int(row["bytes"]) for row in catalog),
            "selected_file_count": len(catalog),
            "first_selected_path": str(catalog[0]["path"]),
            "last_selected_path": str(catalog[-1]["path"]),
        }
        return catalog, evidence, catalog
    if sampling == "evenly_spaced_full_catalog":
        indexes = _distributed_indexes(len(catalog), sample_files)
        selected = [catalog[index] for index in indexes]
        evidence = {
            "path_prefix": path_prefix,
            "sampling": sampling,
            "recursive": recursive,
            "catalog_file_count": len(catalog),
            "catalog_bytes": sum(int(row["bytes"]) for row in catalog),
            "sampled_file_count": len(selected),
            "sample_formula": "floor(i * (catalog_file_count - 1) / (sampled_file_count - 1))",
            "first_sampled_path": str(selected[0]["path"]),
            "last_sampled_path": str(selected[-1]["path"]),
        }
        return selected, evidence, catalog
    evidence = {
        "path_prefix": path_prefix,
        "sampling": sampling,
        "recursive": recursive,
        "catalog_complete": False,
        "sampled_file_count": len(catalog),
    }
    return catalog, evidence, []


def snapshot_inventory(
    *,
    selection_path: str,
    output_path: str,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    selection = _load_json(selection_path)
    if str(selection.get("schema")) != SELECTION_SCHEMA:
        raise ValueError("unsupported pretrain source selection schema")
    if str(selection.get("status") or "").startswith("rejected_"):
        raise ValueError(f"source selection is rejected: {selection.get('status')}")
    policy, policy_path = _load_admission_policy(selection_path, selection)
    forbidden_repo_ids = _forbidden_repo_ids(policy)
    endpoint = str(selection.get("discovery_endpoint") or "").rstrip("/")
    if not endpoint.startswith("https://"):
        raise ValueError("discovery_endpoint must be an https URL")
    sources = selection.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source selection must contain sources")
    active_session = session or _default_session()
    resolved_sources: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source rows must be objects")
        name = str(source.get("name") or "")
        repo_id = str(source.get("repo_id") or "")
        revision = str(source.get("revision") or "").lower()
        if not name or not repo_id or len(revision) != 40:
            raise ValueError(f"invalid pinned source: {name!r}")
        if repo_id.casefold() in forbidden_repo_ids:
            raise ValueError(f"legacy upstream repository is forbidden: {repo_id}")
        if str(source.get("admission_status")) != "admitted":
            raise ValueError(f"source is not admitted: {name}")
        if not str(source.get("license") or "").strip():
            raise ValueError(f"source has no reviewed license: {name}")
        metadata = _repo_metadata(
            endpoint=endpoint,
            repo_id=repo_id,
            session=active_session,
        )
        actual_revision = str(metadata.get("sha") or "").lower()
        if actual_revision != revision:
            raise RuntimeError(
                f"pinned revision mismatch for {repo_id}: "
                f"expected={revision} actual={actual_revision}"
            )
        if bool(metadata.get("private")) or bool(metadata.get("disabled")):
            raise RuntimeError(f"source repository is unavailable: {repo_id}")
        gated = metadata.get("gated")
        if gated not in (False, None, "false"):
            raise RuntimeError(f"gated source is forbidden: {repo_id} gated={gated!r}")
        selectors = source.get("selectors")
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"source has no selectors: {name}")
        files: list[dict[str, Any]] = []
        catalog_files: list[dict[str, Any]] = []
        selector_evidence: list[dict[str, Any]] = []
        for selector in selectors:
            if not isinstance(selector, dict):
                raise ValueError(f"invalid selector for source: {name}")
            selected, evidence, catalog = _selected_tree_files(
                endpoint=endpoint,
                repo_id=repo_id,
                revision=revision,
                selector=selector,
                session=active_session,
            )
            files.extend(selected)
            catalog_files.extend(catalog)
            selector_evidence.append(evidence)
        paths = [str(item["path"]) for item in files]
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"overlapping source selectors produced duplicates: {name}")
        catalog_paths = [str(item["path"]) for item in catalog_files]
        if len(catalog_paths) != len(set(catalog_paths)):
            raise RuntimeError(f"overlapping source catalogs produced duplicates: {name}")
        resolved_sources.append(
            {
                **{key: value for key, value in source.items() if key != "selectors"},
                "selector_evidence": selector_evidence,
                "catalog_files": catalog_files,
                "catalog_file_count": len(catalog_files),
                "catalog_bytes": sum(
                    int(item["bytes"]) for item in catalog_files
                ),
                "files": files,
                "selected_file_count": len(files),
                "selected_bytes": sum(int(item["bytes"]) for item in files),
            }
        )
        print(
            f"[SNAPSHOT] source={name} files={len(files):,} "
            f"bytes={sum(int(item['bytes']) for item in files):,}",
            flush=True,
        )
    selection_bytes = Path(selection_path).resolve().read_bytes()
    report = {
        "schema": INVENTORY_SCHEMA,
        "selection_path": str(Path(selection_path).resolve()),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "admission_policy_path": str(policy_path),
        "admission_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "discovery_endpoint": endpoint,
        "sources": resolved_sources,
        "selected_file_count": sum(
            int(source["selected_file_count"]) for source in resolved_sources
        ),
        "selected_bytes": sum(int(source["selected_bytes"]) for source in resolved_sources),
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
        description="Snapshot pinned Hugging Face files for the new pretrain corpus."
    )
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = snapshot_inventory(
        selection_path=str(args.selection),
        output_path=str(args.output),
    )
    print(
        f"[DONE] files={report['selected_file_count']:,} "
        f"bytes={report['selected_bytes']:,} output={Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
