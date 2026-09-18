from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from huggingface_hub import hf_hub_download
import requests

from ml.core.common.io import write_json_atomic


DEFAULT_HF_ENDPOINT = "https://huggingface.co"
INVENTORY_SCHEMA = "sophia_hf_source_inventory_v2"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(
            "unsupported Hugging Face source inventory; "
            f"expected schema {INVENTORY_SCHEMA!r}"
        )
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source inventory must contain a non-empty sources list")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source inventory rows must be objects")
        name = source.get("name")
        repo_id = source.get("repo_id")
        revision = source.get("revision")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid source name: {name!r}")
        if not isinstance(repo_id, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
            raise ValueError(f"invalid source repo_id: {repo_id!r}")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"invalid source revision: {revision!r}")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"source files must be a non-empty list: {name}")
        for file_spec in files:
            if not isinstance(file_spec, dict):
                raise ValueError(
                    f"source files must use v2 path/bytes/sha256 objects: {name}"
                )
            path_value = file_spec.get("path")
            bytes_value = file_spec.get("bytes")
            sha256_value = file_spec.get("sha256")
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError(f"source file path is required: {name}")
            candidate = Path(path_value)
            if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
                raise ValueError(f"source file path must be relative: {path_value!r}")
            if isinstance(bytes_value, bool) or not isinstance(bytes_value, int) or bytes_value <= 0:
                raise ValueError(f"source file bytes must be > 0: {path_value!r}")
            if not isinstance(sha256_value, str) or not re.fullmatch(
                r"[0-9a-f]{64}", sha256_value
            ):
                raise ValueError(f"source file sha256 is invalid: {path_value!r}")
    return payload


def _prioritize_distributed_files(
    files: list[object], count: int
) -> list[object]:
    requested = int(count)
    if requested <= 0 or not files:
        return list(files)
    selected = min(requested, len(files))
    if selected == 1:
        indexes = [0]
    else:
        indexes = [
            index * (len(files) - 1) // (selected - 1)
            for index in range(selected)
        ]
    selected_indexes = set(indexes)
    return [files[index] for index in indexes] + [
        value for index, value in enumerate(files) if index not in selected_indexes
    ]


def _download_https(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: str,
    endpoint: str = "",
    attempts: int = 12,
) -> str:
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return str(target)
    partial = target.with_name(target.name + ".part")
    url = _dataset_file_url(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
        endpoint=endpoint,
    )
    for attempt in range(1, int(attempts) + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "Sophia-data-builder/1.0"}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()
                status = int(response.status_code)
                if offset > 0 and status != 206:
                    raise RuntimeError(
                        f"server ignored Range for {filename}: "
                        f"offset={offset} status={status}"
                    )
                if offset > 0:
                    content_range = str(response.headers.get("Content-Range") or "")
                    if not content_range.lower().startswith(f"bytes {offset}-"):
                        raise RuntimeError(
                            f"server returned mismatched Content-Range for {filename}: "
                            f"offset={offset} content_range={content_range!r}"
                        )
                append = offset > 0
                mode = "ab" if append else "wb"
                with open(partial, mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(partial, target)
            return str(target)
        except (OSError, RuntimeError, requests.RequestException) as exc:
            if attempt >= int(attempts):
                raise
            print(
                f"[RETRY] attempt={attempt}/{attempts} file={filename} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"unreachable download state: {repo_id}/{filename}")


def _dataset_file_url(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    endpoint: str = "",
) -> str:
    endpoint = str(
        endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    ).rstrip("/")
    return (
        f"{endpoint}/datasets/{quote(repo_id, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(filename, safe='/')}?download=true"
    )


def _parse_curl_response_headers(headers_text: str) -> list[tuple[int, str]]:
    responses: list[tuple[int, str]] = []
    status: int | None = None
    fields: list[str] = []
    normalized = str(headers_text).replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        match = re.match(r"^HTTP/\S+\s+(\d{3})(?:\s|$)", line, re.IGNORECASE)
        if match is not None:
            if status is not None:
                responses.append((status, "\n".join(fields)))
            status = int(match.group(1))
            fields = []
        elif status is not None and line:
            fields.append(line)
    if status is not None:
        responses.append((status, "\n".join(fields)))
    return responses


def _download_curl(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: str,
    endpoint: str = "",
    expected_bytes: int = 0,
    expected_sha256: str = "",
    attempts: int = 120,
) -> str:
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return str(target)
    partial = target.with_name(target.name + ".part")
    expected_size = int(expected_bytes)
    expected_digest = str(expected_sha256 or "").lower()
    if partial.is_file() and expected_size > 0:
        partial_size = int(partial.stat().st_size)
        if partial_size == expected_size:
            if expected_digest and sha256_file(partial) != expected_digest:
                raise RuntimeError(f"completed partial sha256 mismatch: {partial}")
            os.replace(partial, target)
            return str(target)
        if partial_size > expected_size:
            raise RuntimeError(
                f"partial file exceeds expected size: {partial_size}>{expected_size}: {partial}"
            )
    connect_timeout = int(
        str(os.environ.get("SOPHIA_HF_CONNECT_TIMEOUT_SECONDS") or "15")
    )
    if not 1 <= connect_timeout <= 300:
        raise ValueError("SOPHIA_HF_CONNECT_TIMEOUT_SECONDS must be in [1, 300]")
    configured_attempts = str(os.environ.get("SOPHIA_HF_CURL_ATTEMPTS") or "").strip()
    if configured_attempts:
        attempts = int(configured_attempts)
    if not 1 <= int(attempts) <= 10_000:
        raise ValueError("SOPHIA_HF_CURL_ATTEMPTS must be in [1, 10000]")
    low_speed_seconds = int(str(os.environ.get("SOPHIA_HF_LOW_SPEED_SECONDS") or "120"))
    low_speed_bytes = int(str(os.environ.get("SOPHIA_HF_LOW_SPEED_BYTES") or "1024"))
    checkpoint_seconds = int(
        str(os.environ.get("SOPHIA_HF_CURL_CHECKPOINT_SECONDS") or "300")
    )
    if not 10 <= low_speed_seconds <= 3_600:
        raise ValueError("SOPHIA_HF_LOW_SPEED_SECONDS must be in [10, 3600]")
    if not 1 <= low_speed_bytes <= 1024 * 1024:
        raise ValueError("SOPHIA_HF_LOW_SPEED_BYTES must be in [1, 1048576]")
    if not 30 <= checkpoint_seconds <= 3_600:
        raise ValueError("SOPHIA_HF_CURL_CHECKPOINT_SECONDS must be in [30, 3600]")
    resolve_ip = str(os.environ.get("SOPHIA_HF_RESOLVE_IP") or "").strip()
    endpoint_host = urlsplit(
        str(endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT)
    ).hostname
    if resolve_ip and not endpoint_host:
        raise ValueError("download endpoint has no hostname")
    cdn_resolve_ip = str(os.environ.get("SOPHIA_HF_CDN_RESOLVE_IP") or "").strip()
    attempt_body = partial.with_name(partial.name + ".curl-attempt")
    attempt_headers = partial.with_name(partial.name + ".curl-headers")
    url = _dataset_file_url(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
        endpoint=endpoint,
    )
    last_error = ""
    for attempt in range(1, int(attempts) + 1):
        offset = int(partial.stat().st_size) if partial.is_file() else 0
        attempt_body.unlink(missing_ok=True)
        attempt_headers.unlink(missing_ok=True)
        command = [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--connect-timeout",
            str(connect_timeout),
            "--speed-time",
            str(low_speed_seconds),
            "--speed-limit",
            str(low_speed_bytes),
            "--max-time",
            str(checkpoint_seconds),
            "--user-agent",
            "Sophia-data-builder/1.0",
            "--range",
            f"{offset}-",
            "--dump-header",
            str(attempt_headers),
        ]
        if resolve_ip:
            command.extend(("--resolve", f"{endpoint_host}:443:{resolve_ip}"))
        if cdn_resolve_ip:
            command.extend(("--resolve", f"us.aws.cdn.hf.co:443:{cdn_resolve_ip}"))
        command.extend(("--output", str(attempt_body), url))
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        headers_text = (
            attempt_headers.read_text(encoding="iso-8859-1", errors="replace")
            if attempt_headers.is_file()
            else ""
        )
        responses = _parse_curl_response_headers(headers_text)
        status = int(responses[-1][0]) if responses else 0
        final_headers = responses[-1][1] if responses else ""
        content_range = re.search(
            r"^Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$",
            final_headers,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        accepted = False
        body_size = int(attempt_body.stat().st_size) if attempt_body.is_file() else 0
        if status == 206 and content_range is not None:
            response_start = int(content_range.group(1))
            if response_start == offset and body_size > 0:
                if offset == 0:
                    os.replace(attempt_body, partial)
                else:
                    with (
                        partial.open("ab") as output,
                        attempt_body.open("rb") as source,
                    ):
                        while chunk := source.read(4 * 1024 * 1024):
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    attempt_body.unlink(missing_ok=True)
                accepted = True
        elif status == 200 and offset == 0 and body_size > 0:
            os.replace(attempt_body, partial)
            accepted = True

        current_size = int(partial.stat().st_size) if partial.is_file() else 0
        if expected_size > 0 and current_size > expected_size:
            raise RuntimeError(
                f"downloaded partial exceeds expected size: {current_size}>{expected_size}: {partial}"
            )
        complete = expected_size > 0 and current_size == expected_size
        if complete or (
            expected_size <= 0 and int(result.returncode) == 0 and accepted
        ):
            if expected_digest and sha256_file(partial) != expected_digest:
                raise RuntimeError(f"completed partial sha256 mismatch: {partial}")
            attempt_body.unlink(missing_ok=True)
            attempt_headers.unlink(missing_ok=True)
            os.replace(partial, target)
            return str(target)
        stderr = str(getattr(result, "stderr", "") or "").strip()
        last_error = (
            f"curl_exit={int(result.returncode)} status={status} offset={offset} "
            f"received={body_size} retained={current_size} error={stderr}"
        )
        attempt_body.unlink(missing_ok=True)
        attempt_headers.unlink(missing_ok=True)
        if attempt >= int(attempts):
            break
        print(
            f"[RETRY] attempt={attempt}/{attempts} file={filename} "
            f"retained_bytes={current_size} status={status}",
            flush=True,
        )
        time.sleep(min(2 ** min(attempt, 4), 15))
    raise RuntimeError(
        f"curl download exhausted retries for {repo_id}/{filename}: {last_error}"
    )


def _download_inventory_file(
    *,
    file_index: int,
    file_count: int,
    filename_value: object,
    name: str,
    repo_id: str,
    revision: str,
    source_root: Path,
    transport: str,
    endpoint: str,
) -> dict[str, Any]:
    if not isinstance(filename_value, dict):
        raise TypeError("inventory file entry must be a v2 object")
    filename = str(filename_value["path"])
    expected_bytes = int(filename_value["bytes"])
    expected_sha256 = str(filename_value["sha256"])
    print(
        f"[DOWNLOAD] source={name} file={file_index}/{file_count} {filename}",
        flush=True,
    )
    target = source_root / filename
    reuse_verified_target = (
        target.is_file()
        and expected_bytes > 0
        and int(target.stat().st_size) == expected_bytes
        and bool(expected_sha256)
        and sha256_file(target) == expected_sha256
    )
    if reuse_verified_target:
        path = str(target)
    elif str(transport) == "hf":
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(source_root),
        )
    elif str(transport) == "https":
        path = _download_https(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            local_dir=str(source_root),
            endpoint=str(endpoint),
        )
    elif str(transport) == "curl":
        path = _download_curl(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            local_dir=str(source_root),
            endpoint=str(endpoint),
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
    else:
        raise ValueError(f"unsupported download transport: {transport!r}")
    resolved = Path(path).resolve()
    actual_bytes = int(resolved.stat().st_size)
    actual_sha256 = sha256_file(resolved)
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"downloaded size mismatch for {repo_id}/{filename}: "
            f"expected={expected_bytes} actual={actual_bytes}"
        )
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"downloaded sha256 mismatch for {repo_id}/{filename}: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return {
        "filename": filename,
        "path": str(resolved),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
    }


def download_inventory(
    *,
    inventory_path: str,
    output_root: str,
    transport: str = "https",
    endpoint: str = "",
    workers: int = 1,
    priority_distributed_files: int = 0,
) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    inventory_file = Path(inventory_path).expanduser().resolve()
    inventory_sha256 = sha256_file(inventory_file)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker_count = int(workers)
    if worker_count <= 0 or worker_count > 16:
        raise ValueError("workers must be between 1 and 16")
    source_reports: list[dict[str, Any]] = []
    for source in inventory["sources"]:
        if not isinstance(source, dict):
            raise ValueError("source inventory rows must be objects")
        name = str(source.get("name") or "").strip()
        repo_id = str(source.get("repo_id") or "").strip()
        revision = str(source.get("revision") or "").strip()
        files = source.get("files")
        if not name or not repo_id or len(revision) != 40:
            raise ValueError(f"invalid pinned source metadata: {name!r}")
        if not isinstance(files, list) or not files:
            raise ValueError(f"source files must be a non-empty list: {name}")
        ordered_files = _prioritize_distributed_files(
            list(files), int(priority_distributed_files)
        )
        source_root = root / name
        work = [
            {
                "file_index": index,
                "file_count": len(files),
                "filename_value": filename_value,
                "name": name,
                "repo_id": repo_id,
                "revision": revision,
                "source_root": source_root,
                "transport": str(transport),
                "endpoint": str(endpoint),
            }
            for index, filename_value in enumerate(ordered_files, start=1)
        ]
        if worker_count == 1:
            downloaded = [_download_inventory_file(**item) for item in work]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                downloaded = list(
                    executor.map(lambda item: _download_inventory_file(**item), work)
                )
        source_reports.append(
            {
                **{key: value for key, value in source.items() if key != "files"},
                "files": downloaded,
                "total_bytes": sum(int(item["bytes"]) for item in downloaded),
            }
        )
        write_json_atomic(
            root / "acquisition_report.json",
            {
                "schema": "sophia_hf_acquisition_report_v1",
                "inventory_path": str(inventory_file),
                "inventory_sha256": inventory_sha256,
                "output_root": str(root),
                "sources": source_reports,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    report = {
        "schema": "sophia_hf_acquisition_report_v1",
        "inventory_path": str(inventory_file),
        "inventory_sha256": inventory_sha256,
        "output_root": str(root),
        "download_endpoint": str(
            endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
        ).rstrip("/"),
        "download_workers": worker_count,
        "download_priority_distributed_files": int(priority_distributed_files),
        "sources": source_reports,
        "total_bytes": sum(int(item["total_bytes"]) for item in source_reports),
    }
    write_json_atomic(
        root / "acquisition_report.json",
        report,
        ensure_ascii=False,
        sort_keys=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a pinned HF dataset inventory"
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--transport",
        choices=("https", "hf", "curl"),
        default="https",
    )
    parser.add_argument(
        "--endpoint",
        default="",
        help="Optional HTTPS Hugging Face-compatible download endpoint.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--priority-distributed-files",
        type=int,
        default=0,
        help="Download an evenly spaced file prefix before the remaining inventory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = download_inventory(
        inventory_path=str(args.inventory),
        output_root=str(args.output_root),
        transport=str(args.transport),
        endpoint=str(args.endpoint),
        workers=int(args.workers),
        priority_distributed_files=int(args.priority_distributed_files),
    )
    print(
        f"[DONE] sources={len(report['sources'])} bytes={int(report['total_bytes']):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
