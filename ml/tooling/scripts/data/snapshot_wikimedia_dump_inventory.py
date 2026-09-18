from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ml.core.common.io import write_json_atomic


INVENTORY_SCHEMA = "sophia_wikimedia_dump_inventory_v1"
DEFAULT_ENDPOINT = "https://dumps.wikimedia.org"
MULTISTREAM_JOB = "articlesmultistreamdump"
USER_AGENT = "Sophia-data-builder/1.0 (Wikimedia dump provenance audit)"
SUPPORTED_WIKIS: dict[str, dict[str, str]] = {
    "zhwiki": {
        "language": "zh",
        "output_language": "zh-Hans",
        "domain": "recent_simplified_chinese_encyclopedic",
        "normalization": "opencc_t2s_after_wikitext_extraction",
        "canonical_host": "zh.wikipedia.org",
        "license": "cc-by-sa-4.0-and-gfdl-page-level",
        "license_review": (
            "Page ids, revision ids, timestamps, titles, and canonical URLs are "
            "retained in the extracted parquet for attribution lineage."
        ),
        "quality_score": "current_curated_encyclopedic",
        "min_chars": "160",
        "deduplication_priority": "100",
    },
    "enwiki": {
        "language": "en",
        "output_language": "en",
        "domain": "recent_english_encyclopedic",
        "normalization": "identity_after_wikitext_extraction",
        "canonical_host": "en.wikipedia.org",
        "license": "cc-by-sa-4.0-and-gfdl-page-level",
        "license_review": (
            "Page ids, revision ids, timestamps, titles, and canonical URLs are "
            "retained in the extracted parquet for attribution lineage."
        ),
        "quality_score": "current_curated_encyclopedic",
        "min_chars": "160",
        "deduplication_priority": "100",
    },
    "zhwikibooks": {
        "language": "zh",
        "output_language": "zh-Hans",
        "domain": "open_chinese_textbooks_and_humanities",
        "normalization": "opencc_t2s_after_wikitext_extraction",
        "canonical_host": "zh.wikibooks.org",
        "license": "cc-by-sa-4.0-and-gfdl-page-level",
        "license_review": (
            "Chinese Wikibooks text is reusable under CC BY-SA 4.0 and, where "
            "applicable, GFDL; page-level attribution lineage is retained."
        ),
        "quality_score": "current_curated_open_textbook",
        "min_chars": "160",
        "deduplication_priority": "0",
    },
    "zhwikiversity": {
        "language": "zh",
        "output_language": "zh-Hans",
        "domain": "open_chinese_courses_and_humanities",
        "normalization": "opencc_t2s_after_wikitext_extraction",
        "canonical_host": "zh.wikiversity.org",
        "license": "cc-by-sa-4.0-and-gfdl-page-level",
        "license_review": (
            "Chinese Wikiversity text is reusable under CC BY-SA 4.0 and, where "
            "applicable, GFDL; page-level attribution lineage is retained."
        ),
        "quality_score": "current_curated_open_course",
        "min_chars": "160",
        "deduplication_priority": "0",
    },
    "zhwikiquote": {
        "language": "zh",
        "output_language": "zh-Hans",
        "domain": "open_chinese_quotations_and_humanities",
        "normalization": "opencc_t2s_after_wikitext_extraction",
        "canonical_host": "zh.wikiquote.org",
        "license": "cc-by-sa-4.0-and-gfdl-page-level",
        "license_review": (
            "Chinese Wikiquote contributions are reusable under CC BY-SA 4.0 and, "
            "where applicable, GFDL; quotation-level third-party rights remain "
            "subject to the source attribution carried by each page."
        ),
        "quality_score": "current_curated_quotation",
        "min_chars": "160",
        "deduplication_priority": "0",
    },
    "zhwikinews": {
        "language": "zh",
        "output_language": "zh-Hans",
        "domain": "open_chinese_recent_news",
        "normalization": "opencc_t2s_after_wikitext_extraction",
        "canonical_host": "zh.wikinews.org",
        "license": "cc-by-2.5-page-level",
        "license_review": (
            "Chinese Wikinews original text is published under CC BY 2.5; page-level "
            "attribution lineage is retained and embedded third-party media is not "
            "part of the text extraction."
        ),
        "quality_score": "current_curated_news",
        "min_chars": "200",
        "deduplication_priority": "0",
    },
}
_DATE_RE = re.compile(r"^[0-9]{8}$")
_UNSHARDED_PAGE_ID_END = 2**63 - 1


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
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _parse_multistream_filename(
    *, filename: str, wiki: str, snapshot_date: str
) -> tuple[str, int, int, int] | None:
    prefix = re.escape(f"{wiki}-{snapshot_date}-pages-articles-multistream")
    match = re.fullmatch(
        rf"{prefix}(?P<shard>[0-9]+)\.xml-p(?P<start>[0-9]+)p(?P<end>[0-9]+)\.bz2",
        filename,
    )
    role = "articles"
    if match is None:
        match = re.fullmatch(
            rf"{prefix}-index(?P<shard>[0-9]+)\.txt-p(?P<start>[0-9]+)p(?P<end>[0-9]+)\.bz2",
            filename,
        )
        role = "index"
    if match is None:
        unsharded_prefix = f"{wiki}-{snapshot_date}-pages-articles-multistream"
        if filename == f"{unsharded_prefix}.xml.bz2":
            return "articles", 1, 1, _UNSHARDED_PAGE_ID_END
        if filename == f"{unsharded_prefix}-index.txt.bz2":
            return "index", 1, 1, _UNSHARDED_PAGE_ID_END
    if match is None:
        return None
    dump_part = int(match.group("shard"))
    start = int(match.group("start"))
    end = int(match.group("end"))
    if dump_part <= 0 or start <= 0 or end < start:
        raise ValueError(f"invalid Wikimedia multistream range: {filename}")
    return role, dump_part, start, end


def _response_bytes(response: Any, payload: dict[str, Any]) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes) and content:
        return content
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _snapshot_source(
    *,
    wiki: str,
    snapshot_date: str,
    endpoint: str,
    session: requests.Session,
) -> dict[str, Any]:
    metadata = SUPPORTED_WIKIS.get(str(wiki))
    if metadata is None:
        raise ValueError(f"unsupported Wikimedia wiki: {wiki}")
    if not _DATE_RE.fullmatch(str(snapshot_date)):
        raise ValueError("Wikimedia snapshot date must use YYYYMMDD")
    status_url = f"{endpoint.rstrip('/')}/{wiki}/{snapshot_date}/dumpstatus.json"
    response = session.get(
        status_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=(30, 120),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Wikimedia dump status is not an object: {status_url}")
    jobs = payload.get("jobs")
    job = jobs.get(MULTISTREAM_JOB) if isinstance(jobs, dict) else None
    if not isinstance(job, dict) or str(job.get("status")) != "done":
        raise ValueError(
            f"Wikimedia sharded article dump is not complete: {wiki}/{snapshot_date}"
        )
    raw_files = job.get("files")
    if not isinstance(raw_files, dict):
        raise ValueError("Wikimedia dump job has no file inventory")
    files: list[dict[str, Any]] = []
    pairs: dict[tuple[int, int, int], dict[str, str]] = {}
    for filename, raw in raw_files.items():
        parsed = _parse_multistream_filename(
            filename=str(filename), wiki=wiki, snapshot_date=snapshot_date
        )
        if parsed is None:
            raise ValueError(f"unexpected Wikimedia sharded file: {filename}")
        role, dump_part, page_id_start, page_id_end = parsed
        if not isinstance(raw, dict):
            raise ValueError(f"invalid Wikimedia file metadata: {filename}")
        size = int(raw.get("size") or 0)
        sha1 = str(raw.get("sha1") or "").lower()
        md5 = str(raw.get("md5") or "").lower()
        relative_url = str(raw.get("url") or "")
        if (
            size <= 0
            or not re.fullmatch(r"[0-9a-f]{40}", sha1)
            or not re.fullmatch(r"[0-9a-f]{32}", md5)
            or not relative_url.startswith(f"/{wiki}/{snapshot_date}/")
        ):
            raise ValueError(f"incomplete Wikimedia checksum metadata: {filename}")
        url = urljoin(f"{endpoint.rstrip('/')}/", relative_url.lstrip("/"))
        if "/latest/" in url:
            raise ValueError("mutable Wikimedia latest URL may not enter an inventory")
        files.append(
            {
                "filename": filename,
                "role": role,
                "dump_part": dump_part,
                "page_id_start": page_id_start,
                "page_id_end": page_id_end,
                "bytes": size,
                "sha1": sha1,
                "md5": md5,
                "url": url,
            }
        )
        pair_key = (dump_part, page_id_start, page_id_end)
        roles = pairs.setdefault(pair_key, {})
        if role in roles:
            raise ValueError(
                f"duplicate Wikimedia shard role: pair={pair_key} role={role}"
            )
        roles[role] = str(filename)
    ordered_pairs = sorted(pairs, key=lambda value: (value[1], value[2], value[0]))
    previous_end = 0
    shard_indexes: dict[tuple[int, int, int], int] = {}
    for shard_index, pair_key in enumerate(ordered_pairs, start=1):
        roles = pairs[pair_key]
        if set(roles) != {"articles", "index"}:
            raise ValueError(
                f"unpaired Wikimedia multistream shard: {wiki} pair={pair_key}"
            )
        _dump_part, start, end = pair_key
        if start <= previous_end:
            raise ValueError(
                f"overlapping Wikimedia page-id ranges: {wiki} pair={pair_key}"
            )
        previous_end = end
        shard_indexes[pair_key] = shard_index
    for row in files:
        row["shard_index"] = shard_indexes[
            (
                int(row["dump_part"]),
                int(row["page_id_start"]),
                int(row["page_id_end"]),
            )
        ]
    files.sort(key=lambda row: (int(row["shard_index"]), str(row["role"])))
    return {
        "name": f"wikimedia_{wiki}_{snapshot_date}_v1",
        "source_id": f"wikimedia:{wiki}",
        "wiki": wiki,
        "snapshot_date": snapshot_date,
        "dump_completed_at": str(job.get("updated") or ""),
        "dump_status": "done",
        "dumpstatus_url": status_url,
        "dumpstatus_sha256": hashlib.sha256(
            _response_bytes(response, payload)
        ).hexdigest(),
        "namespace_policy": "main_namespace_zero_non_redirect_pages_only",
        "language": metadata["language"],
        "output_language": metadata["output_language"],
        "domain": metadata["domain"],
        "canonical_host": metadata["canonical_host"],
        "script_normalization": metadata["normalization"],
        "license": metadata["license"],
        "license_review": metadata["license_review"],
        "quality_score": metadata["quality_score"],
        "min_chars": int(metadata["min_chars"]),
        "deduplication_priority": int(metadata["deduplication_priority"]),
        "preserve_pre_normalization_content_sha256": True,
        "selected_file_count": len(files),
        "article_shard_count": len(ordered_pairs),
        "selected_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
    }


def snapshot_inventory(
    *,
    wikis: list[str],
    snapshot_date: str,
    output_path: str,
    endpoint: str = DEFAULT_ENDPOINT,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    selected = [str(value).strip() for value in wikis if str(value).strip()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Wikimedia wiki selection must be non-empty and unique")
    client = session or _default_session()
    sources = [
        _snapshot_source(
            wiki=wiki,
            snapshot_date=str(snapshot_date),
            endpoint=str(endpoint),
            session=client,
        )
        for wiki in selected
    ]
    report = {
        "schema": INVENTORY_SCHEMA,
        "status": "complete",
        "snapshot_date": str(snapshot_date),
        "mutable_latest_urls_forbidden": True,
        "source_count": len(sources),
        "selected_file_count": sum(
            int(source["selected_file_count"]) for source in sources
        ),
        "selected_bytes": sum(int(source["selected_bytes"]) for source in sources),
        "sources": sources,
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
        description="Snapshot immutable, checksum-pinned Wikimedia article dumps."
    )
    parser.add_argument("--wiki", action="append", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = snapshot_inventory(
        wikis=list(args.wiki),
        snapshot_date=str(args.snapshot_date),
        output_path=str(args.output),
        endpoint=str(args.endpoint),
    )
    print(
        f"[DONE] sources={report['source_count']} "
        f"files={report['selected_file_count']} bytes={report['selected_bytes']:,} "
        f"output={Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
