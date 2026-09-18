from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.tooling.scripts.data import clean_pretrain_corpus as mod
from ml.tooling.scripts.data import profile_pretrain_corpus as profile_mod
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _lineage(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = tmp_path / "raw"
    raw_file = raw_root / "source" / "data" / "part.parquet"
    raw_file.parent.mkdir(parents=True)
    base = "这是用于验证全量预训练清洗事务的中文文档，包含可靠来源和足够长度。"
    rows = [
        {"text": base + "第一段讨论数学与科学知识。", "score": 0.9},
        {"text": base + "第二段讨论历史与工程知识。", "score": 0.8},
        {"text": base + "第一段讨论数学与科学知识。", "score": 0.9},
    ]
    pq.write_table(pa.Table.from_pylist(rows), raw_file)
    file_sha256 = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    inventory = tmp_path / "inventory.json"
    _write_json(
        inventory,
        {
            "schema": "sophia_hf_source_inventory_v2",
            "sources": [
                {
                    "name": "source",
                    "repo_id": "owner/repo",
                    "revision": "a" * 40,
                    "license": "reviewed",
                    "language": "zh",
                    "domain": "knowledge",
                    "record_format": "parquet",
                    "text_column": "text",
                    "quality_score": "high",
                    "min_chars": 20,
                    "files": [
                        {
                            "path": "data/part.parquet",
                            "bytes": raw_file.stat().st_size,
                            "sha256": file_sha256,
                        }
                    ],
                }
            ],
        },
    )
    acquisition = tmp_path / "acquisition.json"
    _write_json(
        acquisition,
        {
            "schema": "sophia_hf_acquisition_report_v1",
            "inventory_path": str(inventory),
            "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
            "output_root": str(raw_root),
            "sources": [
                {
                    "name": "source",
                    "repo_id": "owner/repo",
                    "revision": "a" * 40,
                    "license": "reviewed",
                    "language": "zh",
                    "domain": "knowledge",
                    "record_format": "parquet",
                    "text_column": "text",
                    "quality_score": "high",
                    "min_chars": 20,
                    "files": [
                        {
                            "filename": "data/part.parquet",
                            "path": str(raw_file),
                            "bytes": raw_file.stat().st_size,
                            "sha256": file_sha256,
                        }
                    ],
                }
            ],
        },
    )
    policy = tmp_path / "policy.json"
    tokenizer_dir = Path(__file__).resolve().parents[2] / "ml" / "modeling" / "text"
    _write_json(
        policy,
        {
            "schema": "sophia_pretrain_data_admission_policy_v1",
            "forbidden_input_roots": [str(tmp_path / "legacy")],
            "required_tokenizer_bundle_sha1": compute_tokenizer_bundle_sha1(
                str(tokenizer_dir)
            ),
            "allow_legacy_assets_as_training_input": False,
        },
    )
    return acquisition, policy


def test_clean_pretrain_corpus_is_resumable_and_deduplicates(tmp_path) -> None:
    acquisition, policy = _lineage(tmp_path)
    output = tmp_path / "clean"

    first = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(output),
        admission_policy=str(policy),
        rows_per_file=1,
    )
    second = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(output),
        admission_policy=str(policy),
        rows_per_file=1,
    )

    assert first["status"] == "complete"
    assert second["status"] == "complete"
    source = second["sources"][0]
    assert source["stats"]["docs_in"] == 3
    assert source["stats"]["docs_kept"] == 2
    assert source["stats"]["drop_exact_duplicate"] == 1
    assert len(source["output_files"]) == 2
    assert all(Path(path).is_file() for path in source["output_files"])
    assert {row["path"] for row in source["output_file_inventory"]} == set(
        source["output_files"]
    )
    assert all(
        row["sha256"] == hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
        for row in source["output_file_inventory"]
    )

    token_profile = profile_mod.profile_pretrain_corpus(
        clean_root=str(output),
        tokenizer_path=str(
            Path(__file__).resolve().parents[2] / "ml" / "modeling" / "text"
        ),
        admission_policy=str(policy),
        output_path=str(tmp_path / "token_profile.json"),
        batch_size=2,
    )
    assert token_profile["status"] == "complete"
    assert token_profile["documents"] == 2
    assert token_profile["tokens"] > 2
    assert token_profile["tokens_by"]["source"]["source"] == token_profile["tokens"]


def test_benchmark_contamination_is_removed_before_dedup_state(tmp_path) -> None:
    acquisition, policy = _lineage(tmp_path)
    benchmark = tmp_path / "benchmark.jsonl"
    prompt = (
        "这是用于验证全量预训练清洗事务的中文文档，包含可靠来源和足够长度。"
        "第一段讨论数学与科学知识。"
    )
    benchmark.write_text(
        json.dumps({"id": "fixture", "prompt": prompt}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "decontaminated"

    report = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(output),
        admission_policy=str(policy),
        benchmark_jsonl=[str(benchmark)],
    )

    stats = report["sources"][0]["stats"]
    assert stats["drop_benchmark_contamination"] == 2
    assert stats["docs_kept"] == 1
    with sqlite3.connect(output / "cleaning_state.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


def test_clean_pretrain_corpus_supports_native_working_state_database(
    tmp_path,
) -> None:
    acquisition, policy = _lineage(tmp_path)
    output = tmp_path / "clean"
    state_database = tmp_path / "native" / "cleaning.sqlite3"

    result = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(output),
        admission_policy=str(policy),
        rows_per_file=2,
        working_state_database=str(state_database),
    )

    assert result["status"] == "complete"
    assert result["state_database"] == str(state_database.resolve())
    assert state_database.is_file()
    assert not (output / "cleaning_state.sqlite3").exists()


def test_resume_publishes_stage_when_interrupt_arrives_after_sqlite_commit(
    tmp_path,
    monkeypatch,
) -> None:
    acquisition, policy = _lineage(tmp_path)
    output = tmp_path / "clean"
    original_commit = mod.SqliteDeduper.commit

    def commit_then_interrupt(deduper: mod.SqliteDeduper) -> None:
        original_commit(deduper)
        raise KeyboardInterrupt

    monkeypatch.setattr(mod.SqliteDeduper, "commit", commit_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        mod.clean_pretrain_corpus(
            acquisition_report=str(acquisition),
            output_root=str(output),
            admission_policy=str(policy),
            rows_per_file=1,
        )
    assert list((output / ".staging").rglob("*.parquet"))

    monkeypatch.setattr(mod.SqliteDeduper, "commit", original_commit)
    resumed = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(output),
        admission_policy=str(policy),
        rows_per_file=1,
    )

    assert resumed["status"] == "complete"
    assert resumed["sources"][0]["stats"]["docs_kept"] == 2
    assert all(Path(path).is_file() for path in resumed["sources"][0]["output_files"])


def test_parallel_preprocessing_matches_serial_cleaning(tmp_path) -> None:
    acquisition, policy = _lineage(tmp_path)
    serial = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(tmp_path / "serial"),
        admission_policy=str(policy),
        rows_per_file=2,
    )
    parallel = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(tmp_path / "parallel"),
        admission_policy=str(policy),
        rows_per_file=2,
        preprocess_workers=2,
    )

    assert parallel["sources"][0]["stats"] == serial["sources"][0]["stats"]

    def document_hashes(report: dict[str, object]) -> list[str]:
        source = report["sources"][0]
        return [
            str(value)
            for output in source["output_files"]
            for value in pq.read_table(output)["document_sha256"].to_pylist()
        ]

    assert document_hashes(parallel) == document_hashes(serial)


def test_parallel_preprocessing_queue_is_bounded_and_ordered() -> None:
    consumed = 0

    def tasks():
        nonlocal consumed
        for index in range(10):
            consumed += 1
            yield (), {"index": index}

    class Completed:
        def __init__(self, value: object) -> None:
            self.value = value

        def result(self) -> object:
            return self.value

    class ImmediateExecutor:
        def submit(self, _function: object, task: object) -> Completed:
            _records, source = task
            return Completed(([{"index": source["index"]}], {}))

    results = mod._bounded_ordered_process_map(
        ImmediateExecutor(),
        tasks(),
        max_pending=3,
    )
    indexes = []
    for produced, (rows, _stats) in enumerate(results, start=1):
        indexes.append(rows[0]["index"])
        assert consumed <= min(produced + 2, 10)

    assert indexes == list(range(10))


def test_code_cleaning_rejects_vendor_generated_and_secret_content() -> None:
    normal = "\n".join(f"def function_{index}(): return {index}" for index in range(20))
    assert (
        mod.clean_source_document(
            normal,
            language="code",
            min_chars=100,
            source_path="src/module.py",
        ).reason
        == ""
    )
    assert (
        mod.clean_source_document(
            normal,
            language="code",
            min_chars=100,
            source_path="node_modules/package/index.js",
        ).reason
        == "code_vendor"
    )
    assert (
        mod.clean_source_document(
            "// Generated by a schema compiler. Do not edit.\n" + normal,
            language="code",
            min_chars=100,
            source_path="src/generated.py",
        ).reason
        == "code_generated"
    )
    assert (
        mod.clean_source_document(
            normal + "\n-----BEGIN PRIVATE KEY-----\nnot-a-training-secret",
            language="code",
            min_chars=100,
            source_path="src/config.py",
        ).reason
        == "code_secret"
    )


def test_quality_proxy_applies_only_to_explicit_web_sources() -> None:
    proxy = mod.ChineseQualityProxy(
        artifact={"fixture": True},
        minimum_cjk_ratio=0.25,
        applicable_source_names=("web_source",),
    )

    assert proxy.applies(cjk_ratio=0.9, language="zh", source_name="web_source")
    assert not proxy.applies(
        cjk_ratio=0.9,
        language="zh",
        source_name="wikimedia_zhwiki_20260701_v1",
    )


def test_curated_quality_exemption_accepts_snapshot_revision(tmp_path) -> None:
    evidence = tmp_path / "evidence.json"
    payload = {
        "schema": "sophia_curated_chinese_source_proxy_exemption_v1",
        "status": "pass",
        "source": {"repo_id": "wikimedia/wikipedia", "revision": "20260701"},
        "gate_results": {"manual_domain_review": True},
    }
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()

    rows = mod._load_curated_source_exemptions(
        config={
            "curated_source_exemptions": [
                {
                    "repo_id": "wikimedia/wikipedia",
                    "revision": "20260701",
                    "reason": "curated snapshot fixture",
                    "evidence_report_path": str(evidence),
                    "evidence_report_sha256": digest,
                }
            ]
        }
    )

    assert rows[0]["revision"] == "20260701"


def test_legacy_database_prefers_hash_verified_native_working_copy(tmp_path) -> None:
    working = tmp_path / "native" / "legacy.sqlite3"
    working.parent.mkdir()
    working.write_bytes(b"verified native copy")
    authoritative = tmp_path / "mounted" / "legacy.sqlite3"
    authoritative.parent.mkdir()
    authoritative.write_bytes(b"different mounted copy")
    expected = hashlib.sha256(working.read_bytes()).hexdigest()

    selected = mod._select_verified_legacy_database_path(
        report={
            "working_database_path": str(working),
            "database_path": str(authoritative),
        },
        expected_sha256=expected,
    )

    assert selected == working.resolve()


def test_admission_policy_rejects_legacy_raw_root(tmp_path) -> None:
    legacy = tmp_path / "legacy"
    policy = tmp_path / "policy.json"
    _write_json(
        policy,
        {
            "schema": "sophia_pretrain_data_admission_policy_v1",
            "forbidden_input_roots": [str(legacy)],
            "allow_legacy_assets_as_training_input": False,
        },
    )

    with pytest.raises(ValueError, match="legacy training input is forbidden"):
        mod._validate_admission_policy(
            acquisition={"output_root": str(legacy / "nested")},
            policy_path=str(policy),
        )


def test_acquisition_source_order_prefers_explicit_deduplication_priority() -> None:
    ordered = mod._ordered_acquisition_sources(
        {
            "sources": [
                {"name": "supporting"},
                {"name": "fineweb"},
                {"name": "wikipedia", "deduplication_priority": 100},
            ]
        }
    )

    assert [source["name"] for source in ordered] == [
        "wikipedia",
        "supporting",
        "fineweb",
    ]

    with pytest.raises(ValueError, match="must be an integer"):
        mod._ordered_acquisition_sources(
            {"sources": [{"name": "invalid", "deduplication_priority": "100"}]}
        )


def test_filesystem_object_sets_accept_samefile_aliases(tmp_path) -> None:
    original = tmp_path / "original"
    alias = tmp_path / "alias"
    original.write_text("same inode", encoding="utf-8")
    alias.hardlink_to(original)

    assert mod._same_filesystem_objects({original.resolve()}, {alias.resolve()})
    assert not mod._same_filesystem_objects(
        {original.resolve()}, {tmp_path / "missing"}
    )


def test_parquet_reader_rejects_falsely_declared_provenance_column(tmp_path) -> None:
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([{"text": "valid text"}]), path)

    with pytest.raises(ValueError, match="missing configured parquet columns.*url"):
        list(
            mod._iter_records(
                path,
                {
                    "record_format": "parquet",
                    "text_column": "text",
                    "url_column": "url",
                },
            )
        )


def test_inventory_rejects_legacy_upstream_repo_id() -> None:
    with pytest.raises(ValueError, match="legacy upstream repository is forbidden"):
        mod._validate_inventory_upstreams(
            inventory={"sources": [{"repo_id": "Legacy/Repo"}]},
            policy={"forbidden_upstream_repo_ids": ["legacy/repo"]},
        )


def test_v7_policy_allows_audited_skypile_but_rejects_mirror() -> None:
    policy_path = (
        Path(__file__).parents[2]
        / "configs"
        / "data"
        / "pretrain_data_admission_policy.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    mod._validate_inventory_upstreams(
        inventory={"sources": [{"repo_id": "Skywork/SkyPile-150B"}]},
        policy=policy,
    )
    with pytest.raises(ValueError, match="legacy upstream repository is forbidden"):
        mod._validate_inventory_upstreams(
            inventory={"sources": [{"repo_id": "hfxunlp/SkyPile-150B"}]},
            policy=policy,
        )


def test_legacy_database_override_is_hash_verified(tmp_path, monkeypatch) -> None:
    report_database = tmp_path / "missing.sqlite3"
    override_database = tmp_path / "cached.sqlite3"
    override_database.write_bytes(b"verified cache")
    expected_sha256 = hashlib.sha256(override_database.read_bytes()).hexdigest()
    monkeypatch.setenv(
        "SOPHIA_LEGACY_EXCLUSION_DATABASE_PATH",
        str(override_database),
    )

    selected = mod._select_verified_legacy_database_path(
        report={"database_path": str(report_database)},
        expected_sha256=expected_sha256,
    )

    assert selected == override_database.resolve()


def test_required_benchmark_decontamination_assets_are_hash_and_count_pinned(
    tmp_path,
) -> None:
    signatures = tmp_path / "signatures.jsonl"
    signatures.write_text(
        json.dumps({"id": "one", "prompt": "a sufficiently long pinned prompt"}) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(signatures.read_bytes()).hexdigest()
    policy = {
        "benchmark_decontamination": {
            "required": True,
            "required_signature_assets": [
                {"path": str(signatures), "sha256": digest, "signatures": 1}
            ],
        }
    }

    assert mod._validate_benchmark_decontamination(
        policy=policy,
        benchmark_paths=[str(signatures)],
    ) == [str(signatures.resolve())]

    with pytest.raises(ValueError, match="missing required"):
        mod._validate_benchmark_decontamination(
            policy=policy,
            benchmark_paths=[],
        )

    policy["benchmark_decontamination"]["required_signature_assets"][0]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        mod._validate_benchmark_decontamination(
            policy=policy,
            benchmark_paths=[str(signatures)],
        )


def test_cleaning_drops_documents_found_in_legacy_content_index(tmp_path) -> None:
    acquisition, policy = _lineage(tmp_path)
    raw_file = tmp_path / "raw" / "source" / "data" / "part.parquet"
    legacy_text = str(pq.read_table(raw_file, columns=["text"])["text"][0].as_py())
    normalized = mod.dedup_normalize(legacy_text)
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()

    legacy_root = tmp_path / "legacy_corpus"
    legacy_root.mkdir()
    database = legacy_root / "hashes.sqlite3"
    connection = sqlite3.connect(str(database))
    try:
        connection.execute(
            "CREATE TABLE hashes (sha256 BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO hashes(sha256) VALUES (?)", (sqlite3.Binary(digest),)
        )
        connection.commit()
    finally:
        connection.close()
    report = legacy_root / "report.json"
    _write_json(
        report,
        {
            "schema": mod.LEGACY_EXCLUSION_SCHEMA,
            "status": "complete",
            "index_schema": mod.LEGACY_INDEX_SCHEMA,
            "corpus_roots": [str(legacy_root)],
            "candidate_files": 1,
            "processed_files": 1,
            "unique_hashes": 1,
            "database_path": str(database),
            "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        },
    )
    policy_payload = json.loads(policy.read_text())
    policy_payload["legacy_content_exclusion"] = {
        "required": True,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "required_corpus_roots": [str(legacy_root)],
        "match_mode": "exact_sha256_of_nfkc_casefold_remove_whitespace",
    }
    _write_json(policy, policy_payload)

    result = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(tmp_path / "clean_with_exclusion"),
        admission_policy=str(policy),
        rows_per_file=2,
    )

    stats = result["sources"][0]["stats"]
    assert stats["drop_legacy_exact_overlap"] == 2
    assert stats["docs_kept"] == 1
    assert result["legacy_content_exclusion_report_sha256"]
    assert result["legacy_content_exclusion_database_sha256"]


def test_cleaning_can_report_legacy_overlap_without_dropping_it(tmp_path) -> None:
    acquisition, policy = _lineage(tmp_path)
    raw_file = tmp_path / "raw" / "source" / "data" / "part.parquet"
    legacy_text = str(pq.read_table(raw_file, columns=["text"])["text"][0].as_py())
    digest = hashlib.sha256(mod.dedup_normalize(legacy_text).encode()).digest()

    legacy_root = tmp_path / "legacy_report_only"
    legacy_root.mkdir()
    database = legacy_root / "hashes.sqlite3"
    connection = sqlite3.connect(str(database))
    try:
        connection.execute(
            "CREATE TABLE hashes (sha256 BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO hashes(sha256) VALUES (?)", (sqlite3.Binary(digest),)
        )
        connection.commit()
    finally:
        connection.close()
    report = legacy_root / "report.json"
    _write_json(
        report,
        {
            "schema": mod.LEGACY_EXCLUSION_SCHEMA,
            "status": "complete",
            "index_schema": mod.LEGACY_INDEX_SCHEMA,
            "corpus_roots": [str(legacy_root)],
            "candidate_files": 1,
            "processed_files": 1,
            "unique_hashes": 1,
            "database_path": str(database),
            "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        },
    )
    policy_payload = json.loads(policy.read_text())
    policy_payload["legacy_content_exclusion"] = {
        "required": True,
        "action": "report",
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "required_corpus_roots": [str(legacy_root)],
        "match_mode": "exact_sha256_of_nfkc_casefold_remove_whitespace",
    }
    _write_json(policy, policy_payload)

    result = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(tmp_path / "clean_with_overlap_report"),
        admission_policy=str(policy),
        rows_per_file=2,
    )

    stats = result["sources"][0]["stats"]
    assert stats["legacy_exact_overlap"] == 2
    assert stats.get("drop_legacy_exact_overlap", 0) == 0
    assert stats["docs_kept"] == 2
    assert result["legacy_content_overlap_action"] == "report"
    flags = []
    for output in result["sources"][0]["output_files"]:
        flags.extend(pq.read_table(output)["legacy_exact_overlap"].to_pylist())
    assert sorted(flags) == [False, True]


def test_legacy_v3_index_matches_near_modified_content(tmp_path) -> None:
    legacy_text = "\n".join(
        f"第{index}节讨论数学、科学、历史、工程和语言知识，并给出不同的上下文示例。"
        for index in range(120)
    )
    modified_text = legacy_text.replace("数学", "算术", 1)
    legacy_signature = mod.near_duplicate_signature(legacy_text, size=8)
    modified_signature = mod.near_duplicate_signature(modified_text, size=8)
    digest = hashlib.sha256(mod.dedup_normalize(legacy_text).encode()).digest()
    database = tmp_path / "legacy-v3.sqlite3"
    connection = sqlite3.connect(str(database))
    try:
        connection.execute(
            "CREATE TABLE hashes (sha256 BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE signatures (sha256 BLOB PRIMARY KEY, signature BLOB NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE bands (band_key BLOB NOT NULL, sha256 BLOB NOT NULL, PRIMARY KEY (band_key, sha256)) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO hashes(sha256) VALUES (?)", (sqlite3.Binary(digest),)
        )
        connection.execute(
            "INSERT INTO signatures(sha256, signature) VALUES (?, ?)",
            (
                sqlite3.Binary(digest),
                sqlite3.Binary(mod._signature_blob(legacy_signature)),
            ),
        )
        connection.executemany(
            "INSERT INTO bands(band_key, sha256) VALUES (?, ?)",
            (
                (sqlite3.Binary(band), sqlite3.Binary(digest))
                for band in mod._band_keys(legacy_signature)
            ),
        )
        connection.commit()
    finally:
        connection.close()

    exclusion = mod.LegacyContentExclusion(
        database_path=database,
        index_schema=mod.LEGACY_NEAR_INDEX_SCHEMA,
    )
    try:
        modified_sha256 = hashlib.sha256(
            mod.dedup_normalize(modified_text).encode()
        ).hexdigest()
        assert not exclusion.contains(modified_sha256)
        assert exclusion.contains_near(modified_signature)
        assert exclusion.classify_batch(
            [
                (digest.hex(), legacy_signature),
                (modified_sha256, modified_signature),
                ("f" * 64, (101, 102, 103, 104, 105, 106, 107, 108)),
            ]
        ) == [(True, False), (False, True), (False, False)]
    finally:
        exclusion.close()


def test_legacy_v2_index_requires_near_lookup_tables(tmp_path) -> None:
    database = tmp_path / "incomplete-v2.sqlite3"
    connection = sqlite3.connect(str(database))
    try:
        connection.execute(
            "CREATE TABLE hashes (sha256 BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="missing tables"):
        mod.LegacyContentExclusion(
            database_path=database,
            index_schema=mod.LEGACY_NEAR_INDEX_SCHEMA,
        )


def test_batched_sqlite_dedup_matches_sequential_order(tmp_path) -> None:
    documents = [
        ("a" * 64, (1, 2, 3, 4, 5, 6, 7, 8)),
        ("b" * 64, (1, 2, 3, 4, 5, 6, 70, 80)),
        ("a" * 64, (1, 2, 3, 4, 5, 6, 7, 8)),
        ("c" * 64, (11, 12, 13, 14, 15, 16, 17, 18)),
    ]
    sequential = mod.SqliteDeduper(tmp_path / "sequential.sqlite3")
    batched = mod.SqliteDeduper(tmp_path / "batched.sqlite3")
    try:
        expected = [
            sequential.add_if_unique(sha256=sha256, signature=signature)
            for sha256, signature in documents
        ]
        actual = batched.add_batch_if_unique(documents)
    finally:
        sequential.close()
        batched.close()

    assert expected == ["", "near_duplicate", "exact_duplicate", ""]
    assert actual == expected


def test_cleaning_batches_and_drops_low_chinese_quality_scores(
    tmp_path, monkeypatch
) -> None:
    acquisition, policy = _lineage(tmp_path)

    class FakeQualityProxy:
        enabled = True
        report_path = None
        report_sha256 = "report-hash"
        model_sha256 = "model-hash"
        labels_sha256 = "labels-hash"
        minimum_score = 0.5
        minimum_cjk_ratio = 0.25
        curated_source_exemptions: tuple[dict[str, str], ...] = ()

        def is_exempt(self, *, repo_id: str, revision: str) -> bool:
            del repo_id, revision
            return False

        def applies(
            self,
            *,
            cjk_ratio: float,
            language: str,
            source_name: str = "",
            repo_id: str = "",
            revision: str = "",
        ) -> bool:
            del source_name, repo_id, revision
            return language != "code" and cjk_ratio >= self.minimum_cjk_ratio

        def score(self, texts: list[str]) -> list[float]:
            return [0.9 if "数学" in text else 0.1 for text in texts]

    monkeypatch.setattr(
        mod,
        "_load_chinese_quality_proxy",
        lambda **_kwargs: FakeQualityProxy(),
    )
    result = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(tmp_path / "quality_clean"),
        admission_policy=str(policy),
        rows_per_file=2,
        quality_score_batch_size=2,
    )

    stats = result["sources"][0]["stats"]
    assert stats["chinese_quality_proxy_docs_scored"] == 3
    assert stats["drop_chinese_quality_proxy"] == 1
    assert stats["drop_exact_duplicate"] == 1
    assert stats["docs_kept"] == 1
    table = pq.read_table(result["sources"][0]["output_files"][0])
    assert table["chinese_quality_proxy_score"].to_pylist() == pytest.approx([0.9])
    assert result["chinese_quality_proxy_model_sha256"] == "model-hash"


def test_cleaning_records_hash_pinned_curated_quality_exemption(
    tmp_path, monkeypatch
) -> None:
    acquisition, policy = _lineage(tmp_path)
    exemption = {
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "reason": "curated fixture",
        "evidence_report_path": str(tmp_path / "evidence.json"),
        "evidence_report_sha256": "evidence-hash",
    }

    class FakeQualityProxy:
        enabled = True
        report_path = None
        report_sha256 = "report-hash"
        model_sha256 = "model-hash"
        labels_sha256 = "labels-hash"
        minimum_score = 0.5
        minimum_cjk_ratio = 0.25

        def __init__(self) -> None:
            self.curated_source_exemptions = (exemption,)

        def is_exempt(self, *, repo_id: str, revision: str) -> bool:
            return repo_id == "owner/repo" and revision == "a" * 40

        def applies(self, **_kwargs: object) -> bool:
            return False

        def score(self, _texts: list[str]) -> list[float]:
            raise AssertionError("curated source must not be proxy scored")

    monkeypatch.setattr(
        mod,
        "_load_chinese_quality_proxy",
        lambda **_kwargs: FakeQualityProxy(),
    )
    result = mod.clean_pretrain_corpus(
        acquisition_report=str(acquisition),
        output_root=str(tmp_path / "quality_exempt_clean"),
        admission_policy=str(policy),
        rows_per_file=2,
    )

    stats = result["sources"][0]["stats"]
    assert stats["chinese_quality_proxy_curated_exempt_docs"] == 3
    assert stats.get("chinese_quality_proxy_docs_scored", 0) == 0
    assert stats["docs_kept"] == 2
    assert result["chinese_quality_proxy_curated_source_exemptions"] == [exemption]
    scores = []
    for output in result["sources"][0]["output_files"]:
        scores.extend(pq.read_table(output)["chinese_quality_proxy_score"].to_pylist())
    assert scores == [None, None]


def test_curated_quality_exemption_requires_passing_hash_pinned_evidence(
    tmp_path,
) -> None:
    evidence = tmp_path / "evidence.json"
    payload = {
        "schema": "sophia_curated_chinese_source_proxy_exemption_v1",
        "status": "pass",
        "source": {"repo_id": "owner/repo", "revision": "a" * 40},
        "gate_results": {"sample": True, "rights": True},
    }
    _write_json(evidence, payload)
    config = {
        "curated_source_exemptions": [
            {
                "repo_id": "owner/repo",
                "revision": "a" * 40,
                "reason": "pinned curated corpus",
                "evidence_report_path": str(evidence),
                "evidence_report_sha256": hashlib.sha256(
                    evidence.read_bytes()
                ).hexdigest(),
            }
        ]
    }

    exemptions = mod._load_curated_source_exemptions(config=config)
    assert exemptions[0]["repo_id"] == "owner/repo"

    config["curated_source_exemptions"][0]["evidence_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        mod._load_curated_source_exemptions(config=config)


def test_quality_proxy_policy_rejects_threshold_below_validated_value(
    tmp_path, monkeypatch
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"text":"样本","label":3}\n', encoding="utf-8")
    model = tmp_path / "model.pkl"
    model.write_bytes(b"fixture")
    report = tmp_path / "report.json"
    _write_json(
        report,
        {
            "schema": mod.QUALITY_PROXY_REPORT_SCHEMA,
            "status": "pass",
            "gate_results": {"test": True},
            "model": {
                "schema": mod.QUALITY_PROXY_MODEL_SCHEMA,
                "path": str(model),
                "sha256": "model-hash",
                "sklearn_version": "1.7.2",
                "threshold": 0.68,
            },
            "source": {
                "labels_path": str(labels),
                "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            },
        },
    )
    monkeypatch.setattr(
        mod,
        "load_quality_proxy",
        lambda *_args, **_kwargs: {
            "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "sklearn_version": "1.7.2",
        },
    )

    with pytest.raises(ValueError, match="may not be lower"):
        mod._load_chinese_quality_proxy(
            policy={
                "chinese_quality_proxy": {
                    "required": True,
                    "report_path": str(report),
                    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                    "minimum_score": 0.67,
                }
            }
        )


def test_quality_proxy_policy_rejects_failed_report(tmp_path) -> None:
    report = tmp_path / "report.json"
    _write_json(
        report,
        {
            "schema": mod.QUALITY_PROXY_REPORT_SCHEMA,
            "status": "fail",
        },
    )

    with pytest.raises(ValueError, match="not a supported pass report"):
        mod._load_chinese_quality_proxy(
            policy={
                "chinese_quality_proxy": {
                    "required": True,
                    "report_path": str(report),
                    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                }
            }
        )


def test_jsonl_gzip_reader_preserves_nested_metadata(tmp_path) -> None:
    path = tmp_path / "data.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"text": "print('ok')", "metadata": {"license": "MIT"}}) + "\n"
        )

    rows = list(
        mod._iter_records(
            path,
            {"record_format": "jsonl_gzip", "text_column": "text"},
        )
    )

    assert rows == [{"text": "print('ok')", "metadata": {"license": "MIT"}}]


def test_jsonl_reader_and_license_whitelist(tmp_path) -> None:
    path = tmp_path / "data.json"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "content": "def permitted_example(value):\n    return value + 1234567890",
                        "licenses": ["MIT", "Apache-2.0"],
                    }
                ),
                json.dumps(
                    {
                        "content": "def copyleft_example(value):\n    return value + 1234567890",
                        "licenses": ["MIT", "GPL-3.0"],
                    }
                ),
                json.dumps(
                    {
                        "content": "def unknown_example(value):\n    return value + 1234567890",
                        "licenses": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = {
        "record_format": "jsonl",
        "text_column": "content",
        "license_column": "licenses",
        "allowed_licenses": ["MIT", "Apache-2.0"],
        "language": "code",
        "min_chars": 1,
    }
    rows = tuple(mod._iter_records(path, source))
    stats: Counter[str] = Counter()

    kept = mod._prepare_clean_candidates(rows, source=source, stats=stats)

    assert [row["text"] for row in kept] == [
        "def permitted_example(value):\n    return value + 1234567890"
    ]
    assert kept[0]["license"] == "Apache-2.0 OR MIT"
    assert stats["drop_disallowed_license"] == 2


def test_candidate_language_ratios_describe_text_after_title_is_added() -> None:
    body = (
        "A university library preserves manuscripts, scientific journals, maps, "
        "and oral histories for future researchers. Careful cataloguing connects "
        "each source with its author, publication context, and evidence. Students "
        "compare competing explanations, reproduce calculations, inspect primary "
        "documents, and explain uncertainty before reaching a conclusion. Digital "
        "collections make this scholarship accessible across many disciplines."
    )
    source = {
        "text_column": "text",
        "title_column": "title",
        "language": "en",
        "min_chars": 1,
    }
    candidates = mod._prepare_clean_candidates(
        ({"text": body, "title": "\u4e2d\u6587\u6807\u9898" * 10},),
        source=source,
        stats=Counter(),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    expected_cjk, expected_latin = mod._language_ratios(candidate["text"])
    assert candidate["final_cjk_ratio"] == expected_cjk
    assert candidate["final_latin_ratio"] == expected_latin
    assert candidate["final_cjk_ratio"] > candidate["decision"].cjk_ratio


def test_acquisition_content_digest_ignores_bundle_serialization() -> None:
    bundle = {
        "derivative_reports": [{"path": "/d/report.json", "sha256": "a" * 64}],
        "sources": [
            {
                "name": "alpha",
                "repo_id": "org/alpha",
                "revision": "rev-a",
                "files": [
                    {"path": "/raw/a1.parquet", "sha256": "1" * 64},
                    {"path": "/raw/a2.parquet", "sha256": "2" * 64},
                ],
            },
            {"name": "beta", "repo_id": "org/beta", "revision": "rev-b", "files": []},
        ],
    }
    baseline = mod._acquisition_content_sha256(bundle)

    # Re-materializing the bundle reorders sources/files and refreshes the
    # embedded derivative hashes without selecting a different corpus.
    reordered = {
        "derivative_reports": [{"path": "/d/report.json", "sha256": "b" * 64}],
        "sources": [
            {"name": "beta", "repo_id": "org/beta", "revision": "rev-b", "files": []},
            {
                "name": "alpha",
                "repo_id": "org/alpha",
                "revision": "rev-a",
                "files": [
                    {"path": "/raw/a2.parquet", "sha256": "2" * 64},
                    {"path": "/raw/a1.parquet", "sha256": "1" * 64},
                ],
            },
        ],
    }
    assert mod._acquisition_content_sha256(reordered) == baseline

    changed_content = json.loads(json.dumps(bundle))
    changed_content["sources"][0]["files"][0]["sha256"] = "9" * 64
    assert mod._acquisition_content_sha256(changed_content) != baseline

    added_file = json.loads(json.dumps(bundle))
    added_file["sources"][1]["files"].append(
        {"path": "/raw/b1.parquet", "sha256": "3" * 64}
    )
    assert mod._acquisition_content_sha256(added_file) != baseline

    changed_revision = json.loads(json.dumps(bundle))
    changed_revision["sources"][0]["revision"] = "rev-a2"
    assert mod._acquisition_content_sha256(changed_revision) != baseline
