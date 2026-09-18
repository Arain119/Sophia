from __future__ import annotations

import io
import json
import tarfile

import pyarrow.parquet as pq

from ml.tooling.scripts.data import build_tokenizer_code_corpus as mod


def _add(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def test_build_corpus_filters_members_and_records_provenance(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    archive_path = archives / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add(archive, "repo-rev/LICENSE", b"MIT license text")
        _add(archive, "repo-rev/src/main.py", b"def answer():\n    return 42\n" * 4)
        _add(archive, "repo-rev/tests/test_main.py", b"assert answer() == 42\n" * 4)
        _add(archive, "repo-rev/src/blob.py", b"bad\x00binary" * 20)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "source",
                        "repo_id": "org/repo",
                        "revision": "rev",
                        "archive_file": archive_path.name,
                        "archive_sha256": mod._sha256_file(archive_path),
                        "license": "MIT",
                        "license_note": "test",
                        "license_files": ["LICENSE"],
                        "domain": "code_python",
                        "include_prefixes": ["src/"],
                        "extensions": [".py"],
                        "exclude_prefixes": [],
                        "max_total_bytes": 10000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = mod.build_corpus(
        inventory_path=inventory,
        archives_dir=archives,
        output_dir=tmp_path / "output",
        max_file_bytes=10000,
        min_chars=20,
    )

    assert report["totals"]["kept_files"] == 1
    assert report["sources"][0]["license_file_sha256"]["LICENSE"]
    table = pq.read_table(report["sources"][0]["output_path"])
    assert table.num_rows == 1
    assert table.column("path").to_pylist() == ["src/main.py"]
    assert table.column("license").to_pylist() == ["MIT"]
