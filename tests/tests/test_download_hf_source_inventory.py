from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.tooling.scripts.data import download_hf_source_inventory as mod


def test_prioritizes_evenly_spaced_files_without_changing_supply() -> None:
    files = list(range(10))

    ordered = mod._prioritize_distributed_files(files, 4)

    assert ordered[:4] == [0, 3, 6, 9]
    assert sorted(ordered) == files
    assert len(ordered) == len(files)


def _fake_curl_response(command: list[str], content: bytes) -> SimpleNamespace:
    output = Path(command[command.index("--output") + 1])
    headers = Path(command[command.index("--dump-header") + 1])
    start = int(command[command.index("--range") + 1].split("-", 1)[0])
    output.write_bytes(content)
    end = start + len(content) - 1
    headers.write_text(
        f"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{end + 1}\r\n\r\n",
        encoding="ascii",
    )
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_download_inventory_pins_revision_and_records_hash(
    monkeypatch, tmp_path
) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "sophia_hf_source_inventory_v2",
                "sources": [
                    {
                        "name": "source_a",
                        "repo_id": "owner/repo",
                        "revision": "a" * 40,
                        "license": "test-license",
                        "files": [
                            {
                                "path": "data/file.parquet",
                                "bytes": len(b"dataset"),
                                "sha256": hashlib.sha256(b"dataset").hexdigest(),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append(dict(kwargs))
        path = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dataset")
        return str(path)

    monkeypatch.setattr(mod, "_download_https", fake_download)
    report = mod.download_inventory(
        inventory_path=str(inventory),
        output_root=str(tmp_path / "out"),
    )

    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["filename"] == "data/file.parquet"
    row = report["sources"][0]["files"][0]
    assert row["sha256"] == hashlib.sha256(b"dataset").hexdigest()
    assert (tmp_path / "out" / "acquisition_report.json").is_file()


def test_load_inventory_rejects_unhashed_v1_files(tmp_path) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "sophia_hf_source_inventory_v1",
                "sources": [
                    {
                        "name": "source_a",
                        "repo_id": "owner/repo",
                        "revision": "a" * 40,
                        "files": ["data/file.parquet"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected schema"):
        mod.load_inventory(str(inventory))


def test_inventory_file_reuses_verified_target_before_hf_transport(
    monkeypatch, tmp_path
) -> None:
    content = b"already-verified"
    target = tmp_path / "source" / "data" / "file.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    monkeypatch.setattr(
        mod,
        "hf_hub_download",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("redownloaded")),
    )

    row = mod._download_inventory_file(
        file_index=1,
        file_count=1,
        filename_value={
            "path": "data/file.parquet",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        name="source",
        repo_id="owner/repo",
        revision="a" * 40,
        source_root=tmp_path / "source",
        transport="hf",
        endpoint="https://huggingface.co",
    )

    assert row["path"] == str(target.resolve())
    assert row["sha256"] == hashlib.sha256(content).hexdigest()


def test_dataset_file_url_respects_endpoint_and_quotes(monkeypatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")

    url = mod._dataset_file_url(
        repo_id="owner/repo",
        revision="a" * 40,
        filename="data/file name.jsonl",
    )

    assert url == (
        "https://mirror.example/datasets/owner/repo/resolve/"
        f"{'a' * 40}/data/file%20name.jsonl?download=true"
    )


def test_curl_header_parser_handles_proxy_tunnel_and_redirect_chain() -> None:
    headers = (
        "HTTP/1.1 200 Connection established\r\n\r\n"
        "HTTP/2 302 Found\r\nLocation: https://cdn.example/file\r\n\r\n"
        "HTTP/1.1 200 Connection established\r\n\r\n"
        "HTTP/2 206 Partial Content\r\n"
        "Content-Range: bytes 5-9/10\r\n\r\n"
    )

    responses = mod._parse_curl_response_headers(headers)

    assert [status for status, _headers in responses] == [200, 302, 200, 206]
    assert "Content-Range: bytes 5-9/10" in responses[-1][1]


def test_https_transport_does_not_truncate_when_range_is_ignored(
    monkeypatch, tmp_path
) -> None:
    partial = tmp_path / "data" / "file.jsonl.part"
    partial.parent.mkdir(parents=True)
    original = b"existing-prefix"
    partial.write_bytes(original)

    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            del chunk_size
            yield b"ignored-range-body"

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(mod.requests, "get", lambda *_args, **_kwargs: Response())

    with pytest.raises(RuntimeError, match="ignored Range"):
        mod._download_https(
            repo_id="owner/repo",
            revision="a" * 40,
            filename="data/file.jsonl",
            local_dir=str(tmp_path),
            attempts=1,
        )

    assert partial.read_bytes() == original


def test_download_curl_uses_resolve_and_records_partial(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(command))
        return _fake_curl_response(command, b"curl-dataset")

    monkeypatch.setenv("SOPHIA_HF_RESOLVE_IP", "192.0.2.10")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    path = mod._download_curl(
        repo_id="owner/repo",
        revision="b" * 40,
        filename="data/file.jsonl",
        local_dir=str(tmp_path),
    )

    assert Path(path).read_bytes() == b"curl-dataset"
    assert "huggingface.co:443:192.0.2.10" in calls[0]
    assert calls[0][calls[0].index("--range") + 1] == "0-"
    assert not Path(path + ".part").exists()


def test_download_curl_always_resumes_across_internal_retries(
    monkeypatch, tmp_path
) -> None:
    calls: list[list[str]] = []
    partial = tmp_path / "data" / "file.jsonl.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"already-downloaded")

    def fake_run(command, *, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(command))
        return _fake_curl_response(command, b"complete")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._download_curl(
        repo_id="owner/repo",
        revision="b" * 40,
        filename="data/file.jsonl",
        local_dir=str(tmp_path),
    )

    command = calls[0]
    assert command[command.index("--range") + 1] == f"{len(b'already-downloaded')}-"
    assert command.count("--range") == 1


def test_download_curl_backs_off_between_failed_attempts(
    monkeypatch, tmp_path
) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is False and capture_output is True and text is True
        calls.append(list(command))
        if len(calls) == 1:
            return SimpleNamespace(returncode=28, stdout="", stderr="timeout")
        return _fake_curl_response(command, b"complete-after-backoff")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.time, "sleep", sleeps.append)
    path = mod._download_curl(
        repo_id="owner/repo",
        revision="b" * 40,
        filename="data/file.jsonl",
        local_dir=str(tmp_path),
        attempts=2,
    )

    assert Path(path).read_bytes() == b"complete-after-backoff"
    assert len(calls) == 2
    assert sleeps == [2]


def test_download_curl_does_not_truncate_partial_when_range_is_ignored(
    monkeypatch, tmp_path
) -> None:
    partial = tmp_path / "data" / "file.jsonl.part"
    partial.parent.mkdir(parents=True)
    original = b"preserve-this-prefix"
    partial.write_bytes(original)

    def fake_run(command, *, check, capture_output, text):
        assert check is False and capture_output is True and text is True
        output = Path(command[command.index("--output") + 1])
        headers = Path(command[command.index("--dump-header") + 1])
        output.write_bytes(b"server-ignored-range")
        headers.write_text(
            "HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n",
            encoding="ascii",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exhausted retries"):
        mod._download_curl(
            repo_id="owner/repo",
            revision="b" * 40,
            filename="data/file.jsonl",
            local_dir=str(tmp_path),
            attempts=1,
        )

    assert partial.read_bytes() == original


def test_download_curl_does_not_publish_completed_partial_with_wrong_hash(
    tmp_path,
) -> None:
    partial = tmp_path / "data" / "file.jsonl.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"wrong")

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        mod._download_curl(
            repo_id="owner/repo",
            revision="b" * 40,
            filename="data/file.jsonl",
            local_dir=str(tmp_path),
            expected_bytes=len(b"wrong"),
            expected_sha256=hashlib.sha256(b"right").hexdigest(),
        )

    assert partial.is_file()
    assert not (tmp_path / "data" / "file.jsonl").exists()


def test_download_curl_can_pin_transient_cdn_resolution(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(command))
        return _fake_curl_response(command, b"cdn-dataset")

    monkeypatch.setenv("SOPHIA_HF_CDN_RESOLVE_IP", "192.0.2.20")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._download_curl(
        repo_id="owner/repo",
        revision="b" * 40,
        filename="data/file.jsonl",
        local_dir=str(tmp_path),
        endpoint="https://mirror.example",
    )

    assert "us.aws.cdn.hf.co:443:192.0.2.20" in calls[0]


def test_download_curl_accepts_bounded_connect_timeout(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(command))
        return _fake_curl_response(command, b"timeout-fixture")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setenv("SOPHIA_HF_CONNECT_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("SOPHIA_HF_CURL_CHECKPOINT_SECONDS", "600")
    mod._download_curl(
        repo_id="owner/repo",
        revision="b" * 40,
        filename="data/file.jsonl",
        local_dir=str(tmp_path),
    )

    command = calls[0]
    assert command[command.index("--connect-timeout") + 1] == "60"
    assert command[command.index("--speed-time") + 1] == "120"
    assert command[command.index("--speed-limit") + 1] == "1024"
    assert command[command.index("--max-time") + 1] == "600"
    # Let curl negotiate HTTP/2 with the Xet CDN. Forcing HTTP/1.1 can make
    # the redirected CDN connection time out on otherwise healthy networks.
    assert "--http1.1" not in command

    monkeypatch.setenv("SOPHIA_HF_CONNECT_TIMEOUT_SECONDS", "301")
    with pytest.raises(ValueError, match="must be in"):
        mod._download_curl(
            repo_id="owner/repo",
            revision="b" * 40,
            filename="data/other.jsonl",
            local_dir=str(tmp_path),
        )


def test_download_curl_rejects_unbounded_checkpoint_interval(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SOPHIA_HF_CURL_CHECKPOINT_SECONDS", "3601")

    with pytest.raises(ValueError, match="CHECKPOINT_SECONDS"):
        mod._download_curl(
            repo_id="owner/repo",
            revision="b" * 40,
            filename="data/file.jsonl",
            local_dir=str(tmp_path),
        )


def test_download_curl_accepts_bounded_low_speed_timeout(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is False and capture_output is True and text is True
        calls.append(list(command))
        return _fake_curl_response(command, b"low-speed-fixture")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setenv("SOPHIA_HF_LOW_SPEED_SECONDS", "300")
    monkeypatch.setenv("SOPHIA_HF_LOW_SPEED_BYTES", "2048")
    mod._download_curl(
        repo_id="owner/repo",
        revision="b" * 40,
        filename="data/file.jsonl",
        local_dir=str(tmp_path),
    )
    command = calls[0]
    assert command[command.index("--speed-time") + 1] == "300"
    assert command[command.index("--speed-limit") + 1] == "2048"

    monkeypatch.setenv("SOPHIA_HF_LOW_SPEED_SECONDS", "9")
    with pytest.raises(ValueError, match="LOW_SPEED_SECONDS"):
        mod._download_curl(
            repo_id="owner/repo",
            revision="b" * 40,
            filename="data/other.jsonl",
            local_dir=str(tmp_path),
        )


def test_download_curl_accepts_bounded_environment_retry_budget(
    monkeypatch, tmp_path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is False and capture_output is True and text is True
        calls.append(list(command))
        return _fake_curl_response(command, b"retry-budget-fixture")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setenv("SOPHIA_HF_CURL_ATTEMPTS", "1000")
    mod._download_curl(
        repo_id="owner/repo",
        revision="b" * 40,
        filename="data/file.jsonl",
        local_dir=str(tmp_path),
    )
    assert len(calls) == 1

    monkeypatch.setenv("SOPHIA_HF_CURL_ATTEMPTS", "10001")
    with pytest.raises(ValueError, match="must be in"):
        mod._download_curl(
            repo_id="owner/repo",
            revision="b" * 40,
            filename="data/other.jsonl",
            local_dir=str(tmp_path),
        )


def test_v2_inventory_verifies_upstream_size_and_hash(monkeypatch, tmp_path) -> None:
    content = b"verified"
    inventory = tmp_path / "inventory_v2.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "sophia_hf_source_inventory_v2",
                "sources": [
                    {
                        "name": "source_a",
                        "repo_id": "owner/repo",
                        "revision": "a" * 40,
                        "license": "test-license",
                        "files": [
                            {
                                "path": "data/file.parquet",
                                "bytes": len(content),
                                "sha256": hashlib.sha256(content).hexdigest(),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_download(**kwargs):
        path = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    monkeypatch.setattr(mod, "_download_https", fake_download)
    report = mod.download_inventory(
        inventory_path=str(inventory),
        output_root=str(tmp_path / "verified"),
        endpoint="https://mirror.example",
    )

    row = report["sources"][0]["files"][0]
    assert row["expected_bytes"] == len(content)
    assert row["expected_sha256"] == hashlib.sha256(content).hexdigest()
    assert report["download_endpoint"] == "https://mirror.example"
