from __future__ import annotations

from pathlib import Path

from ml.tooling.scripts import audit_torch_inductor as mod


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_duplicate_triton_template_names_ignores_backup_dir(tmp_path: Path) -> None:
    kernel_root = tmp_path / "kernel"
    _write(
        kernel_root / "flex_attention.py",
        'TritonTemplate(name="flex_attention")\n',
    )
    _write(
        kernel_root / mod._BACKUP_DIRNAME / "flex_attention.py",
        'TritonTemplate(name="flex_attention")\n',
    )

    duplicates = mod.duplicate_triton_template_names(kernel_root)

    assert duplicates == {}


def test_detect_known_stale_conflicts_finds_flex_and_mm_conflicts(tmp_path: Path) -> None:
    kernel_root = tmp_path / "kernel"
    _write(
        kernel_root / "flex_attention.py",
        'TritonTemplate(name="flex_attention")\nTritonTemplate(name="flex_attention_backward")\n',
    )
    _write(
        kernel_root / "flex" / "flex_attention.py",
        'TritonTemplate(name="flex_attention")\nTritonTemplate(name="flex_attention_backward")\n',
    )
    _write(
        kernel_root / "mm_grouped.py",
        'TritonTemplate(name="grouped_mm")\nTritonTemplate(name="scaled_grouped_mm")\n',
    )
    _write(
        kernel_root / "mm_scaled_grouped.py",
        'TritonTemplate(name="grouped_mm")\nTritonTemplate(name="scaled_grouped_mm")\n',
    )

    duplicates = mod.duplicate_triton_template_names(kernel_root)
    detected = mod.detect_known_stale_conflicts(
        kernel_root=kernel_root,
        duplicates=duplicates,
    )

    assert sorted(item.stale_relpath for item in detected) == [
        "flex_attention.py",
        "mm_scaled_grouped.py",
    ]


def test_repair_known_stale_conflicts_moves_files_to_backup(tmp_path: Path) -> None:
    kernel_root = tmp_path / "kernel"
    _write(kernel_root / "flex_decoding.py", 'TritonTemplate(name="flex_decoding")\n')
    _write(
        kernel_root / "flex" / "flex_decoding.py",
        'TritonTemplate(name="flex_decoding")\n',
    )
    duplicates = mod.duplicate_triton_template_names(kernel_root)
    detected = mod.detect_known_stale_conflicts(
        kernel_root=kernel_root,
        duplicates=duplicates,
    )

    moved = mod.repair_known_stale_conflicts(
        kernel_root=kernel_root,
        conflicts=detected,
    )

    assert len(moved) == 1
    assert (kernel_root / "flex_decoding.py").exists() is False
    assert (kernel_root / mod._BACKUP_DIRNAME / "flex_decoding.py").is_file()
    assert mod.duplicate_triton_template_names(kernel_root) == {}


def test_untracked_torch_files_ignores_bytecode_and_detects_extra_sources(tmp_path: Path) -> None:
    torch_root = tmp_path / "torch"
    dist_info = tmp_path / "torch-0.dist-info"
    _write(dist_info / "RECORD", "torch/kept.py,,\n")
    _write(torch_root / "kept.py", "pass\n")
    _write(torch_root / "__pycache__" / "kept.cpython-312.pyc", "x")
    _write(torch_root / "extra.py", "pass\n")

    untracked = mod.untracked_torch_files(
        torch_root=torch_root,
        record_path=dist_info / "RECORD",
    )

    assert untracked == ["extra.py"]


def test_repair_untracked_torch_files_moves_relative_tree(tmp_path: Path) -> None:
    torch_root = tmp_path / "torch"
    _write(torch_root / "onnx" / "_experimental.py", "pass\n")

    moved = mod.repair_untracked_torch_files(
        torch_root=torch_root,
        untracked_files=["onnx/_experimental.py"],
    )

    assert len(moved) == 1
    assert (torch_root / "onnx" / "_experimental.py").exists() is False
    assert (
        torch_root
        / mod._UNTRACKED_BACKUP_DIRNAME
        / "onnx"
        / "_experimental.py"
    ).is_file()
