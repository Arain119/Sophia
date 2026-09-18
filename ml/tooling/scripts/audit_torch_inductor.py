#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.errors import SophiaUsageError

_BACKUP_DIRNAME = "_sophia_backup_stale_flex"
_UNTRACKED_BACKUP_DIRNAME = "_sophia_backup_untracked_record"
_BACKUP_PREFIX = "_sophia_backup_"


@dataclass(frozen=True)
class StaleKernelConflict:
    stale_relpath: str
    active_relpath: str
    expected_template_names: tuple[str, ...]
    reason: str


KNOWN_STALE_KERNEL_CONFLICTS: tuple[StaleKernelConflict, ...] = (
    StaleKernelConflict(
        stale_relpath="flex_attention.py",
        active_relpath="flex/flex_attention.py",
        expected_template_names=("flex_attention", "flex_attention_backward"),
        reason="top-level flex_attention.py conflicts with the flex/ package implementation",
    ),
    StaleKernelConflict(
        stale_relpath="flex_decoding.py",
        active_relpath="flex/flex_decoding.py",
        expected_template_names=("flex_decoding",),
        reason="top-level flex_decoding.py conflicts with the flex/ package implementation",
    ),
    StaleKernelConflict(
        stale_relpath="mm_scaled_grouped.py",
        active_relpath="mm_grouped.py",
        expected_template_names=("grouped_mm", "scaled_grouped_mm"),
        reason="mm_scaled_grouped.py conflicts with consolidated grouped-mm templates in mm_grouped.py",
    ),
)


def _iter_python_files(kernel_root: Path):
    for path in sorted(kernel_root.rglob("*.py")):
        if any(part.startswith(_BACKUP_PREFIX) for part in path.parts):
            continue
        yield path


def _template_names_in_source(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name != "TritonTemplate":
            continue
        for kw in node.keywords:
            if (
                kw.arg == "name"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                names.add(str(kw.value.value))
    return names


def triton_template_name_index(kernel_root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for path in _iter_python_files(kernel_root):
        rel = path.relative_to(kernel_root).as_posix()
        for name in sorted(_template_names_in_source(path)):
            index.setdefault(name, []).append(rel)
    return index


def duplicate_triton_template_names(kernel_root: Path) -> dict[str, list[str]]:
    duplicates: dict[str, list[str]] = {}
    for name, files in triton_template_name_index(kernel_root).items():
        if len(files) > 1:
            duplicates[name] = list(files)
    return duplicates


def detect_known_stale_conflicts(
    *,
    kernel_root: Path,
    duplicates: dict[str, list[str]],
) -> list[StaleKernelConflict]:
    detected: list[StaleKernelConflict] = []
    for conflict in KNOWN_STALE_KERNEL_CONFLICTS:
        stale = str(conflict.stale_relpath)
        active = str(conflict.active_relpath)
        stale_path = kernel_root / stale
        active_path = kernel_root / active
        if not stale_path.is_file() or not active_path.is_file():
            continue
        overlapping = [
            name
            for name in conflict.expected_template_names
            if stale in duplicates.get(name, []) and active in duplicates.get(name, [])
        ]
        if overlapping:
            detected.append(conflict)
    return detected


def repair_known_stale_conflicts(
    *,
    kernel_root: Path,
    conflicts: list[StaleKernelConflict],
    backup_dirname: str = _BACKUP_DIRNAME,
) -> list[dict[str, str]]:
    backup_root = kernel_root / str(backup_dirname)
    backup_root.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for conflict in conflicts:
        src = kernel_root / str(conflict.stale_relpath)
        dst = backup_root / Path(str(conflict.stale_relpath)).name
        if not src.exists():
            continue
        if dst.exists():
            raise SophiaUsageError(
                "[ERR] stale torch inductor backup destination already exists.\n"
                f"src={src}\n"
                f"dst={dst}"
            )
        shutil.move(str(src), str(dst))
        moved.append(
            {
                "src": str(src),
                "dst": str(dst),
                "reason": str(conflict.reason),
            }
        )
    return moved


def torch_site_root() -> Path:
    import torch

    return Path(torch.__file__).resolve().parent.parent


def torch_record_path() -> Path:
    site_root = torch_site_root()
    matches = sorted(site_root.glob("torch-*.dist-info/RECORD"))
    if not matches:
        raise SophiaUsageError(
            "[ERR] unable to locate torch dist-info RECORD.\n"
            f"site_root={site_root}"
        )
    return matches[0]


def recorded_torch_paths(record_path: Path) -> set[str]:
    recorded: set[str] = set()
    with record_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            rel = str(row[0] or "")
            if rel.startswith("torch/"):
                recorded.add(rel[len("torch/"):])
    return recorded


def untracked_torch_files(
    *,
    torch_root: Path,
    record_path: Path,
) -> list[str]:
    recorded = recorded_torch_paths(record_path)
    extras: list[str] = []
    for path in sorted(torch_root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(torch_root)).replace("\\", "/")
        if any(part.startswith(_BACKUP_PREFIX) for part in path.parts):
            continue
        if "/__pycache__/" in f"/{rel}/" or rel.endswith((".pyc", ".pyo")):
            continue
        if rel not in recorded:
            extras.append(rel)
    return extras


def repair_untracked_torch_files(
    *,
    torch_root: Path,
    untracked_files: list[str],
    backup_dirname: str = _UNTRACKED_BACKUP_DIRNAME,
) -> list[dict[str, str]]:
    backup_root = torch_root / str(backup_dirname)
    backup_root.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for rel in untracked_files:
        src = torch_root / str(rel)
        if not src.exists():
            continue
        dst = backup_root / Path(str(rel))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise SophiaUsageError(
                "[ERR] untracked torch backup destination already exists.\n"
                f"src={src}\n"
                f"dst={dst}"
            )
        shutil.move(str(src), str(dst))
        moved.append(
            {
                "src": str(src),
                "dst": str(dst),
                "reason": "not present in torch dist-info RECORD",
            }
        )
    return moved


def inductor_kernel_root() -> Path:
    import torch

    return Path(torch.__file__).resolve().parent / "_inductor" / "kernel"


def compile_smoke_test() -> None:
    import torch

    importlib.import_module("torch._inductor.lowering")

    @torch.compile
    def _compiled_increment(x):
        return x + 1

    out = _compiled_increment(torch.zeros(3))
    if tuple(out.shape) != (3,):
        raise RuntimeError(f"unexpected compile smoke output shape: {tuple(out.shape)!r}")


def build_report(
    *,
    kernel_root: Path,
    torch_root: Path,
    record_path: Path,
    duplicates: dict[str, list[str]],
    stale_conflicts: list[StaleKernelConflict],
    untracked_files: list[str],
    moved: list[dict[str, str]],
    compile_smoke_ok: bool,
) -> dict[str, Any]:
    return {
        "kind": "torch_inductor_audit",
        "kernel_root": str(kernel_root),
        "torch_root": str(torch_root),
        "record_path": str(record_path),
        "duplicate_template_names": dict(sorted(duplicates.items())),
        "known_stale_conflicts": [asdict(item) for item in stale_conflicts],
        "untracked_record_files": list(untracked_files),
        "moved_files": list(moved),
        "compile_smoke_ok": bool(compile_smoke_ok),
        "summary": {
            "duplicate_count": int(len(duplicates)),
            "known_stale_conflict_count": int(len(stale_conflicts)),
            "untracked_record_file_count": int(len(untracked_files)),
            "moved_count": int(len(moved)),
            "all_passed": bool(not duplicates and not untracked_files and compile_smoke_ok),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the active torch._inductor kernel tree for duplicate Triton template "
            "registrations and optionally move known stale files aside."
        )
    )
    parser.add_argument("--repair", type=int, default=0, choices=[0, 1])
    parser.add_argument("--repair_untracked", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_compile_smoke", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output_json", type=str, default="")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    torch_root = inductor_kernel_root().parents[1]
    kernel_root = inductor_kernel_root()
    record_path = torch_record_path()
    if not kernel_root.is_dir():
        raise SophiaUsageError(
            "[ERR] torch._inductor kernel root is missing.\n"
            f"path={kernel_root}"
        )

    duplicates = duplicate_triton_template_names(kernel_root)
    stale_conflicts = detect_known_stale_conflicts(
        kernel_root=kernel_root,
        duplicates=duplicates,
    )
    untracked_files = untracked_torch_files(
        torch_root=torch_root,
        record_path=record_path,
    )
    moved: list[dict[str, str]] = []
    if int(args.repair) == 1:
        if stale_conflicts:
            moved.extend(
                repair_known_stale_conflicts(
                    kernel_root=kernel_root,
                    conflicts=stale_conflicts,
                )
            )
    if int(args.repair_untracked) == 1:
        if untracked_files:
            moved.extend(
                repair_untracked_torch_files(
                    torch_root=torch_root,
                    untracked_files=untracked_files,
                )
            )
        duplicates = duplicate_triton_template_names(kernel_root)
        stale_conflicts = detect_known_stale_conflicts(
            kernel_root=kernel_root,
            duplicates=duplicates,
        )
        untracked_files = untracked_torch_files(
            torch_root=torch_root,
            record_path=record_path,
        )

    compile_smoke_ok = False
    compile_smoke_error = ""
    if int(args.skip_compile_smoke) != 1:
        try:
            compile_smoke_test()
        except Exception as exc:
            compile_smoke_error = f"{type(exc).__name__}: {exc}"
        else:
            compile_smoke_ok = True
    else:
        compile_smoke_ok = True

    report = build_report(
        kernel_root=kernel_root,
        torch_root=torch_root,
        record_path=record_path,
        duplicates=duplicates,
        stale_conflicts=stale_conflicts,
        untracked_files=untracked_files,
        moved=moved,
        compile_smoke_ok=compile_smoke_ok,
    )
    if str(args.output_json).strip():
        write_json_atomic(str(Path(str(args.output_json)).resolve()), report, make_parents=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    if duplicates:
        raise SophiaUsageError(
            "[ERR] duplicate torch inductor Triton template registrations remain.\n"
            f"kernel_root={kernel_root}\n"
            f"duplicates={json.dumps(duplicates, ensure_ascii=False, sort_keys=True)}"
        )
    if stale_conflicts:
        raise SophiaUsageError(
            "[ERR] known stale torch inductor kernel conflicts remain after audit.\n"
            f"kernel_root={kernel_root}\n"
            f"conflicts={json.dumps([asdict(item) for item in stale_conflicts], ensure_ascii=False, sort_keys=True)}"
        )
    if untracked_files:
        raise SophiaUsageError(
            "[ERR] torch package contains files not present in dist-info RECORD.\n"
            f"torch_root={torch_root}\n"
            f"record_path={record_path}\n"
            f"untracked={json.dumps(untracked_files, ensure_ascii=False, sort_keys=True)}"
        )
    if not compile_smoke_ok:
        raise SophiaUsageError(
            "[ERR] torch.compile smoke test failed after inductor audit.\n"
            f"kernel_root={kernel_root}\n"
            f"error={compile_smoke_error}"
        )


if __name__ == "__main__":
    main()
