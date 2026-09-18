from __future__ import annotations

from ml.training.release_gate_fs_checks import MIN_DISK_FREE_GB, ensure_disk_free


def test_release_disk_floor_preserves_checkpoint_headroom() -> None:
    assert MIN_DISK_FREE_GB == 80.0


def test_ensure_disk_free_uses_nearest_existing_parent(tmp_path) -> None:
    nested_output = tmp_path / "missing" / "run" / "artifacts"

    free_gb = ensure_disk_free(path=str(nested_output), min_free_gb=0.0)

    assert free_gb > 0.0
    assert not nested_output.exists()
