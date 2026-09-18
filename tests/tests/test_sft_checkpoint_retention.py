"""What survives when a multi-epoch SFT run finishes."""

from __future__ import annotations

from pathlib import Path

from ml.core.engine.checkpoint_io import _trim_checkpoints


def _write_checkpoints(ckpt_dir: Path, steps: tuple[int, ...]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (ckpt_dir / f"ckpt_step{step}.pt").write_bytes(b"weights")
        (ckpt_dir / f"ckpt_step{step}.pt.sha256").write_text(
            f"0  ckpt_step{step}.pt\n", encoding="ascii"
        )


def _surviving_steps(ckpt_dir: Path) -> list[int]:
    return sorted(
        int(path.stem.removeprefix("ckpt_step"))
        for path in ckpt_dir.glob("ckpt_step*.pt")
    )


def test_trim_keeps_the_newest_window_when_nothing_is_protected(
    tmp_path: Path,
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    _write_checkpoints(ckpt_dir, (100, 200, 300, 400, 500))

    _trim_checkpoints(ckpt_dir=str(ckpt_dir), save_total_limit=3, prefix="ckpt_step")

    assert _surviving_steps(ckpt_dir) == [300, 400, 500]


def test_trim_spares_a_milestone_outside_the_window(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    _write_checkpoints(ckpt_dir, (100, 200, 300, 400, 500))

    _trim_checkpoints(
        ckpt_dir=str(ckpt_dir),
        save_total_limit=3,
        prefix="ckpt_step",
        keep_steps=(200,),
    )

    assert _surviving_steps(ckpt_dir) == [200, 300, 400, 500]
    assert (ckpt_dir / "ckpt_step200.pt.sha256").exists()


def test_trim_removes_the_sidecar_of_an_expired_checkpoint(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    _write_checkpoints(ckpt_dir, (100, 200, 300, 400))

    _trim_checkpoints(
        ckpt_dir=str(ckpt_dir),
        save_total_limit=2,
        prefix="ckpt_step",
        keep_steps=(400,),
    )

    assert _surviving_steps(ckpt_dir) == [300, 400]
    assert not (ckpt_dir / "ckpt_step100.pt.sha256").exists()


def test_a_three_epoch_run_can_still_reach_its_first_epoch(tmp_path: Path) -> None:
    """Without protection every survivor of this run sits in the last epoch."""
    ckpt_dir = tmp_path / "checkpoints"
    total_steps, save_interval, limit = 7093, 100, 3
    milestones = (2400, 4800)

    written: list[int] = []
    for step in range(save_interval, total_steps + 1, save_interval):
        _write_checkpoints(ckpt_dir, (step,))
        written.append(step)
        _trim_checkpoints(
            ckpt_dir=str(ckpt_dir),
            save_total_limit=limit,
            prefix="ckpt_step",
            keep_steps=milestones,
        )

    survivors = _surviving_steps(ckpt_dir)
    assert survivors == [2400, 4800, 6800, 6900, 7000]
    # The rolling window alone would have left only the final epoch.
    assert min(step for step in survivors if step not in milestones) > 4800
