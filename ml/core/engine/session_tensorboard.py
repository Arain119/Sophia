from __future__ import annotations

from builtins import ExceptionGroup
import os
from numbers import Number

from ml.core.engine.types import MetricsRow


class _SessionTensorBoardLogger:
    def __init__(self, *, output_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard is required for release training, but SummaryWriter "
                f"could not be imported: {exc}"
            ) from exc

        log_dir = os.path.join(str(output_dir), "tensorboard")
        try:
            self._writer = SummaryWriter(log_dir=log_dir)
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard is required for release training, but SummaryWriter "
                f"could not be created at {log_dir!r}: {exc}"
            ) from exc

    def log_metrics_row(self, row: MetricsRow) -> None:
        writer = self._writer
        if writer is None:
            raise RuntimeError("TensorBoard logger is closed")
        stage = str(row.get("stage", "metrics") or "metrics").strip() or "metrics"
        raw_step = row.get("step", 0)
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"TensorBoard step must be an integer, got {raw_step!r}") from exc
        if step < 0:
            raise ValueError(f"TensorBoard step must be >= 0, got {step}")
        for key, value in dict(row).items():
            if key in {"stage", "step"}:
                continue
            if isinstance(value, bool):
                writer.add_scalar(f"{stage}/{key}", float(value), step)
                continue
            if isinstance(value, Number):
                writer.add_scalar(f"{stage}/{key}", float(value), step)

    def close(self) -> None:
        writer = self._writer
        if writer is None:
            return
        errors: list[Exception] = []
        try:
            writer.flush()
        except Exception as exc:
            errors.append(exc)
        try:
            writer.close()
        except Exception as exc:
            errors.append(exc)
        self._writer = None
        if len(errors) == 1:
            raise RuntimeError("TensorBoard logger close failed") from errors[0]
        if errors:
            raise ExceptionGroup("TensorBoard logger close failed", errors)


__all__ = ["_SessionTensorBoardLogger"]
