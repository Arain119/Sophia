from __future__ import annotations

from builtins import BaseExceptionGroup, ExceptionGroup
import importlib.metadata
import json
import math
import os
import platform
import queue
import string
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TextIO

import torch

from ml.core.common.io import write_json_atomic
from ml.core.engine.checkpointing import save_checkpoint_state


def _fmt_duration_s(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if (not math.isfinite(s)) or s < 0:
        return "?"
    total = int(s + 0.5)
    m, sec = divmod(total, 60)
    h, minute = divmod(m, 60)
    d, hour = divmod(h, 24)
    if d > 0:
        return f"{d}d{hour:02d}:{minute:02d}:{sec:02d}"
    return f"{hour:02d}:{minute:02d}:{sec:02d}"


def _sync(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.current_stream(device).synchronize()
    except Exception as exc:
        raise RuntimeError(f"CUDA stream synchronization failed for device={device}") from exc


def _json_default(obj: object) -> str:
    return str(obj)


def _write_json_atomic(path: str, payload: object) -> None:
    write_json_atomic(path, payload, default=_json_default)


class _AsyncCheckpointWriter:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._exc: BaseException | None = None
        self._step: int | None = None
        self._prefix: str | None = None

    def _run(
        self,
        *,
        output_dir: str,
        step: int,
        state: dict[str, object],
        save_total_limit: int,
        staging_dir: str | None,
        prefix: str,
    ) -> None:
        try:
            save_checkpoint_state(
                output_dir=str(output_dir),
                step=int(step),
                state=state,
                save_total_limit=int(save_total_limit),
                staging_dir=staging_dir,
                prefix=str(prefix),
            )
        except BaseException as exc:
            self._exc = exc

    def wait(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join()
        self._thread = None
        self._raise_stored_error()

    def raise_if_failed(self) -> None:
        """Raise a completed background save failure without waiting on a live save."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._raise_stored_error()

    def _raise_stored_error(self) -> None:
        if self._exc is None:
            self._step = None
            self._prefix = None
            return
        exc = self._exc
        self._exc = None
        step = self._step
        self._step = None
        prefix = self._prefix
        self._prefix = None
        raise RuntimeError(
            f"Async checkpoint save failed (prefix={prefix}, step={step}): {exc}"
        ) from exc

    def save(
        self,
        *,
        output_dir: str,
        step: int,
        state: dict[str, object],
        save_total_limit: int,
        staging_dir: str | None = None,
        prefix: str = "ckpt_step",
    ) -> None:
        self.wait()
        self._exc = None
        self._step = int(step)
        self._prefix = str(prefix)
        self._thread = threading.Thread(
            target=self._run,
            kwargs=dict(
                output_dir=str(output_dir),
                step=int(step),
                state=state,
                save_total_limit=int(save_total_limit),
                staging_dir=staging_dir,
                prefix=str(prefix),
            ),
            daemon=True,
        )
        self._thread.start()


class AsyncJsonlWriter:
    def __init__(self, fp: TextIO, *, max_queue: int = 8192) -> None:
        self._fp = fp
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=max(int(max_queue), 1))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._exc: BaseException | None = None
        self._closing = threading.Event()
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                try:
                    self._fp.write(item)
                except BaseException as exc:
                    self._exc = exc
                    break
        finally:
            cleanup_errors: list[BaseException] = []
            try:
                self._fp.flush()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                self._fp.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if cleanup_errors:
                failures: list[BaseException] = []
                if self._exc is not None:
                    failures.append(self._exc)
                failures.extend(cleanup_errors)
                self._exc = BaseExceptionGroup("async metrics writer failed", failures)

    def write_line(self, line: str) -> None:
        if self._closing.is_set():
            raise RuntimeError("Async metrics writer is closing.")
        if self._exc is not None:
            raise RuntimeError(f"Async metrics writer failed: {self._exc}") from self._exc
        try:
            self._q.put_nowait(str(line))
        except queue.Full as exc:
            raise RuntimeError("Async metrics writer queue is full.") from exc

    def close(self) -> None:
        self._closing.set()
        try:
            self._q.put(None, timeout=5.0)
        except queue.Full as exc:
            raise RuntimeError("Async metrics writer did not accept close sentinel.") from exc
        try:
            self._thread.join(timeout=5.0)
        except RuntimeError as exc:
            raise RuntimeError("Async metrics writer thread join failed.") from exc
        if self._thread.is_alive():
            raise RuntimeError("Async metrics writer did not terminate within timeout.")
        if not self._thread.is_alive() and self._exc is not None:
            exc = self._exc
            self._exc = None
            raise RuntimeError(f"Async metrics writer failed: {exc}") from exc


def _write_best_metric(
    *, output_dir: str, step: int, metric_name: str, metric_value: float
) -> None:
    payload = {
        "step": int(step),
        str(metric_name): float(metric_value),
    }
    path = os.path.join(str(output_dir), "best_metric.json")
    _write_json_atomic(path, payload)


def _read_best_metric_value(*, output_dir: str, metric_name: str) -> float | None:
    path = os.path.join(str(output_dir), "best_metric.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read best metric file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"best metric file must contain a JSON object: {path}")
    if str(metric_name) not in payload:
        raise RuntimeError(f"best metric file is missing {metric_name!r}: {path}")
    value = payload[str(metric_name)]
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"best metric {metric_name!r} must be numeric in {path}"
        ) from exc
    if not math.isfinite(val):
        raise RuntimeError(f"best metric {metric_name!r} must be finite in {path}")
    return val


def _copy_best_checkpoint(*, output_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(str(output_dir), "checkpoints")
    src = os.path.join(ckpt_dir, f"ckpt_step{int(step)}.pt")
    src_sidecar = f"{src}.sha256"
    dst = os.path.join(ckpt_dir, "best.pt")
    dst_sidecar = f"{dst}.sha256"
    if not os.path.isfile(src):
        raise FileNotFoundError(f"best checkpoint source does not exist: {src}")
    if not os.path.isfile(src_sidecar):
        raise FileNotFoundError(
            f"best checkpoint source SHA-256 sidecar does not exist: {src_sidecar}"
        )
    tmp = f"{dst}.tmp.{os.getpid()}"
    tmp_sidecar = f"{dst_sidecar}.tmp.{os.getpid()}"
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    try:
        with open(src_sidecar, encoding="ascii") as handle:
            digest = handle.read().split(maxsplit=1)[0].lower()
        if len(digest) != 64 or any(char not in string.hexdigits for char in digest):
            raise RuntimeError(f"invalid best checkpoint source digest: {src_sidecar}")
        with open(tmp_sidecar, "w", encoding="ascii") as handle:
            handle.write(f"{digest}  {os.path.basename(dst)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dst)
        os.replace(tmp_sidecar, dst_sidecar)
    finally:
        for path in (tmp, tmp_sidecar):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class RunRecipeSnapshot:
    values: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_args_dict(cls, args_dict: dict[str, object] | None) -> RunRecipeSnapshot | None:
        if not isinstance(args_dict, dict):
            return None
        values: dict[str, object] = {}
        for key in (
            "seq_len",
            "batch_size",
            "accumulation_steps",
            "target_tokens_per_update",
            "learning_rate",
            "adam_eps",
            "weight_decay",
            "lr_schedule",
            "warmup_steps",
            "min_lr_ratio",
            "save_interval",
            "save_best",
            "enable_checkpoints",
            "eval_interval",
            "eval_steps",
        ):
            if key in args_dict:
                values[key] = args_dict.get(key)
        return None if not values else cls(values=values)

    def to_payload(self) -> dict[str, object]:
        return dict(self.values)


@dataclass(frozen=True)
class RunInputProvenance:
    values: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_args_dict(
        cls,
        args_dict: dict[str, object] | None,
    ) -> RunInputProvenance | None:
        if not isinstance(args_dict, dict):
            return None
        values: dict[str, object] = {}
        for key in (
            "data_path",
            "tokenizer_path",
            "_sophia_train_manifest_sha1",
        ):
            if key in args_dict:
                values[key] = args_dict.get(key)
        return None if not values else cls(values=values)

    def to_payload(self) -> dict[str, object]:
        return dict(self.values)


@dataclass(frozen=True)
class RunInvocationMeta:
    time: float
    start_step: int
    max_steps: int
    device: str
    torch: str
    cuda: str
    python: str
    platform: str
    machine: str
    hostname: str
    pid: int
    extra: dict[str, object] = field(default_factory=dict)
    recipe: RunRecipeSnapshot | None = None
    input_provenance: RunInputProvenance | None = None
    cuda_device_name: str | None = None
    cuda_capability: list[int] | None = None
    cuda_total_memory_gb: float | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "time": float(self.time),
            "start_step": int(self.start_step),
            "max_steps": int(self.max_steps),
            "device": str(self.device),
            "torch": str(self.torch),
            "cuda": str(self.cuda),
            "python": str(self.python),
            "platform": str(self.platform),
            "machine": str(self.machine),
            "hostname": str(self.hostname),
            "pid": int(self.pid),
        }
        payload.update(dict(self.extra))
        if self.recipe is not None:
            payload["recipe"] = self.recipe.to_payload()
        if self.input_provenance is not None:
            payload["input_provenance"] = self.input_provenance.to_payload()
        if self.cuda_device_name is not None:
            payload["cuda_device_name"] = str(self.cuda_device_name)
        if self.cuda_capability is not None:
            payload["cuda_capability"] = [int(x) for x in self.cuda_capability]
        if self.cuda_total_memory_gb is not None:
            payload["cuda_total_memory_gb"] = float(self.cuda_total_memory_gb)
        return payload


def _build_run_invocation_meta(
    *,
    args_dict: dict[str, object] | None,
    start_step: int,
    max_steps: int,
    device: torch.device,
) -> RunInvocationMeta:
    extra: dict[str, object] = {}
    if isinstance(args_dict, dict):
        for key in (
            "_sophia_run_kind",
            "_sophia_base_dtype",
            "_sophia_precision_stack",
            "_sophia_machine_signature",
            "_sophia_resolved_resume_checkpoint",
            "_sophia_requested_target_tokens_per_update",
        ):
            if key in args_dict:
                extra[key] = args_dict.get(key)
    extra["transformers"] = importlib.metadata.version("transformers")

    cuda_device_name: str | None = None
    cuda_capability: list[int] | None = None
    cuda_total_memory_gb: float | None = None
    if device.type == "cuda":
        try:
            idx = device.index
            if idx is None:
                idx = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(int(idx))
            cuda_device_name = str(getattr(props, "name", ""))
            cuda_capability = [
                int(getattr(props, "major", 0)),
                int(getattr(props, "minor", 0)),
            ]
            total_mem = float(getattr(props, "total_memory", 0.0) or 0.0)
            cuda_total_memory_gb = total_mem / float(1024**3)
        except Exception as exc:
            raise RuntimeError(
                f"failed to read CUDA runtime metadata for device={device}"
            ) from exc

    return RunInvocationMeta(
        time=float(time.time()),
        start_step=int(start_step),
        max_steps=int(max_steps),
        device=str(device),
        torch=str(getattr(torch, "__version__", "")),
        cuda=str(getattr(torch.version, "cuda", None)),
        python=str(getattr(sys, "version", "")).split(" ")[0],
        platform=str(getattr(sys, "platform", "")),
        machine=str(getattr(platform, "machine", lambda: "")()),
        hostname=str(getattr(platform, "node", lambda: "")()),
        pid=int(os.getpid()),
        extra=extra,
        recipe=RunRecipeSnapshot.from_args_dict(args_dict),
        input_provenance=RunInputProvenance.from_args_dict(args_dict),
        cuda_device_name=cuda_device_name,
        cuda_capability=cuda_capability,
        cuda_total_memory_gb=cuda_total_memory_gb,
    )


@dataclass(frozen=True)
class MetricsStartEvent:
    meta: RunInvocationMeta

    def to_payload(self) -> dict[str, object]:
        return {"type": "start", **self.meta.to_payload()}


@dataclass(frozen=True)
class MetricsTrainEvent:
    time: float
    step: int
    max_steps: int
    loss: float
    lr: float
    tok_s: int
    grad_norm: float | None
    mem_gb: float
    dt_s: float
    updates: int
    token_weighted: bool
    tokens_total: int | None
    optimizer_diagnostics: dict[str, float] | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "train",
            "time": float(self.time),
            "step": int(self.step),
            "max_steps": int(self.max_steps),
            "loss": float(self.loss),
            "lr": float(self.lr),
            "tok_s": int(self.tok_s),
            "grad_norm": self.grad_norm,
            "mem_gb": float(self.mem_gb),
            "dt_s": float(self.dt_s),
            "updates": int(self.updates),
            "token_weighted": bool(self.token_weighted),
            "tokens_total": self.tokens_total,
        }
        if self.optimizer_diagnostics:
            payload["optimizer_diagnostics"] = {
                str(key): float(value)
                for key, value in self.optimizer_diagnostics.items()
            }
        return payload


@dataclass(frozen=True)
class MetricsEvalEvent:
    time: float
    step: int
    max_steps: int
    val_loss: float
    improved: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "type": "eval",
            "time": float(self.time),
            "step": int(self.step),
            "max_steps": int(self.max_steps),
            "val_loss": float(self.val_loss),
            "improved": bool(self.improved),
        }


@dataclass(frozen=True)
class MetricsScalarEvent:
    time: float
    step: int
    name: str
    value: float

    def to_payload(self) -> dict[str, object]:
        return {
            "type": "metric",
            "time": float(self.time),
            "step": int(self.step),
            "name": str(self.name),
            "value": float(self.value),
        }


@dataclass(frozen=True)
class MetricsEndEvent:
    time: float
    step: int
    max_steps: int

    def to_payload(self) -> dict[str, object]:
        return {
            "type": "end",
            "time": float(self.time),
            "step": int(self.step),
            "max_steps": int(self.max_steps),
        }


class _MetricsLogger:
    def __init__(
        self,
        *,
        output_dir: str,
        args_dict: dict[str, object] | None,
        start_step: int,
        max_steps: int,
        device: torch.device,
        async_write: bool,
    ) -> None:
        self._fp: TextIO | None = None
        self._async: AsyncJsonlWriter | None = None

        metrics_path = os.path.join(str(output_dir), "metrics.jsonl")
        try:
            self._fp = open(metrics_path, "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            raise RuntimeError(f"unable to open metrics.jsonl for append: {exc}") from exc

        if bool(async_write):
            try:
                self._async = AsyncJsonlWriter(self._fp)
            except Exception as exc:
                raise RuntimeError(f"unable to enable async metrics writer: {exc}") from exc

        if args_dict is not None:
            try:
                run_args_path = os.path.join(str(output_dir), "run_args.json")
                if not os.path.exists(run_args_path):
                    _write_json_atomic(run_args_path, args_dict)
                effective_config_path = os.path.join(
                    str(output_dir), "effective_config.json"
                )
                _write_json_atomic(effective_config_path, args_dict)
            except Exception as exc:
                raise RuntimeError(
                    f"unable to write run_args/effective_config JSON: {exc}"
                ) from exc

        try:
            meta = _build_run_invocation_meta(
                args_dict=args_dict,
                start_step=int(start_step),
                max_steps=int(max_steps),
                device=device,
            )
            run_meta_path = os.path.join(str(output_dir), "run_meta.json")
            if not os.path.exists(run_meta_path):
                _write_json_atomic(run_meta_path, meta.to_payload())
        except Exception as exc:
            raise RuntimeError(f"unable to write run_meta.json: {exc}") from exc

        start_meta = _build_run_invocation_meta(
            args_dict=args_dict,
            start_step=int(start_step),
            max_steps=int(max_steps),
            device=device,
        )
        self._write_event(MetricsStartEvent(meta=start_meta).to_payload())

    def _write_event(self, payload: dict[str, object]) -> None:
        if self._fp is None:
            return
        try:
            line = json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n"
            if self._async is not None:
                self._async.write_line(line)
            else:
                self._fp.write(line)
        except Exception as exc:
            self.close()
            raise RuntimeError(f"failed to write metrics.jsonl: {exc}") from exc

    def log_train(
        self,
        *,
        step: int,
        max_steps: int,
        loss: float,
        lr: float,
        tok_s: int,
        grad_norm: float | None,
        mem_gb: float,
        dt_s: float,
        updates: int,
        token_weighted: bool,
        tokens_per_update: int,
        seen_supervised_tokens: torch.Tensor,
        optimizer_diagnostics: dict[str, float] | None = None,
    ) -> None:
        if self._fp is None:
            return
        tokens_total: int | None = int(step) * int(tokens_per_update)
        if bool(token_weighted):
            try:
                tokens_total = int(seen_supervised_tokens.detach().item())
            except Exception:
                tokens_total = None
        self._write_event(
            MetricsTrainEvent(
                time=float(time.time()),
                step=int(step),
                max_steps=int(max_steps),
                loss=float(loss),
                lr=float(lr),
                tok_s=int(tok_s),
                grad_norm=grad_norm,
                mem_gb=float(mem_gb),
                dt_s=float(dt_s),
                updates=int(updates),
                token_weighted=bool(token_weighted),
                tokens_total=tokens_total,
                optimizer_diagnostics=optimizer_diagnostics,
            ).to_payload()
        )

    def log_eval(
        self,
        *,
        step: int,
        max_steps: int,
        val_loss: float,
        improved: bool,
    ) -> None:
        self._write_event(
            MetricsEvalEvent(
                time=float(time.time()),
                step=int(step),
                max_steps=int(max_steps),
                val_loss=float(val_loss),
                improved=bool(improved),
            ).to_payload()
        )

    def log_scalar(self, *, step: int, name: str, value: float) -> None:
        self._write_event(
            MetricsScalarEvent(
                time=float(time.time()),
                step=int(step),
                name=str(name),
                value=float(value),
            ).to_payload()
        )

    def log_end(self, *, step: int, max_steps: int) -> None:
        self._write_event(
            MetricsEndEvent(
                time=float(time.time()),
                step=int(step),
                max_steps=int(max_steps),
            ).to_payload()
        )

    def close(self) -> None:
        close_exc: BaseException | None = None
        async_writer = self._async
        if async_writer is not None:
            try:
                async_writer.close()
            except Exception as exc:
                close_exc = exc
        self._async = None

        fp = self._fp
        if fp is not None and async_writer is None:
            try:
                fp.close()
            except Exception as exc:
                if close_exc is None:
                    close_exc = exc
        self._fp = None
        if close_exc is not None:
            raise RuntimeError(f"metrics logger close failed: {close_exc}") from close_exc


class _TensorBoardLogger:
    def __init__(
        self,
        *,
        output_dir: str,
        args_dict: dict[str, object] | None,
        start_step: int,
    ) -> None:
        self._writer = None

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

        if args_dict is not None:
            try:
                payload = json.dumps(
                    args_dict, ensure_ascii=False, indent=2, default=_json_default
                )
                self._writer.add_text("run/args", payload, global_step=int(start_step))
            except Exception as exc:
                raise RuntimeError("TensorBoard run args write failed.") from exc

    def log_train(
        self,
        *,
        step: int,
        loss: float,
        lr: float,
        tok_s: int,
        grad_norm: float | None,
        mem_gb: float,
        dt_s: float,
        updates: int,
        token_weighted: bool,
        tokens_per_update: int,
        seen_supervised_tokens: torch.Tensor,
        optimizer_diagnostics: dict[str, float] | None = None,
    ) -> None:
        writer = self._writer
        if writer is None:
            return
        writer.add_scalar("train/loss", float(loss), int(step))
        writer.add_scalar("train/lr", float(lr), int(step))
        writer.add_scalar("train/tok_s", float(tok_s), int(step))
        writer.add_scalar("train/mem_gb", float(mem_gb), int(step))
        writer.add_scalar("train/dt_s", float(dt_s), int(step))
        writer.add_scalar("train/updates", float(updates), int(step))
        writer.add_scalar("train/token_weighted", float(bool(token_weighted)), int(step))
        if grad_norm is not None:
            writer.add_scalar("train/grad_norm", float(grad_norm), int(step))
        tokens_total = int(step) * int(tokens_per_update)
        if bool(token_weighted):
            tokens_total = int(seen_supervised_tokens.detach().item())
        writer.add_scalar("train/tokens_total", float(tokens_total), int(step))
        if isinstance(optimizer_diagnostics, dict):
            for name, value in optimizer_diagnostics.items():
                writer.add_scalar(str(name), float(value), int(step))

    def log_eval(self, *, step: int, val_loss: float) -> None:
        writer = self._writer
        if writer is None:
            return
        writer.add_scalar("eval/val_loss", float(val_loss), int(step))

    def log_scalar(self, *, step: int, name: str, value: float) -> None:
        writer = self._writer
        if writer is None:
            return
        writer.add_scalar(str(name), float(value), int(step))

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


__all__ = [
    "AsyncJsonlWriter",
    "MetricsEndEvent",
    "MetricsEvalEvent",
    "MetricsScalarEvent",
    "MetricsStartEvent",
    "MetricsTrainEvent",
    "RunInputProvenance",
    "RunInvocationMeta",
    "RunRecipeSnapshot",
    "_AsyncCheckpointWriter",
    "_MetricsLogger",
    "_TensorBoardLogger",
    "_build_run_invocation_meta",
    "_copy_best_checkpoint",
    "_fmt_duration_s",
    "_json_default",
    "_read_best_metric_value",
    "_sync",
    "_write_best_metric",
    "_write_json_atomic",
]
