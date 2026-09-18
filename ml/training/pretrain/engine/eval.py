from __future__ import annotations

from contextlib import nullcontext

import torch

from ml.training.pretrain.resources import PretrainDataIter
from ml.training.loss_stats import supervised_token_count
from ml.training.model_contracts import require_compute_loss


def evaluate_loss(
    *,
    model: torch.nn.Module,
    data_iter: PretrainDataIter,
    steps: int,
    base_dtype: torch.dtype,
) -> float:
    eval_steps = max(int(steps), 1)
    try:
        dev = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models always have params
        dev = torch.device("cuda")

    loss_sum = torch.zeros((), device=dev, dtype=torch.float32)
    batch_count = 0

    token_weighted_sum = torch.zeros((), device=dev, dtype=torch.float32)
    token_weighted_count = torch.zeros((), device=dev, dtype=torch.int64)
    require_compute_loss(model, context="Sophia evaluate_loss")
    was_training = bool(model.training)
    model.eval()
    amp_ctx = (
        torch.autocast(
            device_type=str(dev.type),
            dtype=base_dtype,
            enabled=(dev.type == "cuda"),
        )
        if dev.type in ("cuda", "cpu")
        else nullcontext()
    )
    try:
        with torch.inference_mode():
            for _ in range(eval_steps):
                batch = next(data_iter)
                model_inputs = dict(batch)
                model_inputs["compute_loss"] = True
                with amp_ctx:
                    out = model(**model_inputs)
                    loss = out.loss
                if not torch.is_tensor(loss):
                    raise RuntimeError("Model did not return a Tensor loss.")
                loss_det = loss.detach().to(dtype=torch.float32)
                loss_sum = loss_sum + loss_det
                batch_count += 1

                labels = batch.get("labels")
                if torch.is_tensor(labels) and labels.ndim == 2 and int(labels.size(1)) > 1:
                    supervised = supervised_token_count(labels, ignore_index=-100)
                    if supervised.device != loss_det.device:
                        supervised = supervised.to(device=loss_det.device)
                    token_weighted_sum = token_weighted_sum + loss_det * supervised.to(
                        dtype=torch.float32
                    )
                    token_weighted_count = token_weighted_count + supervised
    finally:
        model.train(was_training)

    if batch_count <= 0:
        raise RuntimeError("evaluate_loss produced no losses")

    if int(token_weighted_count.item()) > 0:
        denom = token_weighted_count.clamp_min(1).to(dtype=torch.float32)
        return float((token_weighted_sum / denom).item())
    return float((loss_sum / float(batch_count)).item())
