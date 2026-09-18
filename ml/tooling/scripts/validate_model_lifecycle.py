from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import math
from pathlib import Path
import platform
import time
from typing import Any

import torch

from ml.core.common.io import write_json_atomic
from ml.core.engine.checkpointing import load_checkpoint, save_checkpoint
from ml.core.spec import ModelSpec
from ml.integrations.adapters.hf.tokenizer import load_local_tokenizer
from ml.integrations.export.artifacts import export_model_artifacts
from ml.modeling.sophia_decoder import SophiaDecoder, SophiaDecoderConfig
from ml.runtime.model.attention import SophiaKDA, SophiaMLA
from ml.tooling.scripts.export_pretrain_checkpoint import (
    export_checkpoint,
    sha256_file,
)
from ml.training.pretrain.optimizer import create_torch_muon_optimizer
from ml.training.pretrain.output_policy import path_uses_host_mounted_storage
from ml.training.runtime_tools import move_optimizer_state_to_device


REPORT_SCHEMA = "sophia_hybrid_scaled_model_lifecycle_v1"


def scaled_model_spec() -> ModelSpec:
    """Keep the selected hybrid layer pattern and tokenizer at local scale."""
    return ModelSpec.default().with_overrides(
        dim=192,
        n_layers=4,
        num_heads=4,
        head_dim=32,
        ffn_hidden=512,
        kda_decay_rank=32,
        kda_output_gate_rank=32,
        mla_q_rank=48,
        mla_kv_rank=32,
        max_seq_len=128,
        max_batch_size=2,
    )


def _decoder_config(spec: ModelSpec) -> SophiaDecoderConfig:
    config = SophiaDecoderConfig.from_object(spec.to_config())
    config.bos_token_id = 2
    config.eos_token_id = 3
    config.pad_token_id = 0
    config.unk_token_id = 1
    config.return_logits_in_train = False
    config.use_cache = False
    config.tie_word_embeddings = True
    return config


def _build_model(spec: ModelSpec, *, device: torch.device) -> SophiaDecoder:
    model = SophiaDecoder(
        _decoder_config(spec),
        runtime_max_seq_len=int(spec.max_seq_len),
    )
    return model.to(device=device, dtype=torch.bfloat16)


def _build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-6,
        muon_ns_steps=4,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


def _liger_available() -> bool:
    return importlib.util.find_spec("liger_kernel") is not None


def _fla_available() -> bool:
    return importlib.util.find_spec("fla") is not None


def _train_step(
    *,
    model: SophiaDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    input_ids: torch.Tensor,
    use_fused_loss: bool,
) -> tuple[float, float]:
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        labels=input_ids,
        compute_loss=bool(use_fused_loss),
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError(f"non-finite lifecycle loss: {output.loss}")
    output.loss.backward()
    squared_norm = 0.0
    grad_count = 0
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all().item()):
            raise RuntimeError("non-finite gradient in lifecycle validation")
        squared_norm += float(grad.float().square().sum().item())
        grad_count += int(grad.numel())
    if grad_count <= 0:
        raise RuntimeError("lifecycle backward produced no gradients")
    optimizer.step()
    scheduler.step()
    return float(output.loss.detach().float().item()), math.sqrt(squared_norm)


def _logits(model: SophiaDecoder, input_ids: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, return_dict=True)
    if output.logits is None:
        raise RuntimeError("lifecycle forward did not return logits")
    return output.logits.detach().float().cpu()


def _max_state_delta(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    if set(left) != set(right):
        raise RuntimeError("state dict keys differ during lifecycle comparison")
    maximum = 0.0
    for key in sorted(left):
        delta = float(
            (left[key].detach().float().cpu() - right[key].detach().float().cpu())
            .abs()
            .max()
            .item()
        )
        maximum = max(maximum, delta)
    return maximum


def _state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _sha256_json_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_native_output(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if path_uses_host_mounted_storage(str(resolved)):
        raise ValueError(f"lifecycle output must use target-native storage: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"output directory must be empty: {resolved}")


def validate_lifecycle(
    *,
    output_dir: str,
    tokenizer_path: str,
    device_name: str,
    seed: int,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    _require_native_output(output)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("scaled lifecycle validation requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    started = time.time()
    use_fused_loss = _liger_available()
    fla_available = _fla_available()
    spec = scaled_model_spec().with_overrides(
        kda_backend="auto" if fla_available else "reference"
    )
    model = _build_model(spec, device=device)
    parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    tied_before = input_weight.data_ptr() == output_weight.data_ptr()
    if not tied_before:
        raise RuntimeError("input and output embeddings are not tied")

    layers = model.model.layers
    layer_types = [str(layer.layer_type) for layer in layers]
    kda = layers[0].attn
    mla = layers[3].attn
    architecture_checks = {
        "layer_pattern": layer_types,
        "layer_pattern_matches": layer_types == ["kda", "kda", "kda", "mla"],
        "kda_module": isinstance(kda, SophiaKDA),
        "mla_module": isinstance(mla, SophiaMLA),
        "no_positional_encoding": all(
            not hasattr(module, "rotary_embedding") for module in model.modules()
        ),
        "kda_decay_lower_bound": float(kda.lower_bound),
        "kda_decay_bound_matches": float(kda.lower_bound)
        == float(spec.kda_decay_lower_bound),
        "short_conv_kernel": int(kda.conv_kernel),
        "short_conv_kernel_matches": int(kda.conv_kernel)
        == int(spec.short_conv_kernel),
        "head_count": int(kda.num_heads),
        "head_count_matches": int(kda.num_heads) == int(spec.num_heads),
    }
    if not all(
        bool(value)
        for key, value in architecture_checks.items()
        if key
        not in {
            "layer_pattern",
            "kda_decay_lower_bound",
            "short_conv_kernel",
            "head_count",
        }
    ):
        raise RuntimeError(f"architecture semantic check failed: {architecture_checks}")

    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    causal_a = torch.randint(
        9,
        int(spec.vocab_size),
        (1, 32),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    causal_b = causal_a.clone()
    causal_b[:, 16:] = torch.randint(
        9,
        int(spec.vocab_size),
        (1, 16),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    causal_prefix_delta = float(
        (_logits(model, causal_a)[:, :16] - _logits(model, causal_b)[:, :16])
        .abs()
        .max()
        .item()
    )
    if causal_prefix_delta != 0.0:
        raise RuntimeError(
            f"causal prefix changed after suffix mutation: {causal_prefix_delta}"
        )

    train_batch_1 = torch.randint(
        9,
        int(spec.vocab_size),
        (2, 64),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    train_batch_2 = torch.randint(
        9,
        int(spec.vocab_size),
        (2, 64),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    eval_batch = torch.randint(
        9,
        int(spec.vocab_size),
        (1, 48),
        generator=generator,
        dtype=torch.long,
    ).to(device)

    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)
    loss_step_1, grad_norm_step_1 = _train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        input_ids=train_batch_1,
        use_fused_loss=use_fused_loss,
    )
    checkpoint_logits = _logits(model, eval_batch)
    save_checkpoint(
        output_dir=str(output),
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args={"model": asdict(spec), "seed": int(seed)},
        rng={
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device),
        },
        ema=None,
        train_state={"seen_tokens": int(train_batch_1.numel())},
        save_total_limit=2,
    )
    checkpoint_path = output / "checkpoints" / "ckpt_step1.pt"

    tokenizer = load_local_tokenizer(
        str(Path(tokenizer_path).expanduser().resolve()),
        model_max_length=int(ModelSpec.default().max_seq_len),
    )
    parent_export = output / "parent_export"
    export_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(parent_export),
        safe_serialization=True,
    )
    checkpoint_export = output / "checkpoint_export"
    checkpoint_export_report = export_checkpoint(
        checkpoint_path=str(checkpoint_path),
        parent_export=str(parent_export),
        output_dir=str(checkpoint_export),
    )
    checkpoint_reloaded = SophiaDecoder.from_pretrained(
        checkpoint_export,
        device=device,
        dtype=torch.bfloat16,
        runtime_max_seq_len=int(spec.max_seq_len),
    )
    checkpoint_reload_delta = float(
        (_logits(checkpoint_reloaded, eval_batch) - checkpoint_logits)
        .abs()
        .max()
        .item()
    )
    if checkpoint_reload_delta != 0.0:
        raise RuntimeError(
            f"checkpoint export/reload logits mismatch: {checkpoint_reload_delta}"
        )
    del checkpoint_reloaded

    reference_loss_step_2, reference_grad_norm_step_2 = _train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        input_ids=train_batch_2,
        use_fused_loss=use_fused_loss,
    )
    reference_state = _state_to_cpu(model)

    checkpoint = load_checkpoint(str(checkpoint_path), expected_kind="full")
    restored = _build_model(spec, device=device)
    restored.load_state_dict(checkpoint.model, strict=True)
    restored_optimizer = _build_optimizer(restored)
    restored_optimizer.load_state_dict(checkpoint.optimizer)
    move_optimizer_state_to_device(restored_optimizer, str(device))
    restored_scheduler = _build_scheduler(restored_optimizer)
    if checkpoint.scheduler is None:
        raise RuntimeError("checkpoint is missing scheduler state")
    restored_scheduler.load_state_dict(checkpoint.scheduler)
    resumed_loss_step_2, resumed_grad_norm_step_2 = _train_step(
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        input_ids=train_batch_2,
        use_fused_loss=use_fused_loss,
    )
    resume_state_delta = _max_state_delta(reference_state, restored.state_dict())
    if resume_state_delta != 0.0:
        raise RuntimeError(f"resume model state mismatch: {resume_state_delta}")

    final_logits = _logits(restored, eval_batch)
    final_export = output / "final_export"
    export_model_artifacts(
        model=restored,
        tokenizer=tokenizer,
        output_dir=str(final_export),
        safe_serialization=True,
    )
    reloaded = SophiaDecoder.from_pretrained(
        final_export,
        device=device,
        dtype=torch.bfloat16,
        runtime_max_seq_len=int(spec.max_seq_len),
    )
    final_reload_delta = float(
        (_logits(reloaded, eval_batch) - final_logits).abs().max().item()
    )
    tied_after_reload = (
        reloaded.get_input_embeddings().weight.data_ptr()
        == reloaded.get_output_embeddings().weight.data_ptr()
    )
    if final_reload_delta != 0.0 or not tied_after_reload:
        raise RuntimeError(
            "final export/reload equivalence failed: "
            f"logits_delta={final_reload_delta} tied={tied_after_reload}"
        )

    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "scope": (
            "scaled architecture and training lifecycle validation; not a full-model "
            "throughput, memory, context-length, or stability measurement"
        ),
        "timestamp_unix": int(time.time()),
        "duration_seconds": float(time.time() - started),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "loss_backend": (
                "liger_fused_linear_cross_entropy"
                if use_fused_loss
                else "logits_cross_entropy"
            ),
            "kda_backend": str(spec.kda_backend),
        },
        "selected_model": asdict(ModelSpec.default()),
        "scaled_model": asdict(spec),
        "scaled_parameter_count": int(parameter_count),
        "checks": {
            "forward_backward": "pass",
            "finite_gradients": "pass",
            "causal_mask_prefix_invariance": "pass",
            "hybrid_layer_pattern": "pass",
            "bounded_kda_decay": "pass",
            "short_convolution": "pass",
            "nope": "pass",
            "tied_embeddings_before_training": "pass",
            "full_checkpoint_save_load": "pass",
            "optimizer_scheduler_resume_equivalence": "pass",
            "checkpoint_export_reload_equivalence": "pass",
            "final_export_reload_equivalence": "pass",
            "tied_embeddings_after_reload": "pass",
        },
        "metrics": {
            "loss_step_1": loss_step_1,
            "grad_norm_step_1": grad_norm_step_1,
            "reference_loss_step_2": reference_loss_step_2,
            "resumed_loss_step_2": resumed_loss_step_2,
            "reference_grad_norm_step_2": reference_grad_norm_step_2,
            "resumed_grad_norm_step_2": resumed_grad_norm_step_2,
            "causal_prefix_max_abs_delta": causal_prefix_delta,
            "resume_state_max_abs_delta": resume_state_delta,
            "checkpoint_reload_logits_max_abs_delta": checkpoint_reload_delta,
            "final_reload_logits_max_abs_delta": final_reload_delta,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "architecture_checks": architecture_checks,
        "artifacts": {
            "output_dir": str(output),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_export": checkpoint_export_report,
            "final_export": str(final_export),
            "final_model_sha256": sha256_file(final_export / "model.safetensors"),
            "final_config_sha256": _sha256_json_file(final_export / "config.json"),
        },
    }
    write_json_atomic(
        output / "lifecycle_report.json",
        report,
        ensure_ascii=False,
        sort_keys=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the scaled Sophia model/checkpoint/export lifecycle on CUDA."
    )
    parser.add_argument(
        "--output-dir",
        default="out/sophia_lifecycle",
    )
    parser.add_argument("--tokenizer", default="ml/modeling/text")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--report",
        default="",
        help="Optional second copy of the completed JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_lifecycle(
        output_dir=str(args.output_dir),
        tokenizer_path=str(args.tokenizer),
        device_name=str(args.device),
        seed=int(args.seed),
    )
    if str(args.report).strip():
        write_json_atomic(
            Path(str(args.report)).expanduser().resolve(),
            report,
            ensure_ascii=False,
            sort_keys=True,
            make_parents=True,
        )
    print(
        f"[DONE] status={report['status']} output={report['artifacts']['output_dir']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
