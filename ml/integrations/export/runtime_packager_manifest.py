"""Remote-code bundle manifest for self-contained exported model directories."""

from __future__ import annotations

from dataclasses import dataclass

from ml.integrations.adapters.hf.remote_code import (
    HF_MODELING_MODULE,
    HF_RUNTIME_FILENAME,
    HF_SUPPORT_FILENAME,
)

EXPORT_MODEL_SOURCE_FILENAME = "model.py"
EXPORT_MODELING_MODULE = HF_MODELING_MODULE
EXPORT_MODELING_FILENAME = f"{EXPORT_MODELING_MODULE}.py"
EXPORT_RUNTIME_FILENAME = HF_RUNTIME_FILENAME
EXPORT_RUNTIME_CONFIG_FILENAME = "model_config.py"
EXPORT_HF_REMOTE_CODE_FILENAME = "hf_remote_code.py"
EXPORT_HF_SUPPORT_FILENAME = HF_SUPPORT_FILENAME
EXPORT_HF_CACHE_FILENAME = "hf_cache.py"
EXPORT_HF_CONFIG_FILENAME = "hf_config.py"
EXPORT_HF_GENERATION_FILENAME = "hf_generation.py"
EXPORT_HF_LIFECYCLE_FILENAME = "hf_lifecycle.py"
EXPORT_RUNTIME_LINEAR_FILENAME = "runtime_linear.py"
EXPORT_CANONICAL_CONFIG_FILENAME = "canonical_config.py"
EXPORT_MODEL_DIR_FILENAME = "model_dir.py"
EXPORT_LOSS_STATS_FILENAME = "loss_stats.py"
EXPORT_INPUT_MASK_FILENAME = "input_mask.py"
EXPORT_CACHE_DECODE_FILENAME = "cache_decode.py"
EXPORT_CONFIG_PROJECTION_FILENAME = "config_projection.py"
EXPORT_HF_PROJECTION_FILENAME = "hf_projection.py"
EXPORT_RUNTIME_BACKEND_FILENAME = "runtime_backend.py"
EXPORT_DECODER_FORWARD_FILENAME = "decoder_forward.py"
EXPORT_DECODER_TYPES_FILENAME = "decoder_types.py"
EXPORT_DECODER_LOSS_BASIC_FILENAME = "decoder_loss.py"
EXPORT_DECODER_LOSS_FORWARD_FILENAME = "decoder_loss_forward.py"
EXPORT_DECODER_FULL_FILENAME = "decoder_full.py"
EXPORT_PRETRAINED_BUNDLE_FILENAME = "pretrained_bundle.py"
EXPORT_DECODER_OUTPUT_FILENAME = "decoder_output.py"
EXPORT_DECODER_HOST_FILENAME = "decoder_host.py"
EXPORT_DECODER_RUNTIME_FILENAME = "decoder_runtime.py"
EXPORT_SOPHIA_DECODER_FILENAME = "sophia_decoder.py"
EXPORT_MODEL_SEMANTICS_FILENAME = "semantics.py"
EXPORT_RUNTIME_ATTENTION_FILENAME = "model_attention.py"
EXPORT_RUNTIME_BLOCKS_FILENAME = "model_blocks.py"
EXPORT_RUNTIME_OPS_FILENAME = "model_ops.py"
EXPORT_RUNTIME_HOST_FILENAME = "model_runtime.py"
EXPORT_RUNTIME_CONTROL_FILENAME = "model_runtime_control.py"
EXPORT_RUNTIME_STATE_FILENAME = "model_state.py"
EXPORT_RUNTIME_CONTRACTS_FILENAME = "runtime_contracts.py"
EXPORT_RUNTIME_TRANSFORMER_SETUP_FILENAME = "model_transformer_setup.py"


@dataclass(frozen=True)
class RemoteCodeArtifact:
    output_filename: str
    source_relpath: tuple[str, ...]


_REMOTE_CODE_ARTIFACT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (EXPORT_MODELING_FILENAME, ("integrations", "adapters", "hf", EXPORT_MODEL_SOURCE_FILENAME)),
    (EXPORT_RUNTIME_FILENAME, ("runtime", "model", "transformer.py")),
    (EXPORT_RUNTIME_CONFIG_FILENAME, ("runtime", "model", "config.py")),
    (EXPORT_HF_REMOTE_CODE_FILENAME, ("integrations", "adapters", "hf", "remote_code.py")),
    (EXPORT_HF_SUPPORT_FILENAME, ("integrations", "adapters", "hf", "support.py")),
    (EXPORT_HF_CACHE_FILENAME, ("integrations", "adapters", "hf", "cache.py")),
    (EXPORT_HF_CONFIG_FILENAME, ("integrations", "adapters", "hf", "config.py")),
    (EXPORT_HF_GENERATION_FILENAME, ("integrations", "adapters", "hf", "generation.py")),
    (EXPORT_HF_LIFECYCLE_FILENAME, ("integrations", "adapters", "hf", "lifecycle.py")),
    (EXPORT_RUNTIME_ATTENTION_FILENAME, ("runtime", "model", "attention.py")),
    (EXPORT_RUNTIME_BLOCKS_FILENAME, ("runtime", "model", "blocks.py")),
    (EXPORT_RUNTIME_OPS_FILENAME, ("runtime", "model", "ops.py")),
    (EXPORT_RUNTIME_HOST_FILENAME, ("runtime", "model", "runtime_host.py")),
    (
        EXPORT_RUNTIME_CONTRACTS_FILENAME,
        ("runtime", "contracts.py"),
    ),
    (EXPORT_RUNTIME_CONTROL_FILENAME, ("runtime", "model", "runtime_control.py")),
    (EXPORT_RUNTIME_STATE_FILENAME, ("runtime", "model", "state.py")),
    (
        EXPORT_RUNTIME_TRANSFORMER_SETUP_FILENAME,
        ("runtime", "model", "transformer_setup.py"),
    ),
    (EXPORT_RUNTIME_LINEAR_FILENAME, ("runtime", "model", "runtime_linear.py")),
    (EXPORT_CANONICAL_CONFIG_FILENAME, ("modeling", "config.py")),
    (EXPORT_MODEL_DIR_FILENAME, ("integrations", "export", "model_dir.py")),
    (EXPORT_LOSS_STATS_FILENAME, ("modeling", "text", "loss_stats.py")),
    (EXPORT_INPUT_MASK_FILENAME, ("modeling", "input_mask.py")),
    (EXPORT_CACHE_DECODE_FILENAME, ("modeling", "cache_decode.py")),
    (EXPORT_CONFIG_PROJECTION_FILENAME, ("modeling", "config_projection.py")),
    (EXPORT_HF_PROJECTION_FILENAME, ("integrations", "adapters", "hf", "config.py")),
    (EXPORT_RUNTIME_BACKEND_FILENAME, ("modeling", "runtime_backend.py")),
    (EXPORT_DECODER_FORWARD_FILENAME, ("modeling", "decoder_forward.py")),
    (EXPORT_DECODER_TYPES_FILENAME, ("modeling", "decoder_types.py")),
    (EXPORT_DECODER_LOSS_BASIC_FILENAME, ("modeling", "decoder_loss.py")),
    (EXPORT_DECODER_LOSS_FORWARD_FILENAME, ("modeling", "decoder_loss_forward.py")),
    (EXPORT_DECODER_FULL_FILENAME, ("modeling", "decoder_full.py")),
    (EXPORT_PRETRAINED_BUNDLE_FILENAME, ("modeling", "pretrained_bundle.py")),
    (EXPORT_DECODER_OUTPUT_FILENAME, ("modeling", "decoder_output.py")),
    (
        EXPORT_DECODER_RUNTIME_FILENAME,
        ("modeling", "decoder_runtime.py"),
    ),
    (EXPORT_DECODER_HOST_FILENAME, ("modeling", "decoder_host.py")),
    (EXPORT_SOPHIA_DECODER_FILENAME, ("modeling", "sophia_decoder.py")),
    (EXPORT_MODEL_SEMANTICS_FILENAME, ("core", "spec", "semantics", "__init__.py")),
)
REMOTE_CODE_BUNDLE_MANIFEST = tuple(
    RemoteCodeArtifact(
        output_filename=str(output_filename),
        source_relpath=("ml", *source_relpath),
    )
    for output_filename, source_relpath in _REMOTE_CODE_ARTIFACT_SPECS
)


__all__ = [
    "EXPORT_CANONICAL_CONFIG_FILENAME",
    "EXPORT_CACHE_DECODE_FILENAME",
    "EXPORT_CONFIG_PROJECTION_FILENAME",
    "EXPORT_RUNTIME_BACKEND_FILENAME",
    "EXPORT_HF_PROJECTION_FILENAME",
    "EXPORT_DECODER_FORWARD_FILENAME",
    "EXPORT_HF_REMOTE_CODE_FILENAME",
    "EXPORT_DECODER_TYPES_FILENAME",
    "EXPORT_DECODER_LOSS_BASIC_FILENAME",
    "EXPORT_DECODER_LOSS_FORWARD_FILENAME",
    "EXPORT_DECODER_FULL_FILENAME",
    "EXPORT_HF_CACHE_FILENAME",
    "EXPORT_HF_CONFIG_FILENAME",
    "EXPORT_HF_GENERATION_FILENAME",
    "EXPORT_HF_LIFECYCLE_FILENAME",
    "EXPORT_HF_SUPPORT_FILENAME",
    "EXPORT_MODEL_DIR_FILENAME",
    "EXPORT_INPUT_MASK_FILENAME",
    "EXPORT_LOSS_STATS_FILENAME",
    "EXPORT_MODELING_FILENAME",
    "EXPORT_MODELING_MODULE",
    "EXPORT_MODEL_SOURCE_FILENAME",
    "EXPORT_MODEL_SEMANTICS_FILENAME",
    "EXPORT_DECODER_OUTPUT_FILENAME",
    "EXPORT_DECODER_RUNTIME_FILENAME",
    "EXPORT_SOPHIA_DECODER_FILENAME",
    "EXPORT_PRETRAINED_BUNDLE_FILENAME",
    "EXPORT_RUNTIME_ATTENTION_FILENAME",
    "EXPORT_DECODER_HOST_FILENAME",
    "EXPORT_RUNTIME_BLOCKS_FILENAME",
    "EXPORT_RUNTIME_LINEAR_FILENAME",
    "EXPORT_RUNTIME_CONTROL_FILENAME",
    "EXPORT_RUNTIME_CONFIG_FILENAME",
    "EXPORT_RUNTIME_FILENAME",
    "EXPORT_RUNTIME_HOST_FILENAME",
    "EXPORT_RUNTIME_OPS_FILENAME",
    "EXPORT_RUNTIME_STATE_FILENAME",
    "EXPORT_RUNTIME_CONTRACTS_FILENAME",
    "EXPORT_RUNTIME_TRANSFORMER_SETUP_FILENAME",
    "REMOTE_CODE_BUNDLE_MANIFEST",
    "RemoteCodeArtifact",
]
