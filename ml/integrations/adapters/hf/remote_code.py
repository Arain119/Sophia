"""Remote-code naming shared by the HF adapter and export bundle."""

from __future__ import annotations

HF_MODELING_MODULE = "modeling_sophia"
HF_CONFIG_CLASS = "SophiaConfig"
HF_CAUSAL_LM_CLASS = "SophiaForCausalLM"
HF_SUPPORT_FILENAME = "hf_support.py"
HF_RUNTIME_FILENAME = "sophia_runtime.py"

__all__ = [
    "HF_CAUSAL_LM_CLASS",
    "HF_CONFIG_CLASS",
    "HF_MODELING_MODULE",
    "HF_RUNTIME_FILENAME",
    "HF_SUPPORT_FILENAME",
]
