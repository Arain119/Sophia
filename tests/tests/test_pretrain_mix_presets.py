from __future__ import annotations

import math

from ml.tooling.core.pretrain_mix_presets import (
    PRESETS,
    _EN_HUMANIST_CURATED_BASE,
    _PRETRAIN_HUMANIST_TARGETS_FRAC,
    _PRETRAIN_TARGETS_FRAC,
    _ZH_HUMANIST_CURATED_BASE,
)


def test_pretrain_mix_excludes_chat_and_reasoning() -> None:
    assert all("chat" not in key and "reasoning" not in key for key in _PRETRAIN_TARGETS_FRAC)
    assert math.isclose(sum(_PRETRAIN_TARGETS_FRAC.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_pretrain_preset_quotas_sum_to_total() -> None:
    preset = PRESETS["pretrain"]
    quotas = preset.resolve_quotas()
    assert sum(quotas.values()) == int(preset.total_tokens_eos)


def test_pretrain_target_sources_exist_in_preset() -> None:
    preset = PRESETS["pretrain"]
    source_names = {src.name for src in preset.sources}
    target_source_names = {key.split(":", 1)[0] for key in _PRETRAIN_TARGETS_FRAC}
    assert target_source_names.issubset(source_names)


def test_pretrain_humanist_mix_is_closed_and_sources_exist() -> None:
    assert math.isclose(
        sum(_PRETRAIN_HUMANIST_TARGETS_FRAC.values()),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    preset = PRESETS["pretrain_humanist"]
    source_names = {src.name for src in preset.sources}
    target_source_names = {
        key.split(":", 1)[0] for key in _PRETRAIN_HUMANIST_TARGETS_FRAC
    }
    assert target_source_names.issubset(source_names)


def test_humanist_curated_base_dirs_resolve_to_supported_candidates() -> None:
    assert _EN_HUMANIST_CURATED_BASE in {
        "dataset/pretrain_curations/en_discourse_humanist_balanced_v3",
        "dataset/pretrain_curations/en_discourse_humanist_balanced_v2",
        "dataset/pretrain_curations/en_discourse_humanist_balanced",
    }
    assert _ZH_HUMANIST_CURATED_BASE in {
        "dataset/pretrain_curations/zh_discourse_humanist_v4",
        "dataset/pretrain_curations/zh_discourse_humanist_v3",
        "dataset/pretrain_curations/zh_discourse_humanist_v2",
        "dataset/pretrain_curations/zh_discourse_humanist",
    }


def test_pretrain_sources_include_bridge_discourse_docs() -> None:
    expected = {
        "en_discourse_django",
        "en_discourse_flask",
        "en_discourse_fastapi",
        "en_discourse_jinja",
        "en_discourse_sqlalchemy",
        "en_discourse_htmx",
    }
    assert expected.issubset({src.name for src in PRESETS["pretrain"].sources})
    assert expected.issubset({src.name for src in PRESETS["pretrain_humanist"].sources})


def test_pretrain_final_exists_and_sums_to_total() -> None:
    preset = PRESETS["pretrain_final"]
    quotas = preset.resolve_quotas()
    assert sum(quotas.values()) == int(preset.total_tokens_eos)
    assert "zh_news:UNKNOWN" not in preset.targets_frac
    assert "en_news:UNKNOWN" not in preset.targets_frac
    assert "code_cpp:C++" not in preset.targets_frac
    assert "code_java:Java" not in preset.targets_frac
    assert "code_gcclean:Python" in preset.targets_frac
    assert "code_gcclean:HTML" in preset.targets_frac
    assert "code_gcclean:JavaScript" in preset.targets_frac
    assert "code_alpine:HTML" in preset.targets_frac
    assert "code_hyperscript:JavaScript" in preset.targets_frac
    assert "code_pico:CSS" in preset.targets_frac
    assert "code_datastar:TypeScript" in preset.targets_frac
