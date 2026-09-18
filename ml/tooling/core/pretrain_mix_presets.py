from __future__ import annotations

"""Pretraining mixture presets for local token-shard construction.

Presets target token-level quotas, including EOS insertions, matching the shard
builder and streaming pretraining behavior. The local pool is organized under
``dataset/pretrain/{train,val,test}`` by content family.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BucketMode = Literal["user_taxonomy", "column"]


def _resolve_curated_base_dir(*candidates: str) -> str:
    for cand in candidates:
        raw = str(cand or "").strip().replace("\\", "/").strip("/")
        if not raw:
            continue
        if Path(raw).exists():
            return raw
    for cand in candidates:
        raw = str(cand or "").strip().replace("\\", "/").strip("/")
        if raw:
            return raw
    raise ValueError("at least one curated base-dir candidate is required")


def _normalized_targets_frac(raw: dict[str, float]) -> dict[str, float]:
    total = float(sum(float(v) for v in raw.values() if float(v) > 0.0))
    if total <= 0.0:
        raise ValueError("target fractions must sum to > 0")
    return {
        str(key): (float(value) / total)
        for key, value in raw.items()
        if float(value) > 0.0
    }


_EN_HUMANIST_CURATED_BASE = _resolve_curated_base_dir(
    "dataset/pretrain_curations/en_discourse_humanist_balanced_v3",
    "dataset/pretrain_curations/en_discourse_humanist_balanced_v2",
    "dataset/pretrain_curations/en_discourse_humanist_balanced",
)

_ZH_HUMANIST_CURATED_BASE = _resolve_curated_base_dir(
    "dataset/pretrain_curations/zh_discourse_humanist_v4",
    "dataset/pretrain_curations/zh_discourse_humanist_v3",
    "dataset/pretrain_curations/zh_discourse_humanist_v2",
    "dataset/pretrain_curations/zh_discourse_humanist",
)


@dataclass(frozen=True)
class SourcePreset:
    name: str
    parquet_glob: str
    text_col: str
    title_col: str | None = None
    title_sep: str = "\n\n"
    bucket_mode: BucketMode = "user_taxonomy"
    bucket_col: str | None = None
    score_col: str | None = None
    min_score: float | None = None
    max_score: float | None = None


@dataclass(frozen=True)
class PretrainMixPreset:
    name: str
    total_tokens_eos: int
    sources: tuple[SourcePreset, ...]
    targets_frac: dict[str, float]

    def resolve_quotas(self, *, total_tokens_eos: int | None = None) -> dict[str, int]:
        total = int(
            self.total_tokens_eos if total_tokens_eos is None else total_tokens_eos
        )
        if total <= 0:
            raise ValueError("total_tokens_eos must be > 0")
        quotas: dict[str, int] = {}
        for key, value in self.targets_frac.items():
            frac = float(value)
            if frac <= 0:
                continue
            quotas[str(key)] = int(round(frac * float(total)))
        drift = int(total) - int(sum(quotas.values()))
        if drift != 0 and quotas:
            max_key = max(quotas.items(), key=lambda kv: int(kv[1]))[0]
            quotas[max_key] = int(quotas[max_key]) + int(drift)
        return quotas



def _code_source_preset(
    *,
    split: str,
    name: str,
    source_dir: str,
    file_glob: str,
) -> SourcePreset:
    return SourcePreset(
        name=str(name),
        parquet_glob=f"dataset/pretrain/{split}/code/source={source_dir}/{file_glob}",
        text_col="text",
        bucket_mode="column",
        bucket_col="source",
        score_col="score",
        min_score=4.5,
    )


def _make_pretrain_code_sources(split: str) -> tuple[SourcePreset, ...]:
    s = str(split or "").strip()
    if not s:
        raise ValueError("split must be non-empty")
    return (
        _code_source_preset(split=s, name="code_general_py", source_dir="Python", file_glob="python_part_*.parquet"),
        _code_source_preset(split=s, name="code_general_js", source_dir="JavaScript", file_glob="javascript_part_*.parquet"),
        _code_source_preset(split=s, name="code_general_jsx", source_dir="JavaScript%20(JSX)", file_glob="javascriptjsx_part_*.parquet"),
        _code_source_preset(split=s, name="code_general_ts", source_dir="TypeScript", file_glob="typescript_part_*.parquet"),
        _code_source_preset(split=s, name="code_general_sql", source_dir="SQL", file_glob="sql_part_*.parquet"),
        _code_source_preset(split=s, name="code_gcclean", source_dir="*", file_glob="*_gcclean_part_*.parquet"),
        _code_source_preset(split=s, name="code_mdn", source_dir="*", file_glob="*_mdn_part_*.parquet"),
        _code_source_preset(split=s, name="code_cpython", source_dir="*", file_glob="*_cpython_part_*.parquet"),
        _code_source_preset(split=s, name="code_django", source_dir="*", file_glob="*_django_part_*.parquet"),
        _code_source_preset(split=s, name="code_flask", source_dir="*", file_glob="*_flask_part_*.parquet"),
        _code_source_preset(split=s, name="code_fastapi", source_dir="*", file_glob="*_fastapi_part_*.parquet"),
        _code_source_preset(split=s, name="code_jinja", source_dir="*", file_glob="*_jinja_part_*.parquet"),
        _code_source_preset(split=s, name="code_sqlalchemy", source_dir="*", file_glob="*_sqlalchemy_part_*.parquet"),
        _code_source_preset(split=s, name="code_htmx", source_dir="*", file_glob="*_htmx_part_*.parquet"),
        _code_source_preset(split=s, name="code_pico", source_dir="*", file_glob="*_pico_part_*.parquet"),
        _code_source_preset(split=s, name="code_simplecss", source_dir="*", file_glob="*_simplecss_part_*.parquet"),
        _code_source_preset(split=s, name="code_missingcss", source_dir="*", file_glob="*_missingcss_part_*.parquet"),
        _code_source_preset(split=s, name="code_watercss", source_dir="*", file_glob="*_watercss_part_*.parquet"),
        _code_source_preset(split=s, name="code_newcss", source_dir="*", file_glob="*_newcss_part_*.parquet"),
        _code_source_preset(split=s, name="code_sakura", source_dir="*", file_glob="*_sakura_part_*.parquet"),
        _code_source_preset(split=s, name="code_skeleton", source_dir="*", file_glob="*_skeleton_part_*.parquet"),
        _code_source_preset(split=s, name="code_alpine", source_dir="*", file_glob="*_alpine_part_*.parquet"),
        _code_source_preset(split=s, name="code_hyperscript", source_dir="*", file_glob="*_hyperscript_part_*.parquet"),
        _code_source_preset(split=s, name="code_idiomorph", source_dir="*", file_glob="*_idiomorph_part_*.parquet"),
        _code_source_preset(split=s, name="code_milligram", source_dir="*", file_glob="*_milligram_part_*.parquet"),
        _code_source_preset(split=s, name="code_chota", source_dir="*", file_glob="*_chota_part_*.parquet"),
        _code_source_preset(split=s, name="code_spectre", source_dir="*", file_glob="*_spectre_part_*.parquet"),
        _code_source_preset(split=s, name="code_mvp", source_dir="*", file_glob="*_mvp_part_*.parquet"),
        _code_source_preset(split=s, name="code_picnic", source_dir="*", file_glob="*_picnic_part_*.parquet"),
        _code_source_preset(split=s, name="code_datastar", source_dir="*", file_glob="*_datastar_part_*.parquet"),
        _code_source_preset(split=s, name="code_unpoly", source_dir="*", file_glob="*_unpoly_part_*.parquet"),
        _code_source_preset(split=s, name="code_intercooler", source_dir="*", file_glob="*_intercooler_part_*.parquet"),
        _code_source_preset(split=s, name="code_cpp", source_dir="C++", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_ccheader", source_dir="C%2FC++%20Header", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_go", source_dir="Go", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_rust", source_dir="Rust", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_java", source_dir="Java", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_c", source_dir="C", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_kotlin", source_dir="Kotlin", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_csharp", source_dir="C#", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_swift", source_dir="Swift", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_shell", source_dir="Shell", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_dart", source_dir="Dart", file_glob="*.parquet"),
        _code_source_preset(split=s, name="code_php", source_dir="PHP", file_glob="*.parquet"),
    )


def _make_pretrain_sources(split: str) -> tuple[SourcePreset, ...]:
    s = str(split or "").strip()
    if not s:
        raise ValueError("split must be non-empty")
    return (
        *_make_pretrain_code_sources(s),
        SourcePreset(name="math", parquet_glob=f"dataset/pretrain/{s}/math/*.parquet", text_col="text", bucket_mode="column", score_col="score", min_score=0.5),
        SourcePreset(name="translation", parquet_glob=f"dataset/pretrain/{s}/translation/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_general", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_mdn", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_mdn_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_cpython", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_cpython_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_django", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_django_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_flask", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_flask_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_fastapi", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_fastapi_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_jinja", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_jinja_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_sqlalchemy", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_sqlalchemy_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_htmx", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_htmx_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="zh_discourse", parquet_glob=f"dataset/pretrain/{s}/discourse/zh_discourse/zh_discourse_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_wiki", parquet_glob=f"dataset/pretrain/{s}/wiki/en_wiki/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="zh_wiki", parquet_glob=f"dataset/pretrain/{s}/wiki/zh_wiki/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_books", parquet_glob=f"dataset/pretrain/{s}/books/en_books/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(
            name="zh_books",
            parquet_glob=f"dataset/pretrain/{s}/books/zh_books/*.parquet",
            text_col="content",
            title_col="title",
            title_sep="\n\n",
            bucket_mode="column",
        ),
    )


def _make_pretrain_humanist_sources(split: str) -> tuple[SourcePreset, ...]:
    s = str(split or "").strip()
    if not s:
        raise ValueError("split must be non-empty")
    return (
        *_make_pretrain_code_sources(s),
        SourcePreset(name="math", parquet_glob=f"dataset/pretrain/{s}/math/*.parquet", text_col="text", bucket_mode="column", score_col="score", min_score=0.5),
        SourcePreset(name="translation", parquet_glob=f"dataset/pretrain/{s}/translation/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_humanist", parquet_glob=f"{_EN_HUMANIST_CURATED_BASE}/{s}/discourse/en_discourse/en_discourse_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_mdn", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_mdn_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_cpython", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_cpython_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_django", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_django_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_flask", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_flask_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_fastapi", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_fastapi_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_jinja", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_jinja_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_sqlalchemy", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_sqlalchemy_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_discourse_htmx", parquet_glob=f"dataset/pretrain/{s}/discourse/en_discourse/en_discourse_htmx_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="zh_discourse_humanist", parquet_glob=f"{_ZH_HUMANIST_CURATED_BASE}/{s}/discourse/zh_discourse/zh_discourse_part_*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_wiki", parquet_glob=f"dataset/pretrain/{s}/wiki/en_wiki/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="zh_wiki", parquet_glob=f"dataset/pretrain/{s}/wiki/zh_wiki/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(name="en_books", parquet_glob=f"dataset/pretrain/{s}/books/en_books/*.parquet", text_col="content", bucket_mode="column"),
        SourcePreset(
            name="zh_books",
            parquet_glob=f"dataset/pretrain/{s}/books/zh_books/*.parquet",
            text_col="content",
            title_col="title",
            title_sep="\n\n",
            bucket_mode="column",
        ),
    )


_PRETRAIN_TRAIN_SOURCES = _make_pretrain_sources("train")
_PRETRAIN_VAL_SOURCES = _make_pretrain_sources("val")
_PRETRAIN_TEST_SOURCES = _make_pretrain_sources("test")
_PRETRAIN_HUMANIST_TRAIN_SOURCES = _make_pretrain_humanist_sources("train")
_PRETRAIN_HUMANIST_VAL_SOURCES = _make_pretrain_humanist_sources("val")
_PRETRAIN_HUMANIST_TEST_SOURCES = _make_pretrain_humanist_sources("test")



_PRETRAIN_TARGETS_FRAC: dict[str, float] = _normalized_targets_frac({
    # Supply-aligned lightweight web-app mix:
    # - books/wiki remain the main reasoning and worldview backbone
    # - discourse stays present, but tiny tech-doc tails are reduced to quotas the
    #   local pool can actually satisfy under strict no-repeat sampling
    # - code is concentrated in large Python/web buckets; tiny official sources are
    #   retained as "style/reference spice" instead of unrealistic primary quotas
    # - only document-style corpora admitted by the pretraining policy enter this mix
    "zh_books:UNKNOWN": 0.2300,
    "en_books:UNKNOWN": 0.1850,
    "zh_wiki:UNKNOWN": 0.1530,
    "en_wiki:UNKNOWN": 0.1560,
    "zh_discourse:UNKNOWN": 0.0240,
    "en_discourse_general:UNKNOWN": 0.0180,
    "en_discourse_mdn:UNKNOWN": 0.00024,
    "en_discourse_cpython:UNKNOWN": 0.00006,
    "en_discourse_django:UNKNOWN": 0.00004,
    "en_discourse_flask:UNKNOWN": 0.000003,
    "en_discourse_fastapi:UNKNOWN": 0.0000075,
    "en_discourse_jinja:UNKNOWN": 0.0000007,
    "en_discourse_sqlalchemy:UNKNOWN": 0.00002,
    "en_discourse_htmx:UNKNOWN": 0.000009,
    "math:UNKNOWN": 0.0235,
    "translation:UNKNOWN": 0.0050,
    "code_general_py:Python": 0.0340,
    "code_gcclean:Python": 0.10331,
    "code_cpython:Python": 0.00016,
    "code_django:Python": 0.00007,
    "code_flask:Python": 0.000005,
    "code_fastapi:Python": 0.00001,
    "code_jinja:Python": 0.000006,
    "code_sqlalchemy:Python": 0.00010,
    "code_gcclean:HTML": 0.0123463,
    "code_mdn:HTML": 0.00003,
    "code_flask:HTML": 0.00000015,
    "code_jinja:HTML": 0.00000007,
    "code_intercooler:HTML": 0.00012,
    "code_gcclean:CSS": 0.0081086,
    "code_general_js:JavaScript": 0.0036,
    "code_gcclean:JavaScript": 0.013933,
    "code_mdn:JavaScript": 0.00006,
    "code_htmx:JavaScript": 0.00004,
    "code_unpoly:JavaScript": 0.00004,
    "code_intercooler:JavaScript": 0.00002,
    "code_general_jsx:JavaScript (JSX)": 0.0000397,
    "code_general_ts:TypeScript": 0.0024,
    "code_gcclean:TypeScript": 0.0052974,
    "code_gcclean:TypeScript (TSX)": 0.0019,
    "code_general_sql:SQL": 0.0020,
    "code_cpp:C++": 0.0016,
    "code_ccheader:C%2FC++%20Header": 0.0016,
    "code_go:Go": 0.0016,
    "code_rust:Rust": 0.0015,
    "code_java:Java": 0.0012,
    "code_c:C": 0.0008,
    "code_kotlin:Kotlin": 0.0007,
    "code_csharp:C#": 0.0007,
    "code_swift:Swift": 0.0004,
    "code_shell:Shell": 0.0004,
    "code_dart:Dart": 0.0003,
    "code_php:PHP": 0.0005,
})


_PRETRAIN_HUMANIST_TARGETS_FRAC: dict[str, float] = _normalized_targets_frac({
    # Humanistic variant:
    # - the curated zh/en humanist discourse buckets are capped to their real
    #   approximate supply so the preset remains feasible under no-repeat sampling
    # - extra humanistic emphasis therefore comes mainly from books, not by forcing
    #   unrealistically large quotas onto small curated pools
    # - code keeps the same web-app capable core as the main preset
    "zh_books:UNKNOWN": 0.2674,
    "en_books:UNKNOWN": 0.1750,
    "zh_wiki:UNKNOWN": 0.1540,
    "en_wiki:UNKNOWN": 0.1510,
    "zh_discourse_humanist:UNKNOWN": 0.0112,
    "en_discourse_humanist:UNKNOWN": 0.0104,
    "en_discourse_mdn:UNKNOWN": 0.00024,
    "en_discourse_cpython:UNKNOWN": 0.00006,
    "en_discourse_django:UNKNOWN": 0.00004,
    "en_discourse_flask:UNKNOWN": 0.000003,
    "en_discourse_fastapi:UNKNOWN": 0.0000075,
    "en_discourse_jinja:UNKNOWN": 0.0000007,
    "en_discourse_sqlalchemy:UNKNOWN": 0.00002,
    "en_discourse_htmx:UNKNOWN": 0.000009,
    "math:UNKNOWN": 0.0220,
    "translation:UNKNOWN": 0.0050,
    "code_general_py:Python": 0.0340,
    "code_gcclean:Python": 0.10331,
    "code_cpython:Python": 0.00016,
    "code_django:Python": 0.00007,
    "code_flask:Python": 0.000005,
    "code_fastapi:Python": 0.00001,
    "code_jinja:Python": 0.000006,
    "code_sqlalchemy:Python": 0.00010,
    "code_gcclean:HTML": 0.0123463,
    "code_mdn:HTML": 0.00003,
    "code_flask:HTML": 0.00000015,
    "code_jinja:HTML": 0.00000007,
    "code_intercooler:HTML": 0.00012,
    "code_gcclean:CSS": 0.0081086,
    "code_general_js:JavaScript": 0.0036,
    "code_gcclean:JavaScript": 0.013933,
    "code_mdn:JavaScript": 0.00006,
    "code_htmx:JavaScript": 0.00004,
    "code_unpoly:JavaScript": 0.00004,
    "code_intercooler:JavaScript": 0.00002,
    "code_general_jsx:JavaScript (JSX)": 0.0000397,
    "code_general_ts:TypeScript": 0.0024,
    "code_gcclean:TypeScript": 0.0052974,
    "code_gcclean:TypeScript (TSX)": 0.0019,
    "code_general_sql:SQL": 0.0020,
    "code_cpp:C++": 0.0016,
    "code_ccheader:C%2FC++%20Header": 0.0016,
    "code_go:Go": 0.0016,
    "code_rust:Rust": 0.0015,
    "code_java:Java": 0.0012,
    "code_c:C": 0.0008,
    "code_kotlin:Kotlin": 0.0007,
    "code_csharp:C#": 0.0007,
    "code_swift:Swift": 0.0004,
    "code_shell:Shell": 0.0004,
    "code_dart:Dart": 0.0003,
    "code_php:PHP": 0.0005,
})


_PRETRAIN_FINAL_TARGETS_FRAC: dict[str, float] = _normalized_targets_frac({
    "zh_books:UNKNOWN": 0.1970,
    "en_books:UNKNOWN": 0.1740,
    "zh_wiki:UNKNOWN": 0.1490,
    "en_wiki:UNKNOWN": 0.1600,
    "zh_discourse:UNKNOWN": 0.0430,
    "en_discourse_general:UNKNOWN": 0.0230,
    "en_discourse_mdn:UNKNOWN": 0.00024,
    "en_discourse_cpython:UNKNOWN": 0.00005,
    "en_discourse_django:UNKNOWN": 0.00005,
    "en_discourse_flask:UNKNOWN": 0.000004,
    "en_discourse_fastapi:UNKNOWN": 0.000009,
    "en_discourse_jinja:UNKNOWN": 0.000003,
    "en_discourse_sqlalchemy:UNKNOWN": 0.000022,
    "en_discourse_htmx:UNKNOWN": 0.000018,
    "math:UNKNOWN": 0.0450,
    "translation:UNKNOWN": 0.0065,
    "code_general_py:Python": 0.0200,
    "code_gcclean:Python": 0.0990,
    "code_cpython:Python": 0.00014,
    "code_django:Python": 0.00008,
    "code_flask:Python": 0.000008,
    "code_fastapi:Python": 0.000012,
    "code_jinja:Python": 0.000012,
    "code_sqlalchemy:Python": 0.00012,
    "code_gcclean:HTML": 0.0152,
    "code_mdn:HTML": 0.00005,
    "code_flask:HTML": 0.0000002,
    "code_jinja:HTML": 0.0000012,
    "code_intercooler:HTML": 0.00008,
    "code_alpine:HTML": 0.00005,
    "code_hyperscript:HTML": 0.00003,
    "code_idiomorph:HTML": 0.00002,
    "code_pico:HTML": 0.00003,
    "code_missingcss:HTML": 0.00002,
    "code_watercss:HTML": 0.00002,
    "code_newcss:HTML": 0.000015,
    "code_sakura:HTML": 0.00003,
    "code_skeleton:HTML": 0.00002,
    "code_milligram:HTML": 0.00001,
    "code_chota:HTML": 0.000015,
    "code_spectre:HTML": 0.000025,
    "code_mvp:HTML": 0.00001,
    "code_picnic:HTML": 0.000008,
    "code_datastar:HTML": 0.00001,
    "code_gcclean:CSS": 0.0112,
    "code_hyperscript:CSS": 0.00001,
    "code_pico:CSS": 0.00002,
    "code_simplecss:CSS": 0.000015,
    "code_missingcss:CSS": 0.00001,
    "code_watercss:CSS": 0.000015,
    "code_newcss:CSS": 0.000015,
    "code_sakura:CSS": 0.00002,
    "code_skeleton:CSS": 0.000015,
    "code_milligram:CSS": 0.000008,
    "code_chota:CSS": 0.000008,
    "code_spectre:CSS": 0.000015,
    "code_mvp:CSS": 0.000008,
    "code_picnic:CSS": 0.000006,
    "code_intercooler:CSS": 0.000006,
    "code_general_js:JavaScript": 0.0026,
    "code_gcclean:JavaScript": 0.0182,
    "code_mdn:JavaScript": 0.00008,
    "code_htmx:JavaScript": 0.00006,
    "code_unpoly:JavaScript": 0.00005,
    "code_intercooler:JavaScript": 0.00003,
    "code_alpine:JavaScript": 0.00005,
    "code_hyperscript:JavaScript": 0.00002,
    "code_idiomorph:JavaScript": 0.00002,
    "code_missingcss:JavaScript": 0.00001,
    "code_sakura:JavaScript": 0.00001,
    "code_chota:JavaScript": 0.000008,
    "code_picnic:JavaScript": 0.000004,
    "code_general_jsx:JavaScript (JSX)": 0.00002,
    "code_general_ts:TypeScript": 0.0018,
    "code_gcclean:TypeScript": 0.0068,
    "code_gcclean:TypeScript (TSX)": 0.0022,
    "code_datastar:TypeScript": 0.000005,
    "code_general_sql:SQL": 0.0009,
})



PRESETS: dict[str, PretrainMixPreset] = {
    "pretrain": PretrainMixPreset(
        name="pretrain",
        total_tokens_eos=16_000_000_000,
        sources=_PRETRAIN_TRAIN_SOURCES,
        targets_frac=_PRETRAIN_TARGETS_FRAC,
    ),
    "pretrain_val": PretrainMixPreset(
        name="pretrain_val",
        total_tokens_eos=120_000_000,
        sources=_PRETRAIN_VAL_SOURCES,
        targets_frac=_PRETRAIN_TARGETS_FRAC,
    ),
    "pretrain_test": PretrainMixPreset(
        name="pretrain_test",
        total_tokens_eos=20_000_000,
        sources=_PRETRAIN_TEST_SOURCES,
        targets_frac=_PRETRAIN_TARGETS_FRAC,
    ),
    "pretrain_humanist": PretrainMixPreset(
        name="pretrain_humanist",
        total_tokens_eos=16_000_000_000,
        sources=_PRETRAIN_HUMANIST_TRAIN_SOURCES,
        targets_frac=_PRETRAIN_HUMANIST_TARGETS_FRAC,
    ),
    "pretrain_humanist_val": PretrainMixPreset(
        name="pretrain_humanist_val",
        total_tokens_eos=120_000_000,
        sources=_PRETRAIN_HUMANIST_VAL_SOURCES,
        targets_frac=_PRETRAIN_HUMANIST_TARGETS_FRAC,
    ),
    "pretrain_humanist_test": PretrainMixPreset(
        name="pretrain_humanist_test",
        total_tokens_eos=20_000_000,
        sources=_PRETRAIN_HUMANIST_TEST_SOURCES,
        targets_frac=_PRETRAIN_HUMANIST_TARGETS_FRAC,
    ),
    "pretrain_final": PretrainMixPreset(
        name="pretrain_final",
        total_tokens_eos=20_257_182_714,
        sources=_PRETRAIN_TRAIN_SOURCES,
        targets_frac=_PRETRAIN_FINAL_TARGETS_FRAC,
    ),
}


def get_preset(name: str) -> PretrainMixPreset:
    key = str(name or "").strip()
    if not key:
        raise KeyError("Preset name must be non-empty")
    if key not in PRESETS:
        raise KeyError(
            f"Unknown preset: {key!r}. Available: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[key]


__all__ = [
    "BucketMode",
    "PRESETS",
    "PretrainMixPreset",
    "SourcePreset",
    "_EN_HUMANIST_CURATED_BASE",
    "_PRETRAIN_FINAL_TARGETS_FRAC",
    "_PRETRAIN_HUMANIST_TARGETS_FRAC",
    "_PRETRAIN_TARGETS_FRAC",
    "_ZH_HUMANIST_CURATED_BASE",
    "_normalized_targets_frac",
    "get_preset",
]
