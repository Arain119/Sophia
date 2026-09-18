from __future__ import annotations

import json
from pathlib import Path

from ml.tooling.scripts.data import materialize_pretrain_mix_policy as mod


def test_materializes_v7_draft_status() -> None:
    draft = json.loads(
        Path("configs/data/pretrain_mix_policy.json").read_text(
            encoding="utf-8"
        )
    )
    draft["requested_unique_train_tokens"] = 10_000
    sources = {
        member
        for group in draft["target_source_groups"].values()
        for member in group["members"]
    }
    supply = {source: 20_000 for source in sources}
    supply.update(
        {
            "finemath_4plus_v1": 300,
            "common_pile_stackv2_edu_filtered_v2": 200,
            "the_stack_smol_xl_permissive_v1": 200,
            "github_code_clean_permissive_v1": 360,
            "skypile_diverse_chinese_web_v1": 600,
            "opencsg_fineweb_edu_zh_3_4_humanities_v1": 200,
            "zhihu_kol_modern_chinese_qa_v1": 200,
        }
    )
    for source in draft["target_source_groups"]["wikimedia_chinese_combined"][
        "members"
    ]:
        if source != "wikimedia_wikisource_zh_20231201_v2":
            supply[source] = 100
    profile = {
        "schema": "sophia_pretrain_corpus_token_profile_v1",
        "status": "complete",
        "tokenizer_bundle_sha1": "a" * 40,
        "tokens_by": {
            "source_split": {
                f"{source}|train": tokens for source, tokens in supply.items()
            }
        },
    }

    policy = mod.materialize_policy(draft=draft, profile=profile)

    assert policy["policy_version"] == "v7_materialized_from_complete_profile"
    assert policy["status"] == "admitted_pending_supply_resolution"
    adjustments = policy["materialization"]["shortfall_reallocations"]
    assert any(
        row["source_group"] == "skypile_diverse_chinese_web_v1"
        and row["reallocation_target"] == "opencsg_fineweb_edu_zh_4_5_full_v6"
        for row in adjustments
    )
    assert any(
        row["source_group"] == "permissive_code_combined"
        and row["reallocation_target"] == "fineweb_edu_english_dedup"
        for row in adjustments
    )
    assert any(
        row["source_group"] == "zhihu_kol_modern_chinese_qa_v1"
        and row["reallocation_target"] == "wikimedia_chinese_combined"
        for row in adjustments
    )
