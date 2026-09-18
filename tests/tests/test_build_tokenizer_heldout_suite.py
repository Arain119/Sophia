from ml.tooling.scripts.data import build_tokenizer_heldout_suite as mod


def test_integer_quotas_are_exact() -> None:
    quotas = mod._integer_quotas(10, {"large": 7, "small": 3})

    assert quotas == {"large": 7, "small": 3}
    assert sum(quotas.values()) == 10


def test_domain_sources_partitions_all_sources() -> None:
    plan = {
        "source_train_quotas": {"zh": 1, "en": 1, "code": 1, "math": 1}
    }
    policy = {
        "constraints": {
            "source_groups": {
                "chinese": ["zh"],
                "code": ["code"],
                "math_stem": ["math"],
            }
        }
    }

    assert mod._domain_sources(plan=plan, mix_policy=policy) == {
        "native_chinese": ["zh"],
        "english": ["en"],
        "code": ["code"],
        "math": ["math"],
    }
