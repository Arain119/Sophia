from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml.tooling.scripts.data.build_finemath_quality_exclusion import (
    build_finemath_quality_exclusion,
    finemath_template_reason,
)


def test_finemath_template_rules_preserve_real_forum_math() -> None:
    forum = (
        "# Homework Help: inverse hyperbolic secant\n"
        "I derived sech^-1(x) = log((1 + sqrt(1-x^2))/x). "
        "Which branch makes this a function, and why? " * 8
    )
    assert finemath_template_reason(forum) == ""
    assert (
        finemath_template_reason(
            "Get Professional Assignment Help Cheaply. Our team of professional "
            "academic writers can handle your assignment on time. " * 5
        )
        == "finemath_academic_mill"
    )
    assert (
        finemath_template_reason(
            "# What is the date 7 days from August 30?\n"
            "Adding 7 days from Friday August 30, 2024 is Friday September 06, "
            "2024. Calculating 7 days from August 30 by hand is straightforward."
        )
        == "finemath_date_calculator"
    )
    assert (
        finemath_template_reason(
            "116832 in braille: code. QR code Bar code, type 39. "
            "Images of the number. ## Mathematics of no. 116832. " * 3
        )
        == "finemath_number_properties"
    )
    assert (
        finemath_template_reason(
            "This preview shows page 1 of 10. Subscribe to view the full document."
        )
        == "finemath_subscription_preview"
    )


def _file_info(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_build_finemath_quality_exclusion_writes_sorted_manifest(
    tmp_path: Path,
) -> None:
    low_score = " ".join(
        f"A coherent mathematical explanation develops distinct step {index}."
        for index in range(12)
    )
    paywall = " ".join(
        [
            "A calculus derivation begins with a continuous differentiable function.",
            "Subscribe to view the full document.",
            *[
                f"The hidden preview labels a different intermediate result {index}."
                for index in range(10)
            ],
        ]
    )
    good = " ".join(
        f"A complete induction proof explains a distinct logical step {index}."
        for index in range(12)
    )
    paths: dict[str, Path] = {}
    for source, rows in {
        "finemath_3plus_v1": [
            {"text": low_score, "score": 2.5, "token_count": 120, "url": "low"},
            {"text": good, "score": 2.9, "token_count": 140, "url": "good"},
        ],
        "finemath_4plus_v1": [
            {
                "text": paywall,
                "score": 4.0,
                "token_count": 130,
                "url": "paywall",
                "language": "en",
            },
            {
                "text": good + " distinct",
                "score": 4.0,
                "token_count": 150,
                "url": "good-4",
                "language": "en",
            },
        ],
    }.items():
        path = tmp_path / f"{source}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        paths[source] = path
    acquisition = {
        "sources": [
            {
                "name": "finemath_3plus_v1",
                "files": [_file_info(paths["finemath_3plus_v1"])],
                "text_column": "text",
                "language": "en",
                "min_chars": 100,
            },
            {
                "name": "finemath_4plus_v1",
                "files": [_file_info(paths["finemath_4plus_v1"])],
                "text_column": "text",
                "language": "en",
                "language_column": "language",
                "allowed_languages": ["en"],
                "min_chars": 100,
            },
        ]
    }
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    manifest_path = tmp_path / "exclusions.tsv"
    report_path = tmp_path / "report.json"

    report = build_finemath_quality_exclusion(
        acquisition_path=acquisition_path,
        manifest_path=manifest_path,
        report_path=report_path,
        workers=1,
        batch_size=2,
    )

    lines = manifest_path.read_text(encoding="ascii").splitlines()
    assert lines == sorted(lines)
    assert report["manifest"]["entries"] == 2
    assert report["manifest"]["reason_counts"] == {
        "finemath_score_below_threshold": 1,
        "finemath_subscription_preview": 1,
    }
    assert report["sources"]["finemath_3plus_v1"]["counters"][
        "documents_retained"
    ] == 1
    assert report["sources"]["finemath_4plus_v1"]["counters"][
        "documents_retained"
    ] == 1
    assert report_path.is_file()
