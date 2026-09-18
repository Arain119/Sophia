from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import tempfile
import unicodedata
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.corpus_quality import dedup_normalize
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


REPORT_SCHEMA = "sophia_chinese_quality_proxy_report_v2"
MODEL_SCHEMA = "sophia_chinese_quality_proxy_model_v2"
LABEL_THRESHOLD = 3
DEFAULT_RANDOM_SEED = 20260719
DEFAULT_C_VALUES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
_URL_RE = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
_PUNCTUATION = frozenset("，。！？；：,.!?;:")


def _feature_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _numeric_features(texts: list[str]) -> csr_matrix:
    rows: list[list[float]] = []
    for text in texts:
        length = max(len(text), 1)
        cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
        digits = sum(char.isdigit() for char in text)
        latin = sum(char.isascii() and char.isalpha() for char in text)
        punctuation = sum(char in _PUNCTUATION for char in text)
        urls = len(_URL_RE.findall(text))
        rows.append(
            [
                min(math.log2(length + 1) / 16.0, 1.0),
                cjk / float(length),
                digits / float(length),
                latin / float(length),
                min(urls / 5.0, 1.0),
                punctuation / float(length),
            ]
        )
    return csr_matrix(np.asarray(rows, dtype=np.float32))


def _feature_matrix(
    vectorizer: TfidfVectorizer,
    texts: list[str],
    *,
    fit: bool,
) -> csr_matrix:
    lexical = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    return hstack((lexical, _numeric_features(texts)), format="csr", dtype=np.float32)


def _load_deduplicated_labels(path: Path) -> tuple[list[str], list[int], dict[str, int]]:
    rows: dict[bytes, tuple[str, int]] = {}
    label_counts: dict[int, int] = {}
    raw_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on label row {line_number}") from exc
            text = row.get("text") if isinstance(row, dict) else None
            label = row.get("label") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"label row {line_number} has no non-empty text")
            if not isinstance(label, int) or not 0 <= label <= 5:
                raise ValueError(f"label row {line_number} has invalid label: {label!r}")
            normalized = dedup_normalize(text)
            if not normalized:
                raise ValueError(f"label row {line_number} normalizes to empty text")
            digest = hashlib.sha256(normalized.encode("utf-8")).digest()
            previous = rows.get(digest)
            if previous is not None and previous[1] != label:
                raise ValueError(
                    f"conflicting labels for normalized duplicate on row {line_number}: "
                    f"{previous[1]} != {label}"
                )
            rows.setdefault(digest, (_feature_text(text), label))
            label_counts[label] = label_counts.get(label, 0) + 1
    if not rows:
        raise ValueError("quality label file is empty")
    texts = [value[0] for value in rows.values()]
    targets = [int(value[1] >= LABEL_THRESHOLD) for value in rows.values()]
    if len(set(targets)) != 2:
        raise ValueError("quality labels must contain both positive and negative examples")
    stats = {
        "raw_rows": raw_rows,
        "unique_normalized_rows": len(rows),
        "exact_duplicate_rows_removed": raw_rows - len(rows),
        **{f"label_{label}_rows": label_counts.get(label, 0) for label in range(6)},
    }
    return texts, targets, stats


def _split_rows(
    texts: list[str], targets: list[int], *, random_seed: int
) -> dict[str, tuple[list[str], list[int]]]:
    train_texts, holdout_texts, train_y, holdout_y = train_test_split(
        texts,
        targets,
        test_size=0.30,
        random_state=int(random_seed),
        stratify=targets,
    )
    validation_texts, test_texts, validation_y, test_y = train_test_split(
        holdout_texts,
        holdout_y,
        test_size=0.50,
        random_state=int(random_seed) + 1,
        stratify=holdout_y,
    )
    return {
        "train": (list(train_texts), list(train_y)),
        "validation": (list(validation_texts), list(validation_y)),
        "test": (list(test_texts), list(test_y)),
    }


def _split_sha256(texts: list[str], targets: list[int]) -> str:
    digest = hashlib.sha256()
    for text, target in sorted(zip(texts, targets, strict=True)):
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        digest.update(bytes((int(target),)))
    return digest.hexdigest()


def _metrics(targets: list[int], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (np.asarray(scores) >= float(threshold)).astype(np.int8)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    return {
        "threshold": round(float(threshold), 6),
        "rows": len(targets),
        "accepted_rows": int(predictions.sum()),
        "accepted_fraction": round(float(predictions.mean()), 6),
        "accuracy": round(float(accuracy_score(targets, predictions)), 6),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(targets, predictions)), 6
        ),
        "macro_f1": round(
            float(f1_score(targets, predictions, average="macro", zero_division=0)),
            6,
        ),
        "positive_precision": round(
            float(precision_score(targets, predictions, zero_division=0)), 6
        ),
        "positive_recall": round(
            float(recall_score(targets, predictions, zero_division=0)), 6
        ),
        "positive_f1": round(
            float(f1_score(targets, predictions, zero_division=0)), 6
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _select_threshold(
    targets: list[int],
    scores: np.ndarray,
    *,
    minimum_precision: float,
) -> tuple[float, dict[str, Any]]:
    candidates = np.linspace(0.05, 0.95, 181)
    rows = [_metrics(targets, scores, float(value)) for value in candidates]
    eligible = [
        row for row in rows if row["positive_precision"] >= float(minimum_precision)
    ]
    pool = eligible or rows
    best = max(
        pool,
        key=lambda row: (
            row["macro_f1"],
            row["positive_precision"],
            row["positive_recall"],
            row["threshold"],
        ),
    )
    return float(best["threshold"]), {
        **best,
        "minimum_precision_constraint": float(minimum_precision),
        "precision_constraint_satisfied": bool(eligible),
    }


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_quality_proxy(
    model_path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    actual_sha256 = sha256_file(path)
    if not expected_sha256 or actual_sha256 != str(expected_sha256):
        raise ValueError(
            "quality proxy model SHA-256 mismatch: "
            f"expected={expected_sha256!r} actual={actual_sha256}"
        )
    with path.open("rb") as handle:
        artifact = pickle.load(handle)  # noqa: S301 - hash-pinned, locally trained artifact.
    if not isinstance(artifact, dict) or artifact.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported quality proxy model artifact")
    if not isinstance(artifact.get("vectorizer"), TfidfVectorizer):
        raise ValueError("quality proxy artifact has no TF-IDF vectorizer")
    if not isinstance(artifact.get("classifier"), LogisticRegression):
        raise ValueError("quality proxy artifact has no logistic classifier")
    return artifact


def score_quality_texts(artifact: dict[str, Any], texts: list[str]) -> np.ndarray:
    normalized = [_feature_text(text) for text in texts]
    features = _feature_matrix(artifact["vectorizer"], normalized, fit=False)
    return np.asarray(artifact["classifier"].predict_proba(features)[:, 1])


def train_quality_proxy(
    *,
    labels_path: str,
    model_path: str,
    report_path: str,
    repo_id: str,
    revision: str,
    repository_license: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    minimum_validation_precision: float = 0.70,
    minimum_test_macro_f1: float = 0.60,
    minimum_test_positive_precision: float = 0.65,
    minimum_test_positive_recall: float = 0.45,
) -> dict[str, Any]:
    labels = Path(labels_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    if not labels.is_file():
        raise FileNotFoundError(f"quality label file does not exist: {labels}")
    if model.exists() or report_file.exists():
        raise FileExistsError("quality proxy outputs already exist; overwrite is not supported")
    if not repo_id or not revision or not repository_license:
        raise ValueError("repo_id, revision, and repository_license are required")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C values must contain positive values")

    texts, targets, label_stats = _load_deduplicated_labels(labels)
    splits = _split_rows(texts, targets, random_seed=int(random_seed))
    train_texts, train_targets = splits["train"]
    validation_texts, validation_targets = splits["validation"]
    test_texts, test_targets = splits["test"]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        min_df=2,
        max_df=0.995,
        max_features=300_000,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )
    train_features = _feature_matrix(vectorizer, train_texts, fit=True)
    validation_features = _feature_matrix(vectorizer, validation_texts, fit=False)
    candidates: list[dict[str, Any]] = []
    classifiers: list[LogisticRegression] = []
    for c_value in c_values:
        classifier = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=300,
            random_state=int(random_seed),
        )
        classifier.fit(train_features, train_targets)
        validation_scores = classifier.predict_proba(validation_features)[:, 1]
        threshold, validation_metrics = _select_threshold(
            validation_targets,
            validation_scores,
            minimum_precision=float(minimum_validation_precision),
        )
        candidates.append(
            {
                "c": float(c_value),
                "threshold": float(threshold),
                "validation": validation_metrics,
            }
        )
        classifiers.append(classifier)
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["validation"]["precision_constraint_satisfied"],
            candidates[index]["validation"]["macro_f1"],
            candidates[index]["validation"]["positive_precision"],
            candidates[index]["validation"]["positive_recall"],
            -candidates[index]["c"],
        ),
    )
    selected = candidates[selected_index]
    classifier = classifiers[selected_index]
    threshold = float(selected["threshold"])
    test_features = _feature_matrix(vectorizer, test_texts, fit=False)
    test_scores = classifier.predict_proba(test_features)[:, 1]
    test_metrics = _metrics(test_targets, test_scores, threshold)
    gates = {
        "minimum_test_macro_f1": float(minimum_test_macro_f1),
        "minimum_test_positive_precision": float(minimum_test_positive_precision),
        "minimum_test_positive_recall": float(minimum_test_positive_recall),
    }
    gate_results = {
        "test_macro_f1": test_metrics["macro_f1"] >= float(minimum_test_macro_f1),
        "test_positive_precision": test_metrics["positive_precision"]
        >= float(minimum_test_positive_precision),
        "test_positive_recall": test_metrics["positive_recall"]
        >= float(minimum_test_positive_recall),
        "validation_precision_constraint": bool(
            selected["validation"]["precision_constraint_satisfied"]
        ),
    }
    split_report = {
        name: {
            "rows": len(values[1]),
            "positive_rows": int(sum(values[1])),
            "positive_fraction": round(sum(values[1]) / float(len(values[1])), 6),
            "sha256": _split_sha256(*values),
        }
        for name, values in splits.items()
    }
    labels_sha256 = sha256_file(labels)
    artifact = {
        "schema": MODEL_SCHEMA,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "threshold": threshold,
        "positive_label_definition": f"label >= {LABEL_THRESHOLD}",
        "labels_sha256": labels_sha256,
        "random_seed": int(random_seed),
        "selected_c": float(selected["c"]),
        "sklearn_version": sklearn.__version__,
    }
    _atomic_pickle(model, artifact)
    model_sha256 = sha256_file(model)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "pass" if all(gate_results.values()) else "fail",
        "purpose": (
            "Low-cost project proxy for Chinese educational-quality filtering; "
            "not a reproduction of the upstream CCI3-HQ classifier."
        ),
        "source": {
            "repo_id": str(repo_id),
            "revision": str(revision),
            "repository_license": str(repository_license),
            "labels_path": str(labels),
            "labels_sha256": labels_sha256,
        },
        "label_definition": {
            "positive": f"label >= {LABEL_THRESHOLD}",
            "negative": f"label < {LABEL_THRESHOLD}",
        },
        "label_stats": label_stats,
        "splits": split_report,
        "model": {
            "path": str(model),
            "sha256": model_sha256,
            "schema": MODEL_SCHEMA,
            "sklearn_version": sklearn.__version__,
            "features": {
                "type": "tfidf_char_ngrams_plus_numeric_structure",
                "ngram_range": [1, 5],
                "max_features": 300_000,
                "min_document_frequency": 2,
                "numeric_features": [
                    "log2_character_length",
                    "cjk_fraction",
                    "digit_fraction",
                    "latin_fraction",
                    "url_count_capped",
                    "punctuation_fraction",
                ],
                "normalization": "NFKC_then_collapse_whitespace",
            },
            "classifier": "LogisticRegression_liblinear_balanced",
            "selected_c": float(selected["c"]),
            "threshold": threshold,
        },
        "candidate_validation_results": candidates,
        "selected_validation_metrics": selected["validation"],
        "held_out_test_metrics": test_metrics,
        "gates": gates,
        "gate_results": gate_results,
        "limitations": [
            "This proxy is trained on 14K-scale annotations and is not the upstream 0.5B classifier.",
            "A passing score does not establish copyright, license, provenance, or absence of contamination.",
            "Candidate sources still require manual sample review and downstream model comparison.",
        ],
    }
    write_json_atomic(
        report_file,
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a pinned lightweight Chinese educational-quality proxy."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository-license", required=True)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--c", action="append", type=float)
    parser.add_argument("--minimum-validation-precision", type=float, default=0.70)
    parser.add_argument("--minimum-test-macro-f1", type=float, default=0.60)
    parser.add_argument("--minimum-test-positive-precision", type=float, default=0.65)
    parser.add_argument("--minimum-test-positive-recall", type=float, default=0.45)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_quality_proxy(
        labels_path=str(args.labels),
        model_path=str(args.model),
        report_path=str(args.report),
        repo_id=str(args.repo_id),
        revision=str(args.revision),
        repository_license=str(args.repository_license),
        random_seed=int(args.random_seed),
        c_values=tuple(args.c or DEFAULT_C_VALUES),
        minimum_validation_precision=float(args.minimum_validation_precision),
        minimum_test_macro_f1=float(args.minimum_test_macro_f1),
        minimum_test_positive_precision=float(args.minimum_test_positive_precision),
        minimum_test_positive_recall=float(args.minimum_test_positive_recall),
    )
    metrics = report["held_out_test_metrics"]
    print(
        f"[DONE] status={report['status']} macro_f1={metrics['macro_f1']:.4f} "
        f"positive_precision={metrics['positive_precision']:.4f} "
        f"positive_recall={metrics['positive_recall']:.4f}",
        flush=True,
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
