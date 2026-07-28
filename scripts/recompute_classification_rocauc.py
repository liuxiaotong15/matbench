#!/usr/bin/env python
"""Recompute recoverable historical classification ROC-AUC scores.

The historical result files contain both per-sample predictions and scores
computed when the submission was recorded. This migration recovers true labels
from committed validation artifacts, rescoring only submissions that retained
continuous positive-class probabilities. Boolean-only submissions keep their
label-based (degenerate) ROC-AUC values.
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import math
import os
import stat
import tempfile
from collections import defaultdict
from pathlib import Path

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_THRESH, VERSION
from matbench.data_ops import score_array
from matbench.metadata import mbv01_validation
from matbench.util import hash_dictionary

SCRIPT_VERSION = "1.0.0"
SCORER_VERSION = "probability-rocauc-v1"
ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
VALIDATION_PATH = ROOT / "matbench" / "matbench_v0.1_validation.json"
TARGETS_PATH = (
    ROOT
    / "artifacts"
    / "validation"
    / "matbench_v0.1_classification_targets.json.gz"
)
MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "validation"
    / "classification_rocauc_recompute_manifest.json"
)

TASKS = (
    "matbench_glass",
    "matbench_expt_is_metal",
    "matbench_mp_is_metal",
)
LABEL_METRICS = ("accuracy", "balanced_accuracy", "f1")
CONTINUOUS_KIND = "continuous_positive_class_probability"
HARD_LABEL_KIND = "hard_boolean_label"
EXPECTED_RECORD_COUNTS = {
    CONTINUOUS_KIND: 11,
    HARD_LABEL_KIND: 16,
}
EXPECTED_AFFECTED_FILES = {
    "benchmarks/matbench_v0.1_DimeNetPP_kgcnn_v2.1.0/results.json.gz",
    "benchmarks/matbench_v0.1_MegNet_kgcnn_v2.1.0/results.json.gz",
    "benchmarks/matbench_v0.1_SchNet_kgcnn_v2.1.0/results.json.gz",
    "benchmarks/matbench_v0.1_coGN/results.json.gz",
    "benchmarks/matbench_v0.1_coNGN/results.json.gz",
    "benchmarks/matbench_v0.1_modnet_v0.1.10/results.json.gz",
    "benchmarks/matbench_v0.1_modnet_v0.1.12/results.json.gz",
}
EXPECTED_MEANS = {
    ("matbench_glass", "MODNet (v0.1.12)"): 0.989876,
    ("matbench_glass", "MODNet (v0.1.10)"): 0.932948,
    ("matbench_expt_is_metal", "MODNet (v0.1.10)"): 0.972546,
    ("matbench_expt_is_metal", "MODNet (v0.1.12)"): 0.972546,
}
EXPECTED_MP_TOP_FOUR = (
    ("MODNet (v0.1.12)", 0.964159),
    ("coNGN", 0.954047),
    ("MegNet (kgcnn v2.1.0)", 0.953543),
    ("CGCNN v2019", 0.951962),
)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def deterministic_gzip_json(data):
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        handle.write(encoded)
    return output.getvalue()


def deterministic_manifest(data):
    return (
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_if_changed(path, data):
    if path.exists() and path.read_bytes() == data:
        return False

    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(data)
    os.chmod(temp_path, mode)
    os.replace(temp_path, path)
    return True


def prediction_kind(folds):
    values = [value for fold in folds.values() for value in fold["data"].values()]
    if values and all(isinstance(value, bool) for value in values):
        return HARD_LABEL_KIND
    if values and all(
        isinstance(value, float) and not isinstance(value, bool) for value in values
    ):
        return CONTINUOUS_KIND
    types = sorted({type(value).__name__ for value in values})
    raise TypeError(
        "Classification predictions must be uniformly boolean labels or "
        f"floating-point scores; found {types}."
    )


def load_existing_manifest():
    if not MANIFEST_PATH.exists():
        return None
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported manifest schema: {manifest.get('schema_version')}"
        )
    return manifest


def baseline_maps(existing_manifest):
    if existing_manifest is None:
        return {}, {}

    old_scores = {}
    for record in existing_manifest["records"]:
        old_scores[record["record_id"]] = {
            fold["fold"]: fold["old_rocauc"] for fold in record["folds"]
        }
    old_hashes = {
        entry["results_file"]: entry["sha256_before"]
        for entry in existing_manifest["files"]
    }
    return old_scores, old_hashes


def collect_records():
    payloads = {}
    records = []
    for path in sorted(BENCHMARKS_DIR.glob("*/results.json.gz")):
        payload = load_gzip_json(path)
        payloads[path] = payload
        info_path = path.parent / "info.json"
        with info_path.open(encoding="utf-8") as handle:
            algorithm = json.load(handle)["algorithm"]

        for task_name in TASKS:
            task = payload["tasks"].get(task_name)
            if task is None:
                continue
            rel_path = relative(path)
            records.append(
                {
                    "algorithm": algorithm,
                    "kind": prediction_kind(task["results"]),
                    "path": path,
                    "record_id": f"{rel_path}::{task_name}",
                    "results_file": rel_path,
                    "task": task_name,
                }
            )

    counts = defaultdict(int)
    for record in records:
        counts[record["kind"]] += 1
    if len(records) != 27 or dict(counts) != EXPECTED_RECORD_COUNTS:
        raise AssertionError(
            "Expected 27 classification records split into 11 continuous "
            f"and 16 boolean records; found {len(records)} and {dict(counts)}."
        )

    affected_files = {
        record["results_file"]
        for record in records
        if record["kind"] == CONTINUOUS_KIND
    }
    if affected_files != EXPECTED_AFFECTED_FILES:
        raise AssertionError(
            "Continuous prediction files differ from the audited seven: "
            f"{sorted(affected_files)}"
        )
    return payloads, records


def assert_same_score(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{context}: {actual} != {expected}")


def fold_sort_key(fold_name):
    return int(fold_name.removeprefix("fold_"))


def audit_and_recompute(payloads, records, targets, old_scores):
    for record in records:
        task_name = record["task"]
        folds = payloads[record["path"]]["tasks"][task_name]["results"]
        target_map = targets["tasks"][task_name]["targets_by_id"]
        fold_rows = []

        for fold_name in sorted(folds, key=fold_sort_key):
            fold = folds[fold_name]
            prediction_ids = set(fold["data"])
            validation_ids = mbv01_validation.splits[task_name][fold_name].test
            if prediction_ids != set(validation_ids):
                missing = sorted(set(validation_ids) - prediction_ids)
                extra = sorted(prediction_ids - set(validation_ids))
                raise AssertionError(
                    f"{record['record_id']} {fold_name}: prediction IDs do "
                    f"not match validation IDs; missing={missing[:3]}, "
                    f"extra={extra[:3]}."
                )

            true_labels = [target_map[index] for index in validation_ids]
            predictions = [fold["data"][index] for index in validation_ids]
            recomputed = score_array(
                true_labels, predictions, task_type="classification"
            )
            for metric in LABEL_METRICS:
                assert_same_score(
                    float(fold["scores"][metric]),
                    float(recomputed[metric]),
                    f"{record['record_id']} {fold_name} {metric}",
                )

            current_rocauc = float(fold["scores"]["rocauc"])
            new_rocauc = float(recomputed["rocauc"])
            if record["kind"] == HARD_LABEL_KIND:
                assert_same_score(
                    current_rocauc,
                    new_rocauc,
                    f"{record['record_id']} {fold_name} hard-label rocauc",
                )
            else:
                fold["scores"]["rocauc"] = new_rocauc

            baseline = old_scores.get(record["record_id"], {})
            old_rocauc = float(baseline.get(fold_name, current_rocauc))
            fold_rows.append(
                {
                    "fold": fold_name,
                    "new_rocauc": new_rocauc,
                    "old_rocauc": old_rocauc,
                }
            )

        record["folds"] = fold_rows
        record["new_mean_rocauc"] = sum(
            row["new_rocauc"] for row in fold_rows
        ) / len(fold_rows)
        record["old_mean_rocauc"] = sum(
            row["old_rocauc"] for row in fold_rows
        ) / len(fold_rows)


def ranking_details(records, score_key):
    details = {}
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)

    for task_records in by_task.values():
        ordered = sorted(
            task_records,
            key=lambda item: (
                -item[score_key],
                item["algorithm"].casefold(),
                item["results_file"],
            ),
        )
        for record in ordered:
            score = record[score_key]
            better = [
                item
                for item in ordered
                if item[score_key] > score
                and not math.isclose(
                    item[score_key], score, rel_tol=0.0, abs_tol=1e-15
                )
            ]
            tied = sorted(
                item["algorithm"]
                for item in ordered
                if math.isclose(item[score_key], score, rel_tol=0.0, abs_tol=1e-15)
            )
            details[record["record_id"]] = {
                "rank": len(better) + 1,
                "tie": len(tied) > 1,
                "tie_group": tied,
            }
    return details


def validate_expected_results(records):
    indexed = {(record["task"], record["algorithm"]): record for record in records}
    checks = []
    for key, expected in EXPECTED_MEANS.items():
        actual = indexed[key]["new_mean_rocauc"]
        if round(actual, 6) != expected:
            raise AssertionError(
                f"Independent mean check failed for {key}: "
                f"{actual:.6f} != {expected:.6f}"
            )
        checks.append(
            {
                "algorithm": key[1],
                "expected_rounded_mean": expected,
                "observed_mean": actual,
                "task": key[0],
            }
        )

    darwin = indexed[("matbench_expt_is_metal", "Darwin")]
    for algorithm in ("MODNet (v0.1.10)", "MODNet (v0.1.12)"):
        modnet = indexed[("matbench_expt_is_metal", algorithm)]
        if modnet["new_mean_rocauc"] <= darwin["new_mean_rocauc"]:
            raise AssertionError(f"{algorithm} does not rank above Darwin.")

    mp_records = sorted(
        (record for record in records if record["task"] == "matbench_mp_is_metal"),
        key=lambda item: (
            -item["new_mean_rocauc"],
            item["algorithm"].casefold(),
            item["results_file"],
        ),
    )
    observed_top_four = tuple(
        (record["algorithm"], round(record["new_mean_rocauc"], 6))
        for record in mp_records[:4]
    )
    if observed_top_four != EXPECTED_MP_TOP_FOUR:
        raise AssertionError(
            f"Unexpected matbench_mp_is_metal top four: {observed_top_four}"
        )
    return checks


def update_hashes_and_serialize(payloads, records):
    affected_paths = {
        record["path"] for record in records if record["kind"] == CONTINUOUS_KIND
    }
    serialized = {}
    for path in sorted(affected_paths):
        payload = payloads[path]
        unhashed = {key: value for key, value in payload.items() if key != "hash"}
        payload["hash"] = hash_dictionary(unhashed)
        serialized[path] = deterministic_gzip_json(payload)
    return serialized


def manifest_records(records):
    old_ranks = ranking_details(records, "old_mean_rocauc")
    new_ranks = ranking_details(records, "new_mean_rocauc")
    output = []
    for record in sorted(
        records,
        key=lambda item: (
            item["task"],
            item["algorithm"].casefold(),
            item["results_file"],
        ),
    ):
        output.append(
            {
                "algorithm": record["algorithm"],
                "folds": record["folds"],
                "new_mean_rocauc": record["new_mean_rocauc"],
                "new_rank": new_ranks[record["record_id"]],
                "old_mean_rocauc": record["old_mean_rocauc"],
                "old_rank": old_ranks[record["record_id"]],
                "prediction_kind": record["kind"],
                "record_id": record["record_id"],
                "results_file": record["results_file"],
                "task": record["task"],
            }
        )
    return output


def build_manifest(
    records,
    serialized,
    old_hashes,
    targets,
    independent_checks,
):
    records_by_file = defaultdict(list)
    for record in records:
        records_by_file[record["results_file"]].append(record)

    files = []
    for results_file in sorted(records_by_file):
        path = ROOT / results_file
        current_hash = sha256_file(path)
        new_data = serialized.get(path)
        after_hash = sha256_bytes(new_data) if new_data is not None else current_hash
        files.append(
            {
                "changed": new_data is not None,
                "prediction_records": sorted(
                    record["record_id"] for record in records_by_file[results_file]
                ),
                "results_file": results_file,
                "sha256_after": after_hash,
                "sha256_before": old_hashes.get(results_file, current_hash),
            }
        )

    return {
        "audit": {
            "continuous_probability_records_recomputed": 11,
            "hard_label_records_unchanged": 16,
            "submission_task_records": 27,
            "tasks": list(TASKS),
        },
        "benchmark": "matbench_v0.1",
        "canonical_artifacts": {
            "classification_targets": {
                "path": relative(TARGETS_PATH),
                "sha256": sha256_file(TARGETS_PATH),
                "sources": {
                    task: {
                        "n_samples": targets["tasks"][task]["n_samples"],
                        "source_artifact_sha256": targets["tasks"][task][
                            "source_artifact_sha256"
                        ],
                        "target": targets["tasks"][task]["target"],
                    }
                    for task in TASKS
                },
            },
            "validation_splits": {
                "path": relative(VALIDATION_PATH),
                "sha256": sha256_file(VALIDATION_PATH),
            },
        },
        "files": files,
        "independent_checks": {
            "expected_means": independent_checks,
            "matbench_mp_is_metal_top_four": [
                {"algorithm": algorithm, "rounded_mean": score}
                for algorithm, score in EXPECTED_MP_TOP_FOUR
            ],
        },
        "records": manifest_records(records),
        "schema_version": 1,
        "scorer": {
            "classification_threshold": CLF_THRESH,
            "implementation": "matbench.data_ops.score_array",
            "matbench_version": VERSION,
            "rocauc_basis": "positive-class scores before thresholding",
            "version": SCORER_VERSION,
        },
        "script": {
            "path": relative(Path(__file__).resolve()),
            "version": SCRIPT_VERSION,
        },
    }


def verify_loadable(serialized):
    logging.getLogger("matbench").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as temp_dir:
        for path, data in serialized.items():
            candidate = Path(temp_dir) / path.parent.name / path.name
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(data)
            MatbenchBenchmark.from_file(str(candidate))


def report_summary(records, action):
    continuous = [record for record in records if record["kind"] == CONTINUOUS_KIND]
    print(
        f"{action}: audited 27 records (11 continuous, 16 hard-label); "
        f"{len({record['results_file'] for record in continuous})} "
        "result files contain recoverable scores."
    )
    for record in sorted(
        continuous, key=lambda item: (item["task"], item["algorithm"])
    ):
        print(
            f"  {record['task']} / {record['algorithm']}: "
            f"{record['old_mean_rocauc']:.6f} -> "
            f"{record['new_mean_rocauc']:.6f}"
        )


def run(mode):
    existing_manifest = load_existing_manifest()
    old_scores, old_hashes = baseline_maps(existing_manifest)
    payloads, records = collect_records()
    targets = load_gzip_json(TARGETS_PATH)
    audit_and_recompute(payloads, records, targets, old_scores)
    independent_checks = validate_expected_results(records)
    serialized = update_hashes_and_serialize(payloads, records)
    verify_loadable(serialized)
    manifest = build_manifest(
        records,
        serialized,
        old_hashes,
        targets,
        independent_checks,
    )
    manifest_data = deterministic_manifest(manifest)

    if mode == "write":
        changed = []
        for path, data in serialized.items():
            if write_if_changed(path, data):
                changed.append(relative(path))
        if write_if_changed(MANIFEST_PATH, manifest_data):
            changed.append(relative(MANIFEST_PATH))
        report_summary(records, "write")
        print(
            f"Updated {len(changed)} file(s)."
            if changed
            else "No files changed; migration is already up to date."
        )
        return 0

    stale = []
    for path, expected in serialized.items():
        if path.read_bytes() != expected:
            stale.append(relative(path))
    if not MANIFEST_PATH.exists():
        stale.append(relative(MANIFEST_PATH))
    elif MANIFEST_PATH.read_bytes() != manifest_data:
        stale.append(relative(MANIFEST_PATH))

    report_summary(records, "check")
    if stale:
        print("Migration is stale:")
        for path in sorted(stale):
            print(f"  {path}")
        return 1
    print("All recoverable ROC-AUC scores and provenance are up to date.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_const",
        const="check",
        dest="mode",
        help="verify the committed migration without changing files",
    )
    mode.add_argument(
        "--write",
        action="store_const",
        const="write",
        dest="mode",
        help="write recomputed scores and the provenance manifest",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args().mode))
