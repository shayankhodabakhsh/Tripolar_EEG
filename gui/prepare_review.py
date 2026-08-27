"""One-time creation of the blinded review queue and automated QC inventory."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

try:
    from .data_integrity import (
        build_blinded_manifests,
        duplicate_candidates,
        integrity_record,
        load_manifest,
        load_raw_counts,
        resolve_review_recording,
    )
    from .export_reports import (
        CHANNEL_METRIC_FIELDS,
        DUPLICATE_REVIEW_FIELDS,
        append_rows,
        initialize_exports,
        read_csv,
        write_session_html_report,
    )
    from .mne_backend import checkerboard_samples, condition_intervals, load_brainvision, raw_microvolts, validate_preservation
    from .qc_engine import QCThresholds, assess_channel, raw_dtype
except ImportError:
    from data_integrity import (
        build_blinded_manifests,
        duplicate_candidates,
        integrity_record,
        load_manifest,
        load_raw_counts,
        resolve_review_recording,
    )
    from export_reports import (
        CHANNEL_METRIC_FIELDS,
        DUPLICATE_REVIEW_FIELDS,
        append_rows,
        initialize_exports,
        read_csv,
        write_session_html_report,
    )
    from mne_backend import checkerboard_samples, condition_intervals, load_brainvision, raw_microvolts, validate_preservation
    from qc_engine import QCThresholds, assess_channel, raw_dtype


GUI_DIR = Path(__file__).resolve().parent
ROOT = GUI_DIR.parent
PUBLIC = GUI_DIR / "review_data" / "blinded_review_manifest.csv"
PRIVATE = GUI_DIR / "private" / "blinding_key.csv"
OUTPUT = GUI_DIR / "review_output"
THRESHOLDS = GUI_DIR / "qc_thresholds.json"


def write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare(overwrite_manifest: bool = False, compute_metrics: bool = True) -> None:
    if not THRESHOLDS.exists():
        QCThresholds().to_json(THRESHOLDS)
    if overwrite_manifest or not PUBLIC.exists() or not PRIVATE.exists():
        build_blinded_manifests(ROOT / "Data", PUBLIC, PRIVATE, overwrite=overwrite_manifest)
        os.chmod(PRIVATE, 0o600)
    initialize_exports(OUTPUT)
    manifest = load_manifest(PUBLIC)
    integrity_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    thresholds = QCThresholds.from_json(THRESHOLDS)
    for number, public in enumerate(manifest, start=1):
        review_id, session = public["review_id"], public["session"]
        info = resolve_review_recording(review_id, session, PRIVATE)
        recording = load_brainvision(info.header_path, preload=False)
        events = checkerboard_samples(recording)
        rests = condition_intervals(recording.raw, thresholds.rest_max_seconds)
        preservation = validate_preservation(recording)
        if not preservation["all_preserved"]:
            raise RuntimeError(f"MNE preservation check failed for {review_id}: {preservation}")
        if len(events) != int(public["checkerboard_event_count"]):
            raise RuntimeError(f"MNE/custom integrity event-count mismatch for {review_id}")
        if len(rests["eyes_open"]) != int(public["eyes_open_segments"]) or len(rests["eyes_closed"]) != int(public["eyes_closed_segments"]):
            raise RuntimeError(f"MNE/custom integrity condition-marker mismatch for {review_id}")
        integrity_row = {
                "review_id": review_id,
                "session": session,
                "identity_status": public["identity_status"],
                "configuration_status": public["configuration_status"],
                "manual_review_scope": public["manual_review_scope"],
                "authoritative_backend": "mne.io.read_raw_brainvision",
                "mne_preservation_checks": json.dumps(preservation),
                **integrity_record(info),
            }
        integrity_rows.append(integrity_row)
        if not compute_metrics or public["manual_review_scope"] != "IN_SCOPE":
            write_session_html_report(
                OUTPUT / "reports" / f"{review_id}_{session}.html",
                raw=recording.raw,
                review_id=review_id,
                session=session,
                integrity=integrity_row,
                findings=[],
                decisions=[],
                post_review_results=None,
                include_raw_summary=False,
            )
            continue
        counts = np.asarray(load_raw_counts(info, range(8)))
        microvolts = raw_microvolts(recording.raw, picks=range(8))
        record_metric_rows: list[dict[str, object]] = []
        for channel in range(8):
            site = channel // 2 + 1
            signal_type = "tEEG" if channel % 2 == 0 else "eEEG"
            findings, metrics = assess_channel(
                counts[channel], microvolts[channel], f"Ch{channel + 1}", info.sampling_rate_hz,
                raw_dtype(info), thresholds,
            )
            for finding in findings:
                finding_row = {
                        "review_id": review_id,
                        "session": session,
                        "site": site,
                        "signal_type": signal_type,
                        "channel": finding.channel,
                        "metric": finding.rule_id,
                        "status": finding.status,
                        "measured_value": finding.measured_value,
                        "threshold": finding.threshold,
                        "affected_interval": json.dumps(finding.intervals),
                        "supporting_plot": finding.supporting_plot,
                        "explanation": finding.explanation,
                        "contaminated_sample_fraction": metrics["contaminated_sample_fraction"],
                    }
                metric_rows.append(finding_row)
                record_metric_rows.append(finding_row)
        write_session_html_report(
            OUTPUT / "reports" / f"{review_id}_{session}.html",
            raw=recording.raw,
            review_id=review_id,
            session=session,
            integrity=integrity_row,
            findings=record_metric_rows,
            decisions=[],
            post_review_results=None,
            include_raw_summary=False,
        )
        print(f"prepared {number}/{len(manifest)}: {review_id}")
    integrity_fields = list(dict.fromkeys(field for row in integrity_rows for field in row))
    write_csv(GUI_DIR / "review_data" / "data_integrity_manifest.csv", integrity_fields, integrity_rows)
    if compute_metrics:
        write_csv(OUTPUT / "channel_quality_metrics.csv", CHANNEL_METRIC_FIELDS, metric_rows)
    candidates = duplicate_candidates(integrity_rows)
    duplicate_rows = [
        {
            **row,
            "final_decision": "",
            "reason": "",
            "reviewer": "",
            "timestamp": "",
            "notes": "",
            "evidence_json": json.dumps(row),
        }
        for row in candidates
    ]
    # Duplicate assessments are an append-only human audit trail. Preparation
    # may add newly discovered candidates, but never replaces earlier reviews.
    duplicate_path = OUTPUT / "duplicate_review.csv"
    existing_pairs = {
        tuple(sorted((row.get("left_review_id", ""), row.get("right_review_id", ""))))
        for row in read_csv(duplicate_path)
    }
    new_rows = [
        row for row in duplicate_rows
        if tuple(sorted((str(row["left_review_id"]), str(row["right_review_id"])))) not in existing_pairs
    ]
    append_rows(duplicate_path, DUPLICATE_REVIEW_FIELDS, new_rows)
    summary = {
        "records": len(manifest),
        "in_scope_records": sum(row["manual_review_scope"] == "IN_SCOPE" for row in manifest),
        "integrity_rows": len(integrity_rows),
        "channel_metric_rows": len(metric_rows),
        "duplicate_candidates": len(candidates),
        "raw_data_modified": False,
        "decisions_applied": False,
        "authoritative_backend": "mne.io.read_raw_brainvision",
    }
    (GUI_DIR / "review_data" / "PREPARATION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-manifest", action="store_true", help="Re-randomize IDs; invalidates existing review decisions")
    parser.add_argument("--skip-metrics", action="store_true", help="Create manifests without computing continuous QC")
    args = parser.parse_args()
    if args.overwrite_manifest and (OUTPUT / "qc_decisions.csv").exists() and (OUTPUT / "qc_decisions.csv").stat().st_size > 200:
        raise SystemExit("Refusing to re-randomize IDs after decisions exist. Archive the review output explicitly first.")
    prepare(args.overwrite_manifest, not args.skip_metrics)


if __name__ == "__main__":
    main()
