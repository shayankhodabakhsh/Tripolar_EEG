"""Append-only review audit trail, frozen masks, and post-review reports."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import html
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .qc_engine import validate_epoch_decision, validate_site_decision
except ImportError:
    from qc_engine import validate_epoch_decision, validate_site_decision


UTC = timezone.utc
FREEZE_CONFIRMATION = "REVIEW COMPLETE - FREEZE DECISIONS"

QC_DECISION_FIELDS = (
    "review_id", "session", "site", "event_number", "event_time", "decision", "reason",
    "reviewer", "timestamp", "notes", "original_automated_status", "record_type", "signal_type", "revision_id",
)
CHANNEL_METRIC_FIELDS = (
    "review_id", "session", "site", "signal_type", "channel", "metric", "status", "measured_value",
    "threshold", "affected_interval", "supporting_plot", "explanation", "contaminated_sample_fraction",
)
EPOCH_LOG_FIELDS = QC_DECISION_FIELDS
VEP_REVIEW_FIELDS = (
    "review_id", "session", "site", "signal_type", "peak_name", "automated_status", "final_decision",
    "reason", "reviewer", "timestamp", "notes", "metrics_json",
)
DUPLICATE_REVIEW_FIELDS = (
    "left_review_id", "right_review_id", "automated_status", "final_decision", "reason", "reviewer",
    "timestamp", "notes", "evidence_json",
)


@dataclass(frozen=True)
class DecisionRecord:
    review_id: str
    session: str
    site: str
    event_number: str
    event_time: str
    decision: str
    reason: str
    reviewer: str
    timestamp: str
    notes: str
    original_automated_status: str
    record_type: str
    revision_id: str
    signal_type: str = ""


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_exports(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "qc_decisions": output_dir / "qc_decisions.csv",
        "channel_quality_metrics": output_dir / "channel_quality_metrics.csv",
        "epoch_rejection_log": output_dir / "epoch_rejection_log.csv",
        "vep_peak_review": output_dir / "vep_peak_review.csv",
        "duplicate_review": output_dir / "duplicate_review.csv",
    }
    schemas = {
        "qc_decisions": QC_DECISION_FIELDS,
        "channel_quality_metrics": CHANNEL_METRIC_FIELDS,
        "epoch_rejection_log": EPOCH_LOG_FIELDS,
        "vep_peak_review": VEP_REVIEW_FIELDS,
        "duplicate_review": DUPLICATE_REVIEW_FIELDS,
    }
    for name, path in paths.items():
        expected = list(schemas[name])
        if not path.exists() or path.stat().st_size == 0:
            _append_row(path, schemas[name], None)
        else:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            if rows and rows[0] != expected:
                if len(rows) > 1 and name != "channel_quality_metrics":
                    raise RuntimeError(f"Export schema changed after decisions were recorded; migrate explicitly: {path}")
                path.write_text("", encoding="utf-8")
                _append_row(path, schemas[name], None)
    return paths


def _append_row(path: Path, fields: Sequence[str], row: Mapping[str, object] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        if empty:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_site_decision(
    output_dir: Path,
    *,
    review_id: str,
    session: str,
    site: str,
    decision: str,
    reason: str,
    reviewer: str,
    notes: str = "",
    original_automated_status: str = "UNRESOLVED",
) -> DecisionRecord:
    validate_site_decision(decision)
    if not reviewer.strip():
        raise ValueError("Reviewer is required")
    if decision in {"BAD_SITE", "UNCERTAIN", "GOOD_WITH_BAD_SEGMENTS"} and not reason.strip():
        raise ValueError("A reason is required for rejection, uncertainty, or manual override")
    timestamp = utc_timestamp()
    revision_id = hashlib.sha256(
        f"{review_id}|{session}|{site}|site|{timestamp}|{decision}|{reviewer}".encode()
    ).hexdigest()[:16]
    record = DecisionRecord(
        review_id, session, site, "", "", decision, reason.strip(), reviewer.strip(), timestamp,
        notes.strip(), original_automated_status, "site_session", revision_id,
    )
    paths = initialize_exports(output_dir)
    _append_row(paths["qc_decisions"], QC_DECISION_FIELDS, asdict(record))
    return record


def append_epoch_decision(
    output_dir: Path,
    *,
    review_id: str,
    session: str,
    site: str,
    signal_type: str,
    event_number: int,
    event_time: float,
    decision: str,
    reason: str,
    reviewer: str,
    notes: str = "",
    original_automated_status: str = "UNRESOLVED",
) -> DecisionRecord:
    validate_epoch_decision(decision, reason)
    if not reviewer.strip():
        raise ValueError("Reviewer is required")
    timestamp = utc_timestamp()
    revision_id = hashlib.sha256(
        f"{review_id}|{session}|{site}|{signal_type}|{event_number}|{timestamp}|{decision}|{reviewer}".encode()
    ).hexdigest()[:16]
    record = DecisionRecord(
        review_id, session, str(site), str(event_number), f"{event_time:.6f}", decision, reason,
        reviewer.strip(), timestamp, notes.strip(), original_automated_status, "event_epoch", revision_id,
        signal_type,
    )
    paths = initialize_exports(output_dir)
    row = asdict(record)
    _append_row(paths["qc_decisions"], QC_DECISION_FIELDS, row)
    _append_row(paths["epoch_rejection_log"], EPOCH_LOG_FIELDS, row)
    return record


def append_rows(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    for row in rows:
        _append_row(path, fields, row)


def append_vep_peak_review(
    output_dir: Path,
    *,
    review_id: str,
    session: str,
    site: str,
    signal_type: str,
    peak_name: str,
    automated_status: str,
    final_decision: str,
    reason: str,
    reviewer: str,
    notes: str,
    metrics: Mapping[str, object],
) -> None:
    allowed = {"Accept", "Reject", "Uncertain", "Needs expert review"}
    if final_decision not in allowed:
        raise ValueError(f"Unknown peak review decision: {final_decision}")
    if not reviewer.strip():
        raise ValueError("Reviewer is required")
    if final_decision != "Accept" and not reason.strip():
        raise ValueError("A reason is required for rejection, uncertainty, or expert escalation")
    paths = initialize_exports(output_dir)
    _append_row(
        paths["vep_peak_review"],
        VEP_REVIEW_FIELDS,
        {
            "review_id": review_id,
            "session": session,
            "site": site,
            "signal_type": signal_type,
            "peak_name": peak_name,
            "automated_status": automated_status,
            "final_decision": final_decision,
            "reason": reason.strip(),
            "reviewer": reviewer.strip(),
            "timestamp": utc_timestamp(),
            "notes": notes.strip(),
            "metrics_json": json.dumps(metrics, default=str, sort_keys=True),
        },
    )


def append_duplicate_review(
    output_dir: Path,
    *,
    left_review_id: str,
    right_review_id: str,
    automated_status: str,
    final_decision: str,
    reason: str,
    reviewer: str,
    notes: str,
    evidence: Mapping[str, object],
) -> None:
    """Append a human identity/duplicate assessment without replacing evidence."""
    allowed = {"Accept", "Reject", "Uncertain", "Needs expert review"}
    if final_decision not in allowed:
        raise ValueError(f"Unknown duplicate review decision: {final_decision}")
    if not reviewer.strip():
        raise ValueError("Reviewer is required")
    if final_decision != "Accept" and not reason.strip():
        raise ValueError("A reason is required for rejection, uncertainty, or expert escalation")
    paths = initialize_exports(output_dir)
    _append_row(
        paths["duplicate_review"],
        DUPLICATE_REVIEW_FIELDS,
        {
            "left_review_id": left_review_id,
            "right_review_id": right_review_id,
            "automated_status": automated_status,
            "final_decision": final_decision,
            "reason": reason.strip(),
            "reviewer": reviewer.strip(),
            "timestamp": utc_timestamp(),
            "notes": notes.strip(),
            "evidence_json": json.dumps(evidence, default=str, sort_keys=True),
        },
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_decisions(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Resolve append-only revisions without deleting historical rows."""
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    for input_row in rows:
        row = dict(input_row)
        if row.get("record_type") == "event_epoch":
            key = (
                row.get("record_type", ""), row.get("review_id", ""), row.get("session", ""),
                row.get("site", ""), row.get("signal_type", ""), row.get("event_number", ""),
            )
        else:
            key = (row.get("record_type", ""), row.get("review_id", ""), row.get("session", ""), row.get("site", ""))
        if key not in latest or row.get("timestamp", "") >= latest[key].get("timestamp", ""):
            latest[key] = row
    return sorted(latest.values(), key=lambda row: (row.get("review_id", ""), row.get("session", ""), row.get("site", ""), row.get("signal_type", ""), row.get("event_number", "")))


def completion_audit(
    latest: Sequence[Mapping[str, str]],
    expected_sites: Sequence[tuple[str, str, str]],
    expected_epochs: Sequence[tuple[str, str, str, str, int]],
) -> dict[str, object]:
    site_keys = {
        (row.get("review_id", ""), row.get("session", ""), row.get("site", ""))
        for row in latest if row.get("record_type") == "site_session"
    }
    epoch_keys = {
        (
            row.get("review_id", ""), row.get("session", ""), row.get("site", ""),
            row.get("signal_type", ""), int(row.get("event_number", "0")),
        )
        for row in latest if row.get("record_type") == "event_epoch" and row.get("event_number", "").isdigit()
    }
    missing_sites = sorted(set(expected_sites) - site_keys)
    missing_epochs = sorted(set(expected_epochs) - epoch_keys)
    return {
        "complete": not missing_sites and not missing_epochs,
        "expected_site_decisions": len(expected_sites),
        "completed_site_decisions": len(set(expected_sites) & site_keys),
        "expected_epoch_decisions": len(expected_epochs),
        "completed_epoch_decisions": len(set(expected_epochs) & epoch_keys),
        "missing_sites": missing_sites,
        "missing_epochs": missing_epochs,
    }


def freeze_decisions(
    output_dir: Path,
    *,
    reviewer: str,
    confirmation: str,
    completion: Mapping[str, object],
) -> Path:
    """Freeze the latest mask only after explicit phrase and completeness audit."""
    if confirmation != FREEZE_CONFIRMATION:
        raise ValueError(f"Type exactly: {FREEZE_CONFIRMATION}")
    if not completion.get("complete"):
        raise ValueError("Review is incomplete; no rejection mask was frozen")
    paths = initialize_exports(output_dir)
    source_hash = sha256_file(paths["qc_decisions"])
    current = latest_decisions(read_csv(paths["qc_decisions"]))
    freeze_path = output_dir / "frozen_rejection_mask.csv"
    if freeze_path.exists():
        raise FileExistsError("A frozen rejection mask already exists; it cannot be silently replaced")
    fields = list(QC_DECISION_FIELDS) + ["signal_type", "freeze_reviewer", "freeze_timestamp", "source_decisions_sha256"]
    freeze_timestamp = utc_timestamp()
    with freeze_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in current:
            writer.writerow({**row, "freeze_reviewer": reviewer, "freeze_timestamp": freeze_timestamp, "source_decisions_sha256": source_hash})
    metadata = {
        "frozen": True,
        "reviewer": reviewer,
        "timestamp": freeze_timestamp,
        "source_decisions_sha256": source_hash,
        "frozen_mask_sha256": sha256_file(freeze_path),
        "completion": completion,
        "note": "This freezes decisions only; raw data were not modified and results were not regenerated.",
    }
    (output_dir / "review_complete.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return freeze_path


def review_is_frozen(output_dir: Path) -> bool:
    marker = output_dir / "review_complete.json"
    mask = output_dir / "frozen_rejection_mask.csv"
    if not marker.exists() or not mask.exists():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(metadata.get("frozen")) and metadata.get("frozen_mask_sha256") == sha256_file(mask)


def frozen_epoch_mask(
    frozen_mask_path: Path,
    review_id: str,
    session: str,
    site: str,
    signal_type: str,
    event_numbers: Sequence[int],
) -> list[bool]:
    rows = read_csv(frozen_mask_path)
    decisions = {
        int(row["event_number"]): row["decision"]
        for row in rows
        if row.get("record_type") == "event_epoch"
        and row.get("review_id") == review_id
        and row.get("session") == session
        and row.get("site") == str(site)
        and row.get("signal_type") == signal_type
        and row.get("event_number", "").isdigit()
    }
    missing = [int(number) for number in event_numbers if int(number) not in decisions]
    if missing:
        raise ValueError(f"Frozen mask is incomplete for {review_id}/{session}/site {site}/{signal_type}: {missing[:10]}")
    return [decisions[int(number)] == "KEEP" for number in event_numbers]


def paired_frozen_mask(
    frozen_mask_path: Path,
    review_id: str,
    session: str,
    site: str,
    event_numbers: Sequence[int],
) -> list[bool]:
    t = frozen_epoch_mask(frozen_mask_path, review_id, session, site, "tEEG", event_numbers)
    e = frozen_epoch_mask(frozen_mask_path, review_id, session, site, "eEEG", event_numbers)
    return [left and right for left, right in zip(t, e)]


def count_change_report(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare site/trial denominators after applying a frozen mask."""
    keys = sorted({(str(row["review_id"]), str(row["session"])) for row in before} | {(str(row["review_id"]), str(row["session"])) for row in after})
    report = []
    for review_id, session in keys:
        before_rows = [row for row in before if str(row["review_id"]) == review_id and str(row["session"]) == session]
        after_rows = [row for row in after if str(row["review_id"]) == review_id and str(row["session"]) == session]
        before_sites = len({str(row["site"]) for row in before_rows})
        after_sites = len({str(row["site"]) for row in after_rows})
        before_trials = sum(int(row.get("accepted_trials", row.get("trial_count", 0))) for row in before_rows)
        after_trials = sum(int(row.get("accepted_trials", row.get("trial_count", 0))) for row in after_rows)
        report.append(
            {
                "review_id": review_id,
                "session": session,
                "sites_before": before_sites,
                "sites_after": after_sites,
                "site_change": after_sites - before_sites,
                "trials_before": before_trials,
                "trials_after": after_trials,
                "trial_change": after_trials - before_trials,
            }
        )
    return report


def write_session_html_report(
    path: Path,
    *,
    raw,
    review_id: str,
    session: str,
    integrity: Mapping[str, object],
    findings: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
    post_review_results: Mapping[str, object] | None = None,
    include_raw_summary: bool = False,
    figures: Sequence[tuple[object, str, str]] = (),
) -> Path:
    """Write a reproducible MNE Report; source identity stays hidden pre-freeze."""
    import mne  # Lazy import keeps export-only tests lightweight.

    path.parent.mkdir(parents=True, exist_ok=True)

    def table(rows: Sequence[Mapping[str, object]]) -> str:
        if not rows:
            return "<p>No rows.</p>"
        fields = list(rows[0])
        head = "".join(f"<th>{html.escape(str(field))}</th>" for field in fields)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    report = mne.Report(title=f"EEG quality report: {review_id} / {session}", raw_psd=False, verbose="ERROR")
    report.add_html(
        "<p><strong>Raw data were read-only.</strong> Automated statuses are technical findings, not final human decisions. "
        "The authoritative backend is <code>mne.io.read_raw_brainvision()</code>.</p>",
        title="Review safety and provenance",
        section="Overview",
    )
    report.add_html(
        f"<pre>{html.escape(json.dumps(dict(integrity), indent=2, default=str))}</pre>",
        title="Data integrity",
        section="Integrity",
    )
    if include_raw_summary:
        report.add_raw(raw, title="MNE BrainVision recording summary and raw pre-notch PSD", psd=True, butterfly=False)
    report.add_html(table(findings), title="Technical findings", section="Technical QC")
    report.add_html(table(decisions), title="Append-only decision audit trail", section="Human review")
    if post_review_results is None:
        report.add_html("<p>Locked until review is complete.</p>", title="Post-review physiology", section="Post-review")
    else:
        report.add_html(
            f"<pre>{html.escape(json.dumps(post_review_results, indent=2, default=str))}</pre>",
            title="Post-review physiology",
            section="Post-review",
        )
    for figure, title, section in figures:
        report.add_figure(figure, title=title, section=section)
    report.save(path, overwrite=True, open_browser=False, sort_content=False, verbose="ERROR")
    return path


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Best-effort PDF conversion; HTML remains the authoritative report."""
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError:
        return False
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return True
