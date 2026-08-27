from __future__ import annotations

from pathlib import Path

import pytest

from export_reports import (
    FREEZE_CONFIRMATION,
    append_duplicate_review,
    append_epoch_decision,
    append_site_decision,
    completion_audit,
    freeze_decisions,
    initialize_exports,
    latest_decisions,
    read_csv,
    review_is_frozen,
)


def test_append_only_decisions_and_latest_revision(tmp_path: Path):
    append_site_decision(tmp_path, review_id="RV-X", session="SESSION_01", site="1", decision="GOOD", reason="", reviewer="A")
    append_site_decision(tmp_path, review_id="RV-X", session="SESSION_01", site="1", decision="BAD_SITE", reason="persistent rail saturation", reviewer="A")
    rows = read_csv(tmp_path / "qc_decisions.csv")
    assert len(rows) == 2
    assert latest_decisions(rows)[0]["decision"] == "BAD_SITE"


def test_rejection_requires_reason(tmp_path: Path):
    with pytest.raises(ValueError):
        append_epoch_decision(
            tmp_path, review_id="RV-X", session="SESSION_01", site="1", signal_type="tEEG",
            event_number=1, event_time=2.0, decision="REJECT", reason="", reviewer="A",
        )


def test_duplicate_review_is_append_only_and_requires_reason(tmp_path: Path):
    with pytest.raises(ValueError):
        append_duplicate_review(
            tmp_path,
            left_review_id="RV-A",
            right_review_id="RV-B",
            automated_status="WARNING",
            final_decision="Uncertain",
            reason="",
            reviewer="A",
            notes="",
            evidence={"metadata_near_candidate": True},
        )
    append_duplicate_review(
        tmp_path,
        left_review_id="RV-A",
        right_review_id="RV-B",
        automated_status="WARNING",
        final_decision="Needs expert review",
        reason="identity records unavailable",
        reviewer="A",
        notes="first review",
        evidence={"metadata_near_candidate": True},
    )
    append_duplicate_review(
        tmp_path,
        left_review_id="RV-A",
        right_review_id="RV-B",
        automated_status="WARNING",
        final_decision="Reject",
        reason="verified independent acquisitions",
        reviewer="B",
        notes="revision",
        evidence={"metadata_near_candidate": True},
    )
    assert len(read_csv(tmp_path / "duplicate_review.csv")) == 2


def test_freeze_requires_complete_review_and_exact_confirmation(tmp_path: Path):
    initialize_exports(tmp_path)
    incomplete = completion_audit([], [("RV-X", "SESSION_01", "1")], [])
    with pytest.raises(ValueError):
        freeze_decisions(tmp_path, reviewer="A", confirmation=FREEZE_CONFIRMATION, completion=incomplete)
    append_site_decision(tmp_path, review_id="RV-X", session="SESSION_01", site="1", decision="GOOD", reason="", reviewer="A")
    complete = completion_audit(latest_decisions(read_csv(tmp_path / "qc_decisions.csv")), [("RV-X", "SESSION_01", "1")], [])
    freeze_decisions(tmp_path, reviewer="A", confirmation=FREEZE_CONFIRMATION, completion=complete)
    assert review_is_frozen(tmp_path)
