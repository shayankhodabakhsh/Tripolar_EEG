from __future__ import annotations

import json
from pathlib import Path

from gui.notebook_review import NotebookReviewer


ROOT = Path(__file__).resolve().parents[2]


def test_review_notebook_is_valid_and_uses_shared_backend():
    notebook_path = ROOT / "MNE_QC_REVIEW.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    assert "gui.notebook_review" in source
    assert "read_raw_brainvision" not in source  # Processing stays in the tested module.


def test_notebook_overview_stays_blinded_and_preserves_mne_events(tmp_path: Path):
    reviewer = NotebookReviewer(output_dir=tmp_path)
    review_id = reviewer.review_ids[0]
    overview = reviewer.overview(review_id)
    assert overview["review_id"] == review_id
    assert str(overview["configuration"]).startswith("BLINDED")
    assert overview["checkerboard_event_count"] > 0
    assert overview["mne_preservation"]["all_preserved"] is True
    assert set(overview).isdisjoint({"source_session_name", "source_path", "holder_configuration"})


def test_notebook_event_table_uses_recorded_s7_timing(tmp_path: Path):
    reviewer = NotebookReviewer(output_dir=tmp_path)
    table, figure = reviewer.event_timing(reviewer.review_ids[0])
    assert table["event_number"].iloc[0] == 1
    assert table["event_time_seconds"].is_monotonic_increasing
    assert table["previous_interval_seconds"].iloc[1:].notna().all()
    figure.clear()
