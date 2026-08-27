from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from data_integrity import load_raw_counts
from export_reports import write_session_html_report
from mne_backend import (
    CHECKERBOARD_ANNOTATION,
    checkerboard_epochs,
    condition_intervals,
    load_brainvision,
    raw_microvolts,
    review_copy,
    validate_preservation,
)
from qc_engine import QCThresholds


ROOT = Path(__file__).resolve().parents[2]


def first_project_header() -> Path:
    headers = [path for path in sorted((ROOT / "Data").rglob("*.vhdr")) if "Triggers" not in path.stem]
    if not headers:
        pytest.skip("Project BrainVision files are unavailable")
    return headers[0]


def test_mne_brainvision_preserves_project_metadata_annotations_and_events():
    recording = load_brainvision(first_project_header(), preload=False)
    checks = validate_preservation(recording)
    assert checks["all_preserved"]
    assert recording.event_id[CHECKERBOARD_ANNOTATION] == 7
    assert recording.raw.info["sfreq"] == 1000.0
    assert len(recording.raw.annotations) > len(recording.checkerboard_events)
    rests = condition_intervals(recording.raw)
    assert rests["eyes_open"] and rests["eyes_closed"]


def test_mne_signal_scaling_matches_exact_brainvision_counts():
    recording = load_brainvision(first_project_header(), preload=False)
    counts = np.asarray(load_raw_counts(recording.source_info, [0]))[:, :1000]
    expected_uv = counts * recording.source_info.channel_resolutions[0]
    actual_uv = raw_microvolts(recording.raw, picks=[0])[:, :1000]
    assert np.allclose(actual_uv, expected_uv, rtol=0, atol=1e-10)


def test_mne_filter_and_epoch_window_are_authoritative():
    recording = load_brainvision(first_project_header(), preload=False)
    filtered = review_copy(recording.raw, QCThresholds(), picks=[0, 1])
    assert filtered.info["highpass"] == 0.5
    assert filtered.info["lowpass"] == 30.0
    assert np.array_equal(filtered.annotations.description, recording.raw.annotations.description)
    epochs = checkerboard_epochs(filtered, recording.checkerboard_events, QCThresholds(), picks=[0, 1])
    assert epochs.get_data().shape[-1] == 500
    assert epochs.times[0] == -0.1
    assert epochs.times[-1] == 0.399


def test_session_html_is_an_mne_report(tmp_path: Path):
    info = mne.create_info(["tEEG", "eEEG"], 1000.0, ["eeg", "eeg"])
    raw = mne.io.RawArray(np.zeros((2, 2000)), info, verbose="ERROR")
    path = write_session_html_report(
        tmp_path / "report.html",
        raw=raw,
        review_id="RV-TEST",
        session="SESSION_01",
        integrity={"backend": "mne.io.read_raw_brainvision"},
        findings=[],
        decisions=[],
        post_review_results=None,
        include_raw_summary=False,
    )
    content = path.read_text(encoding="utf-8")
    assert "MNE-Python" in content and "EEG quality report: RV-TEST" in content
    assert "mne.io.read_raw_brainvision" in content
