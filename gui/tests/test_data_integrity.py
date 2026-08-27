from __future__ import annotations

from pathlib import Path

import numpy as np

from data_integrity import (
    Marker,
    checkerboard_events,
    condition_intervals,
    event_blocks,
    marker_timing_metrics,
    parse_markers,
)


def test_marker_parser_and_event_code_are_exact(tmp_path: Path):
    marker = tmp_path / "sample.vmrk"
    marker.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,20260101000000000000\n"
        "Mk2=Stimulus,S  7,101,1,0\n"
        "Mk3=Stimulus,S  8,201,1,0\n",
        encoding="utf-8",
    )
    rows = parse_markers(marker)
    assert checkerboard_events(rows).tolist() == [100]


def test_marker_rate_uses_within_block_intervals():
    events = np.array([100, 700, 1300, 10_000, 10_600, 11_200])
    result = marker_timing_metrics(events, 1000.0)
    assert result["block_count"] == 2
    assert result["median_iei_seconds"] == 0.6
    assert np.isclose(result["median_reversal_rate_hz"], 1 / 0.6)


def test_condition_intervals_are_capped_at_30_seconds():
    markers = [
        Marker(1, "Comment", "Eyes Open", 100, 1, 0),
        Marker(2, "Comment", "Eyes Closed", 50_000, 1, 0),
    ]
    result = condition_intervals(markers, 100_000, 1000.0)
    assert result["eyes_open"] == [(100, 30_100)]
    assert result["eyes_closed"] == [(50_000, 80_000)]


def test_event_block_split():
    blocks = event_blocks([0, 600, 1200, 8000, 8600], 1000.0)
    assert [block.tolist() for block in blocks] == [[0, 600, 1200], [8000, 8600]]
