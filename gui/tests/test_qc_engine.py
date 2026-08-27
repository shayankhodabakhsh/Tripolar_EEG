from __future__ import annotations

import numpy as np

from qc_engine import (
    QCThresholds,
    assess_channel,
    common_accepted_event_mask,
    epoch_data,
    flatline_metrics,
    line_noise_metrics,
    rail_saturation_metrics,
    step_metrics,
    zero_phase_review_filter,
)


def test_zero_phase_filter_preserves_shape_and_event_timing():
    fs = 1000.0
    time = np.arange(0, 5, 1 / fs)
    values = np.sin(2 * np.pi * 10 * time) + 0.2 * np.sin(2 * np.pi * 80 * time)
    filtered = zero_phase_review_filter(values, fs, QCThresholds())
    assert filtered.shape == values.shape
    reference = np.sin(2 * np.pi * 10 * time)
    central = slice(500, -500)
    correlation = np.correlate(filtered[central], reference[central], mode="full")
    lag = int(np.argmax(correlation) - (reference[central].size - 1))
    assert abs(lag) <= 1


def test_epoching_reports_incomplete_events_without_padding():
    values = np.arange(1000)
    result = epoch_data(values, [20, 500, 980], 1000.0, QCThresholds())
    assert result.event_numbers.tolist() == [2]
    assert result.incomplete_event_numbers.tolist() == [1, 3]
    assert result.epochs.shape == (1, 500)


def test_rail_saturation_detects_exact_int16_endpoints_and_plateau():
    values = np.zeros(10_000, dtype=np.int16)
    values[100:130] = np.iinfo(np.int16).max
    metrics = rail_saturation_metrics(values, values.dtype, 1000.0, QCThresholds())
    assert metrics["rail_samples"] == 30
    assert metrics["longest_rail_plateau_samples"] == 30
    assert metrics["status"] == "FAIL"


def test_flatline_and_step_are_localized():
    rng = np.random.default_rng(2)
    values = rng.normal(0, 5, 10_000)
    values[2000:5000] = 0
    flat = flatline_metrics(values, 1000.0, QCThresholds())
    assert flat["status"] == "FAIL"
    popped = rng.normal(0, 2, 10_000)
    popped[5000:] += 200
    steps = step_metrics(popped, 1000.0, QCThresholds())
    assert steps["step_count"] >= 1
    assert any(start <= 5 <= end + 0.001 for start, end in steps["step_intervals"])


def test_raw_60hz_metric_sees_synthetic_line_noise():
    fs = 1000.0
    time = np.arange(0, 20, 1 / fs)
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, time.size) + 20 * np.sin(2 * np.pi * 60 * time)
    result = line_noise_metrics(values, fs, QCThresholds())
    assert result["line_60hz_intrusion_db"] > 12
    assert result["status"] == "FAIL"


def test_channel_assessment_does_not_return_accept_or_reject_decision():
    rng = np.random.default_rng(4)
    counts = rng.integers(-20, 20, size=20_000, dtype=np.int16)
    findings, metrics = assess_channel(counts, counts.astype(float) * 0.1, "Ch1", 1000.0, counts.dtype, QCThresholds())
    assert findings
    assert metrics["automated_status"] in {"PASS", "WARNING", "FAIL", "UNRESOLVED"}
    assert "decision" not in metrics


def test_common_mask_is_intersection_of_manual_keep_decisions():
    event_numbers = [1, 2, 3, 4]
    t = {1: "KEEP", 2: "REJECT", 3: "KEEP", 4: "KEEP"}
    e = {1: "KEEP", 2: "KEEP", 3: "REJECT", 4: "KEEP"}
    assert common_accepted_event_mask(event_numbers, t, e).tolist() == [True, False, False, True]
