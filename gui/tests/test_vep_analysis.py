from __future__ import annotations

import numpy as np

from vep_analysis import (
    NEUTRAL_PEAKS,
    analyze_vep_epochs,
    epoch_selection_sensitivity,
    find_candidate_peak,
    paired_waveforms,
)


def synthetic_epochs(n_trials=40, seed=6):
    rng = np.random.default_rng(seed)
    times = np.arange(-100, 400) / 1000
    waveform = (
        -5 * np.exp(-0.5 * ((times - 0.075) / 0.012) ** 2)
        + 10 * np.exp(-0.5 * ((times - 0.110) / 0.015) ** 2)
        - 6 * np.exp(-0.5 * ((times - 0.165) / 0.020) ** 2)
    )
    epochs = waveform + rng.normal(0, 3, (n_trials, times.size))
    return times, epochs


def test_candidate_peak_uses_neutral_label_and_window():
    times, epochs = synthetic_epochs()
    peak = find_candidate_peak(np.mean(epochs, axis=0), times * 1000, NEUTRAL_PEAKS[1])
    assert peak["name"] == "principal positive peak"
    assert 90 <= peak["latency_ms"] < 140


def test_bootstrap_peak_analysis_returns_confidence_and_reliability():
    times, epochs = synthetic_epochs()
    result = analyze_vep_epochs(epochs, times, n_bootstrap=100)
    assert len(result["peak_rows"]) == 3
    assert result["odd_even_correlation"] > 0
    principal = result["peak_rows"][1]
    assert principal["bootstrap_peak_detection_frequency"] > 0.5
    assert principal["bootstrap_latency_ci_low_ms"] <= principal["latency_ms"] <= principal["bootstrap_latency_ci_high_ms"]


def test_paired_waveforms_uses_only_common_mask():
    times, epochs = synthetic_epochs(n_trials=6)
    other = epochs * 0.2
    mask = np.array([True, False, False, True, False, True])
    result = paired_waveforms(epochs, other, mask, times)
    assert result["common_trial_count"] == 3
    assert np.allclose(result["teeg_average_uv"], np.mean(epochs[mask] - np.mean(epochs[mask][:, times < 0], axis=1, keepdims=True), axis=0))


def test_epoch_selection_sensitivity_reports_prespecified_subsets():
    times, epochs = synthetic_epochs()
    rows = epoch_selection_sensitivity(epochs, times)
    assert [row["epoch_subset"] for row in rows] == [
        "all accepted", "odd accepted", "even accepted", "first half", "second half"
    ]
    assert all("latency_spread_across_subsets_ms" in row for row in rows)
    assert all(row["trial_count"] > 0 for row in rows)
