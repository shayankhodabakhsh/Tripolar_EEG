"""Post-review VEP, peak-confidence, spectral, and alpha calculations.

Nothing in this module is needed to make technical rejection decisions.  The
GUI keeps these results locked until a human review is frozen, preventing VEP
detectability or morphology from biasing artifact rejection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

try:
    from .qc_engine import QCThresholds, baseline_correct, zero_phase_review_filter
except ImportError:
    from qc_engine import QCThresholds, baseline_correct, zero_phase_review_filter


@dataclass(frozen=True)
class PeakDefinition:
    name: str
    polarity: str
    window_ms: tuple[float, float]


NEUTRAL_PEAKS = (
    PeakDefinition("first negative peak", "negative", (55.0, 90.0)),
    PeakDefinition("principal positive peak", "positive", (90.0, 140.0)),
    PeakDefinition("subsequent negative peak", "negative", (130.0, 215.0)),
)


def _mask(times_ms: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    return (times_ms >= window[0]) & (times_ms < window[1])


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(stats.pearsonr(left, right).statistic)


def find_candidate_peak(waveform: np.ndarray, times_ms: np.ndarray, definition: PeakDefinition) -> dict[str, object]:
    window_mask = _mask(times_ms, definition.window_ms)
    indices = np.flatnonzero(window_mask)
    if not indices.size or not np.all(np.isfinite(waveform[window_mask])):
        return {
            "name": definition.name,
            "polarity": definition.polarity,
            "window_start_ms": definition.window_ms[0],
            "window_end_ms": definition.window_ms[1],
            "status": "UNRESOLVED",
        }
    local = np.argmax(waveform[window_mask]) if definition.polarity == "positive" else np.argmin(waveform[window_mask])
    index = int(indices[int(local)])
    boundary = index in {int(indices[0]), int(indices[-1])}
    return {
        "name": definition.name,
        "polarity": definition.polarity,
        "window_start_ms": definition.window_ms[0],
        "window_end_ms": definition.window_ms[1],
        "latency_ms": float(times_ms[index]),
        "amplitude_uv": float(waveform[index]),
        "sample_index": index,
        "at_search_window_boundary": boundary,
        "status": "WARNING" if boundary else "PASS",
    }


def split_half_waveforms(epochs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "odd": np.mean(epochs[::2], axis=0) if epochs[::2].size else np.array([]),
        "even": np.mean(epochs[1::2], axis=0) if epochs[1::2].size else np.array([]),
        "first_half": np.mean(epochs[: len(epochs) // 2], axis=0) if len(epochs) >= 2 else np.array([]),
        "second_half": np.mean(epochs[len(epochs) // 2 :], axis=0) if len(epochs) >= 2 else np.array([]),
    }


def split_half_reliability(epochs: np.ndarray, times_ms: np.ndarray, window_ms: tuple[float, float] = (0.0, 300.0)) -> dict[str, float]:
    halves = split_half_waveforms(epochs)
    mask = _mask(times_ms, window_ms)
    odd_even = _safe_correlation(halves["odd"][mask], halves["even"][mask]) if halves["odd"].size and halves["even"].size else float("nan")
    first_second = _safe_correlation(halves["first_half"][mask], halves["second_half"][mask]) if halves["first_half"].size and halves["second_half"].size else float("nan")
    return {"odd_even_correlation": odd_even, "first_second_half_correlation": first_second}


def _peak_prominence(
    average: np.ndarray,
    times_ms: np.ndarray,
    peak: Mapping[str, object],
) -> dict[str, float]:
    if "sample_index" not in peak:
        return {"prominence_vs_baseline_sd": float("nan"), "prominence_vs_late_sd": float("nan")}
    baseline = average[_mask(times_ms, (-100.0, 0.0))]
    late = average[_mask(times_ms, (300.0, 400.0))]
    amplitude = abs(float(peak["amplitude_uv"]) - float(np.mean(baseline)))
    baseline_sd = float(np.std(baseline, ddof=1)) if baseline.size > 1 else float("nan")
    late_sd = float(np.std(late, ddof=1)) if late.size > 1 else float("nan")
    return {
        "prominence_vs_baseline_sd": amplitude / baseline_sd if baseline_sd > 0 else float("nan"),
        "prominence_vs_late_sd": amplitude / late_sd if late_sd > 0 else float("nan"),
    }


def bootstrap_peak_confidence(
    epochs: np.ndarray,
    times_ms: np.ndarray,
    definition: PeakDefinition,
    n_bootstrap: int = 1000,
    seed: int = 20260826,
) -> dict[str, object]:
    if len(epochs) < 2:
        return {
            "bootstrap_latency_ci_low_ms": float("nan"),
            "bootstrap_latency_ci_high_ms": float("nan"),
            "bootstrap_amplitude_ci_low_uv": float("nan"),
            "bootstrap_amplitude_ci_high_uv": float("nan"),
            "bootstrap_peak_detection_frequency": float("nan"),
        }
    rng = np.random.default_rng(seed)
    latencies: list[float] = []
    amplitudes: list[float] = []
    detections = 0
    baseline_mask = _mask(times_ms, (-100.0, 0.0))
    for _ in range(n_bootstrap):
        sample = epochs[rng.integers(0, len(epochs), size=len(epochs))]
        average = np.mean(sample, axis=0)
        peak = find_candidate_peak(average, times_ms, definition)
        if "latency_ms" not in peak:
            continue
        baseline_sd = float(np.std(average[baseline_mask], ddof=1))
        prominence = abs(float(peak["amplitude_uv"]) - float(np.mean(average[baseline_mask])))
        if baseline_sd > 0 and prominence >= 2 * baseline_sd:
            detections += 1
        latencies.append(float(peak["latency_ms"]))
        amplitudes.append(float(peak["amplitude_uv"]))
    return {
        "bootstrap_latency_ci_low_ms": float(np.percentile(latencies, 2.5)) if latencies else float("nan"),
        "bootstrap_latency_ci_high_ms": float(np.percentile(latencies, 97.5)) if latencies else float("nan"),
        "bootstrap_amplitude_ci_low_uv": float(np.percentile(amplitudes, 2.5)) if amplitudes else float("nan"),
        "bootstrap_amplitude_ci_high_uv": float(np.percentile(amplitudes, 97.5)) if amplitudes else float("nan"),
        "bootstrap_peak_detection_frequency": detections / n_bootstrap,
    }


def leave_one_out_dominance(
    epochs: np.ndarray,
    times_ms: np.ndarray,
    definition: PeakDefinition,
) -> dict[str, float]:
    if len(epochs) < 3:
        return {"maximum_leave_one_out_peak_change_fraction": float("nan"), "dominant_trial_index": float("nan")}
    full = find_candidate_peak(np.mean(epochs, axis=0), times_ms, definition)
    if "amplitude_uv" not in full:
        return {"maximum_leave_one_out_peak_change_fraction": float("nan"), "dominant_trial_index": float("nan")}
    full_amp = float(full["amplitude_uv"])
    denominator = max(abs(full_amp), np.finfo(float).eps)
    changes = []
    total = np.sum(epochs, axis=0)
    for index in range(len(epochs)):
        average = (total - epochs[index]) / (len(epochs) - 1)
        peak = find_candidate_peak(average, times_ms, definition)
        changes.append(abs(float(peak.get("amplitude_uv", np.nan)) - full_amp) / denominator)
    maximum_index = int(np.nanargmax(changes))
    return {
        "maximum_leave_one_out_peak_change_fraction": float(changes[maximum_index]),
        "dominant_trial_index": maximum_index + 1,
    }


def bootstrap_waveform_confidence(
    epochs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 20260826,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile interval for the accepted-trial mean waveform."""
    if len(epochs) < 2:
        empty = np.full(epochs.shape[-1], np.nan)
        return empty, empty.copy()
    rng = np.random.default_rng(seed)
    means = np.empty((n_bootstrap, epochs.shape[-1]), dtype=float)
    for start in range(0, n_bootstrap, 100):
        count = min(100, n_bootstrap - start)
        indices = rng.integers(0, len(epochs), size=(count, len(epochs)))
        means[start : start + count] = np.mean(epochs[indices], axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def analyze_vep_epochs(
    accepted_epochs_uv: np.ndarray,
    times_seconds: np.ndarray,
    definitions: Sequence[PeakDefinition] = NEUTRAL_PEAKS,
    n_bootstrap: int = 1000,
    seed: int = 20260826,
) -> dict[str, object]:
    """Analyze already-frozen accepted epochs; does not create an acceptance mask."""
    times_ms = np.asarray(times_seconds) * 1000
    if not len(accepted_epochs_uv):
        return {"analysis_status": "UNRESOLVED", "reason": "No accepted epochs"}
    corrected = baseline_correct(np.asarray(accepted_epochs_uv, dtype=float), times_seconds)
    average = np.mean(corrected, axis=0)
    waveform_ci_low, waveform_ci_high = bootstrap_waveform_confidence(corrected, n_bootstrap, seed)
    halves = split_half_waveforms(corrected)
    reliability = split_half_reliability(corrected, times_ms)
    peak_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for offset, definition in enumerate(definitions):
        peak = find_candidate_peak(average, times_ms, definition)
        peak.update(_peak_prominence(average, times_ms, peak))
        peak.update(bootstrap_peak_confidence(corrected, times_ms, definition, n_bootstrap, seed + offset))
        peak.update(leave_one_out_dominance(corrected, times_ms, definition))
        ci_width = float(peak.get("bootstrap_latency_ci_high_ms", np.nan)) - float(peak.get("bootstrap_latency_ci_low_ms", np.nan))
        peak["bootstrap_latency_ci_width_ms"] = ci_width
        peak["manual_review_flags"] = []
        if peak.get("at_search_window_boundary"):
            peak["manual_review_flags"].append("peak at search-window boundary")
        if np.isfinite(ci_width) and ci_width > 30:
            peak["manual_review_flags"].append("unstable bootstrap latency")
        if float(peak.get("bootstrap_peak_detection_frequency", 0)) < 0.70:
            peak["manual_review_flags"].append("low bootstrap peak-detection frequency")
        if float(peak.get("maximum_leave_one_out_peak_change_fraction", 0)) > 0.25:
            peak["manual_review_flags"].append("average may be dominated by a small number of trials")
        if peak["manual_review_flags"]:
            warnings.extend(peak["manual_review_flags"])
            peak["status"] = "WARNING"
        peak_rows.append(peak)
    if any(np.isfinite(value) and value < 0.3 for value in reliability.values()):
        warnings.append("odd/even or split-half averages disagree")
    return {
        "analysis_status": "WARNING" if warnings else "PASS",
        "average_uv": average,
        "bootstrap_waveform_ci_low_uv": waveform_ci_low,
        "bootstrap_waveform_ci_high_uv": waveform_ci_high,
        "baseline_corrected_epochs_uv": corrected,
        "odd_average_uv": halves["odd"],
        "even_average_uv": halves["even"],
        "first_half_average_uv": halves["first_half"],
        "second_half_average_uv": halves["second_half"],
        "peak_rows": peak_rows,
        "warnings": sorted(set(warnings)),
        **reliability,
    }


def filter_sensitivity(
    raw_epochs_uv: np.ndarray,
    fs: float,
    times_seconds: np.ndarray,
    definition: PeakDefinition = NEUTRAL_PEAKS[1],
    bands: Sequence[tuple[float, float]] = ((0.5, 30.0), (1.0, 20.0), (0.5, 20.0)),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    times_ms = times_seconds * 1000
    for low, high in bands:
        thresholds = QCThresholds(review_low_hz=low, review_high_hz=high)
        filtered = zero_phase_review_filter(raw_epochs_uv, fs, thresholds)
        corrected = baseline_correct(filtered, times_seconds)
        peak = find_candidate_peak(np.mean(corrected, axis=0), times_ms, definition)
        rows.append({"low_hz": low, "high_hz": high, **peak})
    valid_latencies = [float(row["latency_ms"]) for row in rows if "latency_ms" in row]
    spread = max(valid_latencies) - min(valid_latencies) if valid_latencies else float("nan")
    for row in rows:
        row["latency_spread_across_filters_ms"] = spread
        row["sensitivity_status"] = "WARNING" if np.isfinite(spread) and spread > 15 else "PASS"
    return rows


def epoch_selection_sensitivity(
    epochs_uv: np.ndarray,
    times_seconds: np.ndarray,
    definition: PeakDefinition = NEUTRAL_PEAKS[1],
) -> list[dict[str, object]]:
    """Compare a peak across deterministic subsets of frozen accepted epochs.

    This never selects trials by their waveform and never creates a rejection
    mask; it only describes stability after the human mask has been frozen.
    """
    subsets = {
        "all accepted": np.arange(len(epochs_uv)),
        "odd accepted": np.arange(0, len(epochs_uv), 2),
        "even accepted": np.arange(1, len(epochs_uv), 2),
        "first half": np.arange(0, len(epochs_uv) // 2),
        "second half": np.arange(len(epochs_uv) // 2, len(epochs_uv)),
    }
    times_ms = np.asarray(times_seconds) * 1000
    rows: list[dict[str, object]] = []
    for label, indices in subsets.items():
        if indices.size == 0:
            rows.append({"epoch_subset": label, "trial_count": 0, "status": "UNRESOLVED"})
            continue
        corrected = baseline_correct(np.asarray(epochs_uv)[indices], times_seconds)
        peak = find_candidate_peak(np.mean(corrected, axis=0), times_ms, definition)
        rows.append({"epoch_subset": label, "trial_count": int(indices.size), **peak})
    valid_latencies = [float(row["latency_ms"]) for row in rows if "latency_ms" in row]
    spread = max(valid_latencies) - min(valid_latencies) if valid_latencies else float("nan")
    for row in rows:
        row["latency_spread_across_subsets_ms"] = spread
        row["sensitivity_status"] = (
            "UNRESOLVED" if row.get("status") == "UNRESOLVED"
            else "WARNING" if np.isfinite(spread) and spread > 15
            else "PASS"
        )
    return rows


def paired_waveforms(
    teeb_epochs_uv: np.ndarray,
    eeg_epochs_uv: np.ndarray,
    common_mask: np.ndarray,
    times_seconds: np.ndarray,
) -> dict[str, np.ndarray | float]:
    if len(teeb_epochs_uv) != len(eeg_epochs_uv) or len(common_mask) != len(teeb_epochs_uv):
        raise ValueError("Paired inputs and common mask must have equal trial counts")
    if not np.any(common_mask):
        return {"status": "UNRESOLVED"}
    t = np.mean(baseline_correct(teeb_epochs_uv[common_mask], times_seconds), axis=0)
    e = np.mean(baseline_correct(eeg_epochs_uv[common_mask], times_seconds), axis=0)
    window = _mask(times_seconds * 1000, (0.0, 300.0))
    t_scale = np.max(np.abs(t[window]))
    e_scale = np.max(np.abs(e[window]))
    t_normalized = t / t_scale if t_scale > 0 else np.full_like(t, np.nan)
    e_normalized = e / e_scale if e_scale > 0 else np.full_like(e, np.nan)
    return {
        "status": "DESCRIPTIVE",
        "common_trial_count": int(np.sum(common_mask)),
        "teeg_average_uv": t,
        "eeg_average_uv": e,
        "teeg_normalized": t_normalized,
        "eeg_normalized": e_normalized,
        "difference_uv": t - e,
        "normalized_difference": t_normalized - e_normalized,
        "normalized_waveform_correlation": _safe_correlation(t_normalized[window], e_normalized[window]),
        "interpretation": "Similarity is descriptive only; low similarity is not a technical failure because spatial derivations differ.",
    }


def power_spectrum(data_uv: np.ndarray, fs: float, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    import mne

    n = min(nperseg, data_uv.size)
    psd, frequencies = mne.time_frequency.psd_array_welch(
        np.asarray(data_uv, dtype=float),
        sfreq=fs,
        fmin=0,
        fmax=fs / 2,
        n_fft=max(nperseg, n),
        n_per_seg=n,
        n_overlap=0,
        average="mean",
        window="hamming",
        remove_dc=True,
        output="power",
        verbose="ERROR",
    )
    return frequencies, psd


def bandpower(frequencies: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies <= high)
    return float(np.trapezoid(psd[mask], frequencies[mask])) if np.sum(mask) >= 2 else float("nan")


def alpha_metrics(
    filtered_uv: np.ndarray,
    fs: float,
    intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, object]:
    """Alpha physiology from the 0.5--30-Hz branch; callable only post-freeze."""
    result: dict[str, object] = {}
    spectra = {}
    for condition in ("eyes_open", "eyes_closed"):
        pieces = [filtered_uv[start:end] for start, end in intervals.get(condition, []) if end > start]
        if not pieces:
            return {"status": "UNRESOLVED", "reason": f"No {condition} interval"}
        frequencies, psd = power_spectrum(np.concatenate(pieces), fs)
        spectra[condition] = psd
        result[f"{condition}_alpha_power_uv2"] = bandpower(frequencies, psd, 8, 13)
    open_power = float(result["eyes_open_alpha_power_uv2"])
    closed_power = float(result["eyes_closed_alpha_power_uv2"])
    closed_psd = spectra["eyes_closed"]
    alpha_mask = (frequencies >= 8) & (frequencies <= 13)
    peak_index = np.flatnonzero(alpha_mask)[int(np.argmax(closed_psd[alpha_mask]))]
    peak_hz = float(frequencies[peak_index])
    peak_power = bandpower(frequencies, closed_psd, peak_hz - 0.5, peak_hz + 0.5)
    neighbor = np.nanmean(
        [
            bandpower(frequencies, closed_psd, peak_hz - 2.5, peak_hz - 1.0) / 1.5,
            bandpower(frequencies, closed_psd, peak_hz + 1.0, peak_hz + 2.5) / 1.5,
        ]
    )
    result.update(
        {
            "status": "DESCRIPTIVE",
            "alpha_reactivity_closed_open": closed_power / open_power if open_power > 0 else float("nan"),
            "alpha_peak_frequency_hz": peak_hz,
            "alpha_peak_snr_db": float(10 * np.log10(peak_power / neighbor)) if peak_power > 0 and neighbor > 0 else float("nan"),
            "frequencies_hz": frequencies,
            "eyes_open_psd_uv2_hz": spectra["eyes_open"],
            "eyes_closed_psd_uv2_hz": spectra["eyes_closed"],
        }
    )
    return result


def peak_rows_for_export(review_id: str, session: str, site: str, signal_type: str, analysis: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for peak in analysis.get("peak_rows", []):
        rows.append(
            {
                "review_id": review_id,
                "session": session,
                "site": site,
                "signal_type": signal_type,
                **{key: value for key, value in dict(peak).items() if key != "sample_index"},
            }
        )
    return rows


def definitions_as_dicts() -> list[dict[str, object]]:
    return [asdict(definition) for definition in NEUTRAL_PEAKS]
