"""Deterministic, outcome-blind EEG technical quality-control calculations.

This module never decides whether a recording or epoch should be accepted.  It
returns PASS/WARNING/FAIL/UNRESOLVED findings with traceable measurements so a
reviewer can make and document that decision.  VEP averages, alpha results, and
cross-derivation correlations intentionally live outside this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from .data_integrity import RecordingInfo, SUPPORTED_DTYPES
except ImportError:  # Streamlit can execute app.py as a script.
    from data_integrity import RecordingInfo, SUPPORTED_DTYPES


STATUS_ORDER = {"PASS": 0, "UNRESOLVED": 1, "WARNING": 2, "FAIL": 3}
SITE_DECISIONS = ("GOOD", "GOOD_WITH_BAD_SEGMENTS", "BAD_SITE", "UNCERTAIN")
EPOCH_DECISIONS = ("KEEP", "REJECT")
EPOCH_REASONS = (
    "",
    "clipping",
    "flatline",
    "electrode pop",
    "movement/EMG",
    "ocular artifact",
    "invalid event",
    "incomplete epoch",
)


@dataclass(frozen=True)
class QCThresholds:
    """Configurable technical thresholds, expressed in recorded-output units.

    Defaults are transparent starting values, not claims of biological
    normality.  Project validation may revise them without changing the metric
    implementations.  The legacy 5% clipping criterion remains visible as a
    provisional compatibility gate while plateau-based saturation is also
    reported.
    """

    review_low_hz: float = 0.5
    review_high_hz: float = 30.0
    filter_order: int = 4
    epoch_tmin_seconds: float = -0.100
    epoch_tmax_seconds: float = 0.400
    rest_max_seconds: float = 30.0
    clip_warning_fraction: float = 0.0001
    clip_fail_fraction: float = 0.05
    saturation_plateau_ms: float = 20.0
    repeated_rail_samples_warning: int = 3
    flat_window_seconds: float = 2.0
    flat_std_uv: float = 0.5
    flat_fail_fraction: float = 0.05
    line_noise_warning_db: float = 6.0
    line_noise_fail_db: float = 12.0
    step_min_uv: float = 100.0
    step_robust_z: float = 12.0
    step_fail_per_minute: float = 10.0
    drift_warning_ratio: float = 0.5
    drift_fail_ratio: float = 1.0
    muscle_warning_ratio: float = 0.5
    muscle_fail_ratio: float = 1.0
    unusual_amplitude_warning_uv: float = 1000.0
    unusual_amplitude_fail_uv: float = 3000.0
    marker_iei_robust_z: float = 6.0
    marker_duplicate_fraction: float = 0.25
    epoch_p2p_robust_z: float = 6.0
    epoch_high_frequency_ratio: float = 1.0
    epoch_slow_amplitude_uv: float = 250.0

    def validate(self, fs: float | None = None) -> None:
        if not 0 <= self.clip_warning_fraction <= self.clip_fail_fraction <= 1:
            raise ValueError("Clipping fractions must satisfy 0 <= warning <= fail <= 1")
        if not self.epoch_tmin_seconds < 0 < self.epoch_tmax_seconds:
            raise ValueError("Epoch must include pre- and post-event samples")
        if self.review_low_hz <= 0 or self.review_low_hz >= self.review_high_hz:
            raise ValueError("Invalid review passband")
        if fs is not None and self.review_high_hz >= fs / 2:
            raise ValueError("Review high cutoff must be below Nyquist")

    @classmethod
    def from_json(cls, path: Path) -> "QCThresholds":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class QCFinding:
    rule_id: str
    concept: str
    status: str
    measured_value: float | int | str | None
    threshold: str
    channel: str
    start_seconds: float | None
    end_seconds: float | None
    supporting_plot: str
    explanation: str
    intervals: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["intervals"] = json.dumps(self.intervals)
        return row


@dataclass(frozen=True)
class EpochSet:
    epochs: np.ndarray
    event_numbers: np.ndarray
    event_samples: np.ndarray
    incomplete_event_numbers: np.ndarray
    times_seconds: np.ndarray


def worst_status(findings: Iterable[QCFinding]) -> str:
    values = list(findings)
    return max(values, key=lambda finding: STATUS_ORDER[finding.status]).status if values else "UNRESOLVED"


def zero_phase_review_filter(data_uv: np.ndarray, fs: float, thresholds: QCThresholds) -> np.ndarray:
    """Create a non-destructive 0.5--30 Hz review copy with MNE filtering."""
    import mne  # Lazy import keeps metadata-only operations lightweight.

    thresholds.validate(fs)
    return mne.filter.filter_data(
        np.asarray(data_uv, dtype=float),
        sfreq=fs,
        l_freq=thresholds.review_low_hz,
        h_freq=thresholds.review_high_hz,
        method="iir",
        iir_params={"order": thresholds.filter_order, "ftype": "butter", "output": "sos"},
        phase="zero",
        copy=True,
        verbose="ERROR",
    )


def make_mne_review_raw(data_uv: np.ndarray, fs: float, channel_names: Sequence[str]):
    """Wrap a review copy in MNE without modifying the caller's array.

    MNE stores EEG in volts.  It is imported lazily so the metric test suite can
    run in minimal environments; the GUI requirements install it.
    """
    import mne  # type: ignore

    info = mne.create_info(list(channel_names), sfreq=fs, ch_types=["eeg"] * len(channel_names))
    return mne.io.RawArray(np.array(data_uv, dtype=float, copy=True) * 1e-6, info, verbose="ERROR")


def epoch_data(data: np.ndarray, event_samples: Sequence[int], fs: float, thresholds: QCThresholds) -> EpochSet:
    pre = int(round(-thresholds.epoch_tmin_seconds * fs))
    post = int(round(thresholds.epoch_tmax_seconds * fs))
    complete: list[np.ndarray] = []
    numbers: list[int] = []
    samples: list[int] = []
    incomplete: list[int] = []
    for number, event in enumerate(np.asarray(event_samples, dtype=int), start=1):
        start = int(event) - pre
        end = int(event) + post
        if start < 0 or end > data.shape[-1]:
            incomplete.append(number)
            continue
        complete.append(np.asarray(data[..., start:end]))
        numbers.append(number)
        samples.append(int(event))
    shape = (0,) + data.shape[:-1] + (pre + post,)
    array = np.asarray(complete) if complete else np.empty(shape, dtype=data.dtype)
    times = np.arange(-pre, post) / fs
    return EpochSet(
        epochs=array,
        event_numbers=np.asarray(numbers, dtype=int),
        event_samples=np.asarray(samples, dtype=int),
        incomplete_event_numbers=np.asarray(incomplete, dtype=int),
        times_seconds=times,
    )


def baseline_correct(epochs: np.ndarray, times_seconds: np.ndarray) -> np.ndarray:
    baseline = times_seconds < 0
    if not np.any(baseline):
        raise ValueError("No prestimulus samples available for baseline correction")
    return epochs - np.mean(epochs[..., baseline], axis=-1, keepdims=True)


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return []
    edges = np.diff(values.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _intervals_from_sample_mask(
    mask: np.ndarray, fs: float, minimum_samples: int = 1, maximum_intervals: int = 200
) -> tuple[tuple[float, float], ...]:
    """Return a bounded jump list while metrics retain the full sample fraction."""
    regions = ((start / fs, end / fs) for start, end in contiguous_regions(mask) if end - start >= minimum_samples)
    result = []
    for interval in regions:
        result.append(interval)
        if len(result) >= maximum_intervals:
            break
    return tuple(result)


def rail_saturation_metrics(raw_counts: np.ndarray, dtype: np.dtype, fs: float, thresholds: QCThresholds) -> dict[str, object]:
    integer = np.issubdtype(dtype, np.integer)
    if not integer:
        return {
            "rail_fraction": float("nan"),
            "rail_samples": 0,
            "longest_rail_plateau_samples": 0,
            "rail_intervals": (),
            "status": "UNRESOLVED",
            "rail_mask": np.zeros(raw_counts.size, dtype=bool),
        }
    limits = np.iinfo(dtype)
    rail_mask = (raw_counts <= limits.min) | (raw_counts >= limits.max)
    regions = contiguous_regions(rail_mask)
    longest = max((end - start for start, end in regions), default=0)
    fraction = float(np.mean(rail_mask))
    if fraction >= thresholds.clip_fail_fraction or longest / fs * 1000 >= thresholds.saturation_plateau_ms:
        status = "FAIL"
    elif int(np.sum(rail_mask)) >= thresholds.repeated_rail_samples_warning or fraction >= thresholds.clip_warning_fraction:
        status = "WARNING"
    else:
        status = "PASS"
    return {
        "rail_fraction": fraction,
        "rail_samples": int(np.sum(rail_mask)),
        "longest_rail_plateau_samples": longest,
        "rail_intervals": _intervals_from_sample_mask(rail_mask, fs),
        "status": status,
        "rail_mask": rail_mask,
    }


def rolling_window_mask(
    data_uv: np.ndarray,
    fs: float,
    window_seconds: float,
    predicate,
) -> np.ndarray:
    window = max(2, int(round(window_seconds * fs)))
    mask = np.zeros(data_uv.size, dtype=bool)
    for start in range(0, data_uv.size, window):
        end = min(data_uv.size, start + window)
        if predicate(data_uv[start:end]):
            mask[start:end] = True
    return mask


def flatline_metrics(data_uv: np.ndarray, fs: float, thresholds: QCThresholds) -> dict[str, object]:
    mask = rolling_window_mask(
        np.asarray(data_uv, dtype=float),
        fs,
        thresholds.flat_window_seconds,
        lambda values: np.std(values) <= thresholds.flat_std_uv or np.ptp(values) == 0,
    )
    fraction = float(np.mean(mask))
    return {
        "flat_fraction": fraction,
        "flat_intervals": _intervals_from_sample_mask(mask, fs),
        "status": "FAIL" if fraction >= thresholds.flat_fail_fraction else ("WARNING" if np.any(mask) else "PASS"),
        "flat_mask": mask,
    }


def _bandpower(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq <= high)
    return float(np.trapezoid(psd[mask], freq[mask])) if np.sum(mask) >= 2 else float("nan")


def raw_spectrum(data_uv: np.ndarray, fs: float, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    import mne

    n = min(nperseg, data_uv.size)
    if n < 8:
        return np.array([]), np.array([])
    psd, frequencies = mne.time_frequency.psd_array_welch(
        np.asarray(data_uv, dtype=float),
        sfreq=fs,
        fmin=0,
        fmax=fs / 2,
        n_fft=max(nperseg, n),
        n_per_seg=n,
        n_overlap=0,
        average="mean",
        window="hann",
        remove_dc=True,
        output="power",
        verbose="ERROR",
    )
    return frequencies, psd


def line_noise_metrics(data_uv: np.ndarray, fs: float, thresholds: QCThresholds) -> dict[str, object]:
    """Measure 60-Hz interference and harmonics from the raw, pre-filter branch."""
    freq, psd = raw_spectrum(np.asarray(data_uv, dtype=float), fs)
    if not freq.size or fs / 2 <= 60.5:
        return {"status": "UNRESOLVED", "line_60hz_intrusion_db": float("nan"), "harmonics": {}}
    harmonics: dict[int, dict[str, float]] = {}
    for hz in range(60, int(fs / 2), 60):
        power = _bandpower(freq, psd, hz - 0.5, hz + 0.5)
        left = _bandpower(freq, psd, hz - 3, hz - 1) / 2
        right = _bandpower(freq, psd, hz + 1, hz + 3) / 2
        floor = float(np.nanmean([left, right]))
        intrusion = float(10 * np.log10(power / floor)) if power > 0 and floor > 0 else float("nan")
        harmonics[hz] = {"power_uv2": power, "intrusion_db": intrusion}
    primary = harmonics[60]["intrusion_db"]
    status = (
        "UNRESOLVED"
        if not np.isfinite(primary)
        else "FAIL"
        if primary >= thresholds.line_noise_fail_db
        else "WARNING"
        if primary >= thresholds.line_noise_warning_db
        else "PASS"
    )
    return {"status": status, "line_60hz_intrusion_db": primary, "harmonics": harmonics, "frequency_hz": freq, "psd_uv2_hz": psd}


def step_metrics(data_uv: np.ndarray, fs: float, thresholds: QCThresholds) -> dict[str, object]:
    difference = np.diff(np.asarray(data_uv, dtype=float), prepend=float(data_uv[0]))
    center = float(np.median(difference))
    mad = float(np.median(np.abs(difference - center)))
    robust_sigma = 1.4826 * mad
    adaptive = thresholds.step_robust_z * robust_sigma
    effective = max(thresholds.step_min_uv, adaptive)
    mask = np.abs(difference - center) >= effective
    rate = float(np.sum(mask) / (data_uv.size / fs / 60)) if data_uv.size else float("nan")
    status = "FAIL" if rate >= thresholds.step_fail_per_minute else ("WARNING" if np.any(mask) else "PASS")
    return {
        "status": status,
        "step_count": int(np.sum(mask)),
        "steps_per_minute": rate,
        "effective_step_threshold_uv": effective,
        "step_intervals": _intervals_from_sample_mask(mask, fs),
        "step_mask": mask,
    }


def spectral_ratio_metrics(data_uv: np.ndarray, fs: float, thresholds: QCThresholds) -> dict[str, object]:
    freq, psd = raw_spectrum(np.asarray(data_uv, dtype=float), fs)
    if not freq.size:
        return {"status": "UNRESOLVED"}
    reference = _bandpower(freq, psd, 1.0, 30.0)
    drift = _bandpower(freq, psd, 0.05, 0.5)
    muscle_high = min(100.0, fs / 2 - 1)
    muscle = _bandpower(freq, psd, 30.0, muscle_high) if muscle_high > 30 else float("nan")
    drift_ratio = drift / reference if reference > 0 else float("nan")
    muscle_ratio = muscle / reference if reference > 0 else float("nan")
    drift_status = _ratio_status(drift_ratio, thresholds.drift_warning_ratio, thresholds.drift_fail_ratio)
    muscle_status = _ratio_status(muscle_ratio, thresholds.muscle_warning_ratio, thresholds.muscle_fail_ratio)
    return {
        "status": max((drift_status, muscle_status), key=STATUS_ORDER.get),
        "drift_status": drift_status,
        "muscle_status": muscle_status,
        "slow_drift_to_1_30hz_ratio": drift_ratio,
        "muscle_30_100hz_to_1_30hz_ratio": muscle_ratio,
    }


def _ratio_status(value: float, warning: float, fail: float) -> str:
    if not np.isfinite(value):
        return "UNRESOLVED"
    return "FAIL" if value >= fail else ("WARNING" if value >= warning else "PASS")


def amplitude_metrics(data_uv: np.ndarray, fs: float, thresholds: QCThresholds) -> dict[str, object]:
    absolute = np.abs(np.asarray(data_uv, dtype=float))
    maximum = float(np.max(absolute)) if absolute.size else float("nan")
    mask = absolute >= thresholds.unusual_amplitude_warning_uv
    status = (
        "UNRESOLVED"
        if not np.isfinite(maximum)
        else "FAIL"
        if maximum >= thresholds.unusual_amplitude_fail_uv
        else "WARNING"
        if maximum >= thresholds.unusual_amplitude_warning_uv
        else "PASS"
    )
    return {"status": status, "maximum_absolute_uv": maximum, "large_amplitude_intervals": _intervals_from_sample_mask(mask, fs), "large_amplitude_mask": mask}


def marker_quality(events: Sequence[int], fs: float, thresholds: QCThresholds) -> dict[str, object]:
    values = np.asarray(events, dtype=int)
    if values.size < 2:
        return {"status": "UNRESOLVED", "invalid_event_numbers": list(range(1, len(values) + 1)), "reason": "fewer than two events"}
    differences = np.diff(values) / fs
    within = differences[differences <= 5.0]
    if within.size < 2:
        return {"status": "UNRESOLVED", "invalid_event_numbers": [], "reason": "insufficient within-block intervals"}
    median = float(np.median(within))
    mad = float(np.median(np.abs(within - median)))
    robust_sigma = 1.4826 * mad
    tolerance = max(thresholds.marker_duplicate_fraction * median, thresholds.marker_iei_robust_z * robust_sigma)
    invalid: set[int] = set()
    for index, interval in enumerate(differences):
        if interval > 5.0:  # expected block boundary
            continue
        if interval <= thresholds.marker_duplicate_fraction * median or abs(interval - median) > tolerance:
            invalid.update((index + 1, index + 2))
    status = "WARNING" if invalid else "PASS"
    return {
        "status": status,
        "median_iei_seconds": median,
        "minimum_iei_seconds": float(np.min(within)),
        "maximum_iei_seconds": float(np.max(within)),
        "invalid_event_numbers": sorted(invalid),
        "event_count": int(values.size),
    }


def assess_channel(
    raw_counts: np.ndarray,
    raw_uv: np.ndarray,
    channel_name: str,
    fs: float,
    dtype: np.dtype,
    thresholds: QCThresholds,
) -> tuple[list[QCFinding], dict[str, object]]:
    """Calculate technical recording metrics for one continuous channel."""
    rail = rail_saturation_metrics(raw_counts, dtype, fs, thresholds)
    flat = flatline_metrics(raw_uv, fs, thresholds)
    line = line_noise_metrics(raw_uv, fs, thresholds)
    steps = step_metrics(raw_uv, fs, thresholds)
    ratios = spectral_ratio_metrics(raw_uv, fs, thresholds)
    amplitude = amplitude_metrics(raw_uv, fs, thresholds)
    findings = [
        QCFinding(
            "ADC_RAIL_SATURATION", "technical_recording_quality", str(rail["status"]),
            float(rail["rail_fraction"]),
            f"WARNING at >= {thresholds.repeated_rail_samples_warning} rail samples or {thresholds.clip_warning_fraction:.4%}; "
            f"FAIL at >= {thresholds.clip_fail_fraction:.1%} or a >= {thresholds.saturation_plateau_ms:g}-ms rail plateau",
            channel_name,
            rail["rail_intervals"][0][0] if rail["rail_intervals"] else None,
            rail["rail_intervals"][0][1] if rail["rail_intervals"] else None,
            "raw trace at native ADC scale", "Repeated exact ADC endpoints indicate rail saturation; the legacy 5% gate is shown but is not the only criterion.",
            tuple(rail["rail_intervals"]),
        ),
        QCFinding(
            "FLATLINE", "technical_recording_quality", str(flat["status"]), float(flat["flat_fraction"]),
            f"{thresholds.flat_window_seconds:g}-s SD <= {thresholds.flat_std_uv:g} µV; FAIL if >= {thresholds.flat_fail_fraction:.1%} of samples",
            channel_name,
            flat["flat_intervals"][0][0] if flat["flat_intervals"] else None,
            flat["flat_intervals"][0][1] if flat["flat_intervals"] else None,
            "raw trace with flat intervals", "A flat or nearly flat channel can indicate disconnection, a short, or a stalled amplifier channel.",
            tuple(flat["flat_intervals"]),
        ),
        QCFinding(
            "LINE_NOISE_60HZ", "technical_recording_quality", str(line["status"]), float(line["line_60hz_intrusion_db"]),
            f"raw pre-notch 60-Hz/local-floor >= {thresholds.line_noise_warning_db:g} dB WARNING, >= {thresholds.line_noise_fail_db:g} dB FAIL",
            channel_name, None, None, "raw pre-notch PSD", "A narrow 60-Hz peak suggests mains interference. This is measured before notch or low-pass filtering.",
        ),
        QCFinding(
            "ABRUPT_STEP", "technical_recording_quality", str(steps["status"]), float(steps["steps_per_minute"]),
            f"sample step >= max({thresholds.step_min_uv:g} µV, {thresholds.step_robust_z:g} robust SD); FAIL at >= {thresholds.step_fail_per_minute:g}/min",
            channel_name,
            steps["step_intervals"][0][0] if steps["step_intervals"] else None,
            steps["step_intervals"][0][1] if steps["step_intervals"] else None,
            "raw trace around detected step", "Abrupt baseline steps are characteristic of electrode pops or sudden contact changes.",
            tuple(steps["step_intervals"]),
        ),
        QCFinding(
            "SLOW_DRIFT", "technical_recording_quality", str(ratios.get("drift_status", "UNRESOLVED")),
            ratios.get("slow_drift_to_1_30hz_ratio"),
            f"0.05-0.5 Hz / 1-30 Hz power >= {thresholds.drift_warning_ratio:g} WARNING, >= {thresholds.drift_fail_ratio:g} FAIL",
            channel_name, None, None, "raw PSD and long time-scale trace", "Excess very-slow power can reflect polarization, sweat, movement, or unstable contact.",
        ),
        QCFinding(
            "MUSCLE_HIGH_FREQUENCY", "technical_recording_quality", str(ratios.get("muscle_status", "UNRESOLVED")),
            ratios.get("muscle_30_100hz_to_1_30hz_ratio"),
            f"30-100 Hz / 1-30 Hz power >= {thresholds.muscle_warning_ratio:g} WARNING, >= {thresholds.muscle_fail_ratio:g} FAIL",
            channel_name, None, None, "raw PSD and short time-scale trace", "Broad high-frequency power can be muscle activity; this metric is not a physiological diagnosis.",
        ),
        QCFinding(
            "UNUSUALLY_LARGE_AMPLITUDE", "technical_recording_quality", str(amplitude["status"]), amplitude["maximum_absolute_uv"],
            f"absolute recorded output >= {thresholds.unusual_amplitude_warning_uv:g} µV WARNING, >= {thresholds.unusual_amplitude_fail_uv:g} µV FAIL",
            channel_name,
            amplitude["large_amplitude_intervals"][0][0] if amplitude["large_amplitude_intervals"] else None,
            amplitude["large_amplitude_intervals"][0][1] if amplitude["large_amplitude_intervals"] else None,
            "raw trace at fixed scale", "Very large output may be artifact or saturation; tEEG/eEEG gain differences must be considered.",
            tuple(amplitude["large_amplitude_intervals"]),
        ),
    ]
    contaminated_masks = [rail["rail_mask"], flat["flat_mask"], steps["step_mask"], amplitude["large_amplitude_mask"]]
    contaminated = np.logical_or.reduce(contaminated_masks) if contaminated_masks else np.zeros(raw_uv.size, dtype=bool)
    metrics = {
        "channel": channel_name,
        "automated_status": worst_status(findings),
        "contaminated_sample_fraction": float(np.mean(contaminated)),
        **{key: value for group in (rail, flat, steps, ratios, amplitude) for key, value in group.items() if not isinstance(value, np.ndarray)},
        "line_60hz_intrusion_db": line["line_60hz_intrusion_db"],
    }
    return findings, metrics


def epoch_artifact_suggestions(
    raw_epoch_counts: np.ndarray,
    filtered_epochs_uv: np.ndarray,
    dtype: np.dtype,
    times_seconds: np.ndarray,
    event_numbers: Sequence[int],
    invalid_event_numbers: Sequence[int],
    thresholds: QCThresholds,
) -> list[dict[str, object]]:
    """Return explainable suggestions; never convert these into final decisions."""
    if filtered_epochs_uv.ndim != 2:
        raise ValueError("Expected trial x time epochs for one channel")
    p2p = np.ptp(filtered_epochs_uv, axis=1)
    median = float(np.median(p2p)) if p2p.size else float("nan")
    mad = float(np.median(np.abs(p2p - median))) if p2p.size else float("nan")
    p2p_limit = median + thresholds.epoch_p2p_robust_z * 1.4826 * mad if mad > 0 else float("inf")
    integer_info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    baseline = times_seconds < 0
    post = times_seconds >= 0
    rows: list[dict[str, object]] = []
    for index, event_number in enumerate(event_numbers):
        reasons: list[str] = []
        evidence: list[str] = []
        if integer_info is not None and np.any((raw_epoch_counts[index] <= integer_info.min) | (raw_epoch_counts[index] >= integer_info.max)):
            reasons.append("clipping")
            evidence.append("raw epoch contains an exact ADC endpoint")
        if np.std(filtered_epochs_uv[index]) <= thresholds.flat_std_uv:
            reasons.append("flatline")
            evidence.append(f"epoch SD <= {thresholds.flat_std_uv:g} µV")
        difference = np.diff(filtered_epochs_uv[index])
        if difference.size and np.max(np.abs(difference)) >= thresholds.step_min_uv:
            reasons.append("electrode pop")
            evidence.append(f"sample-to-sample step >= {thresholds.step_min_uv:g} µV")
        if p2p[index] > p2p_limit:
            reasons.append("movement/EMG")
            evidence.append(f"epoch p-p {p2p[index]:.3g} µV > robust limit {p2p_limit:.3g} µV")
        if baseline.any() and post.any():
            slow_shift = abs(float(np.mean(filtered_epochs_uv[index, post]) - np.mean(filtered_epochs_uv[index, baseline])))
            if slow_shift >= thresholds.epoch_slow_amplitude_uv:
                reasons.append("ocular artifact")
                evidence.append(f"pre/post mean shift {slow_shift:.3g} µV; ocular origin is only a suggestion without EOG")
        if int(event_number) in set(map(int, invalid_event_numbers)):
            reasons.append("invalid event")
            evidence.append("inter-event timing is a robust outlier or probable duplicate")
        rows.append(
            {
                "event_number": int(event_number),
                "automated_status": "WARNING" if reasons else "PASS",
                "suggested_reasons": "|".join(dict.fromkeys(reasons)),
                "evidence": "; ".join(evidence) if evidence else "no configured technical epoch rule exceeded",
                "peak_to_peak_uv": float(p2p[index]),
                "robust_peak_to_peak_threshold_uv": p2p_limit,
            }
        )
    return rows


def common_accepted_event_mask(
    event_numbers: Sequence[int],
    teeb_decisions: Mapping[int, str],
    eeg_decisions: Mapping[int, str],
) -> np.ndarray:
    """Intersection of manually kept events for paired tEEG/eEEG analysis."""
    return np.asarray(
        [teeb_decisions.get(int(number)) == "KEEP" and eeg_decisions.get(int(number)) == "KEEP" for number in event_numbers],
        dtype=bool,
    )


def validate_epoch_decision(decision: str, reason: str) -> None:
    if decision not in EPOCH_DECISIONS:
        raise ValueError(f"Unknown epoch decision {decision!r}")
    if decision == "REJECT" and reason not in EPOCH_REASONS[1:]:
        raise ValueError("A rejection requires one of the predefined artifact reasons")
    if decision == "KEEP" and reason:
        raise ValueError("A kept epoch must not carry a rejection reason")


def validate_site_decision(decision: str) -> None:
    if decision not in SITE_DECISIONS:
        raise ValueError(f"Unknown site-session decision {decision!r}")


def load_thresholds(path: Path | None = None) -> QCThresholds:
    return QCThresholds.from_json(path) if path and path.exists() else QCThresholds()


def raw_dtype(info: RecordingInfo) -> np.dtype:
    return SUPPORTED_DTYPES[info.binary_format]
