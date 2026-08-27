"""Authoritative MNE-Python backend for BrainVision review and analysis.

All signal samples, channel names, sampling metadata, annotations, events,
filtering, epoching, and spectra used by the GUI originate from
``mne.io.read_raw_brainvision``.  The separate low-level parser is used only
for cryptographic integrity fields and exact ADC integer-rail verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mne
import numpy as np

try:
    from .data_integrity import RecordingInfo, parse_vhdr
    from .qc_engine import EpochSet, QCThresholds
except ImportError:
    from data_integrity import RecordingInfo, parse_vhdr
    from qc_engine import EpochSet, QCThresholds


CHECKERBOARD_ANNOTATION = "Stimulus/S  7"
CHECKERBOARD_EVENT_ID = 7
OPEN_ANNOTATIONS = {"Comment/open", "Comment/eyes open", "Comment/eo"}
CLOSED_ANNOTATIONS = {"Comment/close", "Comment/closed", "Comment/eyes closed", "Comment/ec", "Comment/ce"}


@dataclass(frozen=True)
class MNERecording:
    raw: mne.io.BaseRaw
    source_info: RecordingInfo
    events: np.ndarray
    event_id: dict[str, int]
    checkerboard_events: np.ndarray


def load_brainvision(header_path: Path, preload: bool = False) -> MNERecording:
    """Read a BrainVision record with MNE and cross-check linked-file metadata."""
    source_info = parse_vhdr(Path(header_path))
    raw = mne.io.read_raw_brainvision(
        Path(header_path),
        preload=preload,
        eog=(),
        misc=(),
        scale=1.0,
        ignore_marker_types=False,
        verbose="ERROR",
    )
    if raw.n_times != source_info.n_samples:
        raise ValueError(f"MNE sample count differs from linked-file audit: {raw.n_times} != {source_info.n_samples}")
    if not np.isclose(raw.info["sfreq"], source_info.sampling_rate_hz):
        raise ValueError(f"MNE sampling rate differs from header audit: {raw.info['sfreq']} != {source_info.sampling_rate_hz}")
    if tuple(raw.ch_names) != source_info.channel_labels:
        raise ValueError(f"MNE channel names differ from BrainVision header: {raw.ch_names} != {source_info.channel_labels}")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    checkerboard = events[events[:, 2] == event_id.get(CHECKERBOARD_ANNOTATION, -1)]
    # BrainVision's numeric annotation parser should preserve S7 as event 7.
    if CHECKERBOARD_ANNOTATION in event_id and event_id[CHECKERBOARD_ANNOTATION] != CHECKERBOARD_EVENT_ID:
        raise ValueError(f"Unexpected MNE S7 event mapping: {event_id[CHECKERBOARD_ANNOTATION]}")
    return MNERecording(raw, source_info, events, dict(event_id), checkerboard)


def recording_metadata(recording: MNERecording) -> dict[str, object]:
    raw = recording.raw
    return {
        "mne_reader": "mne.io.read_raw_brainvision",
        "mne_class": type(raw).__name__,
        "channel_names": list(raw.ch_names),
        "channel_types": raw.get_channel_types(),
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "sample_count": int(raw.n_times),
        "duration_seconds": float(raw.times[-1] + 1 / raw.info["sfreq"]),
        "measurement_date": str(raw.info.get("meas_date") or "unavailable"),
        "highpass_header_hz": float(raw.info["highpass"]),
        "lowpass_header_hz": float(raw.info["lowpass"]),
        "line_frequency_header_hz": raw.info.get("line_freq"),
        "original_units": dict(getattr(raw, "_orig_units", {})),
        "annotation_count": len(raw.annotations),
        "annotation_descriptions": sorted(set(map(str, raw.annotations.description))),
        "event_id": recording.event_id,
        "checkerboard_event_count": int(len(recording.checkerboard_events)),
    }


def raw_microvolts(raw: mne.io.BaseRaw, picks: Sequence[int | str] | None = None) -> np.ndarray:
    """Get MNE-loaded EEG samples in µV without changing the Raw object."""
    return raw.get_data(picks=picks, units="uV")


def checkerboard_samples(recording: MNERecording) -> np.ndarray:
    return np.asarray(recording.checkerboard_events[:, 0], dtype=int)


def review_copy(raw: mne.io.BaseRaw, thresholds: QCThresholds, picks: Sequence[int | str] | None = None) -> mne.io.BaseRaw:
    """Create the manuscript-compatible zero-phase 0.5--30-Hz MNE copy."""
    thresholds.validate(float(raw.info["sfreq"]))
    copy = raw.copy().pick(picks).load_data() if picks is not None else raw.copy().load_data()
    copy.filter(
        l_freq=thresholds.review_low_hz,
        h_freq=thresholds.review_high_hz,
        method="iir",
        iir_params={"order": thresholds.filter_order, "ftype": "butter", "output": "sos"},
        phase="zero",
        picks="eeg",
        verbose="ERROR",
    )
    return copy


def checkerboard_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    thresholds: QCThresholds,
    picks: Sequence[int | str] | None = None,
    baseline_correct: bool = False,
) -> mne.Epochs:
    """Build 500-sample −100 to +400-ms-exclusive MNE epochs."""
    fs = float(raw.info["sfreq"])
    tmax_inclusive = thresholds.epoch_tmax_seconds - 1 / fs
    baseline = (thresholds.epoch_tmin_seconds, -1 / fs) if baseline_correct else None
    return mne.Epochs(
        raw,
        events=events,
        event_id={"checkerboard": CHECKERBOARD_EVENT_ID},
        tmin=thresholds.epoch_tmin_seconds,
        tmax=tmax_inclusive,
        baseline=baseline,
        picks=picks,
        preload=True,
        reject=None,
        flat=None,
        proj=False,
        reject_by_annotation=False,
        event_repeated="error",
        verbose="ERROR",
    )


def complete_and_incomplete_event_numbers(epochs: mne.Epochs, total_events: int) -> tuple[np.ndarray, np.ndarray]:
    complete_zero_based = np.asarray(epochs.selection, dtype=int)
    complete = complete_zero_based + 1
    all_numbers = np.arange(1, total_events + 1)
    incomplete = np.setdiff1d(all_numbers, complete)
    return complete, incomplete


def epoch_set_microvolts(epochs: mne.Epochs, total_events: int) -> EpochSet:
    complete, incomplete = complete_and_incomplete_event_numbers(epochs, total_events)
    return EpochSet(
        epochs=epochs.get_data(units="uV"),
        event_numbers=complete,
        event_samples=np.asarray(epochs.events[:, 0], dtype=int),
        incomplete_event_numbers=incomplete,
        times_seconds=np.asarray(epochs.times, dtype=float),
    )


def condition_intervals(raw: mne.io.BaseRaw, maximum_seconds: float = 30.0) -> dict[str, list[tuple[int, int]]]:
    """Derive EO/EC intervals from MNE-preserved annotations."""
    fs = float(raw.info["sfreq"])
    starts: list[tuple[str, int]] = []
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        normalized = str(description).strip().casefold()
        sample = int(raw.time_as_index(float(onset), use_rounding=True)[0])
        if normalized in {value.casefold() for value in OPEN_ANNOTATIONS} or "eyes open" in normalized:
            starts.append(("eyes_open", sample))
        elif normalized in {value.casefold() for value in CLOSED_ANNOTATIONS} or "eyes close" in normalized:
            starts.append(("eyes_closed", sample))
    starts.sort(key=lambda row: row[1])
    intervals: dict[str, list[tuple[int, int]]] = {"eyes_open": [], "eyes_closed": []}
    maximum = int(round(maximum_seconds * fs))
    for index, (condition, start) in enumerate(starts):
        next_start = starts[index + 1][1] if index + 1 < len(starts) else raw.n_times
        end = min(raw.n_times, start + maximum, next_start)
        if end > start:
            intervals[condition].append((start, end))
    return intervals


def psd_welch_microvolts(
    raw: mne.io.BaseRaw,
    picks: Sequence[int | str],
    *,
    fmin: float = 0.0,
    fmax: float | None = None,
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """MNE Welch PSD in µV²/Hz (channel × frequency)."""
    if fmax is None:
        fmax = float(raw.info["sfreq"]) / 2
    spectrum = raw.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks=picks,
        n_fft=min(n_fft, raw.n_times),
        n_per_seg=min(n_fft, raw.n_times),
        window="hann",
        remove_dc=True,
        verbose="ERROR",
    )
    return spectrum.freqs, spectrum.get_data() * 1e12


def concatenate_intervals_raw(
    raw: mne.io.BaseRaw,
    intervals: Sequence[tuple[int, int]],
    picks: Sequence[int | str],
) -> np.ndarray:
    pieces = [raw.get_data(picks=picks, start=start, stop=end, units="uV") for start, end in intervals if end > start]
    return np.concatenate(pieces, axis=-1) if pieces else np.empty((len(picks), 0), dtype=float)


def native_browser(
    recording: MNERecording,
    *,
    picks: Sequence[int | str] | None = None,
    start: float = 0.0,
    duration: float = 20.0,
    block: bool = True,
):
    """Open MNE's native raw browser with preserved annotations and S7 events."""
    return recording.raw.plot(
        events=recording.checkerboard_events,
        event_id={"S7 checkerboard reversal": CHECKERBOARD_EVENT_ID},
        picks=picks,
        start=start,
        duration=duration,
        n_channels=len(picks) if picks is not None else min(11, len(recording.raw.ch_names)),
        scalings={"eeg": 500e-6},
        clipping="transparent",
        show=True,
        block=block,
        title="Read-only MNE BrainVision review (annotations shown)",
    )


def annotation_table(raw: mne.io.BaseRaw) -> list[dict[str, object]]:
    return [
        {"onset_seconds": float(onset), "duration_seconds": float(duration), "description": str(description)}
        for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
    ]


def event_table(recording: MNERecording) -> list[dict[str, object]]:
    reverse = {value: key for key, value in recording.event_id.items()}
    fs = float(recording.raw.info["sfreq"])
    return [
        {
            "sample": int(row[0]),
            "time_seconds": float(row[0] / fs),
            "event_code": int(row[2]),
            "annotation": reverse.get(int(row[2]), "UNRESOLVED"),
        }
        for row in recording.events
    ]


def validate_preservation(recording: MNERecording, expected: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return explicit MNE preservation checks for tests and reports."""
    metadata = recording_metadata(recording)
    checks = {
        "sample_count_preserved": bool(recording.raw.n_times == recording.source_info.n_samples),
        "sampling_rate_preserved": bool(np.isclose(recording.raw.info["sfreq"], recording.source_info.sampling_rate_hz)),
        "channel_names_preserved": bool(tuple(recording.raw.ch_names) == recording.source_info.channel_labels),
        "annotations_preserved": bool(len(recording.raw.annotations) > 0),
        "checkerboard_event_code_preserved": bool(recording.event_id.get(CHECKERBOARD_ANNOTATION) == CHECKERBOARD_EVENT_ID),
    }
    if expected:
        checks.update({f"expected_{key}": metadata.get(key) == value for key, value in expected.items()})
    return {**checks, "all_preserved": all(checks.values())}
