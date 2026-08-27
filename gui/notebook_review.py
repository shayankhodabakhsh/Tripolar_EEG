"""Blinded Jupyter frontend for the MNE-based manual EEG review workflow.

This is deliberately a thin presentation layer. Signal loading, filtering,
epoching, annotations, event extraction, and technical metrics are delegated to
the same independently tested modules used by the Streamlit application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_integrity import (
    channel_map,
    integrity_record,
    load_manifest,
    load_raw_counts,
    marker_timing_metrics,
    resolve_review_recording,
)
from .export_reports import (
    append_epoch_decision,
    append_site_decision,
    completion_audit,
    initialize_exports,
    latest_decisions,
    read_csv,
    review_is_frozen,
)
from .mne_backend import (
    checkerboard_epochs,
    checkerboard_samples,
    condition_intervals,
    epoch_set_microvolts,
    load_brainvision,
    native_browser,
    raw_microvolts,
    review_copy,
    validate_preservation,
)
from .qc_engine import (
    EPOCH_REASONS,
    QCThresholds,
    assess_channel,
    epoch_artifact_suggestions,
    epoch_data,
    marker_quality,
    raw_dtype,
)


GUI_DIR = Path(__file__).resolve().parent
PUBLIC_MANIFEST = GUI_DIR / "review_data" / "blinded_review_manifest.csv"
PRIVATE_KEY = GUI_DIR / "private" / "blinding_key.csv"
OUTPUT_DIR = GUI_DIR / "review_output"
THRESHOLD_PATH = GUI_DIR / "qc_thresholds.json"
SITE_DECISIONS = ("GOOD", "GOOD_WITH_BAD_SEGMENTS", "BAD_SITE", "UNCERTAIN")


def _site_picks(site: int) -> tuple[int, int]:
    if site not in (1, 2, 3, 4):
        raise ValueError("site must be 1, 2, 3, or 4")
    return (site - 1) * 2, (site - 1) * 2 + 1


@dataclass
class NotebookReviewer:
    """Stateful, blinded review helper suitable for notebook cells or widgets."""

    public_manifest: Path = PUBLIC_MANIFEST
    private_key: Path = PRIVATE_KEY
    output_dir: Path = OUTPUT_DIR
    threshold_path: Path = THRESHOLD_PATH
    _recording_cache: dict[tuple[str, str], Any] = field(default_factory=dict, init=False, repr=False)
    _filtered_cache: dict[tuple[str, str], Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.public_manifest.exists() or not self.private_key.exists():
            raise FileNotFoundError("Run `python -m gui.prepare_review` before opening the notebook")
        self.manifest = load_manifest(self.public_manifest)
        self.thresholds = QCThresholds.from_json(self.threshold_path)
        self.paths = initialize_exports(self.output_dir)

    @property
    def review_ids(self) -> list[str]:
        return [row["review_id"] for row in self.manifest]

    def public_record(self, review_id: str) -> dict[str, str]:
        try:
            return next(row for row in self.manifest if row["review_id"] == review_id)
        except StopIteration as exc:
            raise ValueError(f"Unknown blinded review ID: {review_id}") from exc

    def _session(self, review_id: str) -> str:
        return self.public_record(review_id)["session"]

    def recording(self, review_id: str):
        session = self._session(review_id)
        key = (review_id, session)
        if key not in self._recording_cache:
            info = resolve_review_recording(review_id, session, self.private_key)
            self._recording_cache[key] = load_brainvision(info.header_path, preload=False)
        return self._recording_cache[key]

    def filtered(self, review_id: str):
        session = self._session(review_id)
        key = (review_id, session)
        if key not in self._filtered_cache:
            recording = self.recording(review_id)
            self._filtered_cache[key] = review_copy(
                recording.raw,
                self.thresholds,
                picks=range(min(8, len(recording.raw.ch_names))),
            )
        return self._filtered_cache[key]

    def _require_in_scope(self, review_id: str) -> None:
        if self.public_record(review_id).get("manual_review_scope") != "IN_SCOPE":
            raise ValueError("This record is integrity-only because its channel mapping or protocol is unresolved")

    def overview(self, review_id: str) -> dict[str, object]:
        """Return blinded acquisition and preservation facts, never private identity."""
        public = self.public_record(review_id)
        recording = self.recording(review_id)
        events = checkerboard_samples(recording)
        rests = condition_intervals(recording.raw, self.thresholds.rest_max_seconds)
        return {
            "review_id": review_id,
            "session": public["session"],
            "manual_review_scope": public["manual_review_scope"],
            "identity_status": public["identity_status"],
            "configuration": f"BLINDED ({public['configuration_status']})",
            "sample_count": int(recording.raw.n_times),
            "duration_seconds": float(recording.raw.n_times / recording.raw.info["sfreq"]),
            "sampling_rate_hz": float(recording.raw.info["sfreq"]),
            "channel_names": list(recording.raw.ch_names),
            "checkerboard_event_count": int(len(events)),
            "eyes_open_segments": len(rests["eyes_open"]),
            "eyes_closed_segments": len(rests["eyes_closed"]),
            "mne_preservation": validate_preservation(recording),
            "review_frozen": review_is_frozen(self.output_dir),
        }

    def integrity(self, review_id: str) -> dict[str, object]:
        """Return blinded hashes, headers, marker timing, and verified mapping."""
        recording = self.recording(review_id)
        result = integrity_record(recording.source_info)
        result["review_id"] = review_id
        result["session"] = self._session(review_id)
        result["montage"] = channel_map(recording.source_info.n_channels)
        result["configuration"] = "BLINDED"
        return result

    def event_timing(self, review_id: str) -> tuple[pd.DataFrame, plt.Figure]:
        recording = self.recording(review_id)
        events = checkerboard_samples(recording)
        fs = float(recording.raw.info["sfreq"])
        times = events / fs
        intervals = np.r_[np.nan, np.diff(times)]
        frame = pd.DataFrame(
            {
                "event_number": np.arange(1, len(events) + 1),
                "sample": events,
                "event_time_seconds": times,
                "previous_interval_seconds": intervals,
            }
        )
        fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
        ax.plot(frame["event_number"].iloc[1:], intervals[1:], marker="o", ms=3, lw=0.8)
        within = intervals[(intervals <= 5) & np.isfinite(intervals)]
        if within.size:
            ax.axhline(np.median(within), color="#B5485D", ls="--", label="within-block median")
            ax.legend(frameon=False)
        ax.set(xlabel="Event number", ylabel="Inter-event interval (s)", title="S7 checkerboard timing; long gaps delimit blocks")
        return frame, fig

    def technical_findings(self, review_id: str, site: int) -> pd.DataFrame:
        self._require_in_scope(review_id)
        recording = self.recording(review_id)
        info = recording.source_info
        picks = _site_picks(site)
        counts = np.asarray(load_raw_counts(info))[list(picks)]
        raw_uv = raw_microvolts(recording.raw, picks=list(picks))
        rows: list[dict[str, object]] = []
        for signal_type, raw_count, signal_uv in zip(("tEEG", "eEEG"), counts, raw_uv):
            findings, metrics = assess_channel(
                raw_count,
                signal_uv,
                f"Site {site} {signal_type}",
                info.sampling_rate_hz,
                raw_dtype(info),
                self.thresholds,
            )
            for finding in findings:
                rows.append(
                    {
                        "site": site,
                        "signal_type": signal_type,
                        "status": finding.status,
                        "rule": finding.rule_id,
                        "measured_value": finding.measured_value,
                        "threshold": finding.threshold,
                        "affected_intervals": finding.intervals,
                        "supporting_plot": finding.supporting_plot,
                        "what_does_this_mean": finding.explanation,
                        "contaminated_sample_fraction": metrics["contaminated_sample_fraction"],
                    }
                )
        return pd.DataFrame(rows)

    def plot_raw(
        self,
        review_id: str,
        site: int,
        *,
        start_seconds: float = 0.0,
        duration_seconds: float = 20.0,
        scale_uv: float = 500.0,
    ) -> plt.Figure:
        """Plot synchronized raw and zero-phase 0.5–30-Hz MNE copies."""
        self._require_in_scope(review_id)
        recording = self.recording(review_id)
        filtered = self.filtered(review_id)
        picks = _site_picks(site)
        fs = float(recording.raw.info["sfreq"])
        start = max(0, int(round(start_seconds * fs)))
        stop = min(recording.raw.n_times, int(round((start_seconds + duration_seconds) * fs)))
        if stop <= start:
            raise ValueError("The selected raw interval is empty")
        raw_uv = raw_microvolts(recording.raw, picks=list(picks))[:, start:stop]
        filtered_uv = raw_microvolts(filtered, picks=list(picks))[:, start:stop]
        times = np.arange(start, stop) / fs
        event_times = checkerboard_samples(recording) / fs
        visible_events = event_times[(event_times >= times[0]) & (event_times <= times[-1])]
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
        colors = ("#007C83", "#B5485D")
        for channel, (signal_type, color) in enumerate(zip(("tEEG", "eEEG"), colors)):
            offset = channel * 2.2 * scale_uv
            axes[0].plot(times, raw_uv[channel] + offset, color=color, lw=0.65, label=signal_type)
            axes[1].plot(times, filtered_uv[channel] + offset, color=color, lw=0.65, label=signal_type)
        for ax in axes:
            for event_time in visible_events:
                ax.axvline(event_time, color="#6A4C93", lw=0.45, alpha=0.55)
            ax.set_ylim(-1.2 * scale_uv, 3.4 * scale_uv)
            ax.legend(frameon=False, loc="upper right")
            ax.set_ylabel("Fixed scale (µV)")
        axes[0].set_title(f"Site {site}: raw continuous MNE data")
        axes[1].set_title("Separate zero-phase 0.5–30-Hz review copy")
        axes[1].set_xlabel("Recording time (s)")
        return fig

    def open_native_browser(
        self,
        review_id: str,
        site: int,
        *,
        start_seconds: float = 0.0,
        duration_seconds: float = 20.0,
    ):
        """Open MNE's native scroll/zoom browser without saving source edits."""
        self._require_in_scope(review_id)
        return native_browser(
            self.recording(review_id),
            picks=list(_site_picks(site)),
            start=start_seconds,
            duration=duration_seconds,
            block=False,
        )

    def epoch_evidence(
        self,
        review_id: str,
        site: int,
        signal_type: str,
        event_number: int,
        *,
        scale_uv: float = 500.0,
    ) -> tuple[dict[str, object], plt.Figure | None]:
        """Return suggestion evidence and a fixed-scale raw/filtered epoch plot."""
        self._require_in_scope(review_id)
        if signal_type not in {"tEEG", "eEEG"}:
            raise ValueError("signal_type must be tEEG or eEEG")
        recording = self.recording(review_id)
        filtered = self.filtered(review_id)
        info = recording.source_info
        events = checkerboard_samples(recording)
        if event_number < 1 or event_number > len(events):
            raise ValueError(f"event_number must be between 1 and {len(events)}")
        pair_picks = list(_site_picks(site))
        signal_index = 0 if signal_type == "tEEG" else 1
        raw_mne = checkerboard_epochs(recording.raw, recording.checkerboard_events, self.thresholds, picks=pair_picks)
        filtered_mne = checkerboard_epochs(filtered, recording.checkerboard_events, self.thresholds, picks=pair_picks)
        raw_set = epoch_set_microvolts(raw_mne.copy().pick([raw_mne.ch_names[signal_index]]), len(events))
        filtered_set = epoch_set_microvolts(filtered_mne.copy().pick([filtered_mne.ch_names[signal_index]]), len(events))
        event_time = float(events[event_number - 1] / info.sampling_rate_hz)
        if event_number in raw_set.incomplete_event_numbers:
            return {
                "event_number": event_number,
                "event_time": event_time,
                "automated_status": "FAIL",
                "suggested_reasons": "incomplete epoch",
                "evidence": "The fixed −100 to +400-ms window extends outside the recording.",
            }, None
        raw_counts = np.asarray(load_raw_counts(info))[pair_picks[signal_index]]
        raw_count_set = epoch_data(raw_counts, events, info.sampling_rate_hz, self.thresholds)
        marker_status = marker_quality(events, info.sampling_rate_hz, self.thresholds)
        suggestions = epoch_artifact_suggestions(
            raw_count_set.epochs,
            filtered_set.epochs[:, 0, :],
            raw_dtype(info),
            filtered_set.times_seconds,
            filtered_set.event_numbers,
            marker_status.get("invalid_event_numbers", []),
            self.thresholds,
        )
        index = int(np.flatnonzero(raw_set.event_numbers == event_number)[0])
        suggestion = dict(suggestions[index])
        suggestion.update({"event_number": event_number, "event_time": event_time})
        times_ms = filtered_set.times_seconds * 1000
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        ax.plot(times_ms, raw_set.epochs[index, 0], color="#8A8A8A", lw=0.8, label="raw")
        ax.plot(times_ms, filtered_set.epochs[index, 0], color="#007C83", lw=1.0, label="0.5–30 Hz")
        ax.axvspan(-100, 0, color="#EEEEEE", label="baseline")
        ax.axvline(0, color="#6A4C93", ls="--", lw=1, label="S7 onset")
        ax.set(
            xlim=(-100, 400),
            ylim=(-scale_uv, scale_uv),
            xlabel="Time from checkerboard reversal (ms)",
            ylabel="Amplitude (µV)",
            title=f"{review_id} · site {site} {signal_type} · event {event_number} at {event_time:.3f} s",
        )
        ax.legend(frameon=False, ncol=4)
        return suggestion, fig

    def plot_rest(
        self,
        review_id: str,
        site: int,
        condition: str,
        segment_number: int = 1,
        *,
        scale_uv: float = 500.0,
    ) -> plt.Figure:
        self._require_in_scope(review_id)
        if condition not in {"eyes_open", "eyes_closed"}:
            raise ValueError("condition must be eyes_open or eyes_closed")
        recording = self.recording(review_id)
        rests = condition_intervals(recording.raw, self.thresholds.rest_max_seconds)[condition]
        if segment_number < 1 or segment_number > len(rests):
            raise ValueError(f"segment_number must be between 1 and {len(rests)}")
        start, stop = rests[segment_number - 1]
        return self.plot_raw(
            review_id,
            site,
            start_seconds=start / recording.raw.info["sfreq"],
            duration_seconds=(stop - start) / recording.raw.info["sfreq"],
            scale_uv=scale_uv,
        )

    def save_site_decision(
        self,
        review_id: str,
        site: int,
        decision: str,
        *,
        reason: str,
        reviewer: str,
        notes: str = "",
    ) -> None:
        """Append a human site decision; never infer it from technical findings."""
        self._require_in_scope(review_id)
        findings = self.technical_findings(review_id, site)
        rank = {"PASS": 0, "UNRESOLVED": 1, "WARNING": 2, "FAIL": 3}
        original_status = max(findings["status"], key=lambda status: rank[str(status)])
        append_site_decision(
            self.output_dir,
            review_id=review_id,
            session=self._session(review_id),
            site=str(site),
            decision=decision,
            reason=reason,
            reviewer=reviewer,
            notes=notes,
            original_automated_status=str(original_status),
        )

    def save_epoch_decision(
        self,
        review_id: str,
        site: int,
        signal_type: str,
        event_number: int,
        decision: str,
        *,
        reason: str,
        reviewer: str,
        notes: str = "",
    ) -> None:
        """Append a human epoch decision after displaying its MNE evidence."""
        evidence, _ = self.epoch_evidence(review_id, site, signal_type, event_number)
        append_epoch_decision(
            self.output_dir,
            review_id=review_id,
            session=self._session(review_id),
            site=str(site),
            signal_type=signal_type,
            event_number=event_number,
            event_time=float(evidence["event_time"]),
            decision=decision,
            reason=reason,
            reviewer=reviewer,
            notes=notes,
            original_automated_status=str(evidence["automated_status"]),
        )

    def progress(self) -> dict[str, object]:
        in_scope = [row for row in self.manifest if row.get("manual_review_scope") == "IN_SCOPE"]
        expected_sites: list[tuple[str, str, str]] = []
        expected_epochs: list[tuple[str, str, str, str, int]] = []
        for row in in_scope:
            for site in range(1, 5):
                expected_sites.append((row["review_id"], row["session"], str(site)))
                for signal_type in ("tEEG", "eEEG"):
                    expected_epochs.extend(
                        (row["review_id"], row["session"], str(site), signal_type, event_number)
                        for event_number in range(1, int(row["checkerboard_event_count"]) + 1)
                    )
        current = latest_decisions(read_csv(self.paths["qc_decisions"]))
        return completion_audit(current, expected_sites, expected_epochs)


def launch_notebook_reviewer() -> NotebookReviewer:
    """Render a compact ipywidgets reviewer and return its programmable backend."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    reviewer = NotebookReviewer()
    review_id = widgets.Dropdown(options=reviewer.review_ids, description="Review ID")
    site = widgets.Dropdown(options=(1, 2, 3, 4), description="Site")
    signal_type = widgets.ToggleButtons(options=("tEEG", "eEEG"), description="Signal")
    event_number = widgets.BoundedIntText(value=1, min=1, max=1, description="Event")
    start = widgets.FloatText(value=0.0, description="Start (s)")
    duration = widgets.FloatText(value=20.0, description="Duration (s)")
    scale = widgets.Dropdown(options=(25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 3000.0), value=500.0, description="±µV")
    condition = widgets.ToggleButtons(options=("eyes_open", "eyes_closed"), description="Rest")
    segment = widgets.BoundedIntText(value=1, min=1, max=10, description="Segment")
    reviewer_name = widgets.Text(description="Reviewer")
    notes = widgets.Textarea(description="Notes")
    reason = widgets.Text(description="Reason")
    site_decision = widgets.Dropdown(options=("-- select --",) + SITE_DECISIONS, description="Site decision")
    epoch_decision = widgets.Dropdown(options=("-- select --", "KEEP", "REJECT"), description="Epoch decision")
    epoch_reason = widgets.Dropdown(options=("",) + EPOCH_REASONS, description="Reject reason")
    output = widgets.Output()

    def update_event_limit(*_args) -> None:
        public = reviewer.public_record(review_id.value)
        event_number.max = max(1, int(public["checkerboard_event_count"]))
        event_number.value = min(event_number.value, event_number.max)

    review_id.observe(update_event_limit, names="value")
    update_event_limit()

    def run_in_output(action) -> None:
        with output:
            clear_output(wait=True)
            try:
                action()
            except Exception as exc:  # Notebook must keep the controls usable.
                print(f"ERROR: {exc}")

    def show_overview(_button) -> None:
        run_in_output(lambda: display(reviewer.overview(review_id.value)))

    def show_findings(_button) -> None:
        run_in_output(lambda: display(reviewer.technical_findings(review_id.value, site.value)))

    def show_raw(_button) -> None:
        def action() -> None:
            display(reviewer.plot_raw(review_id.value, site.value, start_seconds=start.value, duration_seconds=duration.value, scale_uv=scale.value))
        run_in_output(action)

    def show_events(_button) -> None:
        def action() -> None:
            table, figure = reviewer.event_timing(review_id.value)
            display(table)
            display(figure)
        run_in_output(action)

    def show_epoch(_button) -> None:
        def action() -> None:
            evidence, figure = reviewer.epoch_evidence(review_id.value, site.value, signal_type.value, event_number.value, scale_uv=scale.value)
            display(evidence)
            if figure is not None:
                display(figure)
        run_in_output(action)

    def show_rest(_button) -> None:
        run_in_output(lambda: display(reviewer.plot_rest(review_id.value, site.value, condition.value, segment.value, scale_uv=scale.value)))

    def open_browser(_button) -> None:
        run_in_output(lambda: display(reviewer.open_native_browser(review_id.value, site.value, start_seconds=start.value, duration_seconds=duration.value)))

    def save_site(_button) -> None:
        def action() -> None:
            if site_decision.value == "-- select --":
                raise ValueError("Choose a site decision; no default decision is saved")
            reviewer.save_site_decision(
                review_id.value,
                site.value,
                site_decision.value,
                reason=reason.value,
                reviewer=reviewer_name.value,
                notes=notes.value,
            )
            print("Site decision appended. Prior revisions remain in qc_decisions.csv.")
        run_in_output(action)

    def save_epoch(_button) -> None:
        def action() -> None:
            if epoch_decision.value == "-- select --":
                raise ValueError("Choose an epoch decision; no default decision is saved")
            reviewer.save_epoch_decision(
                review_id.value,
                site.value,
                signal_type.value,
                event_number.value,
                epoch_decision.value,
                reason=epoch_reason.value,
                reviewer=reviewer_name.value,
                notes=notes.value,
            )
            print("Epoch decision appended. No rejection mask was applied.")
        run_in_output(action)

    def show_progress(_button) -> None:
        run_in_output(lambda: display(reviewer.progress()))

    buttons = {
        "Overview": (show_overview, "info"),
        "Technical findings": (show_findings, "warning"),
        "Raw + filtered": (show_raw, ""),
        "Event timing": (show_events, ""),
        "Selected epoch": (show_epoch, ""),
        "EO/EC segment": (show_rest, ""),
        "Native MNE browser": (open_browser, ""),
        "Save site decision": (save_site, "danger"),
        "Save epoch decision": (save_epoch, "danger"),
        "Review progress": (show_progress, "success"),
    }
    controls = []
    for label, (handler, style) in buttons.items():
        button = widgets.Button(description=label, button_style=style)
        button.on_click(handler)
        controls.append(button)

    warning = widgets.HTML(
        "<b>BLINDED TECHNICAL-QC MODE.</b> Automated PASS/WARNING/FAIL values are evidence, not decisions. "
        "VEP averages, alpha results, configuration labels, and spectral correlations are not displayed here."
    )
    display(
        warning,
        widgets.VBox(
            [
                widgets.HBox([review_id, site, signal_type, event_number]),
                widgets.HBox([start, duration, scale, condition, segment]),
                widgets.HBox(controls[:5]),
                widgets.HBox(controls[5:]),
                widgets.HTML("<hr><b>Human decision fields</b>"),
                widgets.HBox([reviewer_name, site_decision, epoch_decision, epoch_reason]),
                widgets.HBox([reason, notes]),
                output,
            ]
        ),
    )
    return reviewer


__all__ = ["NotebookReviewer", "launch_notebook_reviewer"]
