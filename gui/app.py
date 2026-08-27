"""Streamlit EEG/VEP quality-review application.

Start from the repository root with:
    streamlit run gui/app.py

The default state is outcome-blind technical review.  Physiology pages unlock
only when the append-only decisions have been explicitly frozen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from data_integrity import (  # noqa: E402
    channel_map,
    integrity_record,
    load_manifest,
    load_raw_counts,
    marker_timing_metrics,
    resolve_review_recording,
)
from export_reports import (  # noqa: E402
    FREEZE_CONFIRMATION,
    append_duplicate_review,
    append_epoch_decision,
    append_site_decision,
    append_vep_peak_review,
    completion_audit,
    freeze_decisions,
    frozen_epoch_mask,
    initialize_exports,
    latest_decisions,
    paired_frozen_mask,
    read_csv,
    review_is_frozen,
    write_session_html_report,
)
from mne_backend import (  # noqa: E402
    checkerboard_epochs,
    checkerboard_samples,
    condition_intervals,
    epoch_set_microvolts,
    load_brainvision,
    raw_microvolts,
    recording_metadata,
    review_copy,
    validate_preservation,
)
from qc_engine import (  # noqa: E402
    EPOCH_REASONS,
    SITE_DECISIONS,
    QCThresholds,
    assess_channel,
    baseline_correct,
    epoch_artifact_suggestions,
    epoch_data,
    marker_quality,
    raw_dtype,
    raw_spectrum,
    zero_phase_review_filter,
)
from vep_analysis import (  # noqa: E402
    NEUTRAL_PEAKS,
    alpha_metrics,
    analyze_vep_epochs,
    epoch_selection_sensitivity,
    filter_sensitivity,
    paired_waveforms,
)


PUBLIC_MANIFEST = GUI_DIR / "review_data" / "blinded_review_manifest.csv"
PRIVATE_KEY = GUI_DIR / "private" / "blinding_key.csv"
OUTPUT_DIR = GUI_DIR / "review_output"
THRESHOLD_PATH = GUI_DIR / "qc_thresholds.json"

st.set_page_config(page_title="Blinded EEG Quality Review", layout="wide")


@st.cache_resource(show_spinner="Loading BrainVision with MNE-Python (read-only)…")
def load_session(review_id: str, session: str):
    info = resolve_review_recording(review_id, session, PRIVATE_KEY)
    return load_brainvision(info.header_path, preload=False)


@st.cache_resource(show_spinner="Creating MNE zero-phase 0.5–30-Hz review copy…")
def filtered_copy(review_id: str, session: str, thresholds: QCThresholds):
    recording = load_session(review_id, session)
    return review_copy(recording.raw, thresholds, picks=range(min(8, len(recording.raw.ch_names))))


def fixed_scale_plot(
    raw_uv: np.ndarray,
    review_uv: np.ndarray,
    fs: float,
    start_seconds: float,
    duration_seconds: float,
    channel_labels: list[str],
    events: np.ndarray,
    scale_uv: float,
):
    start = max(0, int(round(start_seconds * fs)))
    end = min(raw_uv.shape[-1], int(round((start_seconds + duration_seconds) * fs)))
    times = np.arange(start, end) / fs
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, constrained_layout=True)
    colors = ("#007C83", "#B5485D")
    for axis, values, title in zip(axes, (raw_uv, review_uv), ("Raw continuous output", "Zero-phase 0.5–30-Hz review copy")):
        for number, (row, label) in enumerate(zip(values[:, start:end], channel_labels)):
            offset = number * 2.4 * scale_uv
            axis.plot(times, np.clip(row, -scale_uv, scale_uv) + offset, lw=0.65, color=colors[number % len(colors)])
            axis.text(times[0] if times.size else 0, offset + scale_uv * 0.85, label, fontsize=8)
        for event in events[(events >= start) & (events < end)] / fs:
            axis.axvline(event, color="#6D3BC2", lw=0.6, alpha=0.6)
        axis.set_ylim(-scale_uv, max(scale_uv, (len(channel_labels) - 1) * 2.4 * scale_uv + scale_uv))
        axis.set_ylabel(f"fixed ±{scale_uv:g} µV")
        axis.set_title(title, loc="left")
    axes[-1].set_xlabel("Seconds from recording start; purple = S7 reversal")
    return fig


def epoch_plot(raw_epochs: np.ndarray, filtered_epochs: np.ndarray, times: np.ndarray, scale_uv: float, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, constrained_layout=True)
    for axis, values, label in zip(axes, (raw_epochs, filtered_epochs), ("Raw", "0.5–30 Hz review copy")):
        axis.plot(times * 1000, values, color="#006D77", lw=1)
        axis.axvline(0, color="black", ls="--", lw=0.8)
        axis.axvspan(-100, 0, color="#ECECEC", alpha=0.8)
        axis.set_ylim(-scale_uv, scale_uv)
        axis.set_ylabel(f"{label}\nµV")
    axes[-1].set_xlabel("Time from S7 reversal (ms); gray = baseline")
    fig.suptitle(title)
    return fig


def finding_table(findings):
    return pd.DataFrame(
        [
            {
                "status": row.status,
                "rule": row.rule_id,
                "value": row.measured_value,
                "threshold": row.threshold,
                "first interval (s)": "" if row.start_seconds is None else f"{row.start_seconds:.3f}–{row.end_seconds:.3f}",
                "what does this mean?": row.explanation,
            }
            for row in findings
        ]
    )


def render_status(status: str):
    colors = {"PASS": "#137333", "WARNING": "#A15C00", "FAIL": "#B3261E", "UNRESOLVED": "#5F6368"}
    st.markdown(f"<span style='color:{colors.get(status, '#444')};font-weight:700'>{status}</span>", unsafe_allow_html=True)


def current_decisions():
    paths = initialize_exports(OUTPUT_DIR)
    return latest_decisions(read_csv(paths["qc_decisions"]))


def expected_decision_keys(manifest_rows):
    sites = []
    epochs = []
    for row in manifest_rows:
        if row.get("manual_review_scope") != "IN_SCOPE":
            continue
        for site in range(1, 5):
            sites.append((row["review_id"], row["session"], str(site)))
            for signal_type in ("tEEG", "eEEG"):
                for event_number in range(1, int(row["checkerboard_event_count"]) + 1):
                    epochs.append((row["review_id"], row["session"], str(site), signal_type, event_number))
    return sites, epochs


def synthetic_examples(fs: float = 1000.0):
    rng = np.random.default_rng(20260826)
    time = np.arange(0, 3, 1 / fs)
    clean = 8 * np.sin(2 * np.pi * 10 * time) + rng.normal(0, 2, time.size)
    examples = {"Clean signal": clean}
    clipped = clean * 500
    examples["Clipping"] = np.clip(clipped, -120, 120)
    flat = clean.copy(); flat[900:1900] = 0
    examples["Flatline/disconnection"] = flat
    pop = clean.copy(); pop[1400:] += 90; pop[1700:] -= 90
    examples["Electrode pop"] = pop
    examples["60-Hz contamination"] = clean + 30 * np.sin(2 * np.pi * 60 * time)
    examples["Slow drift"] = clean + 80 * np.sin(2 * np.pi * 0.2 * time)
    muscle = clean.copy(); muscle[1000:1800] += 25 * np.sin(2 * np.pi * 70 * time[1000:1800])
    examples["Muscle activity"] = muscle
    return time, examples


def main():
    st.title("Blinded EEG quality review")
    st.caption("Technical quality, physiology, peak confidence, and data integrity remain separate. Raw BrainVision files are read-only.")
    if not PUBLIC_MANIFEST.exists() or not PRIVATE_KEY.exists():
        st.error("Review manifests are missing. Run `python -m gui.prepare_review` from the repository root.")
        st.stop()
    manifest = load_manifest(PUBLIC_MANIFEST)
    in_scope = [row for row in manifest if row.get("manual_review_scope") == "IN_SCOPE"]
    thresholds = QCThresholds.from_json(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else QCThresholds()
    paths = initialize_exports(OUTPUT_DIR)
    frozen = review_is_frozen(OUTPUT_DIR)

    st.sidebar.header("Review context")
    reviewer = st.sidebar.text_input("Reviewer", value=st.session_state.get("reviewer", ""), help="Saved with every decision")
    st.session_state["reviewer"] = reviewer
    selected_id = st.sidebar.selectbox("Blinded review ID", [row["review_id"] for row in manifest])
    public = next(row for row in manifest if row["review_id"] == selected_id)
    session = public["session"]
    site = st.sidebar.selectbox("Site", ["1", "2", "3", "4"])
    page = st.sidebar.radio(
        "Page",
        [
            "Review overview", "Data integrity", "Raw-signal quality", "Event-locked epochs",
            "Eyes-open / eyes-closed", "VEP morphology", "Spectral and alpha", "Education", "Decisions and export",
        ],
    )
    st.sidebar.info("POST-REVIEW PAGES UNLOCKED" if frozen else "BLINDED TECHNICAL-QC MODE")

    recording = load_session(selected_id, session)
    info = recording.source_info
    counts = np.asarray(load_raw_counts(info))  # low-level access is limited to exact ADC rail checks
    microvolts = raw_microvolts(recording.raw)
    events = checkerboard_samples(recording)
    rests = condition_intervals(recording.raw, thresholds.rest_max_seconds)
    review_raw = filtered_copy(selected_id, session, thresholds)
    review_uv = raw_microvolts(review_raw)

    pair_indices = ((0, 1), (2, 3), (4, 5), (6, 7))[int(site) - 1]
    pair_raw_counts = counts[list(pair_indices)]
    pair_raw_uv = microvolts[list(pair_indices)]
    pair_review_uv = review_uv[list(pair_indices)]
    pair_labels = [f"Site {site} tEEG", f"Site {site} eEEG"]
    dtype = raw_dtype(info)

    if public.get("manual_review_scope") != "IN_SCOPE" and page not in {"Review overview", "Data integrity", "Education", "Decisions and export"}:
        st.warning("This record is retained for data-integrity review only because its channel mapping or common protocol is unresolved. It is not in the mapped site/epoch review queue.")
        st.stop()

    if page == "Review overview":
        st.header(f"Review {selected_id} · {session}")
        st.warning("Outcome-sensitive information is hidden until decisions are frozen: holder configuration, VEP SNR, alpha results, spectral correlations, and averaged VEP morphology.")
        preservation = validate_preservation(recording)
        if not preservation["all_preserved"]:
            st.error(f"MNE preservation check failed: {preservation}")
            st.stop()
        cols = st.columns(5)
        for column, label, value in zip(
            cols,
            ("Duration", "Channels", "S7 events", "Eyes open", "Eyes closed"),
            (f"{info.duration_seconds:.1f} s", info.n_channels, len(events), len(rests["eyes_open"]), len(rests["eyes_closed"])),
        ):
            column.metric(label, value)
        st.subheader("Decision progress")
        expected_sites, expected_epochs = expected_decision_keys(manifest)
        completion = completion_audit(current_decisions(), expected_sites, expected_epochs)
        st.progress(completion["completed_epoch_decisions"] / completion["expected_epoch_decisions"] if completion["expected_epoch_decisions"] else 0)
        st.write(f"Site records: {completion['completed_site_decisions']}/{completion['expected_site_decisions']} · event epochs: {completion['completed_epoch_decisions']}/{completion['expected_epoch_decisions']}")
        st.markdown("Automated statuses mean: **PASS** no configured important problem; **WARNING** review the highlighted evidence; **FAIL** a predefined technical criterion was exceeded; **UNRESOLVED** insufficient evidence. None is a final decision.")

    elif page == "Data integrity":
        st.header("Data integrity (identity blinded)")
        record = integrity_record(info)
        timing = marker_timing_metrics(events, info.sampling_rate_hz)
        left, right = st.columns(2)
        left.subheader("Acquisition")
        left.json(
            {
                "review_id": selected_id,
                "session": session,
                "participant_identity": public["identity_status"],
                "configuration": "BLINDED (status: " + public["configuration_status"] + ")",
                "sample_count": record["sample_count"],
                "duration_seconds": record["duration_seconds"],
                "sampling_rate_hz": record["sampling_rate_hz"],
                "binary_format": record["binary_format"],
                "channel_units": record["channel_units"],
                "channel_resolutions": record["channel_resolutions"],
                "authoritative_backend": "mne.io.read_raw_brainvision",
            }
        )
        if frozen:
            private_row = next(
                row for row in load_manifest(PRIVATE_KEY)
                if row["review_id"] == selected_id and row["session"] == session
            )
            authoritative_manifest = PROJECT_ROOT / "Felt_TCRE_Manuscript_Rebuild_2026" / "derived_data" / "MANUSCRIPT_COHORT_MANIFEST.csv"
            context = {
                "source_session_identity": private_row["source_session_name"],
                "recording_timestamp": private_row["recording_timestamp"],
                "configuration": "UNRESOLVED",
            }
            if authoritative_manifest.exists():
                rows = load_manifest(authoritative_manifest)
                match = next((row for row in rows if row.get("session_id") == private_row["source_session_name"]), None)
                if match:
                    context["configuration"] = match.get("session_configuration_label", "UNRESOLVED")
            left.caption("Unblinded identity/configuration context (available only after review freeze)")
            left.json(context)
        right.subheader("File and marker fingerprints")
        right.json({key: record[key] for key in ("file_hash_vhdr", "file_hash_eeg", "file_hash_vmrk", "marker_sequence_hash")})
        right.subheader("MNE preservation checks")
        right.json(validate_preservation(recording))
        with st.expander("MNE measurement metadata and preserved annotations"):
            st.json(recording_metadata(recording))
        st.subheader("Verified montage")
        st.dataframe(pd.DataFrame(channel_map(info.n_channels)), width="stretch", hide_index=True)
        st.subheader("Marker sequence and timing")
        st.json(timing)
        intervals = np.diff(events) / info.sampling_rate_hz
        fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
        ax.plot(np.arange(1, len(intervals) + 1), intervals, marker="o", ms=3, lw=0.8)
        ax.axhline(np.median(intervals[intervals <= 5]), color="#B5485D", ls="--", label="within-block median")
        ax.set(xlabel="Interval number", ylabel="Inter-event interval (s)", title="S7 inter-event intervals; long gaps delimit blocks")
        ax.legend(frameon=False)
        st.pyplot(fig)
        st.info("No exact duplicate was found in the repository’s frozen 218-file SHA-256 audit. Participant relationships among repeated initials remain unresolved and are not treated as independent participants.")
        duplicate_rows = read_csv(paths["duplicate_review"])
        duplicate_history = [
            row for row in duplicate_rows
            if selected_id in {row.get("left_review_id"), row.get("right_review_id")}
        ]
        candidate_by_pair = {}
        for row in duplicate_history:
            pair_key = tuple(sorted((row.get("left_review_id", ""), row.get("right_review_id", ""))))
            candidate_by_pair.setdefault(pair_key, row)
        candidates = list(candidate_by_pair.values())
        st.subheader("Exact and near-duplicate candidates")
        if not candidates:
            st.write("No hash-, marker-, or metadata-based candidate involves this review ID.")
        else:
            st.dataframe(pd.DataFrame(duplicate_history), width="stretch", hide_index=True)
            candidate_index = st.selectbox(
                "Candidate pair",
                range(len(candidates)),
                format_func=lambda index: (
                    f"{candidates[index]['left_review_id']} / {candidates[index]['right_review_id']}"
                ),
            )
            candidate = candidates[candidate_index]
            duplicate_decision = st.selectbox(
                "Duplicate/identity decision",
                ("Accept", "Reject", "Uncertain", "Needs expert review"),
            )
            duplicate_reason = st.text_input("Duplicate decision reason (required except for Accept)")
            duplicate_notes = st.text_area("Duplicate review notes")
            if st.button("Append duplicate review"):
                try:
                    append_duplicate_review(
                        OUTPUT_DIR,
                        left_review_id=candidate["left_review_id"],
                        right_review_id=candidate["right_review_id"],
                        automated_status=candidate.get("automated_status", "UNRESOLVED"),
                        final_decision=duplicate_decision,
                        reason=duplicate_reason,
                        reviewer=reviewer,
                        notes=duplicate_notes,
                        evidence=json.loads(candidate.get("evidence_json") or "{}"),
                    )
                    st.success("Duplicate review appended; prior evidence and reviews remain in the audit trail.")
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(str(exc))

    elif page == "Raw-signal quality":
        st.header(f"Raw-signal quality · site {site}")
        channel_findings = []
        metrics = []
        for label, raw_count, raw_uv in zip(pair_labels, pair_raw_counts, pair_raw_uv):
            findings, row = assess_channel(raw_count, raw_uv, label, info.sampling_rate_hz, dtype, thresholds)
            channel_findings.append(findings)
            metrics.append(row)
        summary_columns = st.columns(4)
        for index, label in enumerate(pair_labels):
            summary_columns[index].metric(f"{label} contaminated samples", f"{100 * metrics[index]['contaminated_sample_fraction']:.2f}%")
        reviewed_epochs = [
            row for row in current_decisions()
            if row.get("record_type") == "event_epoch" and row.get("review_id") == selected_id
            and row.get("session") == session and row.get("site") == site
        ]
        rejected_epochs = sum(row.get("decision") == "REJECT" for row in reviewed_epochs)
        summary_columns[2].metric("Reviewed epochs rejected", f"{rejected_epochs}/{len(reviewed_epochs)}")
        marker_result = marker_quality(events, info.sampling_rate_hz, thresholds)
        summary_columns[3].metric("Marker timing", marker_result["status"])
        start_default = float(st.session_state.get("jump_start", 0.0))
        cols = st.columns(3)
        start = cols[0].number_input("Start (seconds)", 0.0, max(0.0, info.duration_seconds - 1), min(start_default, max(0.0, info.duration_seconds - 1)), 0.5)
        duration = cols[1].select_slider("Time scale", options=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0], value=10.0)
        scale = cols[2].select_slider("Fixed amplitude scale (±µV)", options=[25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 3000.0], value=500.0)
        st.pyplot(fixed_scale_plot(pair_raw_uv, pair_review_uv, info.sampling_rate_hz, start, duration, pair_labels, events, scale))
        with st.expander("MNE native interactive browser"):
            st.write("For MNE’s full scroll/zoom/annotation browser, run this local sidecar command from the repository root:")
            st.code(f".venv/bin/python -m gui.mne_viewer --review-id {selected_id} --site {site} --start {start:.3f} --duration {duration:.1f}")
            st.caption("The native browser reads the original BrainVision annotations and S7 events. Do not save edits into the raw source files; record decisions in Streamlit.")
        for label, findings in zip(pair_labels, channel_findings):
            st.subheader(label)
            st.dataframe(finding_table(findings), width="stretch", hide_index=True)
            intervals = [(finding.rule_id, finding.start_seconds) for finding in findings if finding.start_seconds is not None]
            if intervals:
                selected = st.selectbox("Affected interval", intervals, format_func=lambda item: f"{item[0]} at {item[1]:.3f} s", key=f"interval_{label}")
                if st.button("Jump to affected interval", key=f"jump_{label}"):
                    st.session_state["jump_start"] = max(0.0, selected[1] - duration / 2)
                    st.rerun()
        with st.expander("Marker completeness, duplication, irregularity, and alignment"):
            st.json(marker_result)
        st.caption("Contaminated-sample percentage is the union of localized rail, flatline, step, and large-amplitude intervals. Spectral warnings are reported separately because they may affect an entire channel.")
        st.subheader("Whole site-session classification")
        decision = st.selectbox("Decision", SITE_DECISIONS, index=3)
        reason = st.text_input("Reason (required except for GOOD)")
        notes = st.text_area("Notes", key="site_notes")
        automated = max((row["automated_status"] for row in metrics), key=lambda status: {"PASS": 0, "UNRESOLVED": 1, "WARNING": 2, "FAIL": 3}[status])
        if st.button("Append site decision"):
            try:
                append_site_decision(OUTPUT_DIR, review_id=selected_id, session=session, site=site, decision=decision, reason=reason, reviewer=reviewer, notes=notes, original_automated_status=automated)
                st.success("Decision appended. Earlier revisions remain in the audit trail.")
            except ValueError as exc:
                st.error(str(exc))

    elif page == "Event-locked epochs":
        st.header(f"Event-locked epoch review · site {site}")
        event_status = marker_quality(events, info.sampling_rate_hz, thresholds)
        raw_count_sets = [epoch_data(pair_raw_counts[index], events, info.sampling_rate_hz, thresholds) for index in range(2)]
        raw_epochs_mne = checkerboard_epochs(recording.raw, recording.checkerboard_events, thresholds, picks=list(pair_indices))
        filtered_epochs_mne = checkerboard_epochs(review_raw, recording.checkerboard_events, thresholds, picks=list(pair_indices))
        raw_sets = [
            epoch_set_microvolts(raw_epochs_mne.copy().pick([raw_epochs_mne.ch_names[index]]), len(events))
            for index in range(2)
        ]
        filtered_sets = [
            epoch_set_microvolts(filtered_epochs_mne.copy().pick([filtered_epochs_mne.ch_names[index]]), len(events))
            for index in range(2)
        ]
        signal_type = st.radio("Signal", ("tEEG", "eEEG"), horizontal=True)
        signal_index = 0 if signal_type == "tEEG" else 1
        raw_set = raw_sets[signal_index]
        filtered_set = filtered_sets[signal_index]
        suggestions = epoch_artifact_suggestions(
            raw_count_sets[signal_index].epochs, filtered_set.epochs[:, 0, :], dtype, filtered_set.times_seconds,
            filtered_set.event_numbers, event_status.get("invalid_event_numbers", []), thresholds,
        )
        event_number = st.number_input("Event number", 1, len(events), 1, 1)
        if event_number in raw_set.incomplete_event_numbers:
            st.error("INCOMPLETE EPOCH: the fixed −100 to +400 ms window extends outside the recording.")
            event_time = float(events[event_number - 1] / info.sampling_rate_hz)
            automated_status = "FAIL"
            suggested = "incomplete epoch"
        else:
            index = int(np.flatnonzero(raw_set.event_numbers == event_number)[0])
            suggestion = suggestions[index]
            event_time = float(raw_set.event_samples[index] / info.sampling_rate_hz)
            automated_status = str(suggestion["automated_status"])
            suggested = str(suggestion["suggested_reasons"])
            render_status(automated_status)
            st.write(suggestion["evidence"])
            fixed_scale = st.select_slider("Fixed epoch scale (±µV)", options=[25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 3000.0], value=500.0)
            st.pyplot(epoch_plot(raw_set.epochs[index, 0], filtered_set.epochs[index, 0], filtered_set.times_seconds, fixed_scale, f"Event {event_number} at {event_time:.3f} s"))
        cols = st.columns(2)
        epoch_decision = cols[0].radio("Epoch decision", ("KEEP", "REJECT"), horizontal=True)
        default_reason = EPOCH_REASONS.index(suggested.split("|")[0]) if suggested and suggested.split("|")[0] in EPOCH_REASONS else 0
        epoch_reason = cols[1].selectbox("Rejection reason", EPOCH_REASONS, index=default_reason)
        epoch_notes = st.text_area("Epoch notes")
        if st.button("Append epoch decision"):
            try:
                append_epoch_decision(
                    OUTPUT_DIR, review_id=selected_id, session=session, site=site, signal_type=signal_type,
                    event_number=int(event_number), event_time=event_time, decision=epoch_decision,
                    reason=epoch_reason, reviewer=reviewer, notes=epoch_notes,
                    original_automated_status=automated_status,
                )
                st.success("Epoch decision appended. No mask has been applied.")
            except ValueError as exc:
                st.error(str(exc))

    elif page == "Eyes-open / eyes-closed":
        st.header(f"Eyes-open and eyes-closed segments · site {site}")
        condition = st.radio("Condition", ("eyes_open", "eyes_closed"), horizontal=True)
        intervals = rests[condition]
        if not intervals:
            st.warning("No marked segment is available.")
        else:
            segment_number = st.slider("Segment", 1, len(intervals), 1)
            start_sample, end_sample = intervals[segment_number - 1]
            start_seconds = start_sample / info.sampling_rate_hz
            duration_seconds = (end_sample - start_sample) / info.sampling_rate_hz
            scale = st.select_slider("Fixed amplitude scale (±µV)", options=[25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 3000.0], value=500.0)
            st.pyplot(fixed_scale_plot(pair_raw_uv, pair_review_uv, info.sampling_rate_hz, start_seconds, duration_seconds, pair_labels, np.array([], dtype=int), scale))
            st.caption("No alpha power or reactivity result is shown during technical QC.")

    elif page == "VEP morphology":
        st.header("VEP morphology and peak confidence")
        if not frozen:
            st.warning("Locked during manual technical QC to prevent outcome-informed rejection. Freeze a complete review to unlock this page.")
            st.stop()
        raw_epochs_mne = checkerboard_epochs(recording.raw, recording.checkerboard_events, thresholds, picks=list(pair_indices))
        filtered_epochs_mne = checkerboard_epochs(review_raw, recording.checkerboard_events, thresholds, picks=list(pair_indices))
        raw_sets = [epoch_set_microvolts(raw_epochs_mne.copy().pick([raw_epochs_mne.ch_names[index]]), len(events)) for index in range(2)]
        filt_sets = [epoch_set_microvolts(filtered_epochs_mne.copy().pick([filtered_epochs_mne.ch_names[index]]), len(events)) for index in range(2)]
        common = np.asarray(paired_frozen_mask(OUTPUT_DIR / "frozen_rejection_mask.csv", selected_id, session, site, filt_sets[0].event_numbers), dtype=bool)
        t_analysis = analyze_vep_epochs(filt_sets[0].epochs[common, 0], filt_sets[0].times_seconds)
        e_analysis = analyze_vep_epochs(filt_sets[1].epochs[common, 0], filt_sets[1].times_seconds)
        st.info(f"Common accepted-event mask: {np.sum(common)}/{len(common)} trials. The same events are used for tEEG and eEEG.")
        if not np.any(common):
            st.stop()
        times_ms = filt_sets[0].times_seconds * 1000
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].imshow(t_analysis["baseline_corrected_epochs_uv"], aspect="auto", extent=[times_ms[0], times_ms[-1], np.sum(common), 1], cmap="RdBu_r")
        axes[0, 0].axvline(0, color="black", lw=.8); axes[0, 0].set(title="tEEG individual-trial heatmap", ylabel="Accepted trial")
        axes[0, 1].imshow(e_analysis["baseline_corrected_epochs_uv"], aspect="auto", extent=[times_ms[0], times_ms[-1], np.sum(common), 1], cmap="RdBu_r")
        axes[0, 1].axvline(0, color="black", lw=.8); axes[0, 1].set(title="eEEG individual-trial heatmap")
        for label, analysis, color in (("tEEG", t_analysis, "#006D77"), ("eEEG", e_analysis, "#B5485D")):
            axes[1, 0].plot(times_ms, analysis["average_uv"], label=f"{label} manuscript 0.5–30 Hz", color=color)
            axes[1, 0].fill_between(times_ms, analysis["bootstrap_waveform_ci_low_uv"], analysis["bootstrap_waveform_ci_high_uv"], color=color, alpha=.15)
            axes[1, 1].plot(times_ms, analysis["odd_average_uv"], color=color, lw=1, label=f"{label} odd")
            axes[1, 1].plot(times_ms, analysis["even_average_uv"], color=color, lw=1, ls="--", label=f"{label} even")
        for axis in axes[1]:
            axis.axvline(0, color="black", ls="--", lw=.8); axis.axvspan(-100, 0, color="#eee"); axis.legend(fontsize=7, frameon=False); axis.set_xlabel("ms")
        axes[1, 0].set_title("Exact manuscript-processing averages")
        axes[1, 1].set_title("Odd-versus-even averages")
        st.pyplot(fig)
        paired = paired_waveforms(filt_sets[0].epochs[:, 0], filt_sets[1].epochs[:, 0], common, filt_sets[0].times_seconds)
        raw_t_average = np.mean(baseline_correct(raw_sets[0].epochs[common, 0], raw_sets[0].times_seconds), axis=0)
        raw_e_average = np.mean(baseline_correct(raw_sets[1].epochs[common, 0], raw_sets[1].times_seconds), axis=0)
        fig_detail, detail_axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        detail_axes[0].plot(times_ms, raw_t_average, color="#006D77", label="tEEG raw/minimal")
        detail_axes[0].plot(times_ms, raw_e_average, color="#B5485D", label="eEEG raw/minimal")
        detail_axes[0].set_title("Unfiltered averages (baseline only)")
        for label, analysis, color in (("tEEG", t_analysis, "#006D77"), ("eEEG", e_analysis, "#B5485D")):
            detail_axes[1].plot(times_ms, analysis["first_half_average_uv"], color=color, label=f"{label} first half")
            detail_axes[1].plot(times_ms, analysis["second_half_average_uv"], color=color, ls="--", label=f"{label} second half")
        detail_axes[1].set_title("First-versus-second-half averages")
        detail_axes[2].plot(times_ms, paired["teeg_normalized"], color="#006D77", label="tEEG normalized")
        detail_axes[2].plot(times_ms, paired["eeg_normalized"], color="#B5485D", label="eEEG normalized")
        detail_axes[2].plot(times_ms, paired["normalized_difference"], color="#5F6368", label="normalized difference")
        detail_axes[2].set_title("Paired normalized overlay and difference")
        for axis in detail_axes:
            axis.axvline(0, color="black", ls="--", lw=.8)
            axis.axvspan(-100, 0, color="#eee")
            axis.set_xlabel("Time from S7 reversal (ms)")
            axis.legend(fontsize=7, frameon=False)
        st.pyplot(fig_detail)
        st.caption("Low tEEG/eEEG waveform similarity is descriptive and is never a technical failure; the spatial derivations differ.")
        st.dataframe(pd.DataFrame(t_analysis["peak_rows"] + e_analysis["peak_rows"]), width="stretch", hide_index=True)
        st.subheader("Human peak-confidence review")
        peak_signal = st.radio("Peak signal", ("tEEG", "eEEG"), horizontal=True, key="peak_signal")
        selected_analysis = t_analysis if peak_signal == "tEEG" else e_analysis
        peak_names = [str(row["name"]) for row in selected_analysis["peak_rows"]]
        peak_name = st.selectbox("Candidate peak", peak_names)
        peak_metrics = next(row for row in selected_analysis["peak_rows"] if row["name"] == peak_name)
        peak_decision = st.selectbox("Peak decision", ("Accept", "Reject", "Uncertain", "Needs expert review"))
        peak_reason = st.text_input("Peak decision reason (required except for Accept)")
        peak_notes = st.text_area("Peak review notes")
        if st.button("Append peak review"):
            try:
                append_vep_peak_review(
                    OUTPUT_DIR,
                    review_id=selected_id,
                    session=session,
                    site=site,
                    signal_type=peak_signal,
                    peak_name=peak_name,
                    automated_status=str(peak_metrics.get("status", "UNRESOLVED")),
                    final_decision=peak_decision,
                    reason=peak_reason,
                    reviewer=reviewer,
                    notes=peak_notes,
                    metrics=peak_metrics,
                )
                st.success("Peak review appended; prior reviews remain in the audit trail.")
            except ValueError as exc:
                st.error(str(exc))
        st.subheader("Filter and epoch-selection sensitivity")
        sensitivity_signal = 0 if peak_signal == "tEEG" else 1
        peak_definition = next(definition for definition in NEUTRAL_PEAKS if definition.name == peak_name)
        filter_rows = filter_sensitivity(
            raw_sets[sensitivity_signal].epochs[common, 0],
            info.sampling_rate_hz,
            raw_sets[sensitivity_signal].times_seconds,
            definition=peak_definition,
        )
        selection_rows = epoch_selection_sensitivity(
            filt_sets[sensitivity_signal].epochs[common, 0],
            filt_sets[sensitivity_signal].times_seconds,
            definition=peak_definition,
        )
        st.caption(f"Sensitivity evidence for {peak_signal}: {peak_name}")
        st.dataframe(pd.DataFrame(filter_rows), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(selection_rows), width="stretch", hide_index=True)
        st.json({"tEEG warnings": t_analysis["warnings"], "eEEG warnings": e_analysis["warnings"], "normalized similarity (descriptive)": paired.get("normalized_waveform_correlation")})

    elif page == "Spectral and alpha":
        st.header("Spectral and alpha results")
        if not frozen:
            st.warning("Locked during manual technical QC. Raw 60-Hz warnings remain available on the technical-quality page, but alpha and paired spectral results do not.")
            st.stop()
        signal_type = st.radio("Signal", ("tEEG", "eEEG"), horizontal=True)
        index = 0 if signal_type == "tEEG" else 1
        frequencies, raw_psd = raw_spectrum(pair_raw_uv[index], info.sampling_rate_hz)
        paired_spectra = [raw_spectrum(pair_raw_uv[pair_index], info.sampling_rate_hz) for pair_index in range(2)]
        alpha = alpha_metrics(pair_review_uv[index], info.sampling_rate_hz, rests)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        axes[0].semilogy(frequencies, raw_psd, color="#006D77"); axes[0].axvline(60, color="#B5485D", ls="--"); axes[0].set_xlim(0, 120); axes[0].set(title="Raw pre-notch PSD", xlabel="Hz", ylabel="µV²/Hz")
        axes[1].semilogy(alpha["frequencies_hz"], alpha["eyes_open_psd_uv2_hz"], label="eyes open"); axes[1].semilogy(alpha["frequencies_hz"], alpha["eyes_closed_psd_uv2_hz"], label="eyes closed"); axes[1].axvspan(8, 13, color="#FFCC66", alpha=.3); axes[1].set_xlim(0.5, 30); axes[1].set(title="0.5–30-Hz branch", xlabel="Hz", ylabel="µV²/Hz"); axes[1].legend(frameon=False)
        for label, (pair_frequency, pair_psd), color in zip(pair_labels, paired_spectra, ("#006D77", "#B5485D")):
            axes[2].semilogy(pair_frequency, pair_psd, color=color, label=label)
        axes[2].set_xlim(.5, 120); axes[2].set(title="Paired raw spectra", xlabel="Hz", ylabel="µV²/Hz"); axes[2].legend(frameon=False)
        st.pyplot(fig)
        st.json({key: value for key, value in alpha.items() if not isinstance(value, np.ndarray)})
        spectral_findings, _ = assess_channel(pair_raw_counts[index], pair_raw_uv[index], pair_labels[index], info.sampling_rate_hz, dtype, thresholds)
        st.subheader("Artifact warnings affecting this spectral estimate")
        st.dataframe(finding_table([finding for finding in spectral_findings if finding.status != "PASS"]), width="stretch", hide_index=True)
        st.info("These are descriptive physiological measures, not technical pass/fail rules. Figures are not manuscript-ready until scientific approval.")

    elif page == "Education":
        st.header("What the technical patterns look like")
        st.warning("All examples on this page are synthetic teaching signals. They never enter project analysis or review decisions.")
        time, examples = synthetic_examples()
        selected = st.selectbox("Example", list(examples))
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        ax.plot(time, examples[selected], lw=.8, color="#006D77")
        ax.set(title=selected, xlabel="Seconds", ylabel="Synthetic amplitude (arbitrary µV)")
        st.pyplot(fig)
        explanations = {
            "Clean signal": "Variable, continuous activity without rails, long flat periods, abrupt steps, or a narrow contaminant dominating the trace.",
            "Clipping": "The waveform repeatedly stops at exactly the ADC maximum or minimum, so the original amplitude is unrecoverable.",
            "Flatline/disconnection": "A long interval has almost no variance, which can mean disconnection, a short, or a stalled channel.",
            "Electrode pop": "A sudden baseline step is consistent with an abrupt contact change.",
            "60-Hz contamination": "A regular 60-Hz oscillation often reflects mains coupling; confirm it in the raw pre-notch spectrum.",
            "Slow drift": "Large slow baseline movement can reflect polarization, sweat, motion, or unstable contact.",
            "Muscle activity": "A burst of broad fast activity can reflect muscle; without auxiliary sensors this remains an artifact interpretation, not a diagnosis.",
        }
        st.write(explanations[selected])
        st.subheader("VEP teaching concepts")
        st.write("A reproducible response appears in many trials and agrees across odd/even and first/second halves. An ambiguous peak moves across filter settings or bootstrap samples, lands at a search-window boundary, or changes markedly when a few trials are removed. Those examples are available only after review is frozen because averaged morphology could bias technical rejection.")

    elif page == "Decisions and export":
        st.header("Audit trail and export")
        decisions = current_decisions()
        expected_sites, expected_epochs = expected_decision_keys(manifest)
        completion = completion_audit(decisions, expected_sites, expected_epochs)
        st.json({key: value for key, value in completion.items() if key not in {"missing_sites", "missing_epochs"}})
        st.dataframe(pd.DataFrame(decisions), width="stretch", hide_index=True)
        st.download_button("Download qc_decisions.csv", paths["qc_decisions"].read_bytes(), "qc_decisions.csv", "text/csv")
        st.download_button("Download epoch_rejection_log.csv", paths["epoch_rejection_log"].read_bytes(), "epoch_rejection_log.csv", "text/csv")
        if not frozen:
            st.subheader("Freeze complete review")
            confirmation = st.text_input(f"Type `{FREEZE_CONFIRMATION}`")
            if st.button("Freeze rejection mask", type="primary"):
                try:
                    freeze_decisions(OUTPUT_DIR, reviewer=reviewer, confirmation=confirmation, completion=completion)
                    st.success("Decisions frozen. Raw data remain unchanged; the mask has not yet been applied to analysis.")
                    st.rerun()
                except (ValueError, FileExistsError) as exc:
                    st.error(str(exc))
        else:
            st.success("Review mask is frozen and hash-verified. Post-review analysis pages are unlocked.")
        if st.button("Create this session’s HTML report"):
            record = integrity_record(info)
            session_decisions = [row for row in decisions if row.get("review_id") == selected_id]
            report = write_session_html_report(
                OUTPUT_DIR / "reports" / f"{selected_id}_{session}.html",
                raw=recording.raw,
                review_id=selected_id, session=session, integrity=record, findings=[], decisions=session_decisions,
                post_review_results={} if frozen else None, include_raw_summary=frozen,
            )
            st.success(f"Wrote {report}")


if __name__ == "__main__":
    main()
