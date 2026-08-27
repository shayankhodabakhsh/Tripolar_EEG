"""Apply a frozen human mask to derived analyses without altering raw data.

This command refuses to run before review completion or while any whole-site
decision remains UNCERTAIN.  It uses each signal's frozen KEEP decisions for
single-signal analyses and the intersection for every paired tEEG/eEEG result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from .data_integrity import (
        load_manifest,
        resolve_review_recording,
    )
    from .export_reports import (
        count_change_report,
        frozen_epoch_mask,
        latest_decisions,
        paired_frozen_mask,
        read_csv,
        review_is_frozen,
        write_session_html_report,
    )
    from .mne_backend import (
        checkerboard_epochs,
        checkerboard_samples,
        condition_intervals,
        epoch_set_microvolts,
        load_brainvision,
        raw_microvolts,
        review_copy,
    )
    from .qc_engine import QCThresholds, line_noise_metrics
    from .vep_analysis import alpha_metrics, analyze_vep_epochs, paired_waveforms
except ImportError:
    from data_integrity import (
        load_manifest,
        resolve_review_recording,
    )
    from export_reports import (
        count_change_report,
        frozen_epoch_mask,
        latest_decisions,
        paired_frozen_mask,
        read_csv,
        review_is_frozen,
        write_session_html_report,
    )
    from mne_backend import (
        checkerboard_epochs,
        checkerboard_samples,
        condition_intervals,
        epoch_set_microvolts,
        load_brainvision,
        raw_microvolts,
        review_copy,
    )
    from qc_engine import QCThresholds, line_noise_metrics
    from vep_analysis import alpha_metrics, analyze_vep_epochs, paired_waveforms


GUI_DIR = Path(__file__).resolve().parent
PUBLIC = GUI_DIR / "review_data" / "blinded_review_manifest.csv"
PRIVATE = GUI_DIR / "private" / "blinding_key.csv"
OUTPUT = GUI_DIR / "review_output"
POST = OUTPUT / "post_review"
FROZEN_MASK = OUTPUT / "frozen_rejection_mask.csv"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scalars(values: dict[str, object], prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, np.generic):
            result[name] = value.item()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            result[name] = "|".join(map(str, value))
    return result


def run() -> dict[str, object]:
    if not review_is_frozen(OUTPUT):
        raise RuntimeError("Review is not complete and hash-verified; no decisions were applied")
    manifest = [row for row in load_manifest(PUBLIC) if row.get("manual_review_scope") == "IN_SCOPE"]
    decisions = latest_decisions(read_csv(FROZEN_MASK))
    site_decisions = {
        (row["review_id"], row["session"], row["site"]): row["decision"]
        for row in decisions if row.get("record_type") == "site_session"
    }
    unresolved = [key for key, decision in site_decisions.items() if decision == "UNCERTAIN"]
    if unresolved:
        raise RuntimeError(f"Resolve {len(unresolved)} UNCERTAIN site decisions before applying the mask")
    thresholds = QCThresholds.from_json(GUI_DIR / "qc_thresholds.json")
    signal_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    before_rows: list[dict[str, object]] = []
    after_rows: list[dict[str, object]] = []
    waveform_arrays: dict[str, np.ndarray] = {}
    for public in manifest:
        review_id, session = public["review_id"], public["session"]
        info = resolve_review_recording(review_id, session, PRIVATE)
        recording = load_brainvision(info.header_path, preload=False)
        events = checkerboard_samples(recording)
        rests = condition_intervals(recording.raw, thresholds.rest_max_seconds)
        filtered_raw = review_copy(recording.raw, thresholds, picks=range(8))
        raw_uv = raw_microvolts(recording.raw, picks=range(8))
        filtered_uv = raw_microvolts(filtered_raw)
        raw_epochs_mne = checkerboard_epochs(recording.raw, recording.checkerboard_events, thresholds, picks=range(8))
        filtered_epochs_mne = checkerboard_epochs(filtered_raw, recording.checkerboard_events, thresholds, picks=range(8))
        raw_epoch_set = epoch_set_microvolts(raw_epochs_mne, len(events))
        filtered_epoch_set = epoch_set_microvolts(filtered_epochs_mne, len(events))
        session_results: dict[str, object] = {}
        for site in range(1, 5):
            site_key = (review_id, session, str(site))
            site_decision = site_decisions.get(site_key)
            if site_decision is None:
                raise RuntimeError(f"Frozen mask lacks whole-site decision {site_key}")
            include_site = site_decision in {"GOOD", "GOOD_WITH_BAD_SEGMENTS"}
            teeb_channel, eeg_channel = (site - 1) * 2, (site - 1) * 2 + 1
            event_numbers = filtered_epoch_set.event_numbers
            before_rows.append({"review_id": review_id, "session": session, "site": site, "trial_count": len(event_numbers)})
            masks: dict[str, np.ndarray] = {}
            for signal_type, channel in (("tEEG", teeb_channel), ("eEEG", eeg_channel)):
                mask = np.asarray(frozen_epoch_mask(FROZEN_MASK, review_id, session, str(site), signal_type, event_numbers), dtype=bool)
                masks[signal_type] = mask
                accepted = mask if include_site else np.zeros_like(mask)
                analysis = analyze_vep_epochs(filtered_epoch_set.epochs[accepted, channel], filtered_epoch_set.times_seconds)
                alpha = alpha_metrics(filtered_uv[channel], info.sampling_rate_hz, rests)
                line = line_noise_metrics(raw_uv[channel], info.sampling_rate_hz, thresholds)
                row = {
                    "review_id": review_id,
                    "session": session,
                    "site": site,
                    "signal_type": signal_type,
                    "site_decision": site_decision,
                    "site_included": include_site,
                    "total_complete_events": len(event_numbers),
                    "accepted_trials": int(np.sum(accepted)),
                    "rejected_trials": int(len(accepted) - np.sum(accepted)),
                    "accepted_trial_fraction": float(np.mean(accepted)),
                    "analysis_branch": "VEP/alpha: zero-phase 0.5-30 Hz; line noise: raw pre-notch/pre-low-pass",
                    **scalars(analysis, "vep_"),
                    **scalars(alpha, "alpha_"),
                    **scalars(line, "raw_line_"),
                }
                signal_rows.append(row)
                for key in ("average_uv", "odd_average_uv", "even_average_uv", "first_half_average_uv", "second_half_average_uv"):
                    if key in analysis:
                        waveform_arrays[f"{review_id}|{session}|site{site}|{signal_type}|{key}"] = np.asarray(analysis[key])
                session_results[f"site{site}_{signal_type}_accepted"] = int(np.sum(accepted))
            common = np.asarray(paired_frozen_mask(FROZEN_MASK, review_id, session, str(site), event_numbers), dtype=bool)
            if not include_site:
                common[:] = False
            if include_site:
                after_rows.append(
                    {
                        "review_id": review_id,
                        "session": session,
                        "site": site,
                        "trial_count": int(np.sum(masks["tEEG"])),
                    }
                )
            paired = paired_waveforms(
                filtered_epoch_set.epochs[:, teeb_channel],
                filtered_epoch_set.epochs[:, eeg_channel],
                common,
                filtered_epoch_set.times_seconds,
            )
            pair_rows.append(
                {
                    "review_id": review_id,
                    "session": session,
                    "site": site,
                    "site_decision": site_decision,
                    "site_included": include_site,
                    "common_accepted_trials": int(np.sum(common)),
                    "common_accepted_trial_fraction": float(np.mean(common)),
                    "mask_rule": "intersection of frozen tEEG KEEP and eEEG KEEP decisions",
                    **scalars(paired, "paired_"),
                }
            )
            for key, value in paired.items():
                if isinstance(value, np.ndarray):
                    waveform_arrays[f"{review_id}|{session}|site{site}|paired|{key}"] = value
        write_session_html_report(
            POST / "reports" / f"{review_id}_{session}.html",
            raw=recording.raw,
            review_id=review_id,
            session=session,
            integrity={"review_id": review_id, "session": session, "raw_data_modified": False},
            findings=[],
            decisions=[row for row in decisions if row.get("review_id") == review_id],
            post_review_results=session_results,
            include_raw_summary=True,
        )
    count_rows = count_change_report(before_rows, after_rows)
    write_csv(POST / "post_review_signal_results.csv", signal_rows)
    write_csv(POST / "post_review_paired_results.csv", pair_rows)
    write_csv(POST / "site_and_trial_count_changes.csv", count_rows)
    np.savez_compressed(POST / "post_review_waveforms.npz", **waveform_arrays)
    summary = {
        "review_records": len(manifest),
        "sites_before": len(before_rows),
        "sites_after": len({(row["review_id"], row["session"], row["site"]) for row in after_rows}),
        "teeg_site_trials_before": sum(int(row["trial_count"]) for row in before_rows),
        "teeg_site_trials_after": sum(int(row["trial_count"]) for row in after_rows),
        "signal_trial_count_before": sum(int(row["trial_count"]) for row in before_rows) * 2,
        "signal_trial_count_after": sum(int(row["accepted_trials"]) for row in signal_rows),
        "common_paired_trials_after": sum(int(row["common_accepted_trials"]) for row in pair_rows),
        "raw_data_modified": False,
        "common_event_mask_for_pairs": True,
        "line_noise_source": "raw pre-notch/pre-low-pass",
        "vep_alpha_source": "zero-phase 0.5-30 Hz",
        "manuscript_figures_approved": False,
    }
    POST.mkdir(parents=True, exist_ok=True)
    (POST / "POST_REVIEW_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
