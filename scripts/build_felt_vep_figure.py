#!/usr/bin/env python3
"""
build_felt_vep_figure.py — felt-setup VEP grand averages for §3.7 of the paper.

Loads TU2 and LS2 from data/Long felt TCRE/, reads the pre-averaged checkerboard
response from each subject's .avg file, plots three rows (felt tEEG, paste tEEG,
paste disc) with N75/P100/N135 markers, and emits a per-subject latency/amplitude
table.

Outputs:
    output/felt_vep/felt_vep_grand.png    — figure for §3.7
    paper/felt_vep_table.md       — markdown table for §3.7 prose
    paper/felt_vep_table.json     — same data, machine-readable

Run from repo root:
    source venv/bin/activate
    python scripts/build_felt_vep_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FELT_DIR  = REPO_ROOT / "data" / "Long felt TCRE"
OUT_DIR   = REPO_ROOT / "output" / "felt_vep"
OUT_FIG   = OUT_DIR / "felt_vep_grand.png"
OUT_MD    = REPO_ROOT / "paper" / "felt_vep_table.md"
OUT_JSON  = REPO_ROOT / "paper" / "felt_vep_table.json"

try:
    from tripolar_eeg.eeg_analysis import (
        discover_subjects, load_subject, FELT_TCRE_CONFIG,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tripolar_eeg.eeg_analysis import (        # noqa: E402
        discover_subjects, load_subject, FELT_TCRE_CONFIG,
    )

# Canonical VEP windows (ms post-stimulus). Searched for nearest negative,
# positive, negative peak respectively.
WIN_N75  = (50, 95)
WIN_P100 = (90, 140)
WIN_N135 = (125, 180)


def find_extremum(t_ms, y, window, kind):
    """Find min ('neg') or max ('pos') in [t0, t1] ms. Return (latency, amplitude)."""
    t0, t1 = window
    mask = (t_ms >= t0) & (t_ms <= t1)
    if not mask.any():
        return float("nan"), float("nan")
    seg = y[mask]
    ts  = t_ms[mask]
    idx = np.argmin(seg) if kind == "neg" else np.argmax(seg)
    return float(ts[idx]), float(seg[idx])


def extract_vep_features(t_ms, y):
    n75 = find_extremum(t_ms, y, WIN_N75,  "neg")
    p100 = find_extremum(t_ms, y, WIN_P100, "pos")
    n135 = find_extremum(t_ms, y, WIN_N135, "neg")
    return {"N75": n75, "P100": p100, "N135": n135}


def channel_indices_by_type(cfg, types):
    return [i for i, t in enumerate(cfg.channel_types) if t in types]


def plot_subject(ax, subject, title):
    avg = subject.get("avg_data")
    t   = subject.get("avg_t")
    if avg is None or t is None:
        ax.text(0.5, 0.5, "No .avg data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return None

    cfg = subject.get("config", FELT_TCRE_CONFIG)
    felt_idx  = channel_indices_by_type(cfg, ["FELT_TEEG"])
    paste_idx = channel_indices_by_type(cfg, ["PASTE_TEEG"])
    disc_idx  = channel_indices_by_type(cfg, ["DISC"])

    felt_avg  = np.nanmean(avg[felt_idx, :], axis=0) if felt_idx else None
    paste_avg = np.nanmean(avg[paste_idx, :], axis=0) if paste_idx else None
    disc_avg  = np.nanmean(avg[disc_idx, :], axis=0) if disc_idx else None

    features = {}
    for label, trace, color in [("Felt tEEG",  felt_avg,  "#3498db"),
                                ("Paste tEEG", paste_avg, "#2ecc71"),
                                ("Disc",       disc_avg,  "#e74c3c")]:
        if trace is None:
            continue
        ax.plot(t, trace, color=color, lw=1.6, label=label)
        feats = extract_vep_features(t, trace)
        features[label] = feats
        for peak_label, (lat, amp) in feats.items():
            if np.isnan(lat):
                continue
            ax.plot(lat, amp, "o", color=color, ms=5, mec="black", mew=0.6)
            ax.annotate(peak_label, (lat, amp), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=color)

    ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
    for w, c in [(WIN_N75, "blue"), (WIN_P100, "red"), (WIN_N135, "purple")]:
        ax.axvspan(w[0], w[1], color=c, alpha=0.05)

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return features


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve the long-felt subject codes from the paper's subject key
    # so the VEP figure uses the same de-identified IDs as Table 2.
    key_csv = REPO_ROOT / "data" / "SUBJECT_KEY.csv"
    raw_to_code = {}
    if key_csv.exists():
        import csv
        with open(key_csv) as f:
            for row in csv.DictReader(f):
                raw_to_code[row["raw_name"]] = row["display_id"]

    discovered = discover_subjects(str(FELT_DIR))
    targets = {"TU2": None, "LS2": None}
    for si in discovered:
        name = si["name"]
        if "TU2" in name and targets["TU2"] is None:
            targets["TU2"] = si
        elif "LS2" in name and targets["LS2"] is None:
            targets["LS2"] = si

    if any(v is None for v in targets.values()):
        missing = [k for k, v in targets.items() if v is None]
        raise SystemExit(f"Could not find {missing} in {FELT_DIR}")

    code_tu2 = raw_to_code.get(targets["TU2"]["name"], "TU2-fallback")
    code_ls2 = raw_to_code.get(targets["LS2"]["name"], "LS2-fallback")
    print(f"Loading {code_tu2}")
    print(f"Loading {code_ls2}")
    subj_tu2 = load_subject(str(FELT_DIR), subject_info=targets["TU2"])
    subj_ls2 = load_subject(str(FELT_DIR), subject_info=targets["LS2"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    feats_tu2 = plot_subject(axes[0], subj_tu2,
                             f"{code_tu2} ({subj_tu2.get('avg_n_segments', '?')} trials)")
    feats_ls2 = plot_subject(axes[1], subj_ls2,
                             f"{code_ls2} ({subj_ls2.get('avg_n_segments', '?')} trials)")
    fig.suptitle("Felt-setup checkerboard VEP grand averages",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_FIG.relative_to(REPO_ROOT)}")

    table = {code_tu2: feats_tu2, code_ls2: feats_ls2}

    md = []
    md.append("# Felt-setup VEP latencies and amplitudes\n")
    md.append("Auto-generated by `scripts/build_felt_vep_figure.py`.\n\n")
    md.append("| Subject | Channel | N75 lat (ms) | N75 amp (µV) | P100 lat (ms) | P100 amp (µV) | N135 lat (ms) | N135 amp (µV) |\n")
    md.append("|--|--|--|--|--|--|--|--|\n")
    for subj_name, feats in table.items():
        if feats is None:
            md.append(f"| {subj_name} | — | n/a | n/a | n/a | n/a | n/a | n/a |\n")
            continue
        for ch_label, peaks in feats.items():
            md.append(f"| {subj_name} | {ch_label} "
                      f"| {peaks['N75'][0]:.1f} | {peaks['N75'][1]:.2f} "
                      f"| {peaks['P100'][0]:.1f} | {peaks['P100'][1]:.2f} "
                      f"| {peaks['N135'][0]:.1f} | {peaks['N135'][1]:.2f} |\n")
    OUT_MD.write_text("".join(md))
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")

    OUT_JSON.write_text(json.dumps(table, indent=2))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
