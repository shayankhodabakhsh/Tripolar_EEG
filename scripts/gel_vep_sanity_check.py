"""
Gel TCRE VEP sanity check using the unified pipeline.

Loads BA-1-10-20-2024 via load_subject + GEL_TCRE_CONFIG, applies only
60 Hz notch + 0.05-55 Hz bandpass (no wavelet, no z-score), epochs around
S 7 checkerboard reversals, baseline-corrects, and plots the per-channel
grand-average VEP for the four electrode-relevant channels (Gel O1/O2 tEEG/eEEG)
plus the Pz disc.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

from tripolar_eeg.eeg_analysis import (
    GEL_TCRE_CONFIG,
    FS,
    load_subject,
    notch_filter,
)


def fir_bandpass(x, low, high, fs, numtaps=None):
    """Zero-phase FIR Hamming bandpass (matches Adeli/Norouzi paper spec).

    A 4th-order IIR Butterworth at low=0.05 Hz / fs=1000 is numerically
    unstable (Wn≈1e-4); firwin with a long Hamming window handles it cleanly.
    """
    if numtaps is None:
        numtaps = int(round(fs * 4 / max(low, 0.05))) | 1
        numtaps = min(numtaps, 16001)
    taps = firwin(numtaps, [low, high], pass_zero=False,
                  window="hamming", fs=fs)
    return filtfilt(taps, [1.0], x)

GEL_DIR = os.path.join(ROOT, "Gel TCRE", "10-20-2024")
SUBJECT_BASENAME = "BA-1-10-20-2024"
SUBJECT_KEY_CSV = os.path.join(ROOT, "data", "SUBJECT_KEY.csv")
OUT_DIR = os.path.join(ROOT, "output", "gel_vep_sanity_check")
OUT_FIG = os.path.join(OUT_DIR, "BA_unified_pipeline.png")


def display_label_for(basename, key_csv=SUBJECT_KEY_CSV, fallback="Gel session"):
    """Return the de-identified display ID for a raw subject basename.

    Falls back to ``fallback`` if the basename is not in SUBJECT_KEY.csv —
    figures must never burn in raw subject identifiers.
    """
    if not os.path.exists(key_csv):
        return fallback
    import csv
    with open(key_csv) as f:
        for row in csv.DictReader(f):
            if row.get("raw_name") == basename or row.get("basename") == basename:
                return row["display_id"]
    return fallback

PRE_S = 0.1
POST_S = 0.5
BASELINE_END_S = 0.0

PEAK_WINDOWS_MS = {
    "N75":  (60, 90),
    "P100": (90, 130),
    "N135": (130, 180),
}


def build_subject_info(gel_dir, basename):
    prefix = os.path.join(gel_dir, basename)
    paths = {
        "name": basename,
        "basename": basename,
        "eeg":  prefix + ".eeg",
        "vhdr": prefix + ".vhdr",
        "vmrk": prefix + ".vmrk",
        "avg":      prefix + "-Triggers.avg",
        "avg_vhdr": prefix + "-Triggers.vhdr",
        "avg_vmrk": prefix + "-Triggers.vmrk",
    }
    for k in ("eeg", "vhdr", "vmrk"):
        if not os.path.exists(paths[k]):
            raise FileNotFoundError(f"Missing required file: {paths[k]}")
    for k in ("avg", "avg_vhdr", "avg_vmrk"):
        if not os.path.exists(paths[k]):
            paths[k] = None
    return paths


def find_peak(t_ms, wave, lo, hi, polarity):
    mask = (t_ms >= lo) & (t_ms <= hi)
    if not np.any(mask):
        return None, None
    seg = wave[mask]
    seg_t = t_ms[mask]
    idx = np.argmin(seg) if polarity == "neg" else np.argmax(seg)
    return float(seg_t[idx]), float(seg[idx])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    subject_info = build_subject_info(GEL_DIR, SUBJECT_BASENAME)
    subj = load_subject(GEL_DIR, subject_info=subject_info, config=GEL_TCRE_CONFIG)

    cfg = subj["config"]
    eeg = subj["eeg"]
    n_ch, n_samples = eeg.shape
    print(f"Loaded {SUBJECT_BASENAME}: {n_ch} channels x {n_samples} samples ({n_samples/FS:.1f} s)")
    print(f"Config: {cfg.name}, scaling = {cfg.amplitude_scaling}")
    assert n_ch == 7 and cfg is GEL_TCRE_CONFIG, "Expected 7-ch Gel TCRE layout"
    print(f"tEEG indices (will be /187 scaled): {cfg.idx_teeg + cfg.idx_paste_teeg}")

    # Pre-filter sanity: peak abs amplitude per channel
    raw_max = np.max(np.abs(eeg), axis=1)
    for i, m in enumerate(raw_max):
        print(f"  ch{i+1} max |x| = {m:9.2f} µV  ({cfg.ch_labels[i]})")

    print("\nApplying 60 Hz notch + 0.05-55 Hz FIR Hamming bandpass (no wavelet, no z-score)...")
    eeg_f = np.empty_like(eeg)
    for c in range(n_ch):
        x = notch_filter(eeg[c], freq=60, fs=FS)
        x = fir_bandpass(x, low=0.05, high=55, fs=FS)
        eeg_f[c] = x

    stim_samples = np.asarray(subj["stim_samples"], dtype=int)
    print(f"\n{len(stim_samples)} S 7 stimulus markers parsed from .vmrk")

    pre_n = int(round(PRE_S * FS))
    post_n = int(round(POST_S * FS))
    epoch_len = pre_n + post_n
    base_n = int(round((BASELINE_END_S - (-PRE_S)) * FS))

    valid = (stim_samples >= pre_n) & (stim_samples + post_n <= n_samples)
    stim_samples = stim_samples[valid]
    print(f"{len(stim_samples)} epochs fit within recording bounds")

    epochs = np.zeros((len(stim_samples), n_ch, epoch_len))
    for i, s in enumerate(stim_samples):
        seg = eeg_f[:, s - pre_n : s + post_n]
        baseline = seg[:, :base_n].mean(axis=1, keepdims=True)
        epochs[i] = seg - baseline

    grand = epochs.mean(axis=0)
    t_ms = (np.arange(epoch_len) - pre_n) / FS * 1000.0

    plot_idx = [0, 1, 2, 3, 6]
    plot_titles = [
        "Ch1 — Gel TCRE O1 (tEEG)",
        "Ch2 — Gel TCRE O1 (eEEG)",
        "Ch3 — Gel TCRE O2 (tEEG)",
        "Ch4 — Gel TCRE O2 (eEEG)",
        "Ch7 — Pz disc (eEEG)",
    ]

    print("\nPeak latencies (baseline-corrected grand average):")
    print(f"{'Channel':<28}{'N75 (ms / µV)':<22}{'P100 (ms / µV)':<22}{'N135 (ms / µV)':<22}")
    peak_summary = {}
    for ci, title in zip(plot_idx, plot_titles):
        wave = grand[ci]
        n75 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["N75"], "neg")
        p100 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["P100"], "pos")
        n135 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["N135"], "neg")
        peak_summary[title] = {"N75": n75, "P100": p100, "N135": n135}

        def fmt(p):
            return f"{p[0]:6.1f} / {p[1]:7.2f}" if p[0] is not None else "    --       "
        print(f"{title:<28}{fmt(n75):<22}{fmt(p100):<22}{fmt(n135):<22}")

    fig, axes = plt.subplots(len(plot_idx), 1, figsize=(8, 11), sharex=True)
    for ax, ci, title in zip(axes, plot_idx, plot_titles):
        wave = grand[ci]
        ax.plot(t_ms, wave, color="#2c3e50", lw=1.5)
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        for lbl, (lo, hi) in PEAK_WINDOWS_MS.items():
            ax.axvspan(lo, hi, color="#f1c40f" if "P100" in lbl else "#3498db", alpha=0.07)
        for lbl in ("N75", "P100", "N135"):
            p = peak_summary[title][lbl]
            if p[0] is not None:
                color = "#27ae60" if lbl == "P100" else "#c0392b"
                ax.plot(p[0], p[1], "o", color=color, ms=6)
                ax.annotate(f"{lbl}\n{p[0]:.0f} ms", xy=(p[0], p[1]),
                            xytext=(6, 6), textcoords="offset points",
                            fontsize=8, color=color)
        ax.set_title(f"{title}  (n={len(stim_samples)} epochs)", fontsize=10)
        ax.set_ylabel("µV")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (ms) — 0 = checkerboard reversal")
    display_id = display_label_for(SUBJECT_BASENAME)
    fig.suptitle(
        f"Gel TCRE VEP sanity check — subject {display_id}\n"
        "Unified pipeline: 60 Hz notch + 0.05-55 Hz bandpass only (no wavelet, no z-score)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_FIG, dpi=130)
    print(f"\nSaved figure: {OUT_FIG}")


if __name__ == "__main__":
    main()
