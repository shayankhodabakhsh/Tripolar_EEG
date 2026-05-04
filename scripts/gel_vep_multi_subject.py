"""
Multi-subject Gel TCRE VEP sanity check with corrected channel handling.

Key corrections versus the original gel_vep_sanity_check.py:
  - Channel layout per Norouzi/fifthTCREGelProcess.py:
        Ch1 outer/O1, Ch2 inner/O1, Ch3 outer/O2, Ch4 inner/O2,
        Ch5 outer/Pz, Ch6 inner/Pz, Ch7 normal EEG (disc).
  - tEEG is *computed* as  tEEG = 16 * inner - outer  (not /187 scaled).
  - Per-subject ADC resolution read from the .vhdr (some recordings use
    0.0488281 µV/bit, not 0.1).
  - Permissive marker parser handles both "Stimulus,S  7" and "Trigger,T  1".

Pipeline (still no wavelet, no z-score):
  notch 60 Hz  ->  zero-phase FIR Hamming bandpass 0.05-55 Hz
  epoch -0.1 .. 0.5 s around each stim, baseline = (-0.1, 0).
"""

import os
import sys
import re
import functools
print = functools.partial(print, flush=True)  # line-flush for piped runs
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, iirnotch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GEL_DIR_DEFAULT = os.path.join(ROOT, "Gel TCRE", "10-20-2024")
OUT_DIR = os.path.join(ROOT, "output", "gel_vep_sanity_check")
SUBJECT_KEY_CSV = os.path.join(ROOT, "data", "SUBJECT_KEY.csv")

# Hard-coded display labels for sessions not registered in SUBJECT_KEY.csv
# (e.g. QC-failed sessions that are documented qualitatively only). Keeps raw
# subject identifiers out of figure titles even when the csv is unavailable.
DISPLAY_OVERRIDES = {
    "BA-2-10-20-2024": "Gel session, excluded (QC fail)",
}


def display_label_for(basename):
    """Return the de-identified display label for a raw subject basename.

    Looks up SUBJECT_KEY.csv first, then DISPLAY_OVERRIDES, then falls back to
    a generic ``Gel session`` label so raw IDs never reach figure titles.
    """
    if basename in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[basename]
    if os.path.exists(SUBJECT_KEY_CSV):
        import csv
        with open(SUBJECT_KEY_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("raw_name") == basename or row.get("basename") == basename:
                    return f"subject {row['display_id']}"
    return "Gel session"

FS = 1000.0
PRE_S = 0.1
POST_S = 0.5
BASELINE_S = (-0.1, 0.0)

PEAK_WINDOWS_MS = {
    "N75":  (60, 90),
    "P100": (90, 130),
    "N135": (130, 180),
}

STIM_DESC_PATTERNS = (
    re.compile(r"^\s*S\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*T\s*\d+\s*$", re.IGNORECASE),
)


def parse_vhdr(vhdr_path):
    n_ch = None
    resolutions = {}
    sampling_us = None
    with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NumberOfChannels"):
                n_ch = int(line.split("=")[1].strip())
            elif line.startswith("SamplingInterval"):
                sampling_us = float(line.split("=")[1].strip())
            else:
                m = re.match(r"^Ch(\d+)\s*=\s*([^,]*),([^,]*),([^,]*),", line)
                if m:
                    ch_idx = int(m.group(1)) - 1
                    try:
                        resolutions[ch_idx] = float(m.group(4))
                    except ValueError:
                        pass
    if n_ch is None:
        raise ValueError(f"NumberOfChannels missing in {vhdr_path}")
    fs = 1e6 / sampling_us if sampling_us else FS
    return n_ch, resolutions, fs


def parse_markers(vmrk_path):
    stim_samples = []
    with open(vmrk_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Mk"):
                continue
            try:
                payload = line[line.index("=") + 1:]
            except ValueError:
                continue
            parts = [p.strip() for p in payload.split(",")]
            if len(parts) < 3:
                continue
            mtype, desc = parts[0], parts[1]
            if mtype not in ("Stimulus", "Trigger"):
                continue
            if not any(p.match(desc) for p in STIM_DESC_PATTERNS):
                continue
            try:
                pos = int(parts[2])
            except ValueError:
                continue
            stim_samples.append(pos)
    return np.asarray(stim_samples, dtype=int)


def load_eeg_uv(eeg_path, n_ch, resolutions):
    raw = np.fromfile(eeg_path, dtype=np.int16)
    n_samples = len(raw) // n_ch
    eeg = raw[: n_samples * n_ch].reshape(n_samples, n_ch).T.astype(np.float64)
    for c in range(n_ch):
        eeg[c] *= resolutions.get(c, 0.1)
    return eeg


def fir_bandpass(x, low, high, fs, numtaps=5001):
    """Zero-phase FIR Hamming bandpass.
    5001 taps at fs=1000 gives a transition band of ~0.4 Hz, enough
    to keep low=0.05 well below the alpha band without making
    filtfilt prohibitively slow.
    """
    if numtaps % 2 == 0:
        numtaps += 1
    taps = firwin(numtaps, [low, high], pass_zero=False, window="hamming", fs=fs)
    return filtfilt(taps, [1.0], x)


def notch(x, freq, fs, q=30):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, x)


def find_peak(t_ms, wave, lo, hi, polarity):
    mask = (t_ms >= lo) & (t_ms <= hi)
    if not np.any(mask):
        return None, None
    seg = wave[mask]
    seg_t = t_ms[mask]
    idx = np.argmin(seg) if polarity == "neg" else np.argmax(seg)
    return float(seg_t[idx]), float(seg[idx])


def process_subject(basename, gel_dir):
    prefix = os.path.join(gel_dir, basename)
    vhdr = prefix + ".vhdr"
    vmrk = prefix + ".vmrk"
    eeg_path = prefix + ".eeg"
    for p in (vhdr, vmrk, eeg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    n_ch, resolutions, fs = parse_vhdr(vhdr)
    if n_ch != 7:
        raise ValueError(f"Expected 7 channels for Gel layout, got {n_ch} in {basename}")
    print(f"\n=== {basename} ===")
    print(f"  fs={fs:.0f} Hz, resolution(ch1)={resolutions.get(0):.6f} µV/bit")

    eeg = load_eeg_uv(eeg_path, n_ch, resolutions)
    n_samples = eeg.shape[1]
    print(f"  duration={n_samples/fs:.1f}s  raw max|x| per ch = "
          + ", ".join(f"{np.max(np.abs(eeg[c])):.0f}" for c in range(n_ch)))

    # Channel meanings (Norouzi mapping)
    out_idx = [0, 2, 4]   # outer rings: O1, O2, Pz
    in_idx  = [1, 3, 5]   # inner rings: O1, O2, Pz
    disc    = 6
    site_names = ["O1", "O2", "Pz"]

    # tEEG = 16 * inner - outer
    teeg = np.zeros((3, n_samples))
    for k in range(3):
        teeg[k] = 16.0 * eeg[in_idx[k]] - eeg[out_idx[k]]
    inner = eeg[in_idx]   # (3, n)
    disc_sig = eeg[disc]

    # Filter all signals (notch + FIR bandpass 0.05-55 Hz)
    def _filt(x):
        return fir_bandpass(notch(x, 60, fs), 0.05, 55, fs)

    teeg_f  = np.stack([_filt(teeg[k])  for k in range(3)])
    inner_f = np.stack([_filt(inner[k]) for k in range(3)])
    disc_f  = _filt(disc_sig)

    stim_samples = parse_markers(vmrk)
    print(f"  {len(stim_samples)} stim markers parsed")
    if len(stim_samples) == 0:
        return None

    pre_n = int(round(PRE_S * fs))
    post_n = int(round(POST_S * fs))
    epoch_len = pre_n + post_n
    base_n0 = int(round((BASELINE_S[0] - (-PRE_S)) * fs))
    base_n1 = int(round((BASELINE_S[1] - (-PRE_S)) * fs))

    valid = (stim_samples >= pre_n) & (stim_samples + post_n <= n_samples)
    stim_samples = stim_samples[valid]
    print(f"  {len(stim_samples)} epochs in-bounds")

    def epoch_avg(sig_2d_or_1d):
        sig = np.atleast_2d(sig_2d_or_1d)
        ep = np.zeros((len(stim_samples), sig.shape[0], epoch_len))
        for i, s in enumerate(stim_samples):
            seg = sig[:, s - pre_n: s + post_n]
            seg = seg - seg[:, base_n0:base_n1].mean(axis=1, keepdims=True)
            ep[i] = seg
        return ep.mean(axis=0)

    teeg_avg  = epoch_avg(teeg_f)
    inner_avg = epoch_avg(inner_f)
    disc_avg  = epoch_avg(disc_f)[0]

    t_ms = (np.arange(epoch_len) - pre_n) / fs * 1000.0

    # Plot: 2 cols (tEEG vs inner-ring/eEEG), 3 rows (O1, O2, Pz) + disc on bottom
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    print(f"  Peak latencies (ms / µV)  — search windows N75 60-90, P100 90-130, N135 130-180")
    rows_summary = []
    for k, site in enumerate(site_names):
        for col, (label, wave) in enumerate(
            [(f"tEEG @ {site} (16·in − out)", teeg_avg[k]),
             (f"Inner ring @ {site}",         inner_avg[k])]
        ):
            ax = axes[k, col]
            ax.plot(t_ms, wave, color="#2c3e50", lw=1.4)
            ax.axvline(0, color="k", lw=0.7, alpha=0.4)
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            for span_lo, span_hi in PEAK_WINDOWS_MS.values():
                ax.axvspan(span_lo, span_hi, color="#f1c40f", alpha=0.06)
            n75  = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["N75"], "neg")
            p100 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["P100"], "pos")
            n135 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["N135"], "neg")
            for nm, pk, color in (("N75", n75, "#c0392b"),
                                   ("P100", p100, "#27ae60"),
                                   ("N135", n135, "#c0392b")):
                if pk[0] is not None:
                    ax.plot(pk[0], pk[1], "o", color=color, ms=5)
                    ax.annotate(f"{nm}\n{pk[0]:.0f}", xy=pk, xytext=(5, 5),
                                textcoords="offset points", fontsize=8, color=color)
            ax.set_title(label, fontsize=10)
            ax.set_ylabel("µV")
            ax.grid(alpha=0.25)
            rows_summary.append((label, n75, p100, n135))

    # Disc on bottom row
    for col in range(2):
        axes[3, col].set_visible(False)
    ax_disc = fig.add_subplot(4, 1, 4)
    ax_disc.plot(t_ms, disc_avg, color="#2c3e50", lw=1.4)
    ax_disc.axvline(0, color="k", lw=0.7, alpha=0.4)
    ax_disc.axhline(0, color="k", lw=0.5, alpha=0.3)
    for span_lo, span_hi in PEAK_WINDOWS_MS.values():
        ax_disc.axvspan(span_lo, span_hi, color="#f1c40f", alpha=0.06)
    n75  = find_peak(t_ms, disc_avg, *PEAK_WINDOWS_MS["N75"], "neg")
    p100 = find_peak(t_ms, disc_avg, *PEAK_WINDOWS_MS["P100"], "pos")
    n135 = find_peak(t_ms, disc_avg, *PEAK_WINDOWS_MS["N135"], "neg")
    for nm, pk, color in (("N75", n75, "#c0392b"),
                           ("P100", p100, "#27ae60"),
                           ("N135", n135, "#c0392b")):
        if pk[0] is not None:
            ax_disc.plot(pk[0], pk[1], "o", color=color, ms=5)
            ax_disc.annotate(f"{nm}\n{pk[0]:.0f}", xy=pk, xytext=(5, 5),
                             textcoords="offset points", fontsize=8, color=color)
    ax_disc.set_title("Disc (Ch7, normal EEG)", fontsize=10)
    ax_disc.set_xlabel("Time (ms) — 0 = checkerboard reversal")
    ax_disc.set_ylabel("µV")
    ax_disc.grid(alpha=0.25)
    rows_summary.append(("Disc (Ch7)", n75, p100, n135))

    display_label = display_label_for(basename)
    fig.suptitle(
        f"Gel TCRE VEP — {display_label}  (n={len(stim_samples)} epochs)\n"
        "Pipeline: 60 Hz notch + 0.05–55 Hz FIR Hamming. tEEG = 16·inner − outer.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # Output filename uses the de-identified display ID rather than the raw
    # basename so the per-subject PNGs themselves do not leak identifiers.
    safe_label = display_label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    out_png = os.path.join(OUT_DIR, f"gel_vep_{safe_label}.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  saved {out_png}")

    print(f"  {'Channel':<32}{'N75 (ms / µV)':<22}{'P100 (ms / µV)':<22}{'N135 (ms / µV)':<22}")
    for label, n75, p100, n135 in rows_summary:
        def fmt(p):
            return f"{p[0]:6.1f} / {p[1]:8.2f}" if p[0] is not None else "    --        "
        print(f"  {label:<32}{fmt(n75):<22}{fmt(p100):<22}{fmt(n135):<22}")

    return rows_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gel-dir", default=GEL_DIR_DEFAULT)
    p.add_argument("--subjects", nargs="+", default=[
        "BA-1-10-20-2024",
        "BA-2-10-20-2024",
        "MN-3-BrainAmp_VEP",
    ])
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    for basename in args.subjects:
        try:
            process_subject(basename, args.gel_dir)
        except Exception as exc:
            print(f"!!! {basename} failed: {exc}")


if __name__ == "__main__":
    main()
