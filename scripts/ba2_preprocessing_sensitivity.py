"""
BA-2 preprocessing sensitivity analysis.

Three preprocessing variants compared on the same BA-2 raw recording, after
correct channel handling (per-subject .vhdr resolution, tEEG = 16·inner − outer):

  A  Adeli/Norouzi original :  notch 60 Hz, bandpass 1-55 Hz FIR Hamming,
                               db8 level-3 wavelet denoising (soft, universal),
                               z-score per channel.
  B  Unified minimal        :  notch 60 Hz, bandpass 0.05-55 Hz FIR Hamming.
                               No wavelet, no z-score.
  C  Intermediate           :  notch 60 Hz, bandpass 0.05-55 Hz FIR Hamming,
                               db8 level-3 wavelet denoising. No z-score.

eeg_analysis.py is NOT modified; a local Trigger/T-aware marker parser is
used since the existing parse_vmrk only matches Stimulus,S markers.
"""

import os
import re
import functools
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import firwin, filtfilt, iirnotch

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEL_DIR = os.path.join(ROOT, "Gel TCRE", "10-20-2024")
SUBJECT = "BA-2-10-20-2024"
SUBJECT_DISPLAY = "Excluded gel session (QC fail)"
OUT_DIR = os.path.join(ROOT, "output", "gel_vep_sanity_check")
OUT_PNG = os.path.join(OUT_DIR, "BA-2_preprocessing_sensitivity.png")

PRE_S = 0.1
POST_S = 0.5
BASELINE_S = (-0.1, 0.0)
PEAK_WINDOWS_MS = {"N75": (60, 95), "P100": (95, 145), "N135": (140, 200)}


def parse_vhdr(vhdr_path):
    n_ch = None
    res = {}
    sint = None
    with open(vhdr_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NumberOfChannels"):
                n_ch = int(line.split("=")[1])
            elif line.startswith("SamplingInterval"):
                sint = float(line.split("=")[1])
            else:
                m = re.match(r"^Ch(\d+)\s*=\s*([^,]*),([^,]*),([^,]*),", line)
                if m:
                    try:
                        res[int(m.group(1)) - 1] = float(m.group(4))
                    except ValueError:
                        pass
    return n_ch, res, 1e6 / sint if sint else 1000.0


def parse_vmrk_permissive(vmrk_path):
    """Patched marker parser accepting Stimulus,S<n> AND Trigger,T<n>."""
    stim = []
    trig_pat = re.compile(r"^\s*[ST]\s*\d+\s*$", re.IGNORECASE)
    with open(vmrk_path, encoding="utf-8", errors="ignore") as f:
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
            if not trig_pat.match(desc):
                continue
            try:
                stim.append(int(parts[2]))
            except ValueError:
                continue
    return np.asarray(stim, dtype=int)


def fir_bandpass(x, low, high, fs, numtaps=5001):
    if numtaps % 2 == 0:
        numtaps += 1
    taps = firwin(numtaps, [low, high], pass_zero=False, window="hamming", fs=fs)
    return filtfilt(taps, [1.0], x)


def notch(x, freq, fs, q=30):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, x)


def wavelet_denoise_universal(x, wavelet="db8", level=3):
    """Soft-threshold wavelet denoising with universal threshold per channel.

    sigma estimated from the finest detail coefficients via MAD/0.6745
    (matches the Adeli/Norouzi script's approach).
    """
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2.0 * np.log(len(x)))
    coeffs_thr = [coeffs[0]] + [
        pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]
    ]
    rec = pywt.waverec(coeffs_thr, wavelet)
    return rec[: len(x)]


def zscore(x):
    return (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)


def find_peak(t_ms, wave, lo, hi, polarity):
    mask = (t_ms >= lo) & (t_ms <= hi)
    if not np.any(mask):
        return (None, None)
    seg = wave[mask]
    seg_t = t_ms[mask]
    idx = np.argmin(seg) if polarity == "neg" else np.argmax(seg)
    return float(seg_t[idx]), float(seg[idx])


def epoch_average(sigs, stim_samples, pre_n, post_n, base_n0, base_n1):
    """sigs: (n_ch, n_samples). returns (n_ch, epoch_len) baseline-corrected mean."""
    n_ch, _ = sigs.shape
    epoch_len = pre_n + post_n
    valid = []
    accum = np.zeros((n_ch, epoch_len))
    for s in stim_samples:
        if s < pre_n or s + post_n > sigs.shape[1]:
            continue
        seg = sigs[:, s - pre_n: s + post_n]
        seg = seg - seg[:, base_n0:base_n1].mean(axis=1, keepdims=True)
        accum += seg
        valid.append(s)
    return accum / max(len(valid), 1), len(valid)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = os.path.join(GEL_DIR, SUBJECT)
    n_ch, res, fs = parse_vhdr(prefix + ".vhdr")
    print(f"{SUBJECT_DISPLAY}: n_ch={n_ch}, fs={fs:.0f} Hz, ch1 res={res[0]:.6f} µV/bit")

    raw_int = np.fromfile(prefix + ".eeg", dtype=np.int16)
    n = len(raw_int) // n_ch
    raw = raw_int[: n * n_ch].reshape(n, n_ch).T.astype(np.float64)
    for c in range(n_ch):
        raw[c] *= res.get(c, 0.1)
    print(f"Loaded {n} samples ({n/fs:.1f} s)")

    out_idx = [0, 2, 4]   # outer rings: O1, O2, Pz
    in_idx = [1, 3, 5]    # inner rings: O1, O2, Pz
    disc_idx = 6

    # Reconstruct three tEEG channels: O1, O2, Pz (16·inner − outer)
    teeg = np.stack([16.0 * raw[in_idx[k]] - raw[out_idx[k]] for k in range(3)])
    inner = raw[in_idx]
    disc = raw[disc_idx]

    # We'll work with three channels of interest at the Pz site for the figure:
    # tEEG@Pz (computed), inner@Pz, disc Ch7. Stack them for parallel processing.
    ch_pack = np.stack([teeg[2], inner[2], disc])
    ch_labels = ["tEEG @ Pz (16·in − out)", "Inner ring @ Pz", "Disc (Ch7)"]
    print(f"Pre-filter raw range per channel of interest:")
    for lbl, sig in zip(ch_labels, ch_pack):
        print(f"  {lbl:<30}  min={sig.min():7.1f}  max={sig.max():7.1f}  std={sig.std():6.1f} µV")

    # Triggers
    trigs = parse_vmrk_permissive(prefix + ".vmrk")
    print(f"Triggers parsed (Trigger,T<n> permissive): {len(trigs)}")

    # Variants: each is a list of step functions applied per-channel
    def variant_A(sig):
        x = notch(sig, 60, fs)
        x = fir_bandpass(x, 1.0, 55.0, fs)
        x = wavelet_denoise_universal(x, wavelet="db8", level=3)
        x = zscore(x)
        return x

    def variant_B(sig):
        x = notch(sig, 60, fs)
        x = fir_bandpass(x, 0.05, 55.0, fs)
        return x

    def variant_C(sig):
        x = notch(sig, 60, fs)
        x = fir_bandpass(x, 0.05, 55.0, fs)
        x = wavelet_denoise_universal(x, wavelet="db8", level=3)
        return x

    variants = [
        ("A: Adeli/Norouzi (notch + 1–55 Hz + db8 wavelet + z-score)", variant_A, "z-units"),
        ("B: Unified minimal (notch + 0.05–55 Hz)",                    variant_B, "µV"),
        ("C: Intermediate (notch + 0.05–55 Hz + db8 wavelet)",         variant_C, "µV"),
    ]

    pre_n = int(round(PRE_S * fs))
    post_n = int(round(POST_S * fs))
    base_n0 = int(round((BASELINE_S[0] - (-PRE_S)) * fs))
    base_n1 = int(round((BASELINE_S[1] - (-PRE_S)) * fs))

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    summary = {}

    for col, (vname, vfn, units) in enumerate(variants):
        print(f"\n--- Variant {vname.split(':')[0]} ---")
        ch_filtered = np.stack([vfn(sig) for sig in ch_pack])
        avg, n_used = epoch_average(ch_filtered, trigs, pre_n, post_n, base_n0, base_n1)
        t_ms = (np.arange(avg.shape[1]) - pre_n) / fs * 1000.0
        print(f"  n epochs used: {n_used}")
        for row, lbl in enumerate(ch_labels):
            wave = avg[row]
            ax = axes[row, col]
            ax.plot(t_ms, wave, color="#2c3e50", lw=1.4)
            ax.axvline(0, color="k", lw=0.7, alpha=0.4)
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            for span_lo, span_hi in PEAK_WINDOWS_MS.values():
                ax.axvspan(span_lo, span_hi, color="#f1c40f", alpha=0.06)
            n75 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["N75"], "neg")
            p100 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["P100"], "pos")
            n135 = find_peak(t_ms, wave, *PEAK_WINDOWS_MS["N135"], "neg")
            for nm, pk, c in (("N75", n75, "#c0392b"),
                              ("P100", p100, "#27ae60"),
                              ("N135", n135, "#c0392b")):
                if pk[0] is not None:
                    ax.plot(pk[0], pk[1], "o", color=c, ms=5)
                    ax.annotate(f"{nm}\n{pk[0]:.0f}", xy=pk, xytext=(5, 5),
                                textcoords="offset points", fontsize=8, color=c)
            if col == 0:
                ax.set_ylabel(f"{lbl}\n({units})", fontsize=9)
            else:
                ax.set_ylabel(units, fontsize=8)
            ax.grid(alpha=0.25)
            summary[(vname.split(":")[0].strip(), lbl)] = {
                "N75": n75, "P100": p100, "N135": n135,
                "wave_range": (float(wave.min()), float(wave.max())),
            }
            print(f"  {lbl:<30}  N75={n75[0]} ({n75[1]})  "
                  f"P100={p100[0]} ({p100[1]})  N135={n135[0]} ({n135[1]})  "
                  f"range=[{wave.min():.2f}, {wave.max():.2f}]")
        axes[0, col].set_title(vname, fontsize=10)

    for ax in axes[2]:
        ax.set_xlabel("Time (ms) — 0 = checkerboard reversal")
    fig.suptitle(
        f"{SUBJECT_DISPLAY}: preprocessing sensitivity "
        f"(n={len(trigs)} Trigger,T 1 events, "
        "tEEG = 16·inner − outer, baseline = −100…0 ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"\nSaved: {OUT_PNG}")
    return summary


if __name__ == "__main__":
    main()
