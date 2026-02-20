"""
eeg_analysis.py — Core analysis engine for Tripolar EEG data.

Loads BrainVision format files (.eeg/.vhdr/.vmrk), performs spectral
decomposition, and extracts key metrics comparing tEEG vs conventional
electrodes for alpha-wave detection.

Usage:
    from eeg_analysis import load_subject, analyze_subject, plot_subject_summary

    subject = load_subject("data/", "SK1")
    results = analyze_subject(subject)
    plot_subject_summary(subject, results, save_dir="figures/SK1/")
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import re
import glob
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════

N_CHANNELS = 11
FS = 1000  # Hz
RESOLUTION = 0.1  # µV per bit

# Channel labels (verify odd=Conv / even=tEEG with hardware docs!)
CH_LABELS = [
    "Ch1 – SW CRE #1 (Conv)",
    "Ch2 – SW CRE #1 (tEEG)",
    "Ch3 – SW CRE #2 (Conv)",
    "Ch4 – SW CRE #2 (tEEG)",
    "Ch5 – SW CRE #3 (Conv)",
    "Ch6 – SW CRE #3 (tEEG)",
    "Ch7 – SW CRE #4 (Conv)",
    "Ch8 – SW CRE #4 (tEEG)",
    "Ch9 – Paste CRE #5 (Conv)",
    "Ch10 – Paste CRE #5 (tEEG)",
    "Ch11 – Standard Disc (Conv)",
]
SHORT_LABELS = [f"Ch{i+1}" for i in range(N_CHANNELS)]

# Electrode type grouping (0-indexed)
IDX_SW_CONV = [0, 2, 4, 6]
IDX_SW_TEEG = [1, 3, 5, 7]
IDX_PASTE_CONV = [8]
IDX_PASTE_TEEG = [9]
IDX_DISC = [10]

# EEG frequency bands
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45),
}
BAND_COLORS = {
    "Delta": "#2c3e50",
    "Theta": "#8e44ad",
    "Alpha": "#e67e22",
    "Beta": "#27ae60",
    "Gamma": "#c0392b",
}

# Color scheme by electrode type
ELEC_LEGEND = [
    Patch(facecolor="#95a5a6", label="SW Conv (Ch1,3,5,7)"),
    Patch(facecolor="#3498db", label="SW tEEG (Ch2,4,6,8)"),
    Patch(facecolor="#9b59b6", label="Paste Conv (Ch9)"),
    Patch(facecolor="#2ecc71", label="Paste tEEG (Ch10)"),
    Patch(facecolor="#e74c3c", label="Standard Disc (Ch11)"),
]

EVENT_LEGEND = [
    Patch(facecolor="red", alpha=0.15, label="Stim blocks (S7)"),
    Patch(facecolor="#3498db", alpha=0.12, label="Eyes closed"),
    Line2D([0], [0], color="#27ae60", linestyle="--", label="Eyes open"),
    Line2D([0], [0], color="#3498db", linestyle="--", label="Eyes close"),
]


def ch_color(i):
    """Get color for channel index (0-based)."""
    if i in IDX_SW_TEEG:
        return "#3498db"
    if i in IDX_PASTE_TEEG:
        return "#2ecc71"
    if i in IDX_DISC:
        return "#e74c3c"
    if i in IDX_PASTE_CONV:
        return "#9b59b6"
    return "#95a5a6"


# ═══════════════════════════════════════════════════════
# SIGNAL PROCESSING HELPERS
# ═══════════════════════════════════════════════════════


def notch_filter(data, freq=60, Q=30, fs=FS):
    """Remove power line noise at given frequency."""
    b, a = signal.iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)


def bandpass_filter(data, low, high, fs=FS, order=4):
    """Butterworth bandpass filter."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, data)


def compute_envelope(data, smooth_s=1.0, fs=FS):
    """Amplitude envelope via Hilbert transform + smoothing."""
    analytic = hilbert(data)
    env = np.abs(analytic)
    kernel = np.ones(int(fs * smooth_s)) / int(fs * smooth_s)
    return np.convolve(env, kernel, mode="same")


def add_event_markers(ax, events_oc, stim_blocks, close_epochs,
                      shade_blocks=True, shade_close=True):
    """Add experimental event markers to any matplotlib axis."""
    if shade_blocks:
        for bs, be in stim_blocks:
            ax.axvspan(bs, be, alpha=0.10, color="red")
    for lbl, samp in events_oc:
        c = "#27ae60" if lbl == "open" else "#3498db"
        ax.axvline(samp / FS, color=c, linewidth=0.7, alpha=0.6, linestyle="--")
    if shade_close:
        for ep in close_epochs:
            ax.axvspan(ep["start_s"], ep["end_s"], alpha=0.06, color="#3498db")


# ═══════════════════════════════════════════════════════
# FILE DISCOVERY & LOADING
# ═══════════════════════════════════════════════════════


def discover_subjects(data_dir):
    """
    Auto-discover all subjects in a data directory.

    Returns a list of dicts with subject info:
        [{'name': 'SK1', 'eeg': '...eeg', 'vhdr': '...vhdr', 'vmrk': '...vmrk',
          'avg': '...avg', 'avg_vhdr': '...vhdr', 'avg_vmrk': '...vmrk'}, ...]
    """
    eeg_files = sorted(glob.glob(os.path.join(data_dir, "*.eeg")))
    subjects = []

    for eeg_path in eeg_files:
        basename = os.path.basename(eeg_path)
        # Skip trigger avg files
        if "Trigger" in basename or "trigger" in basename:
            continue

        prefix = os.path.splitext(eeg_path)[0]  # full path without .eeg
        dir_path = os.path.dirname(eeg_path)

        # Derive name: everything before the date pattern
        name_part = os.path.basename(prefix)
        # Try to extract a clean subject name (before date-like patterns)
        # Handles: "SK1 2-19-2026", "SK1_2-19-2026", "Caitlin 1 2-11-26",
        #          "Hunter1 217-26", "LuciTest", "Gab"
        match = re.match(
            r"^(.+?)[\s_]+\d{1,4}[-]\d{1,2}[-]\d{2,4}", name_part
        )
        if not match:
            # Try without separator (e.g. "Hunter1 217-26")
            match = re.match(r"^(.+?)[\s_]+\d{2,4}-\d{2,4}", name_part)
        if match:
            subj_name = match.group(1).strip().rstrip("_")
        else:
            # Fallback: just use the full basename
            subj_name = name_part

        # Find associated files
        vhdr = prefix + ".vhdr"
        vmrk = prefix + ".vmrk"

        # Find Triggers files (may have slight name variations)
        avg_files = glob.glob(prefix + "-Triggers.avg") + glob.glob(prefix + "-Trigg*.avg")
        avg_vhdr = glob.glob(prefix + "-Triggers.vhdr") + glob.glob(prefix + "-Trigg*.vhdr")
        avg_vmrk = glob.glob(prefix + "-Triggers.vmrk") + glob.glob(prefix + "-Trigg*.vmrk")

        subj = {
            "name": subj_name,
            "basename": name_part,
            "eeg": eeg_path if os.path.exists(eeg_path) else None,
            "vhdr": vhdr if os.path.exists(vhdr) else None,
            "vmrk": vmrk if os.path.exists(vmrk) else None,
            "avg": avg_files[0] if avg_files else None,
            "avg_vhdr": avg_vhdr[0] if avg_vhdr else None,
            "avg_vmrk": avg_vmrk[0] if avg_vmrk else None,
        }
        subjects.append(subj)

    return subjects


def parse_vmrk(vmrk_path):
    """
    Parse a BrainVision .vmrk marker file.

    Returns:
        stim_samples: list of int — sample positions of S7 stimulus triggers
        events_oc: list of (label, sample) — eyes open/close markers
    """
    stim_samples = []
    events_oc = []

    with open(vmrk_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Mk"):
                continue

            # Parse: Mk<n>=<Type>,<Description>,<Position>,<Size>,<Channel>
            eq_pos = line.index("=")
            parts = line[eq_pos + 1 :].split(",")
            if len(parts) < 3:
                continue

            marker_type = parts[0].strip()
            description = parts[1].strip()
            try:
                position = int(parts[2].strip())
            except ValueError:
                continue

            # Stimulus triggers
            if marker_type == "Stimulus" and "S" in description:
                stim_samples.append(position)

            # Eyes open/close comments
            if marker_type == "Comment":
                desc_lower = description.lower()
                if "open" in desc_lower:
                    events_oc.append(("open", position))
                elif "close" in desc_lower or "clos" in desc_lower:
                    events_oc.append(("close", position))

    return stim_samples, events_oc


def load_subject(data_dir, subject_name=None, subject_info=None):
    """
    Load a single subject's data.

    Args:
        data_dir: path to data directory
        subject_name: subject name (will auto-discover files)
        subject_info: dict from discover_subjects() (overrides subject_name)

    Returns:
        dict with all loaded data and metadata
    """
    # Find subject files
    if subject_info is None:
        subjects = discover_subjects(data_dir)
        matches = [s for s in subjects if s["name"] == subject_name or
                   subject_name in s["basename"]]
        if not matches:
            raise FileNotFoundError(
                f"Subject '{subject_name}' not found. Available: "
                f"{[s['name'] for s in subjects]}"
            )
        subject_info = matches[0]

    # Load raw EEG
    raw = np.fromfile(subject_info["eeg"], dtype=np.int16)
    n_samples = len(raw) // N_CHANNELS
    eeg = raw[: n_samples * N_CHANNELS].reshape(n_samples, N_CHANNELS).T.astype(
        np.float64
    )
    eeg *= RESOLUTION  # → µV
    t = np.arange(n_samples) / FS

    # Parse markers
    stim_samples, events_oc = [], []
    if subject_info["vmrk"]:
        stim_samples, events_oc = parse_vmrk(subject_info["vmrk"])

    # Compute stim block boundaries
    stim_blocks = []
    if stim_samples:
        stim_times = np.array(stim_samples) / FS
        # Find gaps > 5s to separate blocks
        gaps = np.where(np.diff(stim_times) > 5)[0]
        block_starts = [0] + (gaps + 1).tolist()
        block_ends = gaps.tolist() + [len(stim_times) - 1]
        for bs_idx, be_idx in zip(block_starts, block_ends):
            stim_blocks.append(
                (stim_times[bs_idx] - 0.5, stim_times[be_idx] + 0.5)
            )

    # Build open/close epochs
    epochs_oc = []
    for i in range(len(events_oc) - 1):
        lbl, start = events_oc[i]
        _, end = events_oc[i + 1]
        epochs_oc.append(
            {
                "label": lbl,
                "start": start,
                "end": end,
                "start_s": start / FS,
                "end_s": end / FS,
                "dur": (end - start) / FS,
            }
        )
    if events_oc:
        last_start = events_oc[-1][1]
        last_end = min(last_start + 30 * FS, n_samples)
        epochs_oc.append(
            {
                "label": events_oc[-1][0],
                "start": last_start,
                "end": last_end,
                "start_s": last_start / FS,
                "end_s": last_end / FS,
                "dur": (last_end - last_start) / FS,
            }
        )

    open_epochs = [ep for ep in epochs_oc if ep["label"] == "open"]
    close_epochs = [ep for ep in epochs_oc if ep["label"] == "close"]

    # Load pre-averaged VEP if available
    avg_data = None
    avg_t = None
    avg_n_segments = None
    if subject_info.get("avg") and os.path.exists(subject_info["avg"]):
        avg_raw = np.fromfile(subject_info["avg"], dtype=np.float32)
        # Parse avg_vhdr to get segment points
        avg_pts = 500  # default
        if subject_info.get("avg_vhdr") and os.path.exists(subject_info["avg_vhdr"]):
            with open(subject_info["avg_vhdr"], "r", errors="ignore") as f:
                for line in f:
                    if "SegmentDataPoints" in line:
                        avg_pts = int(line.split("=")[1].strip())
                    if "AveragedSegments" in line:
                        avg_n_segments = int(line.split("=")[1].strip())
        if len(avg_raw) >= avg_pts * N_CHANNELS:
            avg_data = avg_raw[: avg_pts * N_CHANNELS].reshape(
                avg_pts, N_CHANNELS
            ).T
            avg_t = np.arange(avg_pts) / FS * 1000 - 100  # ms

    return {
        "name": subject_info["name"],
        "basename": subject_info["basename"],
        "info": subject_info,
        "eeg": eeg,
        "t": t,
        "n_samples": n_samples,
        "stim_samples": stim_samples,
        "stim_blocks": stim_blocks,
        "events_oc": events_oc,
        "epochs_oc": epochs_oc,
        "open_epochs": open_epochs,
        "close_epochs": close_epochs,
        "avg_data": avg_data,
        "avg_t": avg_t,
        "avg_n_segments": avg_n_segments,
    }


# ═══════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════


def analyze_subject(subject):
    """
    Run full analysis on a loaded subject.

    Returns dict with per-channel metrics:
        - alpha_snr: alpha SNR in dB (alpha vs neighboring bands)
        - alpha_open: mean alpha power during eyes-open
        - alpha_closed: mean alpha power during eyes-closed
        - alpha_reactivity: ratio closed/open (>1 = Berger effect)
        - alpha_envelopes: smoothed alpha envelope per channel
        - disc_correlation: Pearson r of alpha envelope vs Ch11
        - vep_p2p: VEP peak-to-peak amplitude (if avg available)
    """
    eeg = subject["eeg"]
    n_samples = subject["n_samples"]
    open_epochs = subject["open_epochs"]
    close_epochs = subject["close_epochs"]

    alpha_snr = []
    alpha_open = []
    alpha_closed = []
    alpha_envelopes = []

    for i in range(N_CHANNELS):
        cleaned = notch_filter(eeg[i])

        # ─── Alpha SNR (whole recording) ───
        f, pxx = signal.welch(cleaned, fs=FS, nperseg=4096)
        ap = np.mean(pxx[(f >= 8) & (f <= 13)])
        nap = np.mean(pxx[((f >= 4) & (f < 8)) | ((f > 13) & (f <= 30))])
        alpha_snr.append(10 * np.log10(ap / (nap + 1e-10)))

        # ─── Alpha envelope ───
        alpha_bp = bandpass_filter(cleaned, 8, 13)
        env = compute_envelope(alpha_bp)
        alpha_envelopes.append(env)

        # ─── Eyes open/close alpha power ───
        if open_epochs and close_epochs:
            open_data = np.concatenate(
                [cleaned[ep["start"] : ep["end"]] for ep in open_epochs]
            )
            close_data = np.concatenate(
                [cleaned[ep["start"] : ep["end"]] for ep in close_epochs]
            )
            f_o, pxx_o = signal.welch(open_data, fs=FS, nperseg=4096)
            f_c, pxx_c = signal.welch(close_data, fs=FS, nperseg=4096)
            alpha_open.append(np.mean(pxx_o[(f_o >= 8) & (f_o <= 13)]))
            alpha_closed.append(np.mean(pxx_c[(f_c >= 8) & (f_c <= 13)]))
        else:
            alpha_open.append(np.nan)
            alpha_closed.append(np.nan)

    # ─── Alpha reactivity ───
    alpha_reactivity = np.array(alpha_closed) / (np.array(alpha_open) + 1e-10)

    # ─── Correlation with disc (Ch11) ───
    ds = 10
    ref_env = alpha_envelopes[10][::ds]
    disc_corr = []
    for i in range(N_CHANNELS):
        env_ds = alpha_envelopes[i][::ds]
        r = np.corrcoef(ref_env, env_ds)[0, 1]
        disc_corr.append(r)

    # ─── VEP peak-to-peak ───
    vep_p2p = []
    if subject["avg_data"] is not None:
        for i in range(N_CHANNELS):
            vep_p2p.append(
                subject["avg_data"][i].max() - subject["avg_data"][i].min()
            )
    else:
        vep_p2p = [np.nan] * N_CHANNELS

    return {
        "alpha_snr": np.array(alpha_snr),
        "alpha_open": np.array(alpha_open),
        "alpha_closed": np.array(alpha_closed),
        "alpha_reactivity": np.array(alpha_reactivity),
        "alpha_envelopes": alpha_envelopes,
        "disc_correlation": np.array(disc_corr),
        "vep_p2p": np.array(vep_p2p),
    }


def get_group_metrics(results_dict):
    """
    Compile per-subject results into group-level arrays.

    Args:
        results_dict: dict of {subject_name: results} from analyze_subject()

    Returns:
        dict with arrays of shape (n_subjects, n_channels) for each metric,
        plus subject_names list.
    """
    names = list(results_dict.keys())
    n_subj = len(names)

    metrics = {}
    for key in ["alpha_snr", "alpha_open", "alpha_closed",
                 "alpha_reactivity", "disc_correlation", "vep_p2p"]:
        arr = np.array([results_dict[name][key] for name in names])
        metrics[key] = arr  # shape: (n_subjects, 11)

    metrics["subject_names"] = names
    return metrics


# ═══════════════════════════════════════════════════════
# PER-SUBJECT PLOTTING
# ═══════════════════════════════════════════════════════


def plot_subject_summary(subject, results, save_dir=None, show=True):
    """
    Generate the 4-panel summary dashboard for a single subject.

    Args:
        subject: dict from load_subject()
        results: dict from analyze_subject()
        save_dir: if set, save figures here
        show: if True, call plt.show()
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    name = subject["name"]
    colors = [ch_color(i) for i in range(N_CHANNELS)]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Summary Dashboard — {name}", fontsize=14, fontweight="bold")

    # 1. Alpha SNR
    axes[0].bar(range(N_CHANNELS), results["alpha_snr"], color=colors,
                edgecolor="k", lw=0.5)
    axes[0].set_xticks(range(N_CHANNELS))
    axes[0].set_xticklabels(SHORT_LABELS, fontsize=7)
    axes[0].set_ylabel("dB")
    axes[0].set_title("Alpha SNR")
    axes[0].axhline(0, color="gray", lw=0.5)

    # 2. Alpha Reactivity
    axes[1].bar(range(N_CHANNELS), results["alpha_reactivity"], color=colors,
                edgecolor="k", lw=0.5)
    axes[1].set_xticks(range(N_CHANNELS))
    axes[1].set_xticklabels(SHORT_LABELS, fontsize=7)
    axes[1].set_ylabel("Closed / Open")
    axes[1].set_title("Alpha Reactivity")
    axes[1].axhline(1, color="gray", lw=0.5, ls="--")

    # 3. Disc Correlation
    axes[2].bar(range(N_CHANNELS), results["disc_correlation"], color=colors,
                edgecolor="k", lw=0.5)
    axes[2].set_xticks(range(N_CHANNELS))
    axes[2].set_xticklabels(SHORT_LABELS, fontsize=7)
    axes[2].set_ylabel("Pearson r")
    axes[2].set_title("Corr. with Disc (Ch11)")

    # 4. VEP P2P
    axes[3].bar(range(N_CHANNELS), results["vep_p2p"], color=colors,
                edgecolor="k", lw=0.5)
    axes[3].set_xticks(range(N_CHANNELS))
    axes[3].set_xticklabels(SHORT_LABELS, fontsize=7)
    axes[3].set_ylabel("µV")
    axes[3].set_title("VEP Peak-to-Peak")
    axes[3].legend(handles=ELEC_LEGEND, fontsize=6, loc="upper right")

    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{name}_summary.png"),
                    dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_subject_full(subject, results, save_dir=None, show=True):
    """
    Generate the full set of per-subject figures (spectrograms, band
    decomposition, alpha envelopes, PSD, VEP, etc.).

    This is the plotting equivalent of the single-subject notebook.
    Saves individual figure files to save_dir.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    eeg = subject["eeg"]
    t = subject["t"]
    name = subject["name"]
    events_oc = subject["events_oc"]
    stim_blocks = subject["stim_blocks"]
    close_epochs = subject["close_epochs"]
    open_epochs = subject["open_epochs"]
    alpha_envelopes = results["alpha_envelopes"]

    def _save_or_show(fig, filename):
        if save_dir:
            fig.savefig(os.path.join(save_dir, filename),
                        dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ─── 1. Raw traces ───
    fig, axes = plt.subplots(11, 1, figsize=(18, 22), sharex=True)
    fig.suptitle(f"Raw EEG (µV) — {name}", fontsize=14, fontweight="bold", y=1.0)
    ds = 10
    for i in range(N_CHANNELS):
        ax = axes[i]
        ax.plot(t[::ds], eeg[i, ::ds], lw=0.3, color="#2c3e50")
        ax.set_ylabel(f"Ch{i+1}", fontsize=9)
        ymax = np.percentile(np.abs(eeg[i]), 99.5)
        ax.set_ylim(-ymax, ymax)
        ax.tick_params(labelsize=8)
        add_event_markers(ax, events_oc, stim_blocks, close_epochs)
    axes[0].legend(handles=EVENT_LEGEND, fontsize=7, loc="upper right", ncol=4)
    axes[0].set_xlim(0, t[-1])
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    _save_or_show(fig, f"{name}_raw_traces.png")

    # ─── 2. PSD (notch filtered, 1-30 Hz) ───
    fig, axes = plt.subplots(6, 2, figsize=(16, 20))
    fig.suptitle(f"PSD After 60 Hz Notch — {name}", fontsize=14, fontweight="bold", y=1.0)
    for i in range(N_CHANNELS):
        ax = axes.flatten()[i]
        cleaned = notch_filter(eeg[i])
        f, pxx = signal.welch(cleaned, fs=FS, nperseg=4096)
        mask = (f >= 1) & (f <= 30)
        ax.plot(f[mask], pxx[mask], lw=1.5, color="#2c3e50")
        amask = (f >= 8) & (f <= 13) & mask
        ax.fill_between(f[amask], pxx[amask], alpha=0.4, color="#e67e22", label="Alpha")
        ax.set_title(CH_LABELS[i], fontsize=9, fontweight="bold")
        ax.set_xlabel("Hz", fontsize=8)
        ax.set_ylabel("µV²/Hz", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    axes.flatten()[11].set_visible(False)
    plt.tight_layout()
    _save_or_show(fig, f"{name}_psd.png")

    # ─── 3. Spectrograms ───
    for i in range(N_CHANNELS):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7),
                                        gridspec_kw={"height_ratios": [1, 3]})
        fig.suptitle(f"Spectrogram — {name} — {CH_LABELS[i]}", fontsize=12, fontweight="bold")
        cleaned = notch_filter(eeg[i])
        bp = bandpass_filter(cleaned, 1, 45)
        ax1.plot(t[::ds], bp[::ds], lw=0.3, color="#34495e")
        ax1.set_ylabel("µV"); ax1.set_xlim(0, t[-1])
        add_event_markers(ax1, events_oc, stim_blocks, close_epochs)
        nperseg = 2048
        f_s, t_s, Sxx = signal.spectrogram(cleaned, fs=FS, nperseg=nperseg,
                                            noverlap=nperseg // 2, nfft=4096)
        fm = f_s <= 45
        im = ax2.pcolormesh(t_s, f_s[fm], 10 * np.log10(Sxx[fm] + 1e-10),
                             shading="gouraud", cmap="inferno", vmin=-20)
        ax2.set_ylabel("Hz"); ax2.set_xlabel("Time (s)")
        ax2.set_ylim(1, 45)
        ax2.axhline(8, color="cyan", lw=0.8, ls="--", alpha=0.7)
        ax2.axhline(13, color="cyan", lw=0.8, ls="--", alpha=0.7)
        add_event_markers(ax2, events_oc, stim_blocks, close_epochs, shade_close=False)
        plt.colorbar(im, ax=ax2, label="dB")
        plt.tight_layout()
        _save_or_show(fig, f"{name}_spectrogram_ch{i+1}.png")

    # ─── 4. Alpha envelope ───
    fig, axes = plt.subplots(11, 1, figsize=(17, 24), sharex=True)
    fig.suptitle(f"Alpha Envelope (8–13 Hz) — {name}", fontsize=14, fontweight="bold", y=1.0)
    ds_e = 50
    for i in range(N_CHANNELS):
        ax = axes[i]
        env = alpha_envelopes[i]
        ax.plot(t[::ds_e], env[::ds_e], lw=1, color="#e74c3c")
        ax.fill_between(t[::ds_e], 0, env[::ds_e], alpha=0.2, color="#e74c3c")
        ax.set_ylabel(f"Ch{i+1}", fontsize=9)
        ax.tick_params(labelsize=8)
        add_event_markers(ax, events_oc, stim_blocks, close_epochs)
    axes[0].legend(handles=EVENT_LEGEND, fontsize=7, loc="upper right", ncol=4)
    axes[0].set_xlim(0, t[-1])
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    _save_or_show(fig, f"{name}_alpha_envelope.png")

    # ─── 5. Eyes open vs closed PSD ───
    if open_epochs and close_epochs:
        fig, axes = plt.subplots(6, 2, figsize=(16, 20))
        fig.suptitle(f"Eyes Open vs Closed — {name}", fontsize=14, fontweight="bold", y=1.0)
        for i in range(N_CHANNELS):
            ax = axes.flatten()[i]
            cleaned = notch_filter(eeg[i])
            od = np.concatenate([cleaned[ep["start"]:ep["end"]] for ep in open_epochs])
            cd = np.concatenate([cleaned[ep["start"]:ep["end"]] for ep in close_epochs])
            f_o, pxx_o = signal.welch(od, fs=FS, nperseg=4096)
            f_c, pxx_c = signal.welch(cd, fs=FS, nperseg=4096)
            mask = (f_o >= 1) & (f_o <= 30)
            ax.plot(f_o[mask], pxx_o[mask], lw=1.5, color="#27ae60", label="Open")
            ax.plot(f_c[mask], pxx_c[mask], lw=1.5, color="#3498db", label="Closed")
            ax.set_title(CH_LABELS[i], fontsize=9, fontweight="bold")
            ax.legend(fontsize=7); ax.tick_params(labelsize=7)
        axes.flatten()[11].set_visible(False)
        plt.tight_layout()
        _save_or_show(fig, f"{name}_open_vs_closed.png")

    # ─── 6. VEP ───
    if subject["avg_data"] is not None:
        fig, axes = plt.subplots(6, 2, figsize=(16, 18))
        fig.suptitle(f"VEP (n={subject['avg_n_segments'] or '?'}) — {name}",
                     fontsize=14, fontweight="bold", y=1.0)
        avg_t = subject["avg_t"]
        avg_data = subject["avg_data"]
        for i in range(N_CHANNELS):
            ax = axes.flatten()[i]
            ax.plot(avg_t, avg_data[i], lw=1.5, color=ch_color(i))
            ax.axvline(0, color="red", lw=1, ls="--", alpha=0.7)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title(CH_LABELS[i], fontsize=9, fontweight="bold")
            ax.set_xlim(-100, 400); ax.tick_params(labelsize=7)
        axes.flatten()[11].set_visible(False)
        plt.tight_layout()
        _save_or_show(fig, f"{name}_vep.png")

    # ─── 7. Summary dashboard ───
    plot_subject_summary(subject, results, save_dir=save_dir, show=show)
