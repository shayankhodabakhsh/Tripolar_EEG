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

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════

N_CHANNELS = 11
FS = 1000  # Hz
RESOLUTION = 0.1  # µV per bit

# Channel labels (confirmed: odd=tEEG Laplacian / even=eEEG conventional)
CH_LABELS = [
    "Ch1 – Felt TCRE #1 (tEEG)",
    "Ch2 – Felt TCRE #1 (eEEG)",
    "Ch3 – Felt TCRE #2 (tEEG)",
    "Ch4 – Felt TCRE #2 (eEEG)",
    "Ch5 – Felt TCRE #3 (tEEG)",
    "Ch6 – Felt TCRE #3 (eEEG)",
    "Ch7 – Felt TCRE #4 (tEEG)",
    "Ch8 – Felt TCRE #4 (eEEG)",
    "Ch9 – Paste TCRE #5 (tEEG)",
    "Ch10 – Paste TCRE #5 (eEEG)",
    "Ch11 – Paste Disc (eEEG)",
]
SHORT_LABELS = [f"Ch{i+1}" for i in range(N_CHANNELS)]

# Electrode type grouping (0-indexed)
IDX_FELT_TEEG = [0, 2, 4, 6]       # Ch1,3,5,7 – Felt TCRE tEEG (Laplacian)
IDX_FELT_EEEG = [1, 3, 5, 7]       # Ch2,4,6,8 – Felt TCRE eEEG (conventional)
IDX_PASTE_TEEG = [8]                # Ch9 – Paste TCRE tEEG
IDX_PASTE_EEEG = [9]                # Ch10 – Paste TCRE eEEG
IDX_DISC = [10]                     # Ch11 – Paste conventional disc

# Paired tEEG/eEEG channels from the same TCRE (0-indexed)
# Each tuple: (tEEG_idx, eEEG_idx, label)
TCRE_PAIRS = [
    (0, 1, "Felt TCRE #1"),
    (2, 3, "Felt TCRE #2"),
    (4, 5, "Felt TCRE #3"),
    (6, 7, "Felt TCRE #4"),
    (8, 9, "Paste TCRE #5"),
]

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
    Patch(facecolor="#3498db", label="Felt tEEG (Ch1,3,5,7)"),
    Patch(facecolor="#95a5a6", label="Felt eEEG (Ch2,4,6,8)"),
    Patch(facecolor="#2ecc71", label="Paste tEEG (Ch9)"),
    Patch(facecolor="#9b59b6", label="Paste eEEG (Ch10)"),
    Patch(facecolor="#e74c3c", label="Paste Disc (Ch11)"),
]

EVENT_LEGEND = [
    Patch(facecolor="red", alpha=0.15, label="Stim blocks (S7)"),
    Patch(facecolor="#3498db", alpha=0.12, label="Eyes closed"),
    Line2D([0], [0], color="#27ae60", linestyle="--", label="Eyes open"),
    Line2D([0], [0], color="#3498db", linestyle="--", label="Eyes close"),
]


def ch_color(i):
    """Get color for channel index (0-based)."""
    if i in IDX_FELT_TEEG:
        return "#3498db"
    if i in IDX_PASTE_TEEG:
        return "#2ecc71"
    if i in IDX_DISC:
        return "#e74c3c"
    if i in IDX_PASTE_EEEG:
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


def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size for paired samples.

    Uses the pooled standard deviation. Returns NaN if insufficient data.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    valid = ~(np.isnan(g1) | np.isnan(g2))
    g1, g2 = g1[valid], g2[valid]
    if len(g1) < 2:
        return np.nan
    n1, n2 = len(g1), len(g2)
    s_pooled = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 +
                         (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / s_pooled


def compute_spectrogram_ssim(subject, nperseg=2048, max_freq=45):
    """
    Compute SSIM between paired tEEG/eEEG spectrogram images from the same TCRE.

    For each of the 5 TCRE pairs, we generate log-power spectrograms, normalize
    both to a shared dB range, and compute SSIM on the resulting images.

    Args:
        subject: dict from load_subject()
        nperseg: spectrogram segment length
        max_freq: upper frequency limit (Hz)

    Returns:
        dict with:
            - ssim_values: list of 5 SSIM scores (one per TCRE pair)
            - pair_labels: list of 5 pair labels
            - spectrogram_pairs: list of (Sxx_teeg, Sxx_eeeg, f, t) for plotting
    """
    if not HAS_SSIM:
        warnings.warn("scikit-image not installed — SSIM unavailable. "
                       "Install with: pip install scikit-image")
        return {
            "ssim_values": [np.nan] * len(TCRE_PAIRS),
            "pair_labels": [p[2] for p in TCRE_PAIRS],
            "spectrogram_pairs": [],
        }

    eeg = subject["eeg"]
    ssim_values = []
    pair_labels = []
    spectrogram_pairs = []

    for teeg_idx, eeeg_idx, label in TCRE_PAIRS:
        # Compute spectrograms for both channels
        cleaned_t = notch_filter(eeg[teeg_idx])
        cleaned_e = notch_filter(eeg[eeeg_idx])

        f_s, t_s, Sxx_t = signal.spectrogram(cleaned_t, fs=FS, nperseg=nperseg,
                                               noverlap=nperseg // 2, nfft=4096)
        _, _, Sxx_e = signal.spectrogram(cleaned_e, fs=FS, nperseg=nperseg,
                                          noverlap=nperseg // 2, nfft=4096)

        # Crop to max_freq
        fm = f_s <= max_freq
        Sxx_t_db = 10 * np.log10(Sxx_t[fm] + 1e-10)
        Sxx_e_db = 10 * np.log10(Sxx_e[fm] + 1e-10)

        # Normalize both to the same range [0, 1] for fair SSIM
        vmin = min(Sxx_t_db.min(), Sxx_e_db.min())
        vmax = max(Sxx_t_db.max(), Sxx_e_db.max())
        rng = vmax - vmin if vmax != vmin else 1.0
        img_t = (Sxx_t_db - vmin) / rng
        img_e = (Sxx_e_db - vmin) / rng

        # SSIM with a reasonable window (smaller of image dims)
        win = min(7, min(img_t.shape) - 1)
        if win < 3:
            win = 3
        if win % 2 == 0:
            win -= 1
        score = ssim(img_t, img_e, data_range=1.0, win_size=win)

        ssim_values.append(score)
        pair_labels.append(label)
        spectrogram_pairs.append((Sxx_t_db, Sxx_e_db, f_s[fm], t_s))

    return {
        "ssim_values": ssim_values,
        "pair_labels": pair_labels,
        "spectrogram_pairs": spectrogram_pairs,
    }


def add_event_markers(ax, events_oc, stim_blocks, close_epochs,
                      shade_blocks=True, shade_close=True, text_labels=False):
    """Add experimental event markers to any matplotlib axis.

    Args:
        text_labels: if True, add "Open"/"Closed" text at each event boundary.
                     Best used on spectrograms and alpha-envelope plots.
    """
    if shade_blocks:
        for bs, be in stim_blocks:
            ax.axvspan(bs, be, alpha=0.10, color="red")
    for lbl, samp in events_oc:
        t_sec = samp / FS
        c = "#27ae60" if lbl == "open" else "#3498db"
        ax.axvline(t_sec, color=c, linewidth=0.7, alpha=0.6, linestyle="--")
        if text_labels:
            disp_text = "Open" if lbl == "open" else "Closed"
            ax.text(t_sec, 0.97, disp_text, transform=ax.get_xaxis_transform(),
                    fontsize=7, fontweight="bold", color=c, ha="left", va="top",
                    rotation=90, alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c,
                              alpha=0.7, lw=0.5))
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

    # ─── Spectrogram SSIM (tEEG vs eEEG per TCRE pair) ───
    ssim_results = compute_spectrogram_ssim(subject)

    return {
        "alpha_snr": np.array(alpha_snr),
        "alpha_open": np.array(alpha_open),
        "alpha_closed": np.array(alpha_closed),
        "alpha_reactivity": np.array(alpha_reactivity),
        "alpha_envelopes": alpha_envelopes,
        "disc_correlation": np.array(disc_corr),
        "vep_p2p": np.array(vep_p2p),
        "ssim_values": np.array(ssim_results["ssim_values"]),
        "ssim_pair_labels": ssim_results["pair_labels"],
        "ssim_spectrogram_pairs": ssim_results["spectrogram_pairs"],
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

    # SSIM: shape (n_subjects, 5) — one per TCRE pair
    metrics["ssim_values"] = np.array(
        [results_dict[name]["ssim_values"] for name in names]
    )
    metrics["ssim_pair_labels"] = results_dict[names[0]]["ssim_pair_labels"]

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
        add_event_markers(ax1, events_oc, stim_blocks, close_epochs,
                          text_labels=True)
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
        add_event_markers(ax2, events_oc, stim_blocks, close_epochs, shade_close=False, text_labels=True)
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
        # First pass: compute all PSDs and find global min/max for shared y-scale
        psd_data = []
        global_max = 0
        global_min = np.inf
        for i in range(N_CHANNELS):
            cleaned = notch_filter(eeg[i])
            od = np.concatenate([cleaned[ep["start"]:ep["end"]] for ep in open_epochs])
            cd = np.concatenate([cleaned[ep["start"]:ep["end"]] for ep in close_epochs])
            f_o, pxx_o = signal.welch(od, fs=FS, nperseg=4096)
            f_c, pxx_c = signal.welch(cd, fs=FS, nperseg=4096)
            mask = (f_o >= 1) & (f_o <= 30)
            psd_data.append((f_o, pxx_o, f_c, pxx_c, mask))
            vals = np.concatenate([pxx_o[mask], pxx_c[mask]])
            global_max = max(global_max, vals.max())
            global_min = min(global_min, vals[vals > 0].min())
        # Second pass: plot with shared log y-scale
        for i in range(N_CHANNELS):
            ax = axes.flatten()[i]
            f_o, pxx_o, f_c, pxx_c, mask = psd_data[i]
            ax.semilogy(f_o[mask], pxx_o[mask], lw=1.5, color="#27ae60", label="Open")
            ax.semilogy(f_c[mask], pxx_c[mask], lw=1.5, color="#3498db", label="Closed")
            ax.set_ylim(global_min * 0.5, global_max * 2)
            ax.set_title(CH_LABELS[i], fontsize=9, fontweight="bold")
            ax.set_ylabel("µV²/Hz", fontsize=8)
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

    # ─── 7. Spectrogram SSIM comparison (tEEG vs eEEG per TCRE) ───
    ssim_vals = results.get("ssim_values", [])
    ssim_pairs = results.get("ssim_spectrogram_pairs", [])
    ssim_labels = results.get("ssim_pair_labels", [])

    if ssim_pairs:
        for pi, (Sxx_t_db, Sxx_e_db, f_crop, t_crop) in enumerate(ssim_pairs):
            teeg_idx, eeeg_idx, pair_label = TCRE_PAIRS[pi]
            score = ssim_vals[pi]

            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            fig.suptitle(
                f"Spectrogram Comparison — {name} — {pair_label}  "
                f"(SSIM = {score:.3f})",
                fontsize=13, fontweight="bold",
            )

            vmin = min(Sxx_t_db.min(), Sxx_e_db.min())
            vmax = max(Sxx_t_db.max(), Sxx_e_db.max())

            im0 = axes[0].pcolormesh(t_crop, f_crop, Sxx_t_db,
                                      shading="gouraud", cmap="inferno",
                                      vmin=vmin, vmax=vmax)
            axes[0].set_title(f"Ch{teeg_idx+1} — tEEG (Laplacian)", fontsize=10)
            axes[0].set_ylabel("Hz"); axes[0].set_ylim(1, 45)
            axes[0].axhline(8, color="cyan", lw=0.6, ls="--", alpha=0.5)
            axes[0].axhline(13, color="cyan", lw=0.6, ls="--", alpha=0.5)

            im1 = axes[1].pcolormesh(t_crop, f_crop, Sxx_e_db,
                                      shading="gouraud", cmap="inferno",
                                      vmin=vmin, vmax=vmax)
            axes[1].set_title(f"Ch{eeeg_idx+1} — eEEG (Conv)", fontsize=10)
            axes[1].set_ylabel("Hz"); axes[1].set_ylim(1, 45)
            axes[1].axhline(8, color="cyan", lw=0.6, ls="--", alpha=0.5)
            axes[1].axhline(13, color="cyan", lw=0.6, ls="--", alpha=0.5)

            # Difference map
            diff = Sxx_t_db - Sxx_e_db
            d_abs = max(abs(diff.min()), abs(diff.max()), 1)
            im2 = axes[2].pcolormesh(t_crop, f_crop, diff,
                                      shading="gouraud", cmap="RdBu_r",
                                      vmin=-d_abs, vmax=d_abs)
            axes[2].set_title("Difference (tEEG − eEEG)", fontsize=10)
            axes[2].set_ylabel("Hz"); axes[2].set_ylim(1, 45)
            axes[2].axhline(8, color="black", lw=0.6, ls="--", alpha=0.5)
            axes[2].axhline(13, color="black", lw=0.6, ls="--", alpha=0.5)

            for ax in axes:
                ax.set_xlabel("Time (s)")
            plt.colorbar(im0, ax=axes[0], label="dB", shrink=0.8)
            plt.colorbar(im1, ax=axes[1], label="dB", shrink=0.8)
            plt.colorbar(im2, ax=axes[2], label="ΔdB", shrink=0.8)
            plt.tight_layout()
            _save_or_show(fig, f"{name}_ssim_pair{pi+1}_{pair_label.replace(' ', '_')}.png")

        # SSIM summary bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_ssim = ["#3498db"] * 4 + ["#2ecc71"]  # Felt=blue, Paste=green
        ax.bar(range(len(ssim_vals)), ssim_vals, color=colors_ssim,
               edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(ssim_vals)))
        ax.set_xticklabels(ssim_labels, fontsize=9, rotation=15)
        ax.set_ylabel("SSIM")
        ax.set_title(f"Spectrogram SSIM (tEEG vs eEEG) — {name}",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.axhline(0.8, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.text(len(ssim_vals) - 0.5, 0.81, "high similarity", fontsize=7,
                color="gray", ha="right")
        for j, v in enumerate(ssim_vals):
            ax.text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        _save_or_show(fig, f"{name}_ssim_summary.png")

    # ─── 8. Summary dashboard ───
    plot_subject_summary(subject, results, save_dir=save_dir, show=show)
