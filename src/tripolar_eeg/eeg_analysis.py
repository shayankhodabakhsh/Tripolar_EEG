"""
eeg_analysis.py — Core analysis engine for Tripolar EEG data.

Loads BrainVision format files (.eeg/.vhdr/.vmrk), performs spectral
decomposition, and extracts key metrics comparing tEEG vs conventional
electrodes for alpha-wave detection.

Supports multiple electrode configurations (Felt TCRE, Gel TCRE) via
the ElectrodeConfig system.

Usage:
    from tripolar_eeg.eeg_analysis import load_subject, analyze_subject, plot_subject_summary

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
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# ELECTRODE CONFIGURATION SYSTEM
# ═══════════════════════════════════════════════════════


@dataclass
class ElectrodeConfig:
    """Defines the electrode layout for a recording type."""
    name: str
    n_channels: int
    ch_labels: List[str]
    short_labels: List[str]
    idx_teeg: List[int]
    idx_eeeg: List[int]
    idx_paste_teeg: List[int]
    idx_paste_eeeg: List[int]
    idx_disc: List[int]
    tcre_pairs: List[Tuple[int, int, str]]
    # Per-channel amplitude scaling applied after int16->µV conversion.
    # Maps channel index -> divisor (e.g. {0: 187} means ch0 /= 187).
    amplitude_scaling: Dict[int, float] = field(default_factory=dict)
    # Abstract channel-type tag for each channel (for cross-config comparison)
    channel_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.short_labels:
            self.short_labels = [f"Ch{i+1}" for i in range(self.n_channels)]


FELT_TCRE_CONFIG = ElectrodeConfig(
    name="Felt TCRE",
    n_channels=11,
    ch_labels=[
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
    ],
    short_labels=[f"Ch{i+1}" for i in range(11)],
    idx_teeg=[0, 2, 4, 6],
    idx_eeeg=[1, 3, 5, 7],
    idx_paste_teeg=[8],
    idx_paste_eeeg=[9],
    idx_disc=[10],
    tcre_pairs=[
        (0, 1, "Felt TCRE #1"),
        (2, 3, "Felt TCRE #2"),
        (4, 5, "Felt TCRE #3"),
        (6, 7, "Felt TCRE #4"),
        (8, 9, "Paste TCRE #5"),
    ],
    amplitude_scaling={},
    channel_types=[
        "FELT_TEEG", "FELT_EEEG", "FELT_TEEG", "FELT_EEEG",
        "FELT_TEEG", "FELT_EEEG", "FELT_TEEG", "FELT_EEEG",
        "PASTE_TEEG", "PASTE_EEEG", "DISC",
    ],
)

GEL_TCRE_CONFIG = ElectrodeConfig(
    name="Gel TCRE",
    n_channels=7,
    ch_labels=[
        "Ch1 – Gel TCRE O1 (tEEG)",
        "Ch2 – Gel TCRE O1 (eEEG)",
        "Ch3 – Gel TCRE O2 (tEEG)",
        "Ch4 – Gel TCRE O2 (eEEG)",
        "Ch5 – Paste TCRE Pz (tEEG)",
        "Ch6 – Paste TCRE Pz (eEEG)",
        "Ch7 – Normal EEG Pz (disc)",
    ],
    short_labels=[f"Ch{i+1}" for i in range(7)],
    idx_teeg=[0, 2],
    idx_eeeg=[1, 3],
    idx_paste_teeg=[4],
    idx_paste_eeeg=[5],
    idx_disc=[6],
    tcre_pairs=[
        (0, 1, "Gel TCRE O1"),
        (2, 3, "Gel TCRE O2"),
        (4, 5, "Paste TCRE Pz"),
    ],
    amplitude_scaling={},
    channel_types=[
        "GEL_TEEG", "GEL_EEEG", "GEL_TEEG", "GEL_EEEG",
        "PASTE_TEEG", "PASTE_EEEG", "DISC",
    ],
)

CONFIG_BY_NCHAN = {
    11: FELT_TCRE_CONFIG,
    7: GEL_TCRE_CONFIG,
}


def detect_config_from_vhdr(vhdr_path):
    """Read NumberOfChannels from a .vhdr header and return the matching config."""
    n_ch = None
    with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("NumberOfChannels"):
                n_ch = int(line.split("=")[1].strip())
                break
    if n_ch is None:
        raise ValueError(f"Could not parse NumberOfChannels from {vhdr_path}")
    config = CONFIG_BY_NCHAN.get(n_ch)
    if config is None:
        raise ValueError(
            f"No electrode config for {n_ch} channels. "
            f"Known configs: {list(CONFIG_BY_NCHAN.keys())}"
        )
    return config


# ═══════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE CONSTANTS (Felt TCRE defaults)
# ═══════════════════════════════════════════════════════

N_CHANNELS = 11
FS = 1000  # Hz
RESOLUTION = 0.1  # µV per bit

ADC_CLIP_UV = 3276.6

CH_LABELS = FELT_TCRE_CONFIG.ch_labels
SHORT_LABELS = FELT_TCRE_CONFIG.short_labels

IDX_FELT_TEEG = FELT_TCRE_CONFIG.idx_teeg
IDX_FELT_EEEG = FELT_TCRE_CONFIG.idx_eeeg
IDX_PASTE_TEEG = FELT_TCRE_CONFIG.idx_paste_teeg
IDX_PASTE_EEEG = FELT_TCRE_CONFIG.idx_paste_eeeg
IDX_DISC = FELT_TCRE_CONFIG.idx_disc

TCRE_PAIRS = FELT_TCRE_CONFIG.tcre_pairs

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


def ch_color(i, config=None):
    """Get color for channel index (0-based). Uses config if provided."""
    if config is None:
        config = FELT_TCRE_CONFIG
    if i in config.idx_teeg:
        return "#3498db"
    if i in config.idx_paste_teeg:
        return "#2ecc71"
    if i in config.idx_disc:
        return "#e74c3c"
    if i in config.idx_paste_eeeg:
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


def wavelet_denoise(data, wavelet="db8", level=3):
    """Universal-threshold soft-thresholding wavelet denoise.

    Used only as part of pipeline variant A (the original gel-TCRE pipeline).
    Variant B (primary) and variant C do not apply this step.
    """
    import pywt
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma == 0:
        return np.asarray(data, dtype=float)
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]
    out = pywt.waverec(new_coeffs, wavelet)
    return np.asarray(out[: len(data)], dtype=float)


def fir_bandpass(data, low, high, fs=FS, numtaps=2001, window="hamming"):
    """Zero-phase FIR bandpass via filtfilt of a windowed-sinc kernel.

    Used by pipeline variants A and B. A long FIR is needed for the
    0.05-55 Hz primary band; Butterworth at 0.05 Hz is numerically unstable.
    """
    from scipy.signal import firwin
    if numtaps % 2 == 0:
        numtaps += 1
    taps = firwin(numtaps, [low, high], pass_zero=False, fs=fs, window=window)
    return filtfilt(taps, [1.0], data)


def apply_pipeline_variant(data, variant="B", fs=FS):
    """Apply preprocessing variant A, B (primary), or C to a 1-D channel.

    Variants are defined in the paper's Methods §2.3:
      A: notch + zero-phase FIR bandpass 0.05-55 Hz + Daubechies-8 level-3
         wavelet denoise + per-channel z-score (the original gel-TCRE pipeline).
      B: notch + zero-phase FIR bandpass 0.05-55 Hz only (PRIMARY).
      C: notch only (no bandpass, no denoise, no z-score).

    Returns a 1-D numpy array of the same length as the input.
    """
    v = (variant or "B").upper()
    out = notch_filter(data, fs=fs)
    if v in ("A", "B"):
        out = fir_bandpass(out, 0.05, 55, fs=fs)
    if v == "A":
        out = wavelet_denoise(out, wavelet="db8", level=3)
        std = np.std(out)
        if std > 0:
            out = (out - np.mean(out)) / std
    return out


def compute_envelope(data, smooth_s=1.0, fs=FS):
    """Amplitude envelope via Hilbert transform + smoothing."""
    analytic = hilbert(data)
    env = np.abs(analytic)
    kernel = np.ones(int(fs * smooth_s)) / int(fs * smooth_s)
    return np.convolve(env, kernel, mode="same")


def _max_run_length(mask_1d: np.ndarray) -> int:
    """Return the longest consecutive True-run length in a 1D boolean array."""
    if mask_1d.size == 0:
        return 0
    m = np.asarray(mask_1d, dtype=bool)
    if not np.any(m):
        return 0
    x = np.concatenate(([False], m, [False])).astype(np.int8)
    d = np.diff(x)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return int(m.sum())
    return int(np.max(ends - starts))


def compute_adc_clipping(eeg_uv: np.ndarray, threshold_uv: float = ADC_CLIP_UV,
                         n_channels: int = None):
    """
    Detect likely ADC clipping (saturation) in µV-scaled EEG.

    Args:
        eeg_uv: shape (n_channels, n_samples) in µV (post `RESOLUTION` scaling)
        threshold_uv: values with abs(x) >= threshold are considered clipped
        n_channels: expected channel count (auto-detected from array if None)

    Returns:
        dict with per-channel arrays:
          - clip_count: number of clipped samples
          - clip_fraction: fraction of samples clipped
          - clip_max_run_samples: longest consecutive clipped run length
    """
    x = np.asarray(eeg_uv)
    if n_channels is None:
        n_channels = x.shape[0]
    if x.ndim != 2 or x.shape[0] != n_channels:
        raise ValueError(f"Expected eeg_uv shape ({n_channels}, n_samples), got {x.shape}")
    mask = np.abs(x) >= float(threshold_uv)
    clip_count = mask.sum(axis=1).astype(int)
    n = x.shape[1]
    clip_fraction = clip_count / max(n, 1)
    clip_max_run_samples = np.array([_max_run_length(mask[i]) for i in range(n_channels)], dtype=int)
    return {
        "clip_count": clip_count,
        "clip_fraction": clip_fraction,
        "clip_max_run_samples": clip_max_run_samples,
        "threshold_uv": float(threshold_uv),
    }


def cohens_d(group1, group2):
    """
    Compute Cohen's d effect size for two independent groups.

    Uses the pooled standard deviation. Handles unequal group sizes.
    Returns NaN if insufficient data.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
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

    For each TCRE pair (from subject's config), we generate log-power
    spectrograms, normalize both to a shared dB range, and compute SSIM.

    Args:
        subject: dict from load_subject()
        nperseg: spectrogram segment length
        max_freq: upper frequency limit (Hz)

    Returns:
        dict with:
            - ssim_values: list of SSIM scores (one per TCRE pair)
            - pair_labels: list of pair labels
            - spectrogram_pairs: list of (Sxx_teeg, Sxx_eeeg, f, t) for plotting
    """
    cfg = subject.get("config", FELT_TCRE_CONFIG)
    tcre_pairs = cfg.tcre_pairs

    if not HAS_SSIM:
        warnings.warn("scikit-image not installed — SSIM unavailable. "
                       "Install with: pip install scikit-image")
        return {
            "ssim_values": [np.nan] * len(tcre_pairs),
            "pair_labels": [p[2] for p in tcre_pairs],
            "spectrogram_pairs": [],
        }

    eeg = subject["eeg"]
    ssim_values = []
    pair_labels = []
    spectrogram_pairs = []

    for teeg_idx, eeeg_idx, label in tcre_pairs:
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


def discover_subjects(data_dir, recursive=False, standard_protocol_only=False):
    """
    Auto-discover all subjects in a data directory.

    Args:
        data_dir: path to data directory
        recursive: if True, scan subdirectories (for Gel TCRE date-based folders)
        standard_protocol_only: if True, only include recordings that have
            both stimulus events and eyes open/close markers in their .vmrk

    Returns a list of dicts with subject info:
        [{'name': 'SK1', 'eeg': '...eeg', 'vhdr': '...vhdr', 'vmrk': '...vmrk',
          'avg': '...avg', 'avg_vhdr': '...vhdr', 'avg_vmrk': '...vmrk'}, ...]
    """
    if recursive:
        eeg_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.eeg"),
                                     recursive=True))
    else:
        eeg_files = sorted(glob.glob(os.path.join(data_dir, "*.eeg")))
    subjects = []

    for eeg_path in eeg_files:
        basename = os.path.basename(eeg_path)
        if "Trigger" in basename or "trigger" in basename:
            continue

        prefix = os.path.splitext(eeg_path)[0]

        name_part = os.path.basename(prefix)
        match = re.match(
            r"^(.+?)[\s_]+\d{1,4}[-]\d{1,2}[-]\d{2,4}", name_part
        )
        if not match:
            match = re.match(r"^(.+?)[\s_]+\d{2,4}-\d{2,4}", name_part)
        if match:
            subj_name = match.group(1).strip().rstrip("_").rstrip("-")
        else:
            subj_name = name_part

        vhdr = prefix + ".vhdr"
        vmrk = prefix + ".vmrk"

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

        if standard_protocol_only and subj["vmrk"]:
            stim_samples, events_oc = parse_vmrk(subj["vmrk"])
            has_open = any(lbl == "open" for lbl, _ in events_oc)
            has_close = any(lbl == "close" for lbl, _ in events_oc)
            if not (stim_samples and has_open and has_close):
                continue

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


def load_subject(data_dir, subject_name=None, subject_info=None, config=None):
    """
    Load a single subject's data.

    Args:
        data_dir: path to data directory
        subject_name: subject name (will auto-discover files)
        subject_info: dict from discover_subjects() (overrides subject_name)
        config: ElectrodeConfig to use. If None, auto-detects from .vhdr header.

    Returns:
        dict with all loaded data and metadata, including 'config' key
    """
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

    if config is None and subject_info.get("vhdr"):
        config = detect_config_from_vhdr(subject_info["vhdr"])
    elif config is None:
        config = FELT_TCRE_CONFIG

    n_ch = config.n_channels

    raw = np.fromfile(subject_info["eeg"], dtype=np.int16)
    n_samples = len(raw) // n_ch
    eeg = raw[: n_samples * n_ch].reshape(n_samples, n_ch).T.astype(np.float64)
    eeg *= RESOLUTION

    # Apply per-channel amplitude scaling (e.g. /187 for Gel tEEG)
    for ch_idx, divisor in config.amplitude_scaling.items():
        if ch_idx < n_ch:
            eeg[ch_idx] /= divisor

    t = np.arange(n_samples) / FS

    stim_samples, events_oc = [], []
    if subject_info["vmrk"]:
        stim_samples, events_oc = parse_vmrk(subject_info["vmrk"])

    stim_blocks = []
    if stim_samples:
        stim_times = np.array(stim_samples) / FS
        gaps = np.where(np.diff(stim_times) > 5)[0]
        block_starts = [0] + (gaps + 1).tolist()
        block_ends = gaps.tolist() + [len(stim_times) - 1]
        for bs_idx, be_idx in zip(block_starts, block_ends):
            stim_blocks.append(
                (stim_times[bs_idx] - 0.5, stim_times[be_idx] + 0.5)
            )

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

    avg_data = None
    avg_t = None
    avg_n_segments = None
    if subject_info.get("avg") and os.path.exists(subject_info["avg"]):
        avg_raw = np.fromfile(subject_info["avg"], dtype=np.float32)
        avg_pts = 500
        if subject_info.get("avg_vhdr") and os.path.exists(subject_info["avg_vhdr"]):
            with open(subject_info["avg_vhdr"], "r", errors="ignore") as f:
                for line in f:
                    if "SegmentDataPoints" in line:
                        avg_pts = int(line.split("=")[1].strip())
                    if "AveragedSegments" in line:
                        avg_n_segments = int(line.split("=")[1].strip())
        if len(avg_raw) >= avg_pts * n_ch:
            avg_data = avg_raw[: avg_pts * n_ch].reshape(avg_pts, n_ch).T
            avg_t = np.arange(avg_pts) / FS * 1000 - 100

    return {
        "name": subject_info["name"],
        "basename": subject_info["basename"],
        "info": subject_info,
        "config": config,
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


def analyze_subject(subject, pipeline_variant="B"):
    """
    Run full analysis on a loaded subject.

    Reads electrode config from subject['config'] (defaults to FELT_TCRE_CONFIG).

    Args:
        subject: dict from load_subject().
        pipeline_variant: "A", "B" (default, primary), or "C". See
            apply_pipeline_variant() for the definition of each variant.
            Variant B is the locked primary pipeline used in the main text.

    Returns dict with per-channel metrics:
        - alpha_snr: alpha SNR in dB (alpha vs neighboring bands)
        - alpha_open: mean alpha power during eyes-open
        - alpha_closed: mean alpha power during eyes-closed
        - alpha_reactivity: ratio closed/open (>1 = Berger effect)
        - alpha_envelopes: smoothed alpha envelope per channel
        - disc_correlation: Pearson r of alpha envelope vs disc channel
        - vep_p2p: VEP peak-to-peak amplitude (if avg available)
        - pipeline_variant: which variant produced these metrics
    """
    cfg = subject.get("config", FELT_TCRE_CONFIG)
    n_ch = cfg.n_channels

    eeg = subject["eeg"]
    n_samples = subject["n_samples"]
    open_epochs = subject["open_epochs"]
    close_epochs = subject["close_epochs"]

    clip = compute_adc_clipping(eeg, threshold_uv=ADC_CLIP_UV, n_channels=n_ch)

    alpha_snr = []
    alpha_open = []
    alpha_closed = []
    alpha_envelopes = []

    for i in range(n_ch):
        cleaned = apply_pipeline_variant(eeg[i], variant=pipeline_variant)

        f, pxx = signal.welch(cleaned, fs=FS, nperseg=4096)
        ap = np.mean(pxx[(f >= 8) & (f <= 13)])
        nap = np.mean(pxx[((f >= 4) & (f < 8)) | ((f > 13) & (f <= 30))])
        alpha_snr.append(10 * np.log10(ap / (nap + 1e-10)))

        alpha_bp = bandpass_filter(cleaned, 8, 13)
        env = compute_envelope(alpha_bp)
        alpha_envelopes.append(env)

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

    alpha_reactivity = np.array(alpha_closed) / (np.array(alpha_open) + 1e-10)

    # Correlation with disc channel (last idx_disc entry, or skip if none)
    ds = 10
    disc_corr = []
    if cfg.idx_disc:
        ref_idx = cfg.idx_disc[0]
        ref_env = alpha_envelopes[ref_idx][::ds]
        for i in range(n_ch):
            env_ds = alpha_envelopes[i][::ds]
            r = np.corrcoef(ref_env, env_ds)[0, 1]
            disc_corr.append(r)
    else:
        disc_corr = [np.nan] * n_ch

    vep_p2p = []
    if subject["avg_data"] is not None:
        for i in range(n_ch):
            vep_p2p.append(
                subject["avg_data"][i].max() - subject["avg_data"][i].min()
            )
    else:
        vep_p2p = [np.nan] * n_ch

    ssim_results = compute_spectrogram_ssim(subject)

    return {
        "config": cfg,
        "pipeline_variant": (pipeline_variant or "B").upper(),
        "adc_clip_threshold_uv": clip["threshold_uv"],
        "adc_clip_count": clip["clip_count"],
        "adc_clip_fraction": clip["clip_fraction"],
        "adc_clip_max_run_samples": clip["clip_max_run_samples"],
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

    cfg = subject.get("config", FELT_TCRE_CONFIG)
    n_ch = cfg.n_channels
    name = subject["name"]
    colors = [ch_color(i, cfg) for i in range(n_ch)]
    short_labels = cfg.short_labels

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Summary Dashboard — {name} ({cfg.name})",
                 fontsize=14, fontweight="bold")

    axes[0].bar(range(n_ch), results["alpha_snr"], color=colors,
                edgecolor="k", lw=0.5)
    axes[0].set_xticks(range(n_ch))
    axes[0].set_xticklabels(short_labels, fontsize=7)
    axes[0].set_ylabel("dB")
    axes[0].set_title("Alpha SNR")
    axes[0].axhline(0, color="gray", lw=0.5)

    axes[1].bar(range(n_ch), results["alpha_reactivity"], color=colors,
                edgecolor="k", lw=0.5)
    axes[1].set_xticks(range(n_ch))
    axes[1].set_xticklabels(short_labels, fontsize=7)
    axes[1].set_ylabel("Closed / Open")
    axes[1].set_title("Alpha Reactivity")
    axes[1].axhline(1, color="gray", lw=0.5, ls="--")

    disc_label = f"Corr. with Disc (Ch{cfg.idx_disc[0]+1})" if cfg.idx_disc else "Disc Corr."
    axes[2].bar(range(n_ch), results["disc_correlation"], color=colors,
                edgecolor="k", lw=0.5)
    axes[2].set_xticks(range(n_ch))
    axes[2].set_xticklabels(short_labels, fontsize=7)
    axes[2].set_ylabel("Pearson r")
    axes[2].set_title(disc_label)

    axes[3].bar(range(n_ch), results["vep_p2p"], color=colors,
                edgecolor="k", lw=0.5)
    axes[3].set_xticks(range(n_ch))
    axes[3].set_xticklabels(short_labels, fontsize=7)
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

    Reads electrode config from subject['config']. Works with any layout
    (Felt TCRE 11-ch, Gel TCRE 7-ch, etc.).
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cfg = subject.get("config", FELT_TCRE_CONFIG)
    n_ch = cfg.n_channels
    ch_labels = cfg.ch_labels

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
    fig, axes = plt.subplots(n_ch, 1, figsize=(18, max(12, n_ch * 2)), sharex=True)
    if n_ch == 1:
        axes = [axes]
    fig.suptitle(f"Raw EEG (µV) — {name} ({cfg.name})",
                 fontsize=14, fontweight="bold", y=1.0)
    ds = 10
    for i in range(n_ch):
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

    # ─── 2. PSD — Alpha focus (1–30 Hz), after 60 Hz notch ───
    nrows_psd = (n_ch + 1) // 2
    fig, axes = plt.subplots(nrows_psd, 2, figsize=(16, nrows_psd * 3.3))
    fig.suptitle(f"PSD After 60 Hz Notch — {name} ({cfg.name})",
                 fontsize=14, fontweight="bold", y=1.0)

    sm = None
    alpha_results_sp = None
    try:
        from specparam import SpectralModel  # type: ignore
        sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                           aperiodic_mode="fixed")
        alpha_results_sp = []
    except Exception:
        pass

    for i in range(n_ch):
        ax = axes.flatten()[i]
        cleaned = notch_filter(eeg[i])
        f, pxx = signal.welch(cleaned, fs=FS, nperseg=2048)
        mask = (f >= 1) & (f <= 30)
        ax.semilogy(f[mask], pxx[mask], lw=1.5, color="#2c3e50")
        amask = (f >= 8) & (f <= 13) & mask
        ax.fill_between(f[amask], pxx[amask], alpha=0.4, color="#e67e22", label="Alpha")

        if sm is not None and alpha_results_sp is not None:
            sm.fit(f, pxx, [1, 30])
            peaks = sm.get_params("peak")
            aperiodic = sm.get_params("aperiodic")
            r_squared = float(np.squeeze(sm.get_metrics("gof", "rsquared")))
            peaks = np.asarray(peaks)
            if peaks.size > 0 and peaks.ndim == 1:
                peaks = peaks.reshape(1, -1)
            reliable = True
            try:
                if float(aperiodic[1]) < 0.3:
                    reliable = False
            except Exception:
                reliable = False
            alpha_peaks = []
            if peaks.size > 0:
                alpha_mask_p = (peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)
                alpha_peaks = peaks[alpha_mask_p]
            if len(alpha_peaks) > 0:
                best = alpha_peaks[np.argmax(alpha_peaks[:, 1])]
                alpha_results_sp.append({
                    "channel": ch_labels[i], "alpha_freq": float(best[0]),
                    "alpha_amp": float(best[1]), "offset": float(aperiodic[0]),
                    "exponent": float(aperiodic[1]), "r_squared": r_squared,
                    "reliable": reliable,
                })
                ax.axvline(float(best[0]), color="red", ls="--", alpha=0.5,
                           label=f"α={float(best[0]):.1f} Hz")
            else:
                alpha_results_sp.append({
                    "channel": ch_labels[i], "alpha_freq": None, "alpha_amp": 0.0,
                    "offset": float(aperiodic[0]) if np.size(aperiodic) > 0 else np.nan,
                    "exponent": float(aperiodic[1]) if np.size(aperiodic) > 1 else np.nan,
                    "r_squared": r_squared, "reliable": reliable,
                })

        ax.set_title(ch_labels[i], fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_ch, len(axes.flatten())):
        axes.flatten()[j].set_visible(False)
    plt.tight_layout()
    _save_or_show(fig, f"{name}_psd_alpha_1_30hz.png")

    if alpha_results_sp is not None:
        print(f"\n{'Channel':<35} {'Alpha':<30} {'Aperiodic':<30} {'R²':<8} {'Quality'}")
        print("=" * 110)
        for r in alpha_results_sp:
            if r["alpha_freq"] is not None:
                alpha_str = f"α = {r['alpha_freq']:.2f} Hz, amp = {r['alpha_amp']:.2f}"
            else:
                alpha_str = "No alpha"
            aperiodic_str = f"offset={r['offset']:.2f}, slope={r['exponent']:.2f}"
            quality = "OK" if r["reliable"] else "poor aperiodic fit"
            print(f"{r['channel']:<35} {alpha_str:<30} {aperiodic_str:<30} {r['r_squared']:.2f}    {quality}")

    # ─── 3. Spectrograms — per channel ───
    for i in range(n_ch):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7),
                                        gridspec_kw={"height_ratios": [1, 3]})
        fig.suptitle(f"Spectrogram — {name} — {ch_labels[i]}", fontsize=12, fontweight="bold")
        cleaned = notch_filter(eeg[i])
        bp = bandpass_filter(cleaned, 1, 45)
        ax1.plot(t[::ds], bp[::ds], lw=0.3, color="#34495e")
        ax1.set_ylabel("µV"); ax1.set_xlim(0, t[-1])
        add_event_markers(ax1, events_oc, stim_blocks, close_epochs,
                          text_labels=True)
        nperseg = 2048
        f_s, t_s, Sxx = signal.spectrogram(cleaned, fs=FS, nperseg=nperseg,
                                            noverlap=nperseg // 2, nfft=2048)
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
        _save_or_show(fig, f"{name}_spectrogram_ch{i+1:02d}.png")

    # ─── 4. Band decomposition — per channel ───
    for i in range(n_ch):
        fig, axs = plt.subplots(len(BANDS) + 1, 1, figsize=(16, 10), sharex=True,
                                gridspec_kw={"height_ratios": [2] + [1] * len(BANDS)})
        fig.suptitle(f"Band Decomposition — {name} — {ch_labels[i]}",
                     fontsize=12, fontweight="bold")
        cleaned = notch_filter(eeg[i])
        ds_b = 20
        bp_full = bandpass_filter(cleaned, 1, 45)
        axs[0].plot(t[::ds_b], bp_full[::ds_b], lw=0.3, color="#2c3e50")
        axs[0].set_title("Broadband (1–45 Hz)", fontsize=10)
        axs[0].set_ylabel("µV", fontsize=8)
        for j, (bname, (lo, hi)) in enumerate(BANDS.items()):
            ax = axs[j + 1]
            bpd = bandpass_filter(cleaned, lo, hi)
            ax.plot(t[::ds_b], bpd[::ds_b], lw=0.4, color=BAND_COLORS[bname])
            ax.set_title(f"{bname} ({lo}–{hi} Hz)", fontsize=10, color=BAND_COLORS[bname])
            ax.set_ylabel("µV", fontsize=8)
        for ax in axs:
            ax.set_xlim(0, t[-1])
            ax.tick_params(labelsize=7)
            add_event_markers(ax, events_oc, stim_blocks, close_epochs)
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        _save_or_show(fig, f"{name}_band_decomp_ch{i+1:02d}.png")

    # ─── 5. Alpha envelope with events ───
    fig, axes = plt.subplots(n_ch, 1, figsize=(17, max(12, n_ch * 2.2)), sharex=True)
    if n_ch == 1:
        axes = [axes]
    fig.suptitle(f"Alpha Envelope (8–13 Hz) — {name} ({cfg.name})",
                 fontsize=14, fontweight="bold", y=1.0)
    ds_e = 50
    for i in range(n_ch):
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

    # ─── 6. Eyes open vs closed (absolute + reactivity) ───
    if open_epochs and close_epochs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Alpha Reactivity — {name} ({cfg.name})",
                     fontsize=13, fontweight="bold")
        x = np.arange(n_ch)
        w = 0.35
        ax1.bar(x - w / 2, results["alpha_open"], w, color="#27ae60",
                label="Open", edgecolor="k", lw=0.5)
        ax1.bar(x + w / 2, results["alpha_closed"], w, color="#3498db",
                label="Closed", edgecolor="k", lw=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cfg.short_labels, fontsize=9)
        ax1.set_ylabel("Alpha Power (µV²/Hz)")
        ax1.set_title("Absolute")
        ax1.legend()
        ax1.set_yscale("log")

        ax2.bar(x, results["alpha_reactivity"],
                color=[ch_color(i, cfg) for i in range(n_ch)],
                edgecolor="k", lw=0.5)
        ax2.axhline(1, color="gray", lw=1, ls="--")
        ax2.set_xticks(x)
        ax2.set_xticklabels(cfg.short_labels, fontsize=9)
        ax2.set_ylabel("Closed / Open")
        ax2.set_title("Reactivity (>1 = Berger effect)")
        ax2.legend(handles=ELEC_LEGEND, fontsize=7, loc="upper right")

        plt.tight_layout()
        _save_or_show(fig, f"{name}_alpha_reactivity.png")

    # ─── 7. Eyes open vs closed PSD (shared scale) ───
    if open_epochs and close_epochs:
        nrows_oc = (n_ch + 1) // 2
        fig, axes = plt.subplots(nrows_oc, 2, figsize=(16, nrows_oc * 3.3))
        fig.suptitle(f"Eyes Open vs Closed — {name} ({cfg.name})",
                     fontsize=14, fontweight="bold", y=1.0)
        psd_data = []
        global_max = 0
        global_min = np.inf
        for i in range(n_ch):
            cleaned = notch_filter(eeg[i])
            od = np.concatenate([cleaned[ep["start"]:ep["end"]] for ep in open_epochs])
            cd = np.concatenate([cleaned[ep["start"]:ep["end"]] for ep in close_epochs])
            f_o, pxx_o = signal.welch(od, fs=FS, nperseg=4096)
            f_c, pxx_c = signal.welch(cd, fs=FS, nperseg=4096)
            fmask = (f_o >= 1) & (f_o <= 30)
            psd_data.append((f_o, pxx_o, f_c, pxx_c, fmask))
            vals = np.concatenate([pxx_o[fmask], pxx_c[fmask]])
            global_max = max(global_max, vals.max())
            global_min = min(global_min, vals[vals > 0].min())
        for i in range(n_ch):
            ax = axes.flatten()[i]
            f_o, pxx_o, f_c, pxx_c, fmask = psd_data[i]
            ax.semilogy(f_o[fmask], pxx_o[fmask], lw=1.5, color="#27ae60", label="Open")
            ax.semilogy(f_c[fmask], pxx_c[fmask], lw=1.5, color="#3498db", label="Closed")
            ax.set_ylim(global_min * 0.5, global_max * 2)
            ax.set_title(ch_labels[i], fontsize=9, fontweight="bold")
            ax.set_ylabel("µV²/Hz", fontsize=8)
            ax.legend(fontsize=7); ax.tick_params(labelsize=7)
        for j in range(n_ch, len(axes.flatten())):
            axes.flatten()[j].set_visible(False)
        plt.tight_layout()
        _save_or_show(fig, f"{name}_open_vs_closed.png")

    # ─── 8. Visual evoked potential (comparison panels) ───
    if subject["avg_data"] is not None:
        avg_t = subject["avg_t"]
        avg_data = subject["avg_data"]

        if cfg.name == "Felt TCRE":
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(
                f"VEP Comparison — {name} (n={subject['avg_n_segments'] or '?'})",
                fontsize=13, fontweight="bold",
            )
            ax = axes[0]
            for idx in cfg.idx_teeg:
                ax.plot(avg_t, avg_data[idx], lw=1.2, alpha=0.7, label=f"Ch{idx+1}")
            ax.axvline(0, color="red", lw=1, ls="--", alpha=0.5)
            ax.axhline(0, color="gray", lw=0.3)
            ax.set_title("Felt TCRE tEEG (Laplacian)")
            ax.set_xlabel("ms"); ax.set_ylabel("µV")
            ax.legend(fontsize=8); ax.set_xlim(-100, 400)

            ax = axes[1]
            ax.plot(avg_t, avg_data[9], lw=2, color="#2ecc71", label="Ch10 Paste tEEG")
            ax.plot(avg_t, avg_data[10], lw=2, color="#e74c3c", label="Ch11 Disc")
            ax.plot(avg_t, avg_data[8], lw=1.5, color="#9b59b6", ls="--", label="Ch9 Paste eEEG")
            ax.axvline(0, color="red", lw=1, ls="--", alpha=0.5)
            ax.axhline(0, color="gray", lw=0.3)
            ax.set_title("Paste tEEG vs Disc"); ax.set_xlabel("ms")
            ax.legend(fontsize=8); ax.set_xlim(-100, 400)

            ax = axes[2]
            ax.plot(avg_t, np.mean(avg_data[cfg.idx_teeg], axis=0), lw=2, color="#3498db",
                    label="Avg Felt tEEG")
            ax.plot(avg_t, avg_data[9], lw=2, color="#2ecc71", label="Paste tEEG")
            ax.plot(avg_t, avg_data[10], lw=2, color="#e74c3c", label="Disc")
            ax.axvline(0, color="red", lw=1, ls="--", alpha=0.5)
            ax.axhline(0, color="gray", lw=0.3)
            ax.set_title("Grand Average by Type"); ax.set_xlabel("ms")
            ax.legend(fontsize=8); ax.set_xlim(-100, 400)
            plt.tight_layout()
            _save_or_show(fig, f"{name}_vep_comparison.png")
        else:
            fig, axes_vep = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"VEP Comparison — {name} ({cfg.name}, n={subject['avg_n_segments'] or '?'})",
                fontsize=13, fontweight="bold",
            )
            ax = axes_vep[0]
            for idx in cfg.idx_teeg:
                ax.plot(avg_t, avg_data[idx], lw=1.2, alpha=0.7,
                        label=ch_labels[idx])
            ax.axvline(0, color="red", lw=1, ls="--", alpha=0.5)
            ax.axhline(0, color="gray", lw=0.3)
            ax.set_title(f"{cfg.name} tEEG"); ax.set_xlabel("ms"); ax.set_ylabel("µV")
            ax.legend(fontsize=7); ax.set_xlim(-100, 400)

            ax = axes_vep[1]
            if cfg.idx_paste_teeg:
                ax.plot(avg_t, avg_data[cfg.idx_paste_teeg[0]], lw=2,
                        color="#2ecc71", label="Paste tEEG")
            if cfg.idx_disc:
                ax.plot(avg_t, avg_data[cfg.idx_disc[0]], lw=2,
                        color="#e74c3c", label="Disc")
            ax.axvline(0, color="red", lw=1, ls="--", alpha=0.5)
            ax.axhline(0, color="gray", lw=0.3)
            ax.set_title("Paste tEEG vs Disc"); ax.set_xlabel("ms")
            ax.legend(fontsize=8); ax.set_xlim(-100, 400)
            plt.tight_layout()
            _save_or_show(fig, f"{name}_vep_comparison.png")

    # ─── 9. Disc correlation ───
    disc_ref_label = f"Ch{cfg.idx_disc[0]+1} (Disc)" if cfg.idx_disc else "Disc"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(n_ch), results["disc_correlation"],
           color=[ch_color(i, cfg) for i in range(n_ch)],
           edgecolor="k", lw=0.5)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(cfg.short_labels)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Alpha Envelope Correlation with {disc_ref_label} — {name}",
                 fontsize=13, fontweight="bold")
    for i, r in enumerate(results["disc_correlation"]):
        if not np.isnan(r):
            ax.text(i, r + 0.02, f"{r:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.legend(handles=ELEC_LEGEND, fontsize=7, loc="upper left")
    plt.tight_layout()
    _save_or_show(fig, f"{name}_disc_correlation.png")

    # ─── 10. ADC clipping report ───
    clip_frac = results.get("adc_clip_fraction", None)
    clip_max_run = results.get("adc_clip_max_run_samples", None)
    if clip_frac is not None and clip_max_run is not None:
        clip_pct = 100.0 * np.asarray(clip_frac, dtype=float)
        clip_run_s = np.asarray(clip_max_run, dtype=float) / FS

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle(f"ADC Clipping Report — {name} ({cfg.name})",
                     fontsize=13, fontweight="bold")
        x = np.arange(n_ch)
        colors_clip = [ch_color(i, cfg) for i in range(n_ch)]

        ax1.bar(x, clip_pct, color=colors_clip, edgecolor="k", lw=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cfg.short_labels, fontsize=9)
        ax1.set_ylabel("% samples at/near ADC limit")
        ax1.set_title(f"Clipped samples (threshold ≈ {results.get('adc_clip_threshold_uv', ADC_CLIP_UV):.1f} µV)")
        ax1.set_ylim(0, max(0.5, float(np.nanmax(clip_pct)) * 1.2))

        ax2.bar(x, clip_run_s, color=colors_clip, edgecolor="k", lw=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(cfg.short_labels, fontsize=9)
        ax2.set_ylabel("Longest clipped run (s)")
        ax2.set_title("Worst contiguous saturation")
        ax2.set_ylim(0, max(0.05, float(np.nanmax(clip_run_s)) * 1.2))

        ax2.legend(handles=ELEC_LEGEND, fontsize=7, loc="upper right")
        plt.tight_layout()
        _save_or_show(fig, f"{name}_adc_clipping.png")

        if np.any(clip_pct > 0):
            bad = np.where(clip_pct > 0)[0]
            worst = int(bad[np.argmax(clip_pct[bad])]) if bad.size else None
            if worst is not None:
                print(
                    f"WARNING: ADC clipping detected in {bad.size}/{n_ch} channels. "
                    f"Worst: {ch_labels[worst]} ({clip_pct[worst]:.3f}% samples clipped, "
                    f"max run {clip_run_s[worst]:.3f}s)."
                )

    # ─── 11. Spectrogram SSIM comparison (tEEG vs eEEG per TCRE) ───
    ssim_vals = results.get("ssim_values", [])
    ssim_pairs_data = results.get("ssim_spectrogram_pairs", [])
    ssim_labels = results.get("ssim_pair_labels", [])
    tcre_pairs = cfg.tcre_pairs

    if ssim_pairs_data:
        for pi, (Sxx_t_db, Sxx_e_db, f_crop, t_crop) in enumerate(ssim_pairs_data):
            teeg_idx, eeeg_idx, pair_label = tcre_pairs[pi]
            score = ssim_vals[pi]

            fig, axes_ssim = plt.subplots(1, 3, figsize=(20, 5))
            fig.suptitle(
                f"Spectrogram Comparison — {name} — {pair_label}  "
                f"(SSIM = {score:.3f})",
                fontsize=13, fontweight="bold",
            )

            vmin = min(Sxx_t_db.min(), Sxx_e_db.min())
            vmax = max(Sxx_t_db.max(), Sxx_e_db.max())

            im0 = axes_ssim[0].pcolormesh(t_crop, f_crop, Sxx_t_db,
                                           shading="gouraud", cmap="inferno",
                                           vmin=vmin, vmax=vmax)
            axes_ssim[0].set_title(f"Ch{teeg_idx+1} — tEEG (Laplacian)", fontsize=10)
            axes_ssim[0].set_ylabel("Hz"); axes_ssim[0].set_ylim(1, 45)
            axes_ssim[0].axhline(8, color="cyan", lw=0.6, ls="--", alpha=0.5)
            axes_ssim[0].axhline(13, color="cyan", lw=0.6, ls="--", alpha=0.5)

            im1 = axes_ssim[1].pcolormesh(t_crop, f_crop, Sxx_e_db,
                                           shading="gouraud", cmap="inferno",
                                           vmin=vmin, vmax=vmax)
            axes_ssim[1].set_title(f"Ch{eeeg_idx+1} — eEEG (Conv)", fontsize=10)
            axes_ssim[1].set_ylabel("Hz"); axes_ssim[1].set_ylim(1, 45)
            axes_ssim[1].axhline(8, color="cyan", lw=0.6, ls="--", alpha=0.5)
            axes_ssim[1].axhline(13, color="cyan", lw=0.6, ls="--", alpha=0.5)

            diff = Sxx_t_db - Sxx_e_db
            d_abs = max(abs(diff.min()), abs(diff.max()), 1)
            im2 = axes_ssim[2].pcolormesh(t_crop, f_crop, diff,
                                           shading="gouraud", cmap="RdBu_r",
                                           vmin=-d_abs, vmax=d_abs)
            axes_ssim[2].set_title("Difference (tEEG − eEEG)", fontsize=10)
            axes_ssim[2].set_ylabel("Hz"); axes_ssim[2].set_ylim(1, 45)
            axes_ssim[2].axhline(8, color="black", lw=0.6, ls="--", alpha=0.5)
            axes_ssim[2].axhline(13, color="black", lw=0.6, ls="--", alpha=0.5)

            for ax in axes_ssim:
                ax.set_xlabel("Time (s)")
            plt.colorbar(im0, ax=axes_ssim[0], label="dB", shrink=0.8)
            plt.colorbar(im1, ax=axes_ssim[1], label="dB", shrink=0.8)
            plt.colorbar(im2, ax=axes_ssim[2], label="ΔdB", shrink=0.8)
            plt.tight_layout()
            _save_or_show(fig, f"{name}_ssim_pair{pi+1}_{pair_label.replace(' ', '_')}.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        n_pairs = len(ssim_vals)
        colors_ssim = [("#3498db" if "Paste" not in lbl else "#2ecc71")
                       for lbl in ssim_labels]
        ax.bar(range(n_pairs), ssim_vals, color=colors_ssim,
               edgecolor="k", lw=0.5)
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels(ssim_labels, fontsize=9, rotation=15)
        ax.set_ylabel("SSIM")
        ax.set_title(f"Spectrogram SSIM (tEEG vs eEEG) — {name}",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.axhline(0.8, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.text(n_pairs - 0.5, 0.81, "high similarity", fontsize=7,
                color="gray", ha="right")
        for j, v in enumerate(ssim_vals):
            ax.text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        _save_or_show(fig, f"{name}_ssim_summary.png")

    # ─── 12. Summary dashboard ───
    plot_subject_summary(subject, results, save_dir=save_dir, show=show)
