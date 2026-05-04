"""
report_v2.py

Generate a per-participant QC report (`output/report_v2.txt`) summarizing:
- marker availability (open/close, stim blocks, VEP avg)
- ADC clipping (saturation-like samples and longest clipped run)
- flatline / near-flat segments
- channel similarity (are channels "the same"?)
- alpha metrics (SNR + reactivity), disc correlation

Usage:
  venv/bin/python -m src.report_v2
  venv/bin/python -m src.report_v2 --data-dir ./data --out ./output/report_v2.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Import analysis engine (repo-local)
from src.eeg_analysis import (
    FS,
    N_CHANNELS,
    CH_LABELS,
    SHORT_LABELS,
    ADC_CLIP_UV,
    discover_subjects,
    load_subject,
    analyze_subject,
    notch_filter,
    bandpass_filter,
)


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan%"
    return f"{x*100:.3f}%"


def _robust_std(x: np.ndarray) -> float:
    # Robust scale estimate (MAD → std)
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)


def _max_run_true(mask_1d: np.ndarray) -> int:
    m = np.asarray(mask_1d, dtype=bool)
    if m.size == 0 or not np.any(m):
        return 0
    x = np.concatenate(([False], m, [False])).astype(np.int8)
    d = np.diff(x)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return int(m.sum())
    return int(np.max(ends - starts))


@dataclass
class SimilarityStats:
    median_r: float
    max_r: float
    high_pairs: list[tuple[int, int, float]]
    near_duplicate_pairs: list[tuple[int, int, float]]


def compute_channel_similarity(eeg_uv: np.ndarray, ds: int = 20) -> SimilarityStats:
    """
    Estimate whether channels are unusually similar.

    - Uses notch + 1–45 Hz bandpass.
    - Downsamples to speed up correlation.
    - Computes correlation matrix across channels.
    - Flags:
      - high_pairs: r >= 0.98 (very similar)
      - near_duplicate_pairs: median absolute difference <= 0.5 µV after filtering
        (often indicates duplicated/bridged signals).
    """
    x = np.asarray(eeg_uv, dtype=float)
    if x.shape[0] != N_CHANNELS:
        raise ValueError(f"Expected eeg_uv shape ({N_CHANNELS}, n_samples), got {x.shape}")

    # Filter each channel
    xf = []
    for ch in range(N_CHANNELS):
        y = notch_filter(x[ch])
        y = bandpass_filter(y, 1, 45)
        xf.append(y[::ds])
    xf = np.vstack(xf)

    # Correlation matrix
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(xf)
    # off-diagonal stats
    iu = np.triu_indices(N_CHANNELS, k=1)
    off = C[iu]
    off = off[np.isfinite(off)]
    median_r = float(np.nanmedian(off)) if off.size else float("nan")
    max_r = float(np.nanmax(off)) if off.size else float("nan")

    high_pairs = []
    for i, j in zip(iu[0], iu[1]):
        r = C[i, j]
        if np.isfinite(r) and r >= 0.98:
            high_pairs.append((int(i), int(j), float(r)))
    high_pairs.sort(key=lambda t: t[2], reverse=True)

    near_dup = []
    # near-duplicates based on absolute diff
    for i, j in zip(iu[0], iu[1]):
        d = np.nanmedian(np.abs(xf[i] - xf[j]))
        if np.isfinite(d) and d <= 0.5:
            near_dup.append((int(i), int(j), float(d)))
    near_dup.sort(key=lambda t: t[2])

    return SimilarityStats(
        median_r=median_r,
        max_r=max_r,
        high_pairs=high_pairs[:10],
        near_duplicate_pairs=near_dup[:10],
    )


def compute_flatline_fraction(eeg_uv: np.ndarray, window_s: float = 1.0, eps_uv: float = 1.0):
    """
    Fraction of time where a channel is "near-flat" within sliding windows.
    We mark a window flat if robust std < eps_uv.
    """
    x = np.asarray(eeg_uv, dtype=float)
    win = int(max(1, round(window_s * FS)))
    n = x.shape[1]
    if n < win:
        # trivial case
        rs = np.array([_robust_std(x[i]) for i in range(N_CHANNELS)])
        return (rs < eps_uv).astype(float)

    step = win  # non-overlapping windows (simple + fast)
    flat_fracs = []
    for i in range(N_CHANNELS):
        flat = 0
        total = 0
        for start in range(0, n - win + 1, step):
            seg = x[i, start : start + win]
            total += 1
            if _robust_std(seg) < eps_uv:
                flat += 1
        flat_fracs.append(flat / max(total, 1))
    return np.array(flat_fracs, dtype=float)


def grade_subject(subject: dict, results: dict, sim: SimilarityStats, flat_frac: np.ndarray) -> tuple[str, list[str]]:
    """
    Heuristic grade: Good / Mixed / Poor with reasons.
    """
    reasons: list[str] = []

    clip_frac = np.asarray(results.get("adc_clip_fraction", np.zeros(N_CHANNELS)), dtype=float)
    clip_max_run = np.asarray(results.get("adc_clip_max_run_samples", np.zeros(N_CHANNELS)), dtype=float)
    clip_any = np.nanmax(clip_frac) if clip_frac.size else 0.0
    clip_run_s = np.nanmax(clip_max_run) / FS if clip_max_run.size else 0.0

    if np.isfinite(clip_any) and clip_any > 0.001:  # >0.1% samples clipped
        reasons.append(f"ADC clipping present (worst channel {_fmt_pct(float(clip_any))}, max run {clip_run_s:.2f}s)")

    if np.nanmax(flat_frac) > 0.10:
        reasons.append(f"Flat/near-flat segments detected (worst channel {_fmt_pct(float(np.nanmax(flat_frac)))})")

    if np.isfinite(sim.median_r) and sim.median_r > 0.90:
        reasons.append(f"Channels unusually similar (median r={sim.median_r:.3f}, max r={sim.max_r:.3f})")

    # Marker availability
    if not subject.get("events_oc"):
        reasons.append("No eyes open/close markers")
    if not subject.get("stim_blocks"):
        reasons.append("No stim blocks detected")
    if subject.get("avg_data") is None:
        reasons.append("No averaged VEP file")

    # Alpha metrics sanity
    alpha_snr = np.asarray(results.get("alpha_snr", np.full(N_CHANNELS, np.nan)), dtype=float)
    alpha_react = np.asarray(results.get("alpha_reactivity", np.full(N_CHANNELS, np.nan)), dtype=float)
    if np.nanmedian(alpha_snr) < 0.5:
        reasons.append(f"Weak alpha SNR overall (median {np.nanmedian(alpha_snr):.2f} dB)")
    if subject.get("open_epochs") and subject.get("close_epochs"):
        if np.nanmedian(alpha_react) < 1.05:
            reasons.append(f"Low alpha reactivity overall (median {np.nanmedian(alpha_react):.2f}x)")

    # Grade
    # If any big red flags → Poor; if some minor flags → Mixed; else Good
    severe = False
    if (np.isfinite(clip_any) and clip_any > 0.01) or clip_run_s > 5.0:  # 1% clipped or long saturation
        severe = True
    if np.nanmax(flat_frac) > 0.30:
        severe = True
    if np.isfinite(sim.median_r) and sim.median_r > 0.95:
        severe = True

    if severe:
        return "POOR", reasons
    if reasons:
        return "MIXED", reasons
    return "GOOD", ["No major QC flags detected"]


def write_report_v2(data_dir: Path, out_path: Path) -> Path:
    subjects = discover_subjects(str(data_dir))

    lines: list[str] = []
    lines.append("Tripolar EEG — report_v2")
    lines.append(f"DATA_DIR: {data_dir}")
    lines.append(f"Subjects found: {len(subjects)}")
    lines.append("")

    for s in subjects:
        subj = load_subject(str(data_dir), subject_info=s)
        res = analyze_subject(subj)

        # Similarity + flatline QC
        sim = compute_channel_similarity(subj["eeg"], ds=20)
        flat = compute_flatline_fraction(subj["eeg"], window_s=1.0, eps_uv=1.0)

        grade, reasons = grade_subject(subj, res, sim, flat)

        lines.append("=" * 90)
        lines.append(f"{subj['name']}  |  grade: {grade}")
        lines.append(f"basename: {subj['basename']}")
        lines.append(f"duration: {subj['n_samples']/FS/60:.2f} min")
        lines.append(f"markers: open/close={len(subj.get('events_oc', []))}  stim_blocks={len(subj.get('stim_blocks', []))}")
        lines.append(f"avg VEP: {'yes' if subj.get('avg_data') is not None else 'no'} (n={subj.get('avg_n_segments')})")
        lines.append("")

        # Summaries
        clip_frac = np.asarray(res.get("adc_clip_fraction", np.zeros(N_CHANNELS)), dtype=float)
        clip_run = np.asarray(res.get("adc_clip_max_run_samples", np.zeros(N_CHANNELS)), dtype=float) / FS
        lines.append(f"ADC clip threshold: ~{res.get('adc_clip_threshold_uv', ADC_CLIP_UV):.1f} µV")
        lines.append(f"ADC clipping: max={_fmt_pct(float(np.nanmax(clip_frac)))}  median={_fmt_pct(float(np.nanmedian(clip_frac)))}")
        lines.append(f"Flatline-ish (1s windows, <1µV robust-std): max={_fmt_pct(float(np.nanmax(flat)))}  median={_fmt_pct(float(np.nanmedian(flat)))}")
        lines.append(f"Channel similarity: median r={sim.median_r:.3f}  max r={sim.max_r:.3f}")
        lines.append("")

        # Worst channels
        worst_clip = int(np.nanargmax(clip_frac)) if np.any(np.isfinite(clip_frac)) else 0
        worst_flat = int(np.nanargmax(flat)) if np.any(np.isfinite(flat)) else 0
        lines.append(f"Worst clipping: {CH_LABELS[worst_clip]}  ({_fmt_pct(float(clip_frac[worst_clip]))}, max run {clip_run[worst_clip]:.2f}s)")
        lines.append(f"Worst flatline: {CH_LABELS[worst_flat]}  ({_fmt_pct(float(flat[worst_flat]))})")
        lines.append("")

        # Alpha metrics quick view
        alpha_snr = np.asarray(res.get("alpha_snr", np.full(N_CHANNELS, np.nan)), dtype=float)
        alpha_react = np.asarray(res.get("alpha_reactivity", np.full(N_CHANNELS, np.nan)), dtype=float)
        disc_corr = np.asarray(res.get("disc_correlation", np.full(N_CHANNELS, np.nan)), dtype=float)
        lines.append(f"Alpha SNR (dB): median={np.nanmedian(alpha_snr):.2f}  best={np.nanmax(alpha_snr):.2f}")
        lines.append(f"Alpha reactivity (x): median={np.nanmedian(alpha_react):.2f}  best={np.nanmax(alpha_react):.2f}")
        lines.append(f"Disc correlation: median={np.nanmedian(disc_corr):.3f}  best={np.nanmax(disc_corr):.3f}")
        lines.append("")

        # Similarity details
        if sim.high_pairs:
            pairs = ", ".join([f"{SHORT_LABELS[i]}-{SHORT_LABELS[j]}(r={r:.3f})" for i, j, r in sim.high_pairs[:5]])
            lines.append(f"Very high-corr pairs (top): {pairs}")
        if sim.near_duplicate_pairs:
            pairs = ", ".join([f"{SHORT_LABELS[i]}-{SHORT_LABELS[j]}(Δmed={d:.2f}µV)" for i, j, d in sim.near_duplicate_pairs[:5]])
            lines.append(f"Near-duplicate pairs (top): {pairs}")
        lines.append("")

        lines.append("Reasons / notes:")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=(repo_root / "data"))
    p.add_argument("--out", type=Path, default=(repo_root / "output" / "report_v2.txt"))
    args = p.parse_args()

    out = write_report_v2(args.data_dir, args.out)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

