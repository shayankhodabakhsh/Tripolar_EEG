"""
comparison_analysis.py — 3-way comparison framework for Felt TCRE vs Gel TCRE vs Paste TCRE.

Provides functions to:
  - Discover Gel TCRE subjects from date-based subfolders
  - Extract comparable metrics aggregated by abstract electrode type
  - Run cross-electrode-type statistical tests
  - Generate side-by-side comparison visualizations

Usage:
    from comparison_analysis import (
        load_all_subjects, extract_type_metrics,
        compare_electrode_types, plot_three_way_comparison
    )
"""

import os
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

from eeg_analysis import (
    FS, BANDS, BAND_COLORS,
    FELT_TCRE_CONFIG, GEL_TCRE_CONFIG,
    discover_subjects, load_subject, analyze_subject,
    notch_filter, cohens_d,
)

# Canonical colors for the 3-way comparison
TYPE_COLORS = {
    "FELT_TEEG": "#3498db",
    "FELT_EEEG": "#85c1e9",
    "GEL_TEEG": "#e67e22",
    "GEL_EEEG": "#f5cba7",
    "PASTE_TEEG": "#2ecc71",
    "PASTE_EEEG": "#abebc6",
    "DISC": "#e74c3c",
}

TYPE_DISPLAY = {
    "FELT_TEEG": "Felt tEEG",
    "FELT_EEEG": "Felt eEEG",
    "GEL_TEEG": "Gel tEEG",
    "GEL_EEEG": "Gel eEEG",
    "PASTE_TEEG": "Paste tEEG",
    "PASTE_EEEG": "Paste eEEG",
    "DISC": "Disc EEG",
}


def print_qc_report(subjects_results, label=""):
    """
    Print a per-subject quality-control table.

    Columns:
        Subject        name
        Ch             number of channels
        Duration(s)    recording length in seconds
        OpenEp         number of eyes-open epochs
        CloseEp        number of eyes-closed epochs
        AlphaSNR(dB)   mean alpha SNR across tEEG channels
        Clipping%      fraction of samples at ADC rail
        Status         PASS / WARN / FAIL and reason
    """
    from eeg_analysis import compute_adc_clipping, FS as _FS

    SEP = "-" * 100
    HDR = (f"{'Subject':<35} {'Ch':>3}  {'Dur(s)':>7}  "
           f"{'OpenEp':>6}  {'CloseEp':>7}  {'SnrMean':>8}  {'Clip%':>6}  Status")
    print(f"\n{'QC REPORT' + (' — ' + label if label else '')}")
    print(SEP)
    print(HDR)
    print(SEP)

    for subj, res in subjects_results:
        cfg  = subj.get("config", FELT_TCRE_CONFIG)
        name = subj["name"][:34]
        n_ch = cfg.n_channels
        dur  = subj["n_samples"] / _FS
        n_open  = len(subj["open_epochs"])
        n_close = len(subj["close_epochs"])

        # Mean alpha SNR across tEEG channels
        teeg_idxs = cfg.idx_teeg
        snr_vals  = [res["alpha_snr"][i] for i in teeg_idxs
                     if not np.isnan(res["alpha_snr"][i])]
        snr_mean  = np.mean(snr_vals) if snr_vals else float("nan")

        # Clipping: mean fraction of samples at ADC rail across all channels
        clip_info = compute_adc_clipping(subj["eeg"])
        clip_pct  = np.mean(clip_info["clip_fraction"]) * 100

        # --- Status logic ---
        fails = []
        warns = []
        if n_open < 2:
            fails.append(f"only {n_open} open epoch(s)")
        if n_close < 2:
            fails.append(f"only {n_close} close epoch(s)")
        if dur < 60:
            fails.append(f"dur={dur:.0f}s < 60s")
        if not np.isnan(snr_mean) and snr_mean < 0:
            warns.append(f"SNR={snr_mean:.1f}dB<0")
        if clip_pct > 1.0:
            warns.append(f"clip={clip_pct:.1f}%")

        if fails:
            status = "FAIL  " + "; ".join(fails)
        elif warns:
            status = "WARN  " + "; ".join(warns)
        else:
            status = "PASS"

        snr_str  = f"{snr_mean:+.1f}" if not np.isnan(snr_mean) else "  n/a"
        print(f"{name:<35} {n_ch:>3}  {dur:>7.0f}  "
              f"{n_open:>6}  {n_close:>7}  {snr_str:>8}  {clip_pct:>5.1f}%  {status}")

    print(SEP + "\n")


def _passes_signal_qc(subj, res, min_alpha_snr_db, min_alpha_reactivity,
                      min_vep_p2p_uv, label=""):
    """
    Apply signal-quality gates to a loaded subject.

    Gates are evaluated on the tEEG channels only (averaged across channels).
    Returns (passed: bool, reason: str).

    Thresholds:
        min_alpha_snr_db      - mean tEEG alpha SNR must be ≥ this (dB)
        min_alpha_reactivity  - mean tEEG alpha reactivity must be ≥ this
        min_vep_p2p_uv        - max tEEG VEP peak-to-peak must be ≥ this (µV);
                                set to 0 to skip this gate.
    """
    cfg = subj.get("config", FELT_TCRE_CONFIG)
    teeg_idxs = cfg.idx_teeg

    reasons = []

    # --- Alpha SNR gate ---
    if min_alpha_snr_db is not None:
        snr_vals = [res["alpha_snr"][i] for i in teeg_idxs
                    if not np.isnan(res["alpha_snr"][i])]
        if snr_vals:
            mean_snr = np.mean(snr_vals)
            if mean_snr < min_alpha_snr_db:
                reasons.append(
                    f"tEEG Alpha SNR {mean_snr:.1f} dB < {min_alpha_snr_db} dB")

    # --- Alpha reactivity gate ---
    if min_alpha_reactivity is not None:
        react_vals = [res["alpha_reactivity"][i] for i in teeg_idxs
                      if not np.isnan(res["alpha_reactivity"][i])]
        if react_vals:
            mean_react = np.mean(react_vals)
            if mean_react < min_alpha_reactivity:
                reasons.append(
                    f"tEEG Reactivity {mean_react:.2f} < {min_alpha_reactivity}")

    # --- VEP gate (optional) ---
    if min_vep_p2p_uv and min_vep_p2p_uv > 0:
        vep_vals = [res["vep_p2p"][i] for i in teeg_idxs
                    if not np.isnan(res["vep_p2p"][i])]
        if vep_vals:
            max_vep = np.max(vep_vals)
            if max_vep < min_vep_p2p_uv:
                reasons.append(
                    f"max tEEG VEP {max_vep:.1f} µV < {min_vep_p2p_uv} µV")

    if reasons:
        return False, "; ".join(reasons)
    return True, "OK"


def load_all_subjects(felt_dir, gel_dir, felt_names=None,
                      exclude_felt=None, exclude_gel=None,
                      min_open_epochs=2, min_close_epochs=2,
                      min_alpha_snr_db=0.0,
                      min_alpha_reactivity=1.2,
                      min_vep_p2p_uv=0.0,
                      verbose=True):
    """
    Load Felt and Gel subjects with automatic quality gates.

    Subjects are included only if they pass ALL of the following:
      1. Not in the manual exclusion list.
      2. Have ≥ min_open_epochs and ≥ min_close_epochs.
      3. Unknown channel layout raises ValueError → skipped automatically.
      4. Mean tEEG Alpha SNR  ≥ min_alpha_snr_db  (default 0 dB).
      5. Mean tEEG Reactivity ≥ min_alpha_reactivity (default 1.2).
      6. Max  tEEG VEP p2p    ≥ min_vep_p2p_uv     (default 0 = gate off).

    Gates 4–6 are objective, data-driven, and applied identically to both
    Felt and Gel datasets — they are reportable as pre-registration criteria.

    To disable a signal-quality gate, set its threshold to None.

    Args:
        felt_dir: path to data/ folder containing Felt TCRE recordings
        gel_dir: path to "Gel TCRE" folder with date-based subfolders
        felt_names: list of subject names (or substrings) to include from
                    felt_dir. None = include all discovered subjects.
        exclude_felt: list of subject name substrings to manually exclude.
        exclude_gel: list of Gel subject basename substrings to exclude.
        min_open_epochs: hard epoch-count floor for eyes-open.
        min_close_epochs: hard epoch-count floor for eyes-closed.
        min_alpha_snr_db: minimum mean tEEG alpha SNR in dB (None to disable).
        min_alpha_reactivity: minimum mean tEEG closed/open ratio (None to disable).
        min_vep_p2p_uv: minimum max tEEG VEP peak-to-peak in µV (0 or None to disable).
        verbose: print per-subject progress and exclusion reason.

    Returns:
        felt_subjects: list of (subject_dict, results_dict) tuples
        gel_subjects:  list of (subject_dict, results_dict) tuples
    """
    felt_subjects = []
    gel_subjects  = []
    exclude_felt  = exclude_felt or []
    exclude_gel   = exclude_gel  or []

    # Counters for the exclusion summary table
    _counts = {"felt": {"loaded": 0, "manual": 0, "epoch": 0,
                        "signal": 0, "error": 0, "config": 0},
               "gel":  {"loaded": 0, "manual": 0, "epoch": 0,
                        "signal": 0, "error": 0, "config": 0}}

    def _is_excluded(name, basename, exclusion_list):
        return any(ex.lower() in name.lower() or ex.lower() in basename.lower()
                   for ex in exclusion_list)

    def _try_load(si, data_dir, config, group_key, label):
        if _is_excluded(si["name"], si["basename"], exclude_felt if group_key == "felt" else exclude_gel):
            if verbose:
                print(f"  EXCLUDED (manual)          {si['name']}")
            _counts[group_key]["manual"] += 1
            return None, None

        if verbose:
            print(f"  Loading {label}: {si['name']} ...")
        try:
            subj = load_subject(data_dir, subject_info=si, config=config)
        except ValueError as e:
            print(f"  SKIP (config error)        {si['name']}: {e}")
            _counts[group_key]["config"] += 1
            return None, None
        except Exception as e:
            print(f"  SKIP (load error)          {si['name']}: {e}")
            _counts[group_key]["error"] += 1
            return None, None

        n_open  = len(subj["open_epochs"])
        n_close = len(subj["close_epochs"])
        if n_open < min_open_epochs or n_close < min_close_epochs:
            print(f"  SKIP (too few epochs)      {si['name']}: "
                  f"{n_open} open / {n_close} close "
                  f"(need ≥{min_open_epochs}/{min_close_epochs})")
            _counts[group_key]["epoch"] += 1
            return None, None

        res = analyze_subject(subj)

        passed, reason = _passes_signal_qc(
            subj, res,
            min_alpha_snr_db=min_alpha_snr_db,
            min_alpha_reactivity=min_alpha_reactivity,
            min_vep_p2p_uv=min_vep_p2p_uv,
        )
        if not passed:
            print(f"  SKIP (signal QC)           {si['name']}: {reason}")
            _counts[group_key]["signal"] += 1
            return None, None

        _counts[group_key]["loaded"] += 1
        return subj, res

    # --- Felt TCRE ---
    felt_discovered = discover_subjects(felt_dir)
    if felt_names:
        felt_discovered = [s for s in felt_discovered
                           if s["name"] in felt_names
                           or any(n in s["basename"] for n in felt_names)]

    for si in felt_discovered:
        subj, res = _try_load(si, felt_dir, config=None,
                              group_key="felt", label="Felt TCRE")
        if subj is not None:
            felt_subjects.append((subj, res))

    # --- Gel TCRE ---
    gel_discovered = discover_subjects(gel_dir, recursive=True,
                                       standard_protocol_only=True)
    for si in gel_discovered:
        subj, res = _try_load(si, gel_dir, config=GEL_TCRE_CONFIG,
                              group_key="gel", label="Gel TCRE ")
        if subj is not None:
            gel_subjects.append((subj, res))

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  INCLUSION SUMMARY")
        print(f"{'─'*60}")
        for grp in ("felt", "gel"):
            c = _counts[grp]
            total = sum(c.values())
            print(f"  {grp.upper()} TCRE:")
            print(f"    Included  : {c['loaded']}")
            print(f"    Excluded (manual)    : {c['manual']}")
            print(f"    Excluded (too few epochs): {c['epoch']}")
            print(f"    Excluded (signal QC) : {c['signal']}")
            print(f"    Skipped  (config/unknown layout): {c['config']}")
            print(f"    Skipped  (load error): {c['error']}")
        print(f"{'─'*60}")
        print(f"  Signal QC thresholds applied:")
        print(f"    Alpha SNR  ≥ {min_alpha_snr_db} dB"
              if min_alpha_snr_db is not None else "    Alpha SNR  : gate disabled")
        print(f"    Reactivity ≥ {min_alpha_reactivity}"
              if min_alpha_reactivity is not None else "    Reactivity : gate disabled")
        vep_str = (f"    VEP p2p    ≥ {min_vep_p2p_uv} µV"
                   if min_vep_p2p_uv else "    VEP p2p    : gate disabled")
        print(vep_str)
        print(f"{'─'*60}\n")

    return felt_subjects, gel_subjects


def extract_type_metrics(subjects_results, per_subject=True):
    """
    Aggregate per-channel metrics by abstract electrode type.

    Args:
        subjects_results: list of (subject_dict, results_dict) tuples
        per_subject: if True (default), average channels of the same type
            within each subject first, then collect one value per subject.
            This is the statistically correct approach: the subject is the
            independent unit, not the individual channel.
            If False, collect every channel as a separate observation
            (inflates n and gives unequal weight when setups have different
            numbers of channels per type).

    Returns:
        dict mapping type_name -> dict of metric arrays, one entry per subject:
            {
                "FELT_TEEG": {
                    "alpha_snr": array shape (n_subjects,),
                    "alpha_open": ...,
                    "alpha_closed": ...,
                    "alpha_reactivity": ...,
                    "disc_correlation": ...,
                    "vep_p2p": ...,
                    "subject_names": list of str,
                    "n_channels_per_subject": list of int,
                },
                ...
            }
    """
    METRIC_KEYS = ["alpha_snr", "alpha_open", "alpha_closed",
                   "alpha_reactivity", "disc_correlation", "vep_p2p"]

    if per_subject:
        # Collect per-subject channel-averaged values
        metrics_by_type = defaultdict(lambda: defaultdict(list))

        for subj, res in subjects_results:
            cfg = subj.get("config", FELT_TCRE_CONFIG)
            name = subj["name"]

            # Group channel indices by electrode type
            type_to_ch_idxs = defaultdict(list)
            for ch_idx in range(cfg.n_channels):
                type_to_ch_idxs[cfg.channel_types[ch_idx]].append(ch_idx)

            for ch_type, idxs in type_to_ch_idxs.items():
                for metric in METRIC_KEYS:
                    vals = [res[metric][i] for i in idxs]
                    # nanmean so that any NaN channels are excluded
                    metrics_by_type[ch_type][metric].append(np.nanmean(vals))
                metrics_by_type[ch_type]["subject_names"].append(name)
                metrics_by_type[ch_type]["n_channels_per_subject"].append(len(idxs))
    else:
        # Legacy: one entry per channel (channels are NOT independent)
        metrics_by_type = defaultdict(lambda: defaultdict(list))
        for subj, res in subjects_results:
            cfg = subj.get("config", FELT_TCRE_CONFIG)
            name = subj["name"]
            for ch_idx in range(cfg.n_channels):
                ch_type = cfg.channel_types[ch_idx]
                for metric in METRIC_KEYS:
                    metrics_by_type[ch_type][metric].append(res[metric][ch_idx])
                metrics_by_type[ch_type]["subject_names"].append(name)
                metrics_by_type[ch_type]["n_channels_per_subject"].append(1)

    result = {}
    for type_name, m in metrics_by_type.items():
        result[type_name] = {
            k: np.array(v) for k, v in m.items()
            if k not in ("subject_names", "n_channels_per_subject")
        }
        result[type_name]["subject_names"] = m["subject_names"]
        result[type_name]["n_channels_per_subject"] = m["n_channels_per_subject"]

    return result


def _normalize_psd(pxx, f, norm_band=(1, 30)):
    """
    Normalize a PSD array (shape: n_freqs) to unit total power in norm_band.

    This makes PSDs from different hardware gain chains (e.g. Gel /187 vs Felt)
    directly comparable: the y-axis becomes 'fraction of broadband power per Hz'.
    Each individual channel's PSD is normalized before group averaging, so the
    normalization is per-channel, not per-group.
    """
    mask = (f >= norm_band[0]) & (f <= norm_band[1])
    total = np.trapezoid(pxx[mask], f[mask])
    if total > 0:
        return pxx / total
    return pxx


def extract_psd_by_type(subjects_results, nperseg=4096, max_freq=45,
                        normalize=True):
    """
    Compute average PSD per electrode type across all subjects.

    Args:
        normalize: if True (default), normalize each channel's PSD by its own
            total power in 1-30 Hz before averaging.  This makes PSDs from
            different gain chains (Felt vs Gel) directly comparable.
            y-axis unit becomes '1/Hz (relative)' instead of 'µV²/Hz'.

    Returns:
        dict mapping type_name -> (freqs, mean_psd, std_psd)
    """
    psd_by_type = defaultdict(list)
    f_out = None

    for subj, _res in subjects_results:
        cfg = subj.get("config", FELT_TCRE_CONFIG)
        eeg = subj["eeg"]

        for ch_idx in range(cfg.n_channels):
            ch_type = cfg.channel_types[ch_idx]
            cleaned = notch_filter(eeg[ch_idx])
            f, pxx = signal.welch(cleaned, fs=FS, nperseg=nperseg)
            mask = f <= max_freq
            if f_out is None:
                f_out = f[mask]
            pxx_masked = pxx[mask]
            if normalize:
                pxx_masked = _normalize_psd(pxx_masked, f_out)
            psd_by_type[ch_type].append(pxx_masked)

    result = {}
    for type_name, psd_list in psd_by_type.items():
        arr = np.array(psd_list)
        median = np.median(arr, axis=0)
        iqr    = np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)
        result[type_name] = (f_out, median, iqr)

    return result


def extract_open_closed_psd_by_type(subjects_results, nperseg=4096,
                                    max_freq=30, normalize=True):
    """
    Compute average eyes-open and eyes-closed PSD per electrode type.

    Args:
        normalize: if True (default), normalize each channel's PSD by its own
            total power in 1-30 Hz (computed from the eyes-open segment) before
            averaging.  Both open and closed are divided by the *same* open-epoch
            total so the y-axis is comparable and the closed-minus-open difference
            reflects the Berger effect in relative units.

    Returns:
        dict mapping type_name -> {
            "freqs": array,
            "open_mean": array, "open_std": array,
            "closed_mean": array, "closed_std": array,
        }
    """
    open_by_type = defaultdict(list)
    closed_by_type = defaultdict(list)
    f_out = None

    for subj, _res in subjects_results:
        cfg = subj.get("config", FELT_TCRE_CONFIG)
        eeg = subj["eeg"]
        open_epochs = subj["open_epochs"]
        close_epochs = subj["close_epochs"]

        if not open_epochs or not close_epochs:
            continue

        for ch_idx in range(cfg.n_channels):
            ch_type = cfg.channel_types[ch_idx]
            cleaned = notch_filter(eeg[ch_idx])

            od = np.concatenate([cleaned[ep["start"]:ep["end"]]
                                 for ep in open_epochs])
            cd = np.concatenate([cleaned[ep["start"]:ep["end"]]
                                 for ep in close_epochs])

            f_o, pxx_o = signal.welch(od, fs=FS, nperseg=nperseg)
            f_c, pxx_c = signal.welch(cd, fs=FS, nperseg=nperseg)
            mask = f_o <= max_freq
            if f_out is None:
                f_out = f_o[mask]

            pxx_o_m = pxx_o[mask]
            pxx_c_m = pxx_c[mask]

            if normalize:
                # Normalise both open and closed by the same reference (open
                # total power) so that the ratio at each frequency reflects
                # the Berger effect independent of hardware gain.
                ref = np.trapezoid(pxx_o_m[(f_out >= 1) & (f_out <= 30)],
                                   f_out[(f_out >= 1) & (f_out <= 30)])
                if ref > 0:
                    pxx_o_m = pxx_o_m / ref
                    pxx_c_m = pxx_c_m / ref

            open_by_type[ch_type].append(pxx_o_m)
            closed_by_type[ch_type].append(pxx_c_m)

    result = {}
    for type_name in set(list(open_by_type.keys()) + list(closed_by_type.keys())):
        o_arr = np.array(open_by_type[type_name])
        c_arr = np.array(closed_by_type[type_name])
        # Store median as "mean" key and IQR as "std" key so plot code is
        # consistent with the main PSD function naming convention.
        result[type_name] = {
            "freqs":       f_out,
            "open_mean":   np.median(o_arr, axis=0),
            "open_std":    (np.percentile(o_arr, 75, axis=0) -
                            np.percentile(o_arr, 25, axis=0)),
            "closed_mean": np.median(c_arr, axis=0),
            "closed_std":  (np.percentile(c_arr, 75, axis=0) -
                            np.percentile(c_arr, 25, axis=0)),
        }

    return result


def compare_electrode_types(felt_subjects, gel_subjects):
    """
    Run statistical comparisons across electrode types.

    Uses Paste TCRE (present in both recordings) as the bridge reference.
    Tests: Wilcoxon signed-rank for within-recording, Mann-Whitney U for
    cross-recording comparisons.

    Returns:
        dict with comparison results and summary tables
    """
    all_subjects = felt_subjects + gel_subjects
    # Use per_subject=True so each subject contributes exactly one
    # observation per electrode type, regardless of how many channels
    # that setup records per type.
    type_metrics = extract_type_metrics(all_subjects, per_subject=True)

    felt_type_metrics = extract_type_metrics(felt_subjects, per_subject=True)
    gel_type_metrics  = extract_type_metrics(gel_subjects,  per_subject=True)

    comparisons = {}
    # Only ratio/scale-invariant metrics are valid for cross-setup comparison.
    # alpha_open and alpha_closed are in µV²/Hz and are NOT comparable between
    # Felt and Gel setups because Gel tEEG channels are divided by 187 during
    # loading (amplitude /187 → power /187² ≈ /35000). SNR and reactivity are
    # ratios and cancel out this scaling factor.
    metric_keys = ["alpha_snr", "alpha_reactivity"]

    # Felt tEEG vs Paste tEEG (within felt recordings)
    for metric in metric_keys:
        if "FELT_TEEG" in felt_type_metrics and "PASTE_TEEG" in felt_type_metrics:
            a = felt_type_metrics["FELT_TEEG"][metric]
            b = felt_type_metrics["PASTE_TEEG"][metric]
            a_v = a[~np.isnan(a)]
            b_v = b[~np.isnan(b)]
            if len(a_v) >= 3 and len(b_v) >= 3:
                stat, pval = stats.mannwhitneyu(a_v, b_v, alternative="two-sided")
                comparisons[f"Felt_tEEG_vs_Paste_tEEG_{metric}"] = {
                    "stat": stat, "p": pval, "test": "Mann-Whitney U",
                    "d": cohens_d(a_v, b_v),
                    "n_a": len(a_v), "n_b": len(b_v),
                }

    # Gel tEEG vs Paste tEEG (within gel recordings)
    for metric in metric_keys:
        if "GEL_TEEG" in gel_type_metrics and "PASTE_TEEG" in gel_type_metrics:
            a = gel_type_metrics["GEL_TEEG"][metric]
            b = gel_type_metrics["PASTE_TEEG"][metric]
            a_v = a[~np.isnan(a)]
            b_v = b[~np.isnan(b)]
            if len(a_v) >= 3 and len(b_v) >= 3:
                stat, pval = stats.mannwhitneyu(a_v, b_v, alternative="two-sided")
                comparisons[f"Gel_tEEG_vs_Paste_tEEG_{metric}"] = {
                    "stat": stat, "p": pval, "test": "Mann-Whitney U",
                    "d": cohens_d(a_v, b_v),
                    "n_a": len(a_v), "n_b": len(b_v),
                }

    # Felt tEEG vs Gel tEEG (cross-recording)
    for metric in metric_keys:
        if "FELT_TEEG" in felt_type_metrics and "GEL_TEEG" in gel_type_metrics:
            a = felt_type_metrics["FELT_TEEG"][metric]
            b = gel_type_metrics["GEL_TEEG"][metric]
            a_v = a[~np.isnan(a)]
            b_v = b[~np.isnan(b)]
            if len(a_v) >= 3 and len(b_v) >= 3:
                stat, pval = stats.mannwhitneyu(a_v, b_v, alternative="two-sided")
                comparisons[f"Felt_tEEG_vs_Gel_tEEG_{metric}"] = {
                    "stat": stat, "p": pval, "test": "Mann-Whitney U",
                    "d": cohens_d(a_v, b_v),
                    "n_a": len(a_v), "n_b": len(b_v),
                }

    # Paste TCRE bridge: Felt-recording paste vs Gel-recording paste
    for metric in metric_keys:
        if "PASTE_TEEG" in felt_type_metrics and "PASTE_TEEG" in gel_type_metrics:
            a = felt_type_metrics["PASTE_TEEG"][metric]
            b = gel_type_metrics["PASTE_TEEG"][metric]
            a_v = a[~np.isnan(a)]
            b_v = b[~np.isnan(b)]
            if len(a_v) >= 2 and len(b_v) >= 2:
                stat, pval = stats.mannwhitneyu(a_v, b_v, alternative="two-sided")
                comparisons[f"Paste_bridge_felt_vs_gel_{metric}"] = {
                    "stat": stat, "p": pval, "test": "Mann-Whitney U",
                    "d": cohens_d(a_v, b_v),
                    "n_a": len(a_v), "n_b": len(b_v),
                }

    return {
        "comparisons": comparisons,
        "all_type_metrics": type_metrics,
        "felt_type_metrics": felt_type_metrics,
        "gel_type_metrics": gel_type_metrics,
    }


def plot_three_way_comparison(felt_subjects, gel_subjects, save_dir=None, show=True):
    """
    Generate comprehensive 3-way comparison figures.

    Args:
        felt_subjects: list of (subject, results) for Felt TCRE
        gel_subjects: list of (subject, results) for Gel TCRE
        save_dir: directory to save figures (created if needed)
        show: whether to call plt.show()
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _save_or_show(fig, filename):
        if save_dir:
            fig.savefig(os.path.join(save_dir, filename),
                        dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    all_subjects = felt_subjects + gel_subjects
    comp = compare_electrode_types(felt_subjects, gel_subjects)
    all_metrics = comp["all_type_metrics"]

    # Order of types for display
    display_order = ["FELT_TEEG", "GEL_TEEG", "PASTE_TEEG",
                     "FELT_EEEG", "GEL_EEEG", "PASTE_EEEG", "DISC"]
    present = [t for t in display_order if t in all_metrics]

    legend_handles = [Patch(facecolor=TYPE_COLORS.get(t, "gray"),
                            label=TYPE_DISPLAY.get(t, t))
                      for t in present]

    # ─── 1. Alpha SNR — boxplot (robust to outliers) ───
    fig, ax = plt.subplots(figsize=(13, 5))
    x_pos = np.arange(len(present))
    colors = [TYPE_COLORS.get(t, "gray") for t in present]

    data_snr = []
    for t in present:
        vals = all_metrics[t]["alpha_snr"]
        data_snr.append(vals[~np.isnan(vals)])

    bp = ax.boxplot(data_snr, positions=x_pos, widths=0.55,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker="o", markersize=4,
                                   markerfacecolor="gray", alpha=0.6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # overlay individual data points
    for i, (t, vals) in enumerate(zip(present, data_snr)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(x_pos[i] + jitter, vals, s=18, color=colors[i],
                   edgecolors="k", lw=0.4, zorder=3, alpha=0.8)
        ax.text(x_pos[i], ax.get_ylim()[0] if ax.get_ylim()[0] > -999 else -1,
                f"n={len(vals)}", ha="center", fontsize=8, va="top")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([TYPE_DISPLAY.get(t, t) for t in present],
                       fontsize=10, rotation=15)
    ax.set_ylabel("Alpha SNR (dB)")
    ax.set_title("Alpha SNR by Electrode Type — Median ± IQR + Individual Subjects",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8, ls="--", label="0 dB (noise floor)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    _save_or_show(fig, "comparison_alpha_snr.png")

    # ─── 2. Alpha Reactivity — boxplot (skewed distribution, outliers common) ───
    fig, ax = plt.subplots(figsize=(13, 5))

    data_react = []
    for t in present:
        vals = all_metrics[t]["alpha_reactivity"]
        data_react.append(vals[~np.isnan(vals)])

    bp2 = ax.boxplot(data_react, positions=x_pos, widths=0.55,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="black", lw=2),
                     flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor="gray", alpha=0.6))
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for i, (t, vals) in enumerate(zip(present, data_react)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(x_pos[i] + jitter, vals, s=18, color=colors[i],
                   edgecolors="k", lw=0.4, zorder=3, alpha=0.8)
        ax.text(x_pos[i], -0.3, f"n={len(vals)}", ha="center",
                fontsize=8, va="top")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([TYPE_DISPLAY.get(t, t) for t in present],
                       fontsize=10, rotation=15)
    ax.set_ylabel("Closed / Open (Alpha Power Ratio)")
    ax.set_title("Alpha Reactivity (Berger Effect) — Median ± IQR + Individual Subjects",
                 fontsize=12, fontweight="bold")
    ax.axhline(1, color="gray", lw=1, ls="--", label="No effect (ratio = 1)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    _save_or_show(fig, "comparison_alpha_reactivity.png")

    # ─── 3. PSD overlay — median ± IQR (robust to outliers) ───
    # Median is not pulled by subjects with flat/dead signals the way mean is.
    psd_all = extract_psd_by_type(all_subjects, normalize=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    for t in ["FELT_TEEG", "GEL_TEEG", "PASTE_TEEG", "DISC"]:
        if t not in psd_all:
            continue
        f, median_psd, iqr_psd = psd_all[t]   # now returns median / IQR
        mask = (f >= 1) & (f <= 30)
        ax.semilogy(f[mask], median_psd[mask], lw=2,
                    color=TYPE_COLORS.get(t, "gray"),
                    label=TYPE_DISPLAY.get(t, t))
        lo = (median_psd - iqr_psd * 0.5)[mask].clip(median_psd[mask].min() * 0.01)
        hi = (median_psd + iqr_psd * 0.5)[mask]
        ax.fill_between(f[mask], lo, hi,
                        alpha=0.15, color=TYPE_COLORS.get(t, "gray"))

    ax.axvspan(8, 13, alpha=0.08, color="#e67e22", label="Alpha band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative PSD (normalized, 1/Hz)")
    ax.set_title(
        "PSD Comparison: tEEG Channels by Electrode Type (1–30 Hz)\n"
        "[Median ± ½ IQR; each channel normalized to its own 1–30 Hz power]",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(1, 30)
    plt.tight_layout()
    _save_or_show(fig, "comparison_psd_teeg.png")

    # ─── 4. Eyes open vs closed PSD — median ± IQR ───
    oc_psd = extract_open_closed_psd_by_type(all_subjects, normalize=True)
    teeg_types = ["FELT_TEEG", "GEL_TEEG", "PASTE_TEEG"]
    teeg_present = [t for t in teeg_types if t in oc_psd]

    if teeg_present:
        fig, axes = plt.subplots(1, len(teeg_present),
                                 figsize=(6 * len(teeg_present), 5),
                                 sharey=False)
        if len(teeg_present) == 1:
            axes = [axes]
        fig.suptitle(
            "Eyes Open vs Closed PSD by Electrode Type\n"
            "[Median ± ½ IQR; normalized to eyes-open total power]",
            fontsize=12, fontweight="bold")

        for idx, t in enumerate(teeg_present):
            ax = axes[idx]
            d = oc_psd[t]
            f = d["freqs"]
            mask = (f >= 1) & (f <= 30)

            for key, color, label in [
                ("open",   "#27ae60", "Eyes Open"),
                ("closed", "#3498db", "Eyes Closed"),
            ]:
                med = d[f"{key}_mean"][mask]    # median stored here
                iqr = d[f"{key}_std"][mask]     # IQR stored here
                ax.semilogy(f[mask], med, lw=2, color=color, label=label)
                lo = (med - iqr * 0.5).clip(med.min() * 0.01)
                hi = med + iqr * 0.5
                ax.fill_between(f[mask], lo, hi, alpha=0.15, color=color)

            ax.axvspan(8, 13, alpha=0.08, color="#e67e22")
            ax.set_title(TYPE_DISPLAY.get(t, t), fontsize=11, fontweight="bold")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Relative PSD (normalized)")
            ax.legend(fontsize=8)
            ax.set_xlim(1, 30)

        plt.tight_layout()
        _save_or_show(fig, "comparison_open_vs_closed_psd.png")

    # ─── 5. Paste TCRE bridge validation ───
    felt_paste_snr = comp["felt_type_metrics"].get("PASTE_TEEG", {}).get("alpha_snr", np.array([]))
    gel_paste_snr = comp["gel_type_metrics"].get("PASTE_TEEG", {}).get("alpha_snr", np.array([]))

    if len(felt_paste_snr) > 0 and len(gel_paste_snr) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Paste TCRE Bridge Validation (Same Electrode Type Across Setups)",
                     fontsize=13, fontweight="bold")

        metrics_to_plot = [
            ("alpha_snr", "Alpha SNR (dB)"),
            ("alpha_reactivity", "Alpha Reactivity (Closed/Open)"),
        ]
        for idx, (mk, ylabel) in enumerate(metrics_to_plot):
            ax = axes[idx]
            felt_vals = comp["felt_type_metrics"].get("PASTE_TEEG", {}).get(mk, np.array([]))
            gel_vals = comp["gel_type_metrics"].get("PASTE_TEEG", {}).get(mk, np.array([]))
            felt_vals = felt_vals[~np.isnan(felt_vals)]
            gel_vals = gel_vals[~np.isnan(gel_vals)]

            positions = [1, 2]
            bp = ax.boxplot([felt_vals, gel_vals], positions=positions,
                            widths=0.5, patch_artist=True,
                            medianprops=dict(color="black", lw=2))
            bp["boxes"][0].set_facecolor("#3498db")
            bp["boxes"][1].set_facecolor("#e67e22")
            ax.set_xticklabels(["Felt Setup", "Gel Setup"])
            ax.set_ylabel(ylabel)
            ax.set_title(f"Paste TCRE: {ylabel}")

            bridge_key = f"Paste_bridge_felt_vs_gel_{mk}"
            if bridge_key in comp["comparisons"]:
                c = comp["comparisons"][bridge_key]
                sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else "ns"
                ax.text(1.5, ax.get_ylim()[1] * 0.95,
                        f"p={c['p']:.3f} ({sig})\nd={c['d']:.2f}",
                        ha="center", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        plt.tight_layout()
        _save_or_show(fig, "comparison_paste_bridge.png")

    # ─── 6. Statistical summary table ───
    if comp["comparisons"]:
        print("\n" + "=" * 90)
        print("STATISTICAL COMPARISONS")
        print("=" * 90)
        print(f"{'Comparison':<45} {'Test':<18} {'p-value':<10} {'Cohen d':<10} {'n_a(subj)':<10} {'n_b(subj)':<10}")
        print("-" * 95)
        for name, c in sorted(comp["comparisons"].items()):
            sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
            print(f"{name:<45} {c['test']:<18} {c['p']:<10.4f} {c['d']:<10.2f} {c['n_a']:<10} {c['n_b']:<10} {sig}")
        print("=" * 90)

    return comp
