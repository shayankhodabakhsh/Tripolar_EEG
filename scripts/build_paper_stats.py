#!/usr/bin/env python3
"""
build_paper_stats.py — refresh the §3 numbers and the comparison figures.

Pools all felt sessions (long + short) and gel sessions that pass objective
QC (epoch-count floor + alpha SNR >= -1 dB + reactivity >= 1.0), runs the
three-way comparison in tripolar_eeg.comparison_analysis, and emits:

  paper/auto_stats.md     — markdown tables ready to paste into main.tex
  paper/auto_stats.json   — the same numbers, machine-readable
  output/three_way_comparison/    — refreshed PNGs (incl. comparison_ssim.png)

Run from repo root:
    source venv/bin/activate
    python scripts/build_paper_stats.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT  = Path(__file__).resolve().parent.parent
# Pool long-felt and short-felt recordings into a single felt cohort. Long-felt
# sessions live under data/Long felt TCRE/; short-felt sessions live in data/
# directly. Each subject is tagged "long" or "short" by load_all_subjects.
FELT_DIR   = REPO_ROOT / "data"
GEL_DIR    = REPO_ROOT / "Gel TCRE"
OUT_FIGS   = REPO_ROOT / "output" / "three_way_comparison"
OUT_MD     = REPO_ROOT / "paper" / "auto_stats.md"
OUT_JSON   = REPO_ROOT / "paper" / "auto_stats.json"

# Prefer the installed package; fall back to src-layout discovery so the
# script can be run before `pip install -e .`.
try:
    from tripolar_eeg.comparison_analysis import (
        load_all_subjects, extract_type_metrics, compare_electrode_types,
        plot_three_way_comparison, print_qc_report, assign_display_ids,
        TYPE_DISPLAY,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tripolar_eeg.comparison_analysis import (        # noqa: E402
        load_all_subjects, extract_type_metrics, compare_electrode_types,
        plot_three_way_comparison, print_qc_report, assign_display_ids,
        TYPE_DISPLAY,
    )

# Pooled felt cohort (long + short) with objective QC gates.
FELT_SUBJECTS = None
EXCLUDE_FELT  = ["LuciTest"]
EXCLUDE_GEL   = []

# Signal-quality gates. Drop "terrible" subjects on data-driven thresholds:
#   - At least 2 eyes-open and 2 eyes-closed epochs (data integrity).
#   - Mean tEEG alpha SNR >= -1 dB (lets through borderline-negative cases
#     for transparency, drops only clearly broken sessions).
#   - Mean tEEG alpha reactivity >= 1.0 (Berger effect detectable).
QC_KW = dict(
    min_open_epochs=2,
    min_close_epochs=2,
    min_alpha_snr_db=-1.0,
    min_alpha_reactivity=1.0,
    min_vep_p2p_uv=None,
)


def median_with_bca(values, n_boot=1000, seed=0):
    """Median + BCa 95% CI. Returns (median, lo, hi, n)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float(v[0]), float(v[0]), 1
    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (v,),
        np.median,
        n_resamples=n_boot,
        confidence_level=0.95,
        method="BCa",
        random_state=rng,
        vectorized=False,
    )
    return float(np.median(v)), float(res.confidence_interval.low), float(res.confidence_interval.high), n


def fmt_ci(med, lo, hi, n, digits=2):
    if n == 0:
        return "n/a"
    if n == 1:
        return f"{med:.{digits}f} (n=1)"
    return f"{med:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}] (n={n})"


def main():
    print("=" * 72)
    print("Building paper/auto_stats.md from locked cohort")
    print("=" * 72)

    felt_subjects, gel_subjects = load_all_subjects(
        felt_dir=str(FELT_DIR),
        gel_dir=str(GEL_DIR),
        felt_names=FELT_SUBJECTS,
        exclude_felt=EXCLUDE_FELT,
        exclude_gel=EXCLUDE_GEL,
        felt_recursive=True,
        verbose=True,
        **QC_KW,
    )

    # Assign de-identified display codes (LF01.., SF01.., G01..) and
    # write the private subject key inside data/ (which is gitignored).
    key_csv = REPO_ROOT / "data" / "SUBJECT_KEY.csv"
    assign_display_ids(felt_subjects, gel_subjects, key_csv_path=str(key_csv))
    print(f"Wrote {key_csv.relative_to(REPO_ROOT)} (private; do not commit)")

    print_qc_report(felt_subjects, label="Felt (long + short, QC-passing)")
    print_qc_report(gel_subjects,  label="Gel (QC-passing)")

    long_felt  = [(s, r) for s, r in felt_subjects if s.get("felt_cohort") == "long"]
    short_felt = [(s, r) for s, r in felt_subjects if s.get("felt_cohort") == "short"]
    print(f"\n  Felt cohort split: long n={len(long_felt)}, short n={len(short_felt)}")

    cmp = compare_electrode_types(felt_subjects, gel_subjects)
    type_metrics_all = cmp["all_type_metrics"]
    felt_tm = cmp["felt_type_metrics"]
    gel_tm  = cmp["gel_type_metrics"]
    comparisons = cmp["comparisons"]

    # --- Per-class medians + BCa CIs ---
    metric_keys = ["alpha_snr", "alpha_reactivity", "ssim"]
    classes_in_order = ["FELT_TEEG", "GEL_TEEG", "PASTE_TEEG",
                        "FELT_EEEG", "GEL_EEEG", "PASTE_EEEG", "DISC"]

    summary = {}
    for cls in classes_in_order:
        if cls not in type_metrics_all:
            continue
        summary[cls] = {}
        for k in metric_keys + ["alpha_open", "alpha_closed", "disc_correlation", "vep_p2p"]:
            if k not in type_metrics_all[cls]:
                continue
            med, lo, hi, n = median_with_bca(type_metrics_all[cls][k])
            summary[cls][k] = {"median": med, "ci_lo": lo, "ci_hi": hi, "n": n}

    # --- Reactivity > 1 counts ---
    react_gt1 = {}
    for cls in classes_in_order:
        if cls in type_metrics_all and "alpha_reactivity" in type_metrics_all[cls]:
            v = np.asarray(type_metrics_all[cls]["alpha_reactivity"], dtype=float)
            v = v[~np.isnan(v)]
            react_gt1[cls] = {"n_gt1": int((v > 1.0).sum()), "n_total": int(v.size)}

    # --- Paste bridge: Felt-paste vs Gel-paste ---
    paste_bridge = {k: comparisons.get(k) for k in (
        "Paste_bridge_felt_vs_gel_alpha_snr",
        "Paste_bridge_felt_vs_gel_alpha_reactivity",
        "Paste_bridge_felt_vs_gel_ssim",
    )}

    # --- Refresh figures ---
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    print(f"\nRegenerating figures in {OUT_FIGS} ...")
    plot_three_way_comparison(felt_subjects, gel_subjects,
                              save_dir=str(OUT_FIGS), show=False)

    # --- Write markdown ---
    md = []
    md.append("# Auto-generated stats for paper/main.tex\n")
    md.append("Source: `scripts/build_paper_stats.py` on the pooled-felt cohort ")
    md.append("(felt = long + short post-QC; gel = QC-passing). Re-run to refresh.\n")

    md.append("\n## §3.1 Cohort post-QC\n")
    md.append("| Cohort | n loaded | Subjects |\n|--|--|--|\n")
    md.append(f"| Felt (all) | {len(felt_subjects)} | {', '.join(s.get('display_id') or s['name'] for s, _ in felt_subjects)} |\n")
    md.append(f"| &nbsp;&nbsp;Long felt | {len(long_felt)} | {', '.join(s.get('display_id') or s['name'] for s, _ in long_felt)} |\n")
    md.append(f"| &nbsp;&nbsp;Short felt | {len(short_felt)} | {', '.join(s.get('display_id') or s['name'] for s, _ in short_felt)} |\n")
    md.append(f"| Gel       | {len(gel_subjects)}  | {', '.join(s.get('display_id') or s['name'] for s, _ in gel_subjects)} |\n")

    md.append("\n## §3.2–§3.5 Per-class medians (BCa 95% CI)\n")
    md.append("| Class | Alpha SNR (dB) | Reactivity (closed/open) | SSIM |\n|--|--|--|--|\n")
    for cls in classes_in_order:
        if cls not in summary:
            continue
        s = summary[cls]
        snr  = fmt_ci(s["alpha_snr"]["median"], s["alpha_snr"]["ci_lo"],
                      s["alpha_snr"]["ci_hi"], s["alpha_snr"]["n"], 2) \
                if "alpha_snr" in s else "n/a"
        rea  = fmt_ci(s["alpha_reactivity"]["median"], s["alpha_reactivity"]["ci_lo"],
                      s["alpha_reactivity"]["ci_hi"], s["alpha_reactivity"]["n"], 2) \
                if "alpha_reactivity" in s else "n/a"
        ssm  = fmt_ci(s["ssim"]["median"], s["ssim"]["ci_lo"],
                      s["ssim"]["ci_hi"], s["ssim"]["n"], 3) \
                if "ssim" in s else "n/a"
        md.append(f"| {TYPE_DISPLAY.get(cls, cls)} | {snr} | {rea} | {ssm} |\n")

    md.append("\n## §3.3 Reactivity > 1 (Berger effect detected) per class\n")
    md.append("| Class | n with reactivity > 1 | n total |\n|--|--|--|\n")
    for cls in classes_in_order:
        if cls in react_gt1:
            md.append(f"| {TYPE_DISPLAY.get(cls, cls)} | {react_gt1[cls]['n_gt1']} | {react_gt1[cls]['n_total']} |\n")

    md.append("\n## §3.6 Paste-TCRE cross-setup bridge (Mann–Whitney U)\n")
    md.append("| Metric | n_felt | n_gel | U | p | Cohen d |\n|--|--|--|--|--|--|\n")
    for label, c in paste_bridge.items():
        metric = label.replace("Paste_bridge_felt_vs_gel_", "")
        if c is None:
            md.append(f"| {metric} | — | — | — | n/a (insufficient n) | — |\n")
        else:
            md.append(f"| {metric} | {c['n_a']} | {c['n_b']} | {c['stat']:.1f} | {c['p']:.3f} | {c['d']:.2f} |\n")

    md.append("\n## All Mann–Whitney comparisons\n")
    md.append("| Contrast | n_a | n_b | U | p | Cohen d |\n|--|--|--|--|--|--|\n")
    for label, c in comparisons.items():
        md.append(f"| {label} | {c['n_a']} | {c['n_b']} | {c['stat']:.1f} | {c['p']:.3f} | {c['d']:.2f} |\n")

    OUT_MD.write_text("".join(md))
    print(f"\nWrote {OUT_MD.relative_to(REPO_ROOT)}")

    # Long-vs-short felt breakdown for supplement S6.
    breakdown = {}
    for label, subset in [("long_felt", long_felt), ("short_felt", short_felt)]:
        if not subset:
            continue
        sub_metrics = extract_type_metrics(subset, per_subject=True)
        breakdown[label] = {}
        for cls in ("FELT_TEEG", "FELT_EEEG", "PASTE_TEEG", "PASTE_EEEG", "DISC"):
            if cls not in sub_metrics:
                continue
            breakdown[label][cls] = {}
            for k in ("alpha_snr", "alpha_reactivity", "ssim"):
                if k not in sub_metrics[cls]:
                    continue
                med, lo, hi, n = median_with_bca(sub_metrics[cls][k])
                breakdown[label][cls][k] = {
                    "median": med, "ci_lo": lo, "ci_hi": hi, "n": n,
                }

    md.append("\n## Long-vs-short felt breakdown (supplement S6)\n")
    md.append("Per-class median (BCa 95% CI) computed separately for the long-felt ")
    md.append("and short-felt sub-cohorts. Used to show that the construction ")
    md.append("comparison is not driven by either sub-cohort alone.\n\n")
    for label, subset in [("Long felt", long_felt), ("Short felt", short_felt)]:
        if not subset:
            continue
        key = "long_felt" if label == "Long felt" else "short_felt"
        md.append(f"### {label} (n={len(subset)})\n")
        md.append("| Class | Alpha SNR (dB) | Reactivity | SSIM |\n|--|--|--|--|\n")
        for cls in ("FELT_TEEG", "FELT_EEEG", "PASTE_TEEG", "PASTE_EEEG", "DISC"):
            if cls not in breakdown[key]:
                continue
            row = [TYPE_DISPLAY.get(cls, cls)]
            for k, digits in (("alpha_snr", 2), ("alpha_reactivity", 2), ("ssim", 3)):
                if k not in breakdown[key][cls]:
                    row.append("n/a")
                    continue
                d = breakdown[key][cls][k]
                row.append(fmt_ci(d["median"], d["ci_lo"], d["ci_hi"], d["n"], digits))
            md.append("| " + " | ".join(row) + " |\n")
        md.append("\n")

    payload = {
        "cohort": {
            "felt": [s.get("display_id") or s["name"] for s, _ in felt_subjects],
            "long_felt":  [s.get("display_id") or s["name"] for s, _ in long_felt],
            "short_felt": [s.get("display_id") or s["name"] for s, _ in short_felt],
            "gel":  [s.get("display_id") or s["name"] for s, _ in gel_subjects],
        },
        "summary": summary,
        "long_short_breakdown": breakdown,
        "reactivity_gt1": react_gt1,
        "comparisons": {k: v for k, v in comparisons.items()},
    }

    def _to_jsonable(o):
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(x) for x in o]
        if isinstance(o, np.generic):
            return o.item()
        return o

    OUT_JSON.write_text(json.dumps(_to_jsonable(payload), indent=2))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
