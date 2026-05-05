#!/usr/bin/env python3
"""
build_sensitivity_figure.py — cohort-level A/B/C preprocessing sensitivity for
supplement section S1.

Re-runs the locked cohort under each pipeline variant and produces:

  output/sensitivity/sensitivity_abc.png   — 3-row panel (alpha SNR / reactivity / SSIM)
  paper/sensitivity_abc.md         — markdown table of per-class medians + IQR
  paper/sensitivity_abc.json       — same data, machine-readable
  paper/figs/sensitivity_abc.png   — copied for the bundled paper folder

Run from repo root:
    source venv/bin/activate
    python scripts/build_sensitivity_figure.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FELT_DIR  = REPO_ROOT / "data"
GEL_DIR   = REPO_ROOT / "Gel TCRE"
OUT_DIR   = REPO_ROOT / "output" / "sensitivity"
OUT_FIG   = OUT_DIR / "sensitivity_abc.png"
OUT_MD    = REPO_ROOT / "paper" / "sensitivity_abc.md"
OUT_JSON  = REPO_ROOT / "paper" / "sensitivity_abc.json"
FIGS_COPY = REPO_ROOT / "paper" / "figs" / "sensitivity_abc.png"

try:
    from tripolar_eeg.comparison_analysis import (
        load_all_subjects, extract_type_metrics, TYPE_DISPLAY, TYPE_COLORS,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tripolar_eeg.comparison_analysis import (        # noqa: E402
        load_all_subjects, extract_type_metrics, TYPE_DISPLAY, TYPE_COLORS,
    )

VARIANTS = [
    ("A", "Notch + bandpass + wavelet + z-score"),
    ("B", "Notch + bandpass (PRIMARY)"),
    ("C", "Notch only"),
]

CLASSES = ["FELT_TEEG", "GEL_TEEG", "PASTE_TEEG"]
METRICS = [
    ("alpha_snr",         "Alpha SNR (dB)"),
    ("alpha_reactivity",  "Reactivity (closed/open)"),
    ("ssim",              "SSIM (tEEG vs eEEG)"),
]


def run_one_variant(variant):
    print(f"\n=== Variant {variant}: loading + analysing cohort ===")
    felt, gel = load_all_subjects(
        felt_dir=str(FELT_DIR),
        gel_dir=str(GEL_DIR),
        felt_names=None,
        exclude_felt=["LuciTest"],
        exclude_gel=[],
        min_open_epochs=2, min_close_epochs=2,
        min_alpha_snr_db=-1.0, min_alpha_reactivity=1.0, min_vep_p2p_uv=None,
        pipeline_variant=variant,
        felt_recursive=True,
        verbose=False,
    )
    tm = extract_type_metrics(felt + gel, per_subject=True)
    return tm, len(felt), len(gel)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for v, _ in VARIANTS:
        tm, n_felt, n_gel = run_one_variant(v)
        results[v] = {"metrics": tm, "n_felt": n_felt, "n_gel": n_gel}

    # --- Plot ---
    n_metrics = len(METRICS)
    n_variants = len(VARIANTS)
    fig, axes = plt.subplots(n_metrics, n_variants,
                             figsize=(4.0 * n_variants, 3.4 * n_metrics),
                             sharey="row")
    if n_metrics == 1:
        axes = axes[np.newaxis, :]
    if n_variants == 1:
        axes = axes[:, np.newaxis]

    for col, (variant, vlabel) in enumerate(VARIANTS):
        tm = results[variant]["metrics"]
        for row, (metric, mlabel) in enumerate(METRICS):
            ax = axes[row, col]
            data, labels, colors = [], [], []
            for cls in CLASSES:
                if cls not in tm or metric not in tm[cls]:
                    continue
                vals = np.asarray(tm[cls][metric], dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                data.append(vals)
                labels.append(TYPE_DISPLAY.get(cls, cls))
                colors.append(TYPE_COLORS.get(cls, "#888"))
            if not data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                ax.set_xticks([])
                continue
            bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                            showmeans=False)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for i, vals in enumerate(data):
                jitter = (np.random.default_rng(0).normal(0, 0.04, size=len(vals)))
                ax.scatter(np.full_like(vals, i + 1, dtype=float) + jitter, vals,
                           color="black", s=14, alpha=0.75, zorder=3)
            if row == 0:
                ax.set_title(f"Variant {variant}\n{vlabel}", fontsize=10)
            if col == 0:
                ax.set_ylabel(mlabel)
            ax.tick_params(axis="x", labelrotation=18)
            ax.grid(True, axis="y", alpha=0.3)
            if metric == "alpha_reactivity":
                ax.axhline(1.0, color="red", lw=0.8, ls="--", alpha=0.5)

    fig.suptitle("Pipeline-variant sensitivity (locked cohort, subject = unit)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)
    shutil.copy(OUT_FIG, FIGS_COPY)
    print(f"\nWrote {OUT_FIG.relative_to(REPO_ROOT)}  (also copied to paper/figs/)")

    # --- Markdown table ---
    md = ["# Cohort-level pipeline-variant sensitivity (A/B/C)\n",
          "Auto-generated by `scripts/build_sensitivity_figure.py`. ",
          "Per-class median across subjects under each variant. n is the number ",
          "of subjects contributing a non-NaN value.\n\n"]
    for metric, mlabel in METRICS:
        md.append(f"\n## {mlabel}\n")
        md.append("| Class | Variant A | Variant B (primary) | Variant C |\n|--|--|--|--|\n")
        for cls in CLASSES:
            row = [TYPE_DISPLAY.get(cls, cls)]
            for v, _ in VARIANTS:
                tm = results[v]["metrics"]
                if cls not in tm or metric not in tm[cls]:
                    row.append("n/a")
                    continue
                vals = np.asarray(tm[cls][metric], dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    row.append("n/a")
                else:
                    row.append(f"{np.median(vals):.2f} (n={vals.size})")
            md.append("| " + " | ".join(row) + " |\n")
    OUT_MD.write_text("".join(md))
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")

    payload = {
        v: {
            "n_felt": results[v]["n_felt"],
            "n_gel": results[v]["n_gel"],
            "per_class_per_metric": {
                cls: {
                    m: {
                        "values": np.asarray(results[v]["metrics"].get(cls, {}).get(m, []),
                                             dtype=float).tolist(),
                    }
                    for m, _ in METRICS
                    if m in results[v]["metrics"].get(cls, {})
                }
                for cls in CLASSES
                if cls in results[v]["metrics"]
            },
        }
        for v, _ in VARIANTS
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
