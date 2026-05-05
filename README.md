# Tripolar EEG

A methods-harmonized analysis pipeline for **Tripolar Concentric Ring Electrode
(TCRE)** EEG, comparing three electrolyte/coupling constructions:
**saline-soaked felt**, **conductive gel**, and **paste**. The repository
backs the manuscript in [`paper/`](paper/) (target journal: MDPI *Sensors*).

TCREs provide an on-board surface-Laplacian derivation (tEEG) alongside a
conventional outer-ring recording (eEEG). This project's contribution is a
single, deliberately minimal preprocessing pipeline and a cross-setup paste
bridge that makes the three constructions readable on like-for-like
endpoints (alpha SNR, alpha reactivity, spectrogram SSIM).

---

## Repository layout

```
Tripolar_EEG/
├── README.md                  ← this file
├── LICENSE                    ← MIT
├── CITATION.cff               ← machine-readable citation
├── pyproject.toml             ← package metadata + pinned deps
├── .gitignore
│
├── src/tripolar_eeg/          ← installable analysis package
│   ├── __init__.py
│   ├── eeg_analysis.py            core engine (load, filter, analyze, plot)
│   └── comparison_analysis.py     three-way comparison + QC + stats
│
├── scripts/                   ← runnable scripts (`python scripts/<name>.py`)
│   ├── build_paper_stats.py        refresh §3 numbers + comparison figures
│   ├── build_felt_vep_figure.py    felt-setup VEP grand averages (§3.7)
│   ├── build_sensitivity_figure.py A/B/C pipeline-variant sensitivity (S1)
│   ├── gel_vep_sanity_check.py     qualitative gel-VEP supplement panels
│   ├── gel_vep_multi_subject.py
│   ├── ba2_preprocessing_sensitivity.py
│   └── report_v2.py                CLI batch QC report
│
├── notebooks/                 ← interactive analysis
│   ├── single_subject_analysis_v2.ipynb
│   ├── group_analysis.ipynb
│   └── three_way_comparison.ipynb
│
├── paper/                     ← Sensors manuscript + supplement
│   ├── main.tex
│   ├── supplement.tex
│   ├── refs.bib
│   ├── figs/                       camera-ready figures
│   ├── auto_stats.{md,json}        auto-filled §3 stats
│   ├── felt_vep_table.{md,json}
│   ├── sensitivity_abc.{md,json}
│   ├── README.md
│   └── SUBMISSION_CHECKLIST.md
│
├── data/                      ← gitignored raw recordings; see README_DATA.md
├── Gel TCRE/                  ← gitignored gel cohort
├── output/                    ← gitignored auto-generated figures + reports
└── archive/                   ← old reports + abandoned notebooks (not on the path)
```

The `data/` and `Gel TCRE/` directories are not distributed with the
repository — they are access-controlled under URI's IRB protocol. See
[`data/README_DATA.md`](data/README_DATA.md) to request access.

---

## Quickstart

```bash
git clone <repo-url> Tripolar_EEG
cd Tripolar_EEG
python -m venv venv && source venv/bin/activate
pip install -e .
```

Once raw recordings are in `data/` and `Gel TCRE/` (request access — see
above), reproduce the manuscript figures and tables:

```bash
python scripts/build_paper_stats.py
python scripts/build_felt_vep_figure.py
python scripts/build_sensitivity_figure.py
```

Build the paper PDF:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && pdflatex supplement
```

The default build mode (`\previewtrue` in `paper/main.tex`) compiles
without the official MDPI Sensors template. To produce the
submission-ready format, download the template from
<https://www.mdpi.com/authors/latex>, copy the `Definitions/` folder into
`paper/`, and flip `\previewtrue` to `\previewfalse` in both `main.tex`
and `supplement.tex`.

---

## Key design choices

- **Subject is the unit of analysis.** Per-channel measurements are
  averaged within subject before group statistics.
- **Cross-setup endpoints are scale-invariant.** Alpha SNR (a band ratio,
  in dB), alpha reactivity (closed/open ratio), and spectrogram SSIM
  (a normalised image-similarity index) remain valid under arbitrary
  per-channel amplitude scaling.
- **Minimal preprocessing.** The primary pipeline (variant B) is
  60-Hz notch + zero-phase FIR 0.05–55 Hz bandpass, no wavelet, no
  z-score, no ICA, no automated bad-channel interpolation. Adaptive
  cleaners would behave differently across constructions and confound
  the comparison.
- **Three pipeline variants.** Variant A adds Daubechies-8 wavelet
  denoising and per-channel z-score (the original gel-TCRE pipeline);
  variant C is notch-only. Conclusions hold across A/B/C — see
  supplement §S1.
- **Objective QC.** Subjects pass if they have ≥2 eyes-open and ≥2
  eyes-closed epochs, mean tEEG alpha SNR ≥ −1 dB, and mean tEEG
  reactivity ≥ 1.0. Per-channel ADC clipping >5 % drops only the
  affected channels for that subject. Exclusions are reported
  transparently.

---

## Authors

- **Shayan Khodabakhsh** (lead, analysis pipeline) — skhodabakhsh@uri.edu
- **Behtom Adeli** (gel-cohort recordings, gel-TCRE pipeline)
- **Maryam Norouzi** (felt-cohort recordings, electrode housing and 3D printing)
- **Walter G. Besio** (PI, supervision)

Additional acknowledgements are listed in `paper/main.tex`.

## Citation

If you use this software or the accompanying paper, please cite as
described in [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
