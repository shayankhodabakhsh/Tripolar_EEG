# Tripolar EEG Analysis — Felt vs Gel vs Paste TCRE Comparison

Multi-electrode-type comparison of **tripolar concentric ring electrodes (TCRE)** using three
electrolyte/construction variants — **Felt**, **Gel**, and **Paste** — evaluated on
alpha-band (8–13 Hz) detection, visual evoked potentials (VEP), and spectral fidelity.

---

## Background

Tripolar Concentric Ring Electrodes (TCREs) have 3 concentric rings and output two
derivations from a single placement:

- **eEEG (conventional)** — outer ring vs. remote mastoid, identical to standard disc EEG
- **tEEG (Laplacian)** — surface Laplacian computed on-board from all 3 rings, acting as a
  spatial high-pass filter that suppresses volume conduction and sharpens local source detection

This project compares **three electrolyte/construction types** in a unified analysis pipeline:

| Type | Electrolyte | Subjects | Channels |
|------|-------------|----------|----------|
| Felt TCRE | Saltwater-soaked felt pad | All loadable 11-ch subjects (Long + Short unless filtered) | 11 ch |
| Gel TCRE | Conductive gel (new design with 3D housing) | Standard-protocol Gel subjects discovered recursively | 7 ch |
| Paste TCRE | Conductive paste | Present in **both** recording setups (bridge electrode) | — |

The **Paste TCRE** is recorded in both setups, acting as a bridge reference to validate
cross-recording comparability.

---

## Recent Changes (Apr 2026)

- Added Gel-TCRE integration into the same pipeline used for Felt/Paste analyses using `ElectrodeConfig`.
- Added objective QC gates in `load_all_subjects`: minimum open/close epochs, minimum alpha SNR, minimum reactivity, optional VEP gate.
- Added per-subject QC reporting (`print_qc_report`) with duration, epoch counts, SNR, and clipping.
- Switched group comparisons to per-subject aggregation (subject is the independent statistical unit).
- Restricted cross-setup inferential statistics to scale-invariant metrics.
- Replaced `np.trapz` with `np.trapezoid` for NumPy 2.0 compatibility.
- Reworked PSD plots to avoid ribbon artifacts: thin individual traces + thick group median with hard y-floor.
- Reworked alpha SNR/reactivity plots to boxplots + individual points (reactivity axis cap with outlier annotation).
- Added spectrogram SSIM into three-way comparisons as a scale-invariant endpoint (`comparison_ssim.png` + statistical tests).
- Updated `.gitignore` to exclude data-heavy files/folders, binary EEG data, outputs, venv, and LaTeX artifacts.

---

## Latest Implemented Updates (This Iteration)

- **SSIM in group pipeline:** `extract_type_metrics(...)` now carries a per-subject, per-type SSIM value derived from each subject's `ssim_values` and TCRE pair map.
- **SSIM in stats:** `compare_electrode_types(...)` now includes `ssim` in the same statistical table as alpha SNR and alpha reactivity.
- **SSIM figure output:** `plot_three_way_comparison(...)` now writes `comparison_ssim.png` (boxplot + individual subject points, y-range 0-1).
- **Cross-setup metric policy:** cross-setup inference is explicitly limited to scale-invariant endpoints (alpha SNR, alpha reactivity, SSIM).
- **QC behavior now documented:** objective subject filtering is based on epoch count + signal quality gates before subjects enter comparison stats.
- **Current local data note:** the pipeline only analyzes subjects physically present under the configured `FELT_DATA_DIR` / `GEL_DATA_DIR` paths.

---

## Repository Structure

```
Tripolar_EEG/
│
├── README.md                              ← This file
├── requirements.txt                       ← Python dependencies
├── .gitignore                             ← Excludes data/, output/, Gel TCRE/, venv/
├── TCRE_Gel.pdf                           ← Gel TCRE study reference paper
│
├── src/                                   ← All analysis code
│   ├── eeg_analysis.py                    ← Core engine (load, filter, analyze, plot)
│   ├── comparison_analysis.py             ← 3-way comparison framework (NEW)
│   │
│   ├── single_subject_analysis_v2.ipynb   ← Run a single Felt TCRE subject
│   ├── group_analysis.ipynb               ← Group-level Felt TCRE statistics
│   ├── three_way_comparison.ipynb         ← Felt vs Gel vs Paste comparison (NEW)
│   │
│   └── report_v2.py                       ← CLI batch QC report for all subjects
│
├── data/                                  ← [gitignored] Felt TCRE BrainVision files
│   ├── TU2_long_felt_TCRE_4-7-2026.*      ← .eeg / .vhdr / .vmrk
│   ├── LS2-long-felt-TCRE-4-7-2026.*
│   └── (+ 13 other subjects)
│
├── Gel TCRE/                              ← [gitignored] Gel TCRE recordings + original code
│   ├── 10-20-2024/                        ← Date-based session folders
│   ├── 11-8-2024/
│   ├── 11-15-2024/
│   ├── ... (through 11-26-2024)
│   ├── Untitled-1.py                      ← Original Gel TCRE processing script
│   ├── fifthTCREGelProcess.py             ← Alternative Gel TCRE pipeline
│   └── checkerboard/                      ← MATLAB VEP stimulation scripts
│
├── output/                                ← [gitignored] Generated figures and reports
│   └── three_way_comparison/              ← Output from three_way_comparison.ipynb
│
├── my_report/                             ← LaTeX reports (PDF + source)
└── venv/                                  ← [gitignored] Python environment
```

---

## Quickstart

### 1. Activate the environment

```bash
cd /home/shayankh1996/Desktop/Tripolar_EEG/Tripolar_EEG
source venv/bin/activate
```

### 2. Single-subject analysis (Felt TCRE)

Open and run `src/single_subject_analysis_v2.ipynb`.
Set `DATA_DIR` and `SUBJECT` at the top of the notebook:

```python
DATA_DIR = "../data"
SUBJECT  = "TU2"   # or "LS2", "AH felt TCRE", etc.
```

All figures are saved to `output/<subject_basename>/single_subject/`.

### 3. Three-way comparison (Felt vs Gel vs Paste)

Open and run `src/three_way_comparison.ipynb`.
The defaults are pre-configured:

```python
FELT_DATA_DIR = "../data"
GEL_DATA_DIR  = "../Gel TCRE"

# None => include all discovered Felt recordings
FELT_SUBJECTS = None

# manual exclusion lists (substring match)
EXCLUDE_FELT = ["LuciTest"]   # add bad subjects here
EXCLUDE_GEL  = []
```

The loader also performs hard quality gates:
- Skip subjects with unknown channel layout (e.g. 9-ch prototypes without a config)
- Skip subjects with fewer than 2 eyes-open or 2 eyes-closed epochs
- Print a QC report (duration, epoch counts, mean alpha SNR, clipping %)

Figures are saved to `output/three_way_comparison/`.

### 4. Group-level analysis (Felt TCRE only)

Open and run `src/group_analysis.ipynb`. This loads all subjects in `data/`,
runs the full pipeline, and produces paired statistical tests.

### 5. Batch QC report (CLI)

```bash
cd /home/shayankh1996/Desktop/Tripolar_EEG/Tripolar_EEG
source venv/bin/activate
python src/report_v2.py --data-dir data/ --out output/report_v2.txt
```

---

## Channel Maps

### Felt TCRE (11 channels)

| Ch | Label | Type | Electrolyte |
|----|-------|------|-------------|
| 1 | Felt TCRE #1 (tEEG) | Laplacian | Felt pad |
| 2 | Felt TCRE #1 (eEEG) | Conventional | Felt pad |
| 3 | Felt TCRE #2 (tEEG) | Laplacian | Felt pad |
| 4 | Felt TCRE #2 (eEEG) | Conventional | Felt pad |
| 5 | Felt TCRE #3 (tEEG) | Laplacian | Felt pad |
| 6 | Felt TCRE #3 (eEEG) | Conventional | Felt pad |
| 7 | Felt TCRE #4 (tEEG) | Laplacian | Felt pad |
| 8 | Felt TCRE #4 (eEEG) | Conventional | Felt pad |
| 9 | Paste TCRE #5 (tEEG) | Laplacian | Paste |
| 10 | Paste TCRE #5 (eEEG) | Conventional | Paste |
| 11 | Paste Disc | Conventional disc | Paste |

### Gel TCRE (7 channels)

| Ch | Label | Type | Location |
|----|-------|------|----------|
| 1 | Gel TCRE O1 (tEEG) | Laplacian | O1 |
| 2 | Gel TCRE O1 (eEEG) | Conventional | O1 |
| 3 | Gel TCRE O2 (tEEG) | Laplacian | O2 |
| 4 | Gel TCRE O2 (eEEG) | Conventional | O2 |
| 5 | Paste TCRE Pz (tEEG) | Laplacian | Pz |
| 6 | Paste TCRE Pz (eEEG) | Conventional | Pz |
| 7 | Normal EEG Pz (disc) | Conventional disc | Pz |

> Channels 1, 3, 5 are divided by 187 on load to normalize hardware tEEG amplitude.

---

## Signal Processing Pipeline

All recordings share these parameters:

| Parameter | Value |
|-----------|-------|
| Sampling rate | 1000 Hz |
| Resolution | 0.1 µV / bit (INT16) |
| Format | BrainVision (.eeg + .vhdr + .vmrk) |
| Preprocessing | 60 Hz notch filter (Q=30) |
| Welch PSD | nperseg = 4096 (0.244 Hz resolution) |
| Alpha band | 8–13 Hz |
| SSIM spectrograms | nperseg = 2048, max_freq = 45 Hz |

### Analysis steps

```
Raw INT16 binary
       │
       ▼
  Scale → µV  (+  /187 for Gel tEEG)
       │
       ▼
  Parse .vmrk  →  stim blocks, open/close epochs
       │
       ▼
  60 Hz notch
       │
       ├──► Welch PSD (4096)  →  Alpha SNR (dB)
       │
       ├──► Open/close epochs →  Alpha power open & closed
       │                          Alpha reactivity = closed/open
       │
       ├──► Bandpass 8-13 Hz  →  Hilbert envelope
       │                          Correlation with disc channel
       │
       ├──► Spectrogram SSIM  →  tEEG vs eEEG similarity per TCRE pair
       │
       └──► Pre-averaged .avg →  VEP peak-to-peak (if available)
```

---

## Code Architecture

### `src/eeg_analysis.py` — Core engine

```
ElectrodeConfig (dataclass)
├── FELT_TCRE_CONFIG   — 11-ch preset
└── GEL_TCRE_CONFIG    — 7-ch preset, with /187 scaling on tEEG channels

detect_config_from_vhdr(vhdr_path)  — auto-selects config from header

discover_subjects(data_dir, recursive, standard_protocol_only)
load_subject(data_dir, subject_name, subject_info, config)
    └── returns subject dict with "config" key attached

analyze_subject(subject)
    └── reads config from subject["config"], works for any layout

plot_subject_summary(subject, results, ...)
plot_subject_full(subject, results, ...)
    └── all plots auto-sized to actual channel count
```

Backward-compatible constants (`N_CHANNELS`, `CH_LABELS`, `TCRE_PAIRS`, etc.) are
aliases that still point to the Felt TCRE values — existing notebooks need no changes.

### `src/comparison_analysis.py` — 3-way comparison

```
load_all_subjects(
    felt_dir, gel_dir, felt_names,
    exclude_felt, exclude_gel,
    min_open_epochs, min_close_epochs
)
    └── loads + analyzes both datasets with automatic exclusions + QC gates

print_qc_report(subjects_results)
    └── prints per-subject QC table (PASS/WARN/FAIL)

extract_type_metrics(subjects_results, per_subject=True)
    └── aggregates by abstract type: FELT_TEEG, GEL_TEEG, PASTE_TEEG,
        FELT_EEEG, GEL_EEEG, PASTE_EEEG, DISC
    └── default is per-subject averaging (statistically correct unit = subject)

extract_psd_by_type(subjects_results, normalize=True)
extract_open_closed_psd_by_type(subjects_results, normalize=True)
    └── PSDs are normalized to each channel's own broadband power (1–30 Hz)
       so Felt/Gel are comparable despite Gel /187 amplitude scaling

compare_electrode_types(felt_subjects, gel_subjects)
    └── Mann-Whitney U tests across all type pairs
        includes Paste TCRE bridge validation

plot_three_way_comparison(felt_subjects, gel_subjects, save_dir)
    └── generates comparison figures:
        comparison_alpha_snr.png
        comparison_alpha_reactivity.png
        comparison_ssim.png
        comparison_psd_teeg.png
        comparison_open_vs_closed_psd.png
        comparison_paste_bridge.png
        (+ statistical summary printed to console)
```

---

## Output Files

Running `src/three_way_comparison.ipynb` produces these figures in
`output/three_way_comparison/`:

| File | Contents |
|------|----------|
| `comparison_alpha_snr.png` | Boxplot + individual points: Alpha SNR (dB) by electrode type (n = subjects) |
| `comparison_alpha_reactivity.png` | Boxplot + individual points: Closed/Open ratio by electrode type (n = subjects) |
| `comparison_ssim.png` | Boxplot + individual points: spectrogram SSIM by electrode type (n = subjects) |
| `comparison_psd_teeg.png` | Normalized PSD overlay (tEEG + disc), 1–30 Hz |
| `comparison_open_vs_closed_psd.png` | Normalized open vs closed PSD per type |
| `comparison_paste_bridge.png` | Boxplots: Paste TCRE metrics across both setups |
| `gel_individual/<basename>/` | Per-subject summary dashboards for Gel subjects |

Running `src/single_subject_analysis_v2.ipynb` produces figures in
`output/<subject_basename>/single_subject/`:

| File | Contents |
|------|----------|
| `*_raw_traces.png` | All channels, full recording |
| `*_psd_alpha_1_30hz.png` | Per-channel PSD with alpha highlight |
| `*_spectrogram_ch??.png` | Per-channel time-frequency spectrogram |
| `*_band_decomp_ch??.png` | Per-channel delta/theta/alpha/beta/gamma |
| `*_alpha_envelope.png` | Alpha envelope with eyes open/close events |
| `*_open_vs_closed.png` | Eyes open vs closed PSD comparison |
| `*_alpha_reactivity.png` | Alpha reactivity bar chart |
| `*_vep_comparison.png` | VEP waveform panels |
| `*_disc_correlation.png` | Alpha envelope correlation with disc |
| `*_adc_clipping.png` | ADC saturation report |
| `*_ssim_pair*.png` | tEEG vs eEEG spectrogram comparison |
| `*_ssim_summary.png` | SSIM bar chart for all TCRE pairs |
| `*_summary.png` | 4-panel dashboard |

---

## Key Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Alpha SNR** | `10 * log10(alpha_power / neighbor_bands)` in dB | Higher = cleaner alpha relative to background |
| **Alpha Reactivity** | `alpha_closed / alpha_open` | > 1 = Berger effect detected |
| **Disc Correlation** | Pearson r of alpha envelope vs disc channel | Tracks same neural events as gold standard |
| **VEP Peak-to-Peak** | max − min of averaged evoked potential | Stimulus-locked response amplitude |
| **Spectrogram SSIM** | Structural similarity between tEEG and eEEG spectrograms per TCRE pair | Scale-invariant; supports cross-setup comparison without amplitude calibration |

---

## Interpreting The Current Figures

Your comparison output contains **electrode classes**, not three separate Felt/Gel cohorts:

- Felt tEEG / Felt eEEG
- Gel tEEG / Gel eEEG
- Paste tEEG / Paste eEEG (bridge in both setups)
- Disc EEG (reference)

So when you see multiple Felt/Gel bars, they are different **signal derivations** (tEEG vs eEEG), not duplicate groups.

### Long Felt vs Short Felt

- `FELT_SUBJECTS = None` means all discoverable Felt recordings are considered.
- If both Long Felt and Short Felt are present in 11-channel format, both are included unless manually excluded.
- 9-channel prototypes (e.g., Gab/LS1/TU) are currently skipped automatically because no 9-channel map is defined.

If you want a pure Long-Felt analysis for publication, set:

```python
FELT_SUBJECTS = ["TU2", "LS2", "HS Long Felt TCRE"]
```

or explicitly exclude all short sessions via `EXCLUDE_FELT`.

### What each figure is saying

- `comparison_alpha_snr.png`: all classes show positive SNR (alpha is detectable); compare central tendency with caution because variance is high.
- `comparison_alpha_reactivity.png`: Berger effect (>1) is present overall; some groups have large spread indicating subject/session heterogeneity.
- `comparison_ssim.png`: summarizes tEEG-vs-eEEG spectrogram structural similarity by electrode class; this metric is dimensionless and directly comparable across setups.
- `comparison_psd_teeg.png`: normalized PSD curves are now shape-comparable across Felt/Gel despite gain differences.
- `comparison_open_vs_closed_psd.png`: closed-eye alpha bump (8–13 Hz) should exceed open-eye; this validates physiological behavior.
- `comparison_paste_bridge.png`: Paste Felt vs Paste Gel similarity is the key cross-setup sanity check.

---

## Are These Results Publishable?

Short answer: **potentially publishable as a pilot / methods-validation result**, but not yet as a definitive performance claim.

Current strengths:
- Unified pipeline and harmonized preprocessing
- Cross-setup bridge electrode (Paste) for comparability checks
- Per-subject statistics (correct unit of analysis)
- QC/exclusion workflow documented and reproducible

Current limitations to state explicitly:
- Mixed Felt populations (Long + Short) unless filtered
- Small and imbalanced sample sizes in some subsets
- High variance / outliers in Gel and Paste reactivity metrics
- Some sessions removed by quality filters (must report exclusion counts and reasons)

Minimum checklist before submission:
1. Lock a cohort definition (Long-only vs Long+Short) before final stats.
2. Freeze exclusion rules and report them transparently.
3. Re-run all figures/tables with final cohort.
4. Include robustness/sensitivity analysis (with and without borderline subjects).
5. Frame conclusions as exploratory if n remains limited.

---

## Dependencies

```
numpy
scipy
matplotlib
scikit-image     # for SSIM computation
specparam==2.0.0rc6  # optional: spectral parameterization
```

Install:

```bash
source venv/bin/activate
pip install numpy scipy matplotlib scikit-image
pip install specparam==2.0.0rc6  # optional
```

---

## What Was Changed (Integration Summary)

### Problem
The Gel TCRE project used a separate MNE-based pipeline with different channel counts (7 vs 11),
different amplitude scaling, and no common comparison framework.

### Solution: Unified ElectrodeConfig System

`eeg_analysis.py` was extended with an `ElectrodeConfig` dataclass:

```python
# Auto-detects which config to use from the .vhdr header
subj = load_subject("data/", "TU2")              # → FELT_TCRE_CONFIG (11 ch)
subj = load_subject("Gel TCRE/", subject_info=si) # → GEL_TCRE_CONFIG (7 ch)
```

Key compatibility facts confirmed:

| | Felt TCRE | Gel TCRE |
|--|-----------|---------|
| Sampling rate | 1000 Hz | 1000 Hz |
| Resolution | 0.1 µV/bit | 0.1 µV/bit |
| File format | BrainVision INT16 | BrainVision INT16 |
| Paradigm | Checkerboard + eyes open/close | Checkerboard + eyes open/close |
| Paste TCRE present | Yes (Ch9–11) | Yes (Ch5–7) — bridge electrode |
| tEEG scaling | Hardware output | Hardware output / 187 |

### No Breaking Changes
All existing notebooks (`single_subject_analysis_v2.ipynb`, `group_analysis.ipynb`) run
unchanged. The `ElectrodeConfig` system is additive — backward-compatible constants
`N_CHANNELS`, `CH_LABELS`, `TCRE_PAIRS` still work as before.
