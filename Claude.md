# Tripolar EEG — Comparison Paper Project Context

## What we're building
A Sensors MDPI manuscript comparing three TCRE electrolyte/construction
types — Paste, Gel, and saline-soaked Felt — using the existing pipeline
in src/. This is a METHODS / PILOT paper, not a powered RCT.

## Cohort decisions (locked in by Dr. Besio)
- Long Felt: ONLY TU2 and LS2. Do NOT include short Felt or HS/RK/SG.
- Gel: all subjects discoverable under "Gel TCRE/" that pass standard QC.
- Paste TCRE is the cross-study bridge — it appears in BOTH setups.

## VEP handling (UNDER REVIEW with Dr. Besio)
- Original Gel TCRE pipeline (Adeli/Norouzi) recovers clean VEPs but uses:
  - Different tEEG computation: tEEG = 16·in − out
  - Wavelet denoising (db8, level 3)
  - Per-epoch z-score normalization (twice)
  - Different sampling/filter handling that triggers MNE warnings
- These pipeline differences make Gel VEP amplitudes NOT directly comparable
  to Felt VEP amplitudes from the project's main pipeline.
- Decision pending: harmonize by re-processing Gel raw data through the
  unified pipeline, or report VEPs separately per dataset.
- Until decision is made, do NOT mix Gel and Felt VEP data in any
  comparison figure or statistic.

## Preprocessing decisions (locked)
Primary analysis pipeline is intentionally minimal to avoid adaptive
cleaners that could behave differently across media:

- 60 Hz notch filter (IIR, Q=30) — keep
- Zero-phase FIR bandpass 0.05–55 Hz (Hamming) — keep
- Wavelet denoising (Daubechies-8, level 3) — DROPPED for primary analysis
- Per-channel z-score normalization — only AFTER metric computation,
  never before SNR or PSD calculations
- ICA, ASR, automated bad-channel interpolation — NOT used. Adaptive
  cleaners would treat clean Paste vs noisy Felt differently and
  confound the media comparison.
- Re-referencing — NOT used. Mastoid reference for disc; TCRE
  Laplacian is internal.

Sensitivity analyses (report in supplement, not main results):
- Variant A: notch + bandpass + wavelet + z-score (original pipeline)
- Variant B: notch + bandpass only (PRIMARY)
- Variant C: notch only

Conclusions must be checked across all three variants before submission.

## QC and exclusion rules
- Per-channel ADC clipping > 5% → channel EXCLUDED from group stats
  for that subject (subject stays in for other channels).
- Subjects with fewer than 2 eyes-open or 2 eyes-closed epochs → excluded.
- Report exclusion counts and reasons transparently in Methods.

## Critical methodological constraints
- Cross-setup comparisons MUST use scale-invariant metrics only:
  alpha SNR (dB), alpha reactivity (closed/open ratio), spectrogram SSIM.
  Raw amplitudes are NOT comparable (Gel has /187 hardware scaling).
- VEPs from the Gel dataset are EXCLUDED from analysis — the amplifier
  pipeline used during Gel collection produced unreliable evoked responses.
  VEP results are reported only for Felt + Paste in the Felt setup,
  using TU2 and LS2.
- Subject is the unit of statistical analysis, not channel or epoch.
- Group statistics: Mann-Whitney U for between-media; bootstrap BCa
  confidence intervals (1000 resamples) on medians for reporting.
  No parametric tests on n=2 Long Felt.

## Files of record
- src/eeg_analysis.py        — core engine, ElectrodeConfig system
- src/comparison_analysis.py — three-way comparison pipeline
- src/three_way_comparison.ipynb — main figure-generation notebook
- TCRE_Gel.pdf               — companion Gel TCRE paper for context/refs

## Manuscript style
- Sensors MDPI LaTeX template in paper_sensors/
- Apply the humanizer skill when drafting any prose.
- Reference style: numeric brackets [1], BibTeX in refs.bib.
- Frame all claims as descriptive/pilot given small n; no causal language.
- Keep sentences direct; this is a methods paper, not a theoretical piece.

## What Claude should NEVER do
- Do not include short Felt or HS/RK/SG subjects in any analysis cell.
- Do not compare raw VEP amplitudes between Gel and Felt setups.
- Do not apply ICA, ASR, or adaptive bad-channel interpolation.
- Do not interpolate clipped samples — exclude affected channels instead.
- Do not modify files in data/ or "Gel TCRE/" — read-only.
- Do not commit anything in output/ or paper_sensors/figures/raw/.