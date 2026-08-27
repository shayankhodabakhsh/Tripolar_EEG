# QC Rules, Thresholds, and Repository Audit

## Authoritative project conventions

The audit used the frozen analysis in `Felt_TCRE_Manuscript_Rebuild_2026/scripts/manuscript_analysis.py`, the linked continuous BrainVision headers/markers under `Data/`, and the existing identity/deduplication audit.

| Item | Authoritative finding |
|---|---|
| Checkerboard code | BrainVision `Stimulus, S  7`; MNE preserves it as annotation `Stimulus/S  7` and numeric event ID `7`. |
| Checkerboard rate | Session-specific marker-derived median: **1.618–1.789 reversals/s**. The files do not support imposing 1.93 reversals/s. |
| Blocks and counts | 30–80 S7 reversals in 2–4 blocks among protocol-complete sessions; a >5-s gap defines a block. |
| Epoch | −100 ms through +400 ms, end-exclusive; 500 samples at 1000 Hz; −100 to 0 ms baseline. |
| Authoritative backend | `mne.io.read_raw_brainvision()` supplies samples, channel names, sampling metadata, original annotations, numeric events, and measurement metadata. |
| Review/analysis copy | MNE fourth-order Butterworth IIR, zero-phase forward/backward 0.5–30 Hz. Raw remains unchanged. |
| Rest segments | Eyes-open/eyes-closed comment marker through next condition marker or 30 s, whichever occurs first. |
| Verified 11-channel map | Sites 1–4: Ch1/2, Ch3/4, Ch5/6, Ch7/8 = paired tEEG/eEEG; Ch9/10 = reference paste TCRE pair; Ch11 = reference disc eEEG. |
| Unresolved map | Three 9-channel recordings have numeric headers without an independently documented derivation map. They remain integrity-only and are not pooled into mapped endpoints. |
| Units | Multiplexed `INT_16`, 1000 Hz, 0.1 µV/count in every in-scope header. Exact rails are −3276.8 and +3276.7 µV. No `/187` rescaling applies to this felt dataset; that convention belongs to the separate Gel setup. |
| Identity | Filename tokens are not treated as verified participants. Repeated-folder/session independence remains unresolved unless a participant crosswalk is provided. |

The 1.93-rps discrepancy is resolved in favor of the observed marker stream. Several checkerboard program variants exist elsewhere in the repository, but no source revision is linked to an individual 2026 felt acquisition. The recorded S7 positions are therefore the only session-specific timing evidence.

## Existing automated rejection logic

The manuscript script’s historical automated logic was:

1. Extract raw and filtered −100 to +400-ms epochs.
2. Mark an event-channel epoch unacceptable if any raw sample equals the integer ADC minimum/maximum, if any filtered absolute sample reaches 90% of the inferred ADC span (2949.12 µV at 0.1 µV/count), or if its filtered peak-to-peak amplitude exceeds the channel/session median plus `6 × 1.4826 × MAD`.
3. Mark a continuous channel non-usable if data are nonfinite/constant, continuous exact-rail occupancy exceeds 5%, fewer than 20 events remain, or accepted-event yield is below 70%.
4. Use the intersection of tEEG/eEEG automated masks for paired morphology.

Those criteria generated prior results. The GUI displays independent deterministic findings and the original automated status, but it never converts either into a human decision.

## Review of the 5% clipping rule

The 5% continuous exact-rail criterion is reproducible but is not technically or scientifically sufficient as a sole rule. At a 400-second duration it permits roughly 20 seconds of irrecoverably saturated data; conversely, a brief rail plateau centered on a critical event could matter despite occupying far below 5% of the recording. No equipment manual, prospective validation, or literature source was found in the repository that justifies 5% for this amplifier/paradigm.

The GUI therefore retains 5% as a clearly labeled **legacy compatibility FAIL threshold**, while adding:

- WARNING for at least three exact rail samples or ≥0.01% rail occupancy; and
- FAIL for any continuous rail plateau ≥20 ms.

These additional values are engineering starting points and must be validated. Exact ADC endpoints themselves are equipment-derived, not empirically tuned to improve an outcome.

## Configurable technical rules

All values live in `gui/qc_thresholds.json`. Changing them changes findings, never historical decisions. A threshold revision should be versioned, justified, and followed by sensitivity analysis.

| Rule | Default measurement and status | Basis and limitation |
|---|---|---|
| ADC rail saturation | Exact raw integer min/max. WARNING: ≥3 samples or ≥0.01%; FAIL: ≥5% or ≥20-ms plateau. | ADC endpoints follow directly from INT16 × 0.1 µV. Fractions/plateau are provisional; 5% is legacy. |
| Flatline | Non-overlapping 2-s window SD ≤0.5 µV or zero range. WARNING if present; FAIL if ≥5% samples. | Engineering disconnection screen. Must be validated against reviewed project intervals. |
| 60-Hz line intrusion | Raw 60±0.5-Hz power relative to mean 57–59 and 61–63-Hz 1-Hz-equivalent floor. WARNING ≥6 dB; FAIL ≥12 dB. Harmonics through Nyquist also reported. | 6/12 dB are interpretable 4×/16× power ratios, not biological limits. Project distribution is extremely high, so expert validation is essential. |
| Abrupt step/pop | First difference ≥max(100 µV, 12 robust SD); FAIL at ≥10/min, otherwise WARNING. | Provisional recorded-output engineering rule; paired derivations have different gains. |
| Slow drift | Raw 0.05–0.5-Hz / 1–30-Hz power. WARNING ≥0.5; FAIL ≥1.0. | Provisional relative-power rule; avoids raw amplitude comparison. |
| Muscle/high frequency | Raw 30–100-Hz / 1–30-Hz power. WARNING ≥0.5; FAIL ≥1.0. | Provisional; mains harmonics can inflate it and no EMG reference is available. |
| Large amplitude | Absolute recorded output. WARNING ≥1000 µV; FAIL ≥3000 µV. | Provisional and derivation-dependent; always inspect ADC rails and fixed-scale trace. |
| Marker timing | Within-block IEI robust outlier beyond max(25% median IEI, 6 robust SD), or interval ≤25% median. | Project marker distribution; >5-s gaps are expected block boundaries. |
| Epoch robust p-p | Median + 6 × 1.4826 × MAD, displayed as a suggestion. | Preserves prior robust logic but does not reject. |
| Possible ocular artifact | Pre/post mean shift ≥250 µV. | Suggestion only; no EOG channel makes source attribution unresolved. |

## Empirical project distribution before manual review

The preparation run evaluated 96 mapped channels (12 records × 8 paired outputs), producing 672 rule rows. This is a diagnostic baseline, not validation and not a reason to move thresholds until desired results appear.

- Exact-rail occupancy: median 0.02%, 75th percentile 0.64%, 95th percentile 6.37%, maximum 17.53%.
- Raw 60-Hz intrusion: median 27.38 dB, range 7.47–45.20 dB; 94/96 channels exceeded the provisional 12-dB FAIL level.
- Flatline fraction: median 0%; maximum 6.64%.
- Slow-drift ratio: median 0.322; 95th percentile 1.535.
- 30–100/1–30-Hz ratio: median 1.625; the extreme tail is likely influenced by raw line/harmonic energy and must not be called EMG without plot review.

The very high line-noise and high-frequency flag rates demonstrate why the GUI presents rule-specific evidence instead of collapsing metrics into an unexplained composite score.

## VEP peak rules (post-freeze only)

Neutral windows reproduce the current manuscript code: first negative 55–90 ms, principal positive 90–140 ms, subsequent negative 130–215 ms. A candidate is flagged for review when it is on a window boundary, bootstrap time CI exceeds 30 ms, bootstrap detection frequency is below 70%, leave-one-out peak change exceeds 25%, odd/even or first/second-half reliability is poor, or filter or accepted-epoch-subset sensitivity shifts timing by more than 15 ms. These are confidence flags, not clinical reference limits.

The repository lacks check width, display calibration, field size, viewing distance, eye tested, refraction, and other facts required for a standard clinical component interpretation. The [2025 ISCEV VEP standard](https://pmc.ncbi.nlm.nih.gov/articles/PMC12436483/) specifies a standard reversal rate of 2.0±0.2 reversals/s and emphasizes stimulus/reporting details and waveform uncertainty. Although the observed project rate overlaps the lower edge of that rate range, missing acquisition details prevent automatic N75/P100/N145 labels.

## Method references and their role

- The [PREP pipeline paper](https://pubmed.ncbi.nlm.nih.gov/26150785/) supports explicit early-stage line-noise and bad-channel handling, but this GUI does not import PREP’s adaptive referencing/interpolation because that would alter the project’s minimal pipeline.
- [FASTER](https://pubmed.ncbi.nlm.nih.gov/20654646/) is precedent for metric-based artifact screening. Its fully automated rejection/ICA workflow is not used here.
- [MNE-Python’s Raw documentation](https://mne.tools/stable/generated/mne.io.Raw.html) documents maintained EEG containers and zero/linear-phase filtering behavior. MNE is the authoritative processing backend and stores EEG in volts; the exact integer stream is consulted only to identify literal ADC endpoints.
- The [ISCEV VEP standard](https://pmc.ncbi.nlm.nih.gov/articles/PMC12436483/) informs cautious peak reporting and required stimulus metadata. This project is not represented as a standard clinical VEP protocol.

Thresholds must be validated against hand-labeled intervals, synthetic injection tests, sensitivity analysis, and an independent experienced EEG reviewer. They must not be tuned to raise VEP SNR, alpha reactivity, tEEG/eEEG agreement, or any desired manuscript result.
