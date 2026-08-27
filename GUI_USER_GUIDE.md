# EEG/VEP Quality Review GUI User Guide

## Purpose and safety boundary

This local application supports structured human review; it does not decide what to reject. Raw BrainVision `.eeg`, `.vhdr`, and `.vmrk` files are opened read-only. The public queue uses random review IDs. Source filenames and holder configuration are kept in `gui/private/blinding_key.csv`, which the GUI uses only to locate a file and never displays.

Technical quality, physiological-response detectability, VEP peak confidence, and duplicate/identity status are separate concepts. During the initial technical review, the GUI hides holder configuration, VEP SNR, alpha results, spectral correlations, and averaged VEP morphology. These pages unlock only after a complete review is explicitly frozen.

## Install and start

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m gui.prepare_review
streamlit run gui/app.py
```

`prepare_review` is safe to rerun: it keeps the existing blind, recomputes derived technical metrics, and does not change decisions. Never use `--overwrite-manifest` after review begins; changing the random IDs would invalidate the audit trail. The command refuses to do this when decisions exist.

The repository-local environment created during implementation can be used directly:

```bash
.venv/bin/streamlit run gui/app.py
```

### Jupyter/MNE notebook option

For a notebook-based review using the same MNE backend and CSV audit trail:

```bash
cd /home/tnlab_sk/Desktop/Tripolar_EEG
.venv/bin/python -m ipykernel install --prefix .venv --name tripolar-eeg --display-name "Tripolar EEG (.venv)"
.venv/bin/python -m jupyterlab MNE_QC_REVIEW.ipynb
```

The kernel-registration command writes only under the project’s `.venv` and ensures the notebook uses the environment containing MNE. It is safe to rerun. Open `MNE_QC_REVIEW.ipynb` if Jupyter shows its file browser, verify that the kernel is **Tripolar EEG (.venv)**, then run the first code cell. The notebook displays guided `ipywidgets` controls for the blinded record, site, signal, event, fixed scales, evidence, and decisions. Its **Native MNE browser** button opens MNE's scroll/zoom viewer when the local display backend supports it. The notebook and Streamlit append to the same CSVs, so use one reviewer name consistently and do not review the same item concurrently in both interfaces.

The notebook intentionally contains no VEP-average, alpha, configuration, or correlation view during technical review. It cannot freeze or apply a mask. Complete/freeze the audited review through the existing export workflow, and run `python -m gui.post_review` only afterward.

The preparation step creates:

- `gui/review_data/blinded_review_manifest.csv`: public random IDs and non-identifying acquisition facts.
- `gui/review_data/data_integrity_manifest.csv`: hashes and integrity facts indexed by random ID.
- `gui/private/blinding_key.csv`: private backend lookup key, mode `0600`.
- `gui/review_output/channel_quality_metrics.csv`: automated technical findings, not decisions.
- Empty, schema-valid decision and review exports.

## Open and navigate a session

1. Enter your reviewer name in the sidebar. It is required for every saved decision.
2. Select a random review ID and site 1–4.
3. Begin at **Review overview**. It shows acquisition counts and review progress but no outcome-sensitive result.
4. Use **Data integrity** to check hashes, sample count, duration, sampling rate, channel headers, montage status, marker count, S7 timing, and identity/configuration status. The configuration value remains blinded.
5. Use **Raw-signal quality** to inspect synchronized raw and zero-phase 0.5–30-Hz copies. Purple vertical lines are checkerboard S7 reversals. Change the time and fixed amplitude scales as needed.
6. Use **Event-locked epochs** to inspect every tEEG and eEEG epoch from −100 to +400 ms at a fixed scale.
7. Use **Eyes-open / eyes-closed** to inspect the marker-defined resting segments. Each is capped at 30 seconds or the next condition marker.

MNE-Python is authoritative for signal samples, channel names, sampling metadata, annotations, numeric events, filtering, epoching, and PSDs. The integrity page shows explicit preservation checks. For MNE’s native scroll/zoom/annotation browser, copy the sidecar command shown on the raw-quality page, for example:

```bash
.venv/bin/python -m gui.mne_viewer --review-id RV-XXXXXXXX --site 1 --start 0 --duration 20
```

The low-level BrainVision reader is used only for SHA-256/linkage facts and exact integer ADC endpoints. Do not save native-browser annotations back into source files; record them through the separate review audit trail.

If an exact or near-duplicate candidate is listed on **Data integrity**, inspect its evidence and append `Accept`, `Reject`, `Uncertain`, or `Needs expert review`. A reason is mandatory except for `Accept`; revisions append new rows and do not overwrite earlier assessments. No current blinded record pair met the hash-, marker-, or metadata-based candidate rules.

## Interpret automated statuses

- **PASS**: no configured important technical problem was detected.
- **WARNING**: the measured value requires human inspection and may still be usable.
- **FAIL**: a predefined technical threshold was exceeded. This is still not a final rejection.
- **UNRESOLVED**: the data or metadata are insufficient for a reliable automated status.

Each finding shows the rule, measured value, threshold, affected channel/interval, recommended supporting plot, and a plain-language explanation. Select an affected interval and press **Jump to affected interval** to center the trace on it.

The main warnings mean:

- **ADC rail saturation**: samples equal the exact integer ADC minimum or maximum. The original amplitude is not recoverable at those samples.
- **Flatline**: a window has extremely low variance, consistent with disconnection, a short, or a stalled channel.
- **60-Hz intrusion**: the raw pre-notch spectrum has a 60-Hz peak above its neighboring frequency floor.
- **Abrupt step**: a large sample-to-sample baseline change, consistent with an electrode pop or sudden contact change.
- **Slow drift**: 0.05–0.5-Hz power is large relative to 1–30-Hz power.
- **Muscle/high frequency**: broad 30–100-Hz power is large relative to 1–30-Hz power. This suggests artifact; it is not a physiological diagnosis.
- **Unusually large amplitude**: recorded-output amplitude exceeds the configured limit. tEEG and eEEG gain differences must be considered.
- **Invalid event**: the within-block inter-event interval is a robust timing outlier or probable duplicate.

Automated epoch suggestions are starting points only. Ocular origin is especially uncertain because these recordings do not contain a dedicated EOG channel.

## Record decisions

### Whole site-session record

On **Raw-signal quality**, assign exactly one:

- `GOOD`
- `GOOD_WITH_BAD_SEGMENTS`
- `BAD_SITE`
- `UNCERTAIN`

A reason is required for every option except `GOOD`. Add notes when the rationale needs context.

### Individual event-locked site epoch

On **Event-locked epochs**, select tEEG or eEEG, inspect the event, then assign `KEEP` or `REJECT`. A rejection requires exactly one reason:

- clipping
- flatline
- electrode pop
- movement/EMG
- ocular artifact
- invalid event
- incomplete epoch

Every save appends a new audit row with review ID, session, site, event number, event time, decision, reason, reviewer, timestamp, notes, original automated status, and revision ID. Correcting a decision appends a revision; it never overwrites history.

## Freeze the review

The **Decisions and export** page shows completion counts. Freezing is disabled until every in-scope site and every tEEG/eEEG site-event has a decision. Type the exact phrase shown by the application and press **Freeze rejection mask**.

Freezing creates a hash-verified `frozen_rejection_mask.csv` and `review_complete.json`. It does not modify raw data and does not run analysis. If any whole-site decision remains `UNCERTAIN`, post-review analysis will stop until an expert records and freezes a resolved review.

## Review VEP peaks after freezing

The unlocked **VEP morphology** page uses the intersection of tEEG and eEEG `KEEP` events for every paired comparison. It displays:

- individual-trial heatmaps;
- the exact 0.5–30-Hz manuscript-processing averages;
- odd/even averages;
- simultaneous tEEG/eEEG traces;
- normalized overlays and differences;
- neutral peak names and prespecified search windows;
- peak amplitude/time, baseline and late-noise prominence;
- bootstrap time intervals and detection frequency;
- split-half reliability;
- search-window-boundary warnings;
- leave-one-out trial dominance;
- filter sensitivity; and
- accepted-epoch subset sensitivity (odd/even and first/second halves).

Do not interpret low tEEG/eEEG waveform similarity as electrode failure. The two outputs use different spatial derivations. N75/P100/N145 labels are intentionally not assigned because check size, viewing geometry, eye tested, display calibration, and other standard clinical details are unavailable.

## Apply the frozen mask and regenerate derived results

Only after review and expert resolution are complete:

```bash
python -m gui.post_review
```

This command refuses to run without a hash-verified frozen mask or while any site is `UNCERTAIN`. It writes only under `gui/review_output/post_review/` and reports site/trial counts before and after review. It uses:

- the common tEEG/eEEG accepted-event intersection for all paired results;
- raw pre-notch/pre-low-pass samples for 60-Hz and harmonic measurements; and
- the zero-phase 0.5–30-Hz branch for VEP and alpha analyses.

It does not alter raw data and labels outputs as not yet approved for the manuscript.

## Export files

The application and post-review command produce:

- `qc_decisions.csv`
- `channel_quality_metrics.csv`
- `epoch_rejection_log.csv`
- `vep_peak_review.csv`
- `duplicate_review.csv`
- reproducible per-session `mne.Report` HTML reports
- post-review signal, paired, waveform, and count-change files

HTML is the authoritative session-report format. PDF export is available when WeasyPrint and its system libraries are installed.

Do not copy GUI-generated results into the manuscript until the calculations have been reproduced, validation is documented, an EEG expert has reviewed a subset, and the scientific team approves the figures and interpretation.
