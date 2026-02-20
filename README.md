# Tripolar EEG Alpha Wave Analysis — SK1

Comparison of **tripolar concentric ring electrodes (tEEG)** vs. **conventional disc electrodes** for detecting visually-induced and spontaneous alpha waves.

---

## Background

This project analyzes EEG data recorded using tripolar concentric ring electrodes (CREs) developed by CREmedical. Unlike standard disc electrodes that measure voltage at a single point, each tripolar CRE has 3 concentric rings and outputs two derivations:

- **Conventional derivation** — outer ring referenced to a remote mastoid electrode (behaves like a standard EEG electrode)
- **tEEG (Laplacian) derivation** — computed on-board from the 3 rings, producing a surface Laplacian that acts as a spatial high-pass filter, suppressing volume conduction and improving focal source detection

The goal is to determine whether the tripolar tEEG electrodes can detect alpha-band (8–13 Hz) activity as effectively as the gold-standard conventional disc electrode.

---

## Experiment

**Subject:** SK1  
**Date:** February 19, 2026, 10:58 AM  
**Amplifier:** BrainAmp (Brain Products)

### Paradigm

| Phase | Description | Duration |
|-------|-------------|----------|
| Checkerboard VEP | Fixation on reversing checkerboard pattern with central red dot. 3 blocks of 20 reversals (~600 ms ISI) | ~12 s per block |
| Eyes Open / Close | Alternating eyes-open and eyes-closed resting periods to elicit spontaneous alpha (Berger effect) | ~30 s each, 3–4 cycles |

### Recording Setup

| Parameter | Value |
|-----------|-------|
| Channels | 11 data channels |
| Sampling rate | 1000 Hz |
| Resolution | 0.1 µV/bit (INT16) |
| Hardware filters | 0.1 Hz HP (10 s time constant), 250 Hz LP, notch OFF |
| Reference | Mastoid (behind ear) — dedicated amplifier input |
| Ground | Base of skull — dedicated amplifier input |

### Channel Map

> ⚠️ The odd=conventional / even=tEEG pairing is **inferred** from signal amplitude characteristics. Verify with hardware documentation.

| Channel | Electrode | Derivation | Electrolyte |
|---------|-----------|------------|-------------|
| Ch1 | Tripolar CRE #1 | Conventional | Saltwater |
| Ch2 | Tripolar CRE #1 | tEEG (Laplacian) | Saltwater |
| Ch3 | Tripolar CRE #2 | Conventional | Saltwater |
| Ch4 | Tripolar CRE #2 | tEEG (Laplacian) | Saltwater |
| Ch5 | Tripolar CRE #3 | Conventional | Saltwater |
| Ch6 | Tripolar CRE #3 | tEEG (Laplacian) | Saltwater |
| Ch7 | Tripolar CRE #4 | Conventional | Saltwater |
| Ch8 | Tripolar CRE #4 | tEEG (Laplacian) | Saltwater |
| Ch9 | Tripolar CRE #5 | Conventional | Conductive paste |
| Ch10 | Tripolar CRE #5 | tEEG (Laplacian) | Conductive paste |
| Ch11 | Standard disc | Conventional | Conductive paste |

---

## Repository Structure

```
.
├── README.md                              ← You are here
├── EEG_Decomposition_Notebook_SK1.ipynb   ← Main analysis notebook (run this)
│
├── data/                                  ← Raw data files (BrainVision format)
│   ├── SK1_2-19-2026.eeg                  ← Raw EEG binary (11ch × 450,240 samples, INT16)
│   ├── SK1_2-19-2026.vhdr                 ← Header file (channel info, sampling rate, resolution)
│   ├── SK1_2-19-2026.vmrk                 ← Marker file (stimulus triggers + eyes open/close events)
│   ├── SK1_2-19-2026-Triggers.avg         ← Pre-averaged VEP (74 segments, float32, filtered)
│   ├── SK1_2-19-2026-Triggers.vhdr        ← Header for the averaged file
│   ├── SK1_2-19-2026-Triggers.vmrk        ← Markers for the averaged file
│   └── Recorder.Setting                   ← BrainVision Recorder workspace reference
```

### Data Files Explained

**BrainVision format** uses a trio of files (`.vhdr` + `.vmrk` + `.eeg`):

| File | Format | Contents |
|------|--------|----------|
| `.eeg` | Binary (INT16, multiplexed) | The actual EEG samples. Channels are interleaved: `[ch1_t0, ch2_t0, ..., ch11_t0, ch1_t1, ...]` |
| `.vhdr` | Text (INI-style) | Header metadata: number of channels, sampling rate, resolution (µV/bit), hardware filter settings |
| `.vmrk` | Text (INI-style) | Event markers with sample-accurate timing: stimulus triggers (S7), eyes-open/close comments |

**Averaged file** (`-Triggers.*`):

| File | Format | Contents |
|------|--------|----------|
| `-Triggers.avg` | Binary (IEEE float32, multiplexed) | Stimulus-locked average across 74 checkerboard triggers. 500 timepoints (−100 to +400 ms). Already filtered: 0.5–30 Hz bandpass, 60 Hz notch, baseline-corrected |
| `-Triggers.vhdr` | Text | Header for the averaged data (confirms `SegmentDataPoints=500`, `AveragedSegments=74`) |
| `-Triggers.vmrk` | Text | Time-zero marker at sample 101 (= 100 ms, the stimulus onset) |

**Recorder.Setting** — Points to the BrainVision Recorder workspace file (`BAmp_Felt_VEP_11ch_500ms-wnotch.rwksp`). Not needed for analysis.

---

## Notebook Structure

The Jupyter notebook (`EEG_Decomposition_Notebook_SK1.ipynb`) is organized into 14 sections:

| # | Section | What It Does |
|---|---------|--------------|
| 1 | Setup & Data Loading | Load raw `.eeg` binary file, scale to µV, load pre-averaged VEP |
| 2 | Helper Functions | Notch filter (60 Hz), bandpass filter, Hilbert envelope, band definitions |
| 3 | Channel Statistics | Per-channel min/max/std, identifies clipping channels |
| 4 | Raw Time Series | All 11 channels with stimulus and eyes-open/close event markers overlaid |
| 5 | Power Spectral Density | Full-range PSD (0–80 Hz) + alpha-focused PSD (1–30 Hz after 60 Hz notch) |
| 6 | Spectrograms | Per-channel time-frequency decomposition (STFT) with event markers |
| 7 | Band Decomposition | Per-channel decomposition into delta/theta/alpha/beta/gamma |
| 8 | Alpha Envelope | Instantaneous alpha power over time (Hilbert transform) with event shading |
| 9 | Eyes Open vs Closed | **Core analysis:** PSD comparison + alpha reactivity ratio (Berger effect) |
| 10 | Visual Evoked Potential | Pre-averaged VEP waveforms, per-channel and overlaid by electrode type |
| 11 | Alpha Dynamics Comparison | Normalized alpha envelopes: tEEG vs conventional vs disc |
| 12 | Cross-Channel Correlation | Pearson correlation of each channel's alpha envelope with Ch11 (disc) |
| 13 | Summary Statistics | Dashboard: alpha SNR, reactivity, disc correlation, VEP amplitude |
| 14 | Key Findings | Discussion of results, caveats, and suggested next steps |

---

## Getting Started

### Requirements

Only standard scientific Python libraries are needed:

```
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```

### Running the Notebook

1. Clone or download this repository
2. Place the data files in the same directory as the notebook (or update `EEG_FILE` and `AVG_FILE` paths in cell 2)
3. Open the notebook:
   ```bash
   jupyter notebook EEG_Decomposition_Notebook_SK1.ipynb
   ```
4. Run all cells (`Cell → Run All`) or step through one at a time

### Configuration

All configurable parameters are in **cell 2** of the notebook:

```python
EEG_FILE = 'SK1_2-19-2026.eeg'       # Path to raw EEG binary
AVG_FILE = 'SK1_2-19-2026-Triggers.avg'  # Path to pre-averaged VEP
N_CHANNELS = 11
FS = 1000          # Sampling rate (Hz)
RESOLUTION = 0.1   # µV per bit
```

Channel labels and electrode-type groupings (`IDX_SW_CONV`, `IDX_SW_TEEG`, etc.) are also defined here. **Update these if the channel mapping differs from what's documented above.**

---

## Key Metrics

The notebook computes four metrics to compare electrode types:

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Alpha SNR** | Ratio of alpha power (8–13 Hz) to neighboring bands (4–8 + 13–30 Hz) in dB | Higher SNR = cleaner alpha detection relative to background |
| **Alpha Reactivity** | Ratio of alpha power during eyes-closed vs eyes-open epochs | Values >1 indicate the Berger effect is detected — a basic validity check |
| **Correlation with Disc** | Pearson r between a channel's alpha envelope and Ch11 | High r means the electrode tracks the same neural events as the gold standard |
| **VEP Peak-to-Peak** | Max − min of the averaged evoked potential (−100 to +400 ms) | Measures the electrode's ability to detect stimulus-locked neural responses |

---

## Known Limitations

- **Channel mapping is inferred**, not confirmed from hardware documentation. The odd=conventional / even=tEEG pairing is based on amplitude characteristics (conventional channels clip at ±3276.7 µV; tEEG channels have 10–50× lower amplitude)
- **Conventional channels clip** — Ch1, 3, 5, 7 frequently saturate the 16-bit ADC, which may bias power estimates upward
- **No artifact rejection** — eye blinks, muscle activity, and movement artifacts are not removed. Consider ICA for cleaner results
- **Limited eyes-closed epochs** — only 3 eyes-closed periods (~30 s each), limiting statistical power
- **Electrode positions not documented** — the scalp locations of each CRE are not recorded in these files
- **Last epoch is truncated** — Eyes Open #4 (342.5 s) has no closing marker; the notebook assumes 30 s duration

---

## Potential Next Steps

- Confirm channel mapping with hardware documentation or professor
- Apply artifact rejection (ICA or threshold-based)
- Add statistical testing (paired t-tests or permutation tests across epochs)
- Record electrode positions for topographic mapping
- Repeat with additional subjects for group-level analysis
