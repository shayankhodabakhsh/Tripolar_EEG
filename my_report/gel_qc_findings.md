# Gel TCRE — Data Quality Findings

Date: 2026-04-27. Source dataset: `Gel TCRE/10-20-2024/`. Pipeline used for
amplitude statistics: raw int16 → per-channel `.vhdr` resolution → no filtering.
For VEP characterization: 60 Hz notch + zero-phase FIR Hamming bandpass
0.05–55 Hz; epochs −0.1 to 0.5 s; baseline (−0.1, 0).

## ADC resolution discrepancy across the 10-20-2024 session

The 8 recordings collected on the same day use two different ADC resolutions.
Both `int16` ceilings (±32767 counts) are valid, but the µV ceiling differs by
~2.05×:

| Subject                                   | µV/bit     | ±µV ceiling | fs   |
|-------------------------------------------|-----------:|------------:|-----:|
| BA-1-10-20-2024                           | 0.100000   | ±3276.7     | 1000 |
| BA-2-10-20-2024                           | 0.048828   | ±1600.0     | 1000 |
| MN-1-Vamp10-20-2024                       | 0.048828   | ±1600.0     | 1000 |
| MN-2-30Hz-Vamp                            | 0.048828   | ±1600.0     | 1000 |
| MN-3-BrainAmp_VEP                         | 0.100000   | ±3276.7     | 1000 |
| MN-4-foreheadREF-VEP-10-20-2024           | 0.100000   | ±3276.7     | 1000 |
| MN-5-BrainAmp-VEP-PZref-10-20-2024        | 0.100000   | ±3276.7     | 1000 |
| MN-6-Vamp-VEP-10-20-2024                  | 0.048828   | ±1600.0     | 1000 |

The current `eeg_analysis.py` hard-codes `RESOLUTION = 0.1` in
`load_subject`, which means subjects with 0.0488281 µV/bit are read with µV
amplitudes inflated by ~2.05×. Affected subjects: BA-2, MN-1, MN-2, MN-6
(half of the 10-20-2024 session).

## BA-2 raw amplitude / ADC clipping per channel

Duration 2155.0 s, fs=1000 Hz, resolution=0.0488281 µV/bit (ceiling ±1600.0 µV).
"rail %" = fraction of samples at exactly ±32767 int16. "≥99% rail %" =
fraction within 1% of rail.

| Ch  | min int16 | max int16 | mean (µV) | std (µV) | min µV  | max µV  | rail %  | ≥99% rail %  |
|-----|----------:|----------:|----------:|---------:|--------:|--------:|--------:|-------------:|
| Ch1 |   −32768  |    32767  |    450.5  |   793.3  | −1600.0 | +1600.0 |  0.00   |    0.50      |
| Ch2 |   −32768  |    32767  |    434.1  |   798.0  | −1600.0 | +1600.0 |  0.00   |    0.50      |
| Ch3 |   −32768  |    32767  |    409.7  |   667.5  | −1600.0 | +1600.0 |  0.00   |    0.16      |
| Ch4 |   −32768  |    32767  |    445.8  |   762.4  | −1600.0 | +1600.0 |  0.00   |    0.41      |
| Ch5 |   −32768  |    32767  |   −353.5  |   743.3  | −1600.0 | +1600.0 |  0.00   |    0.50      |
| Ch6 |   −32768  |    32767  |    354.8  |   678.7  | −1600.0 | +1600.0 |  0.00   |    0.22      |
| Ch7 |   −32768  |    32767  |     79.7  |  1060.6  | −1600.0 | +1600.0 |  0.00   |    0.99      |

All seven channels span the full ±1600 µV range, but no channel sits pinned
at the rail for any meaningful fraction of samples (max 0.99% on Ch7, all
others ≤0.50%). Standard deviations are 600–1100 µV. Channel-to-channel
DC offsets range from −353 to +450 µV.

## BA-1 raw amplitude / ADC clipping per channel (reference)

Duration 647.3 s, fs=1000 Hz, resolution=0.1 µV/bit (ceiling ±3276.7 µV).

| Ch  | min int16 | max int16 | mean (µV) | std (µV) | min µV  | max µV  | rail %  | ≥99% rail %  |
|-----|----------:|----------:|----------:|---------:|--------:|--------:|--------:|-------------:|
| Ch1 |   −32767  |    32767  |   −11.8   |   799.0  | −3276.7 | +3276.7 |  3.32   |    3.35      |
| Ch2 |    −9822  |    13538  |    22.0   |    59.6  |  −982.2 | +1353.8 |  0.00   |    0.00      |
| Ch3 |   −32767  |    21335  |     8.5   |   219.3  | −3276.7 | +2133.5 |  0.00   |    0.00      |
| Ch4 |     −615  |     1467  |    32.5   |     8.6  |   −61.5 |  +146.7 |  0.00   |    0.00      |
| Ch5 |   −32767  |    12797  |     0.1   |   185.7  | −3276.7 | +1279.7 |  0.00   |    0.00      |
| Ch6 |   −15988  |     2726  |    15.8   |    25.2  | −1598.8 |  +272.6 |  0.00   |    0.00      |
| Ch7 |    −2153  |     3935  |   −35.9   |    47.4  |  −215.3 |  +393.5 |  0.00   |    0.00      |

BA-1 saturation is asymmetric and channel-specific:
- Ch1 (outer ring O1) is rail-pinned on 3.32% of samples (continuous saturation).
- Ch3 (outer O2) and Ch5 (outer Pz) reach the negative rail but not the positive.
- Inner-ring channels (Ch2, Ch4, Ch6) and disc (Ch7) stay well within range
  on BA-1.

## BA-2 vs BA-1 saturation pattern

- BA-1: outer-ring channels saturate (3% on Ch1, asymmetric on Ch3/Ch5);
  inner-ring channels and disc stay clean.
- BA-2: every channel including the disc reaches ±1600 µV (the full int16
  range at this resolution). No channel sits at the rail continuously,
  but std is 6–10× higher than BA-1's inner channels.

These are different failure modes. BA-1's outer-ring saturation is consistent
with the high-impedance hypothesis the team has been investigating
(the central pellet-to-skin contact is the higher-impedance path on the
outer ring). BA-2's whole-channel high-amplitude pattern is consistent with
either a reference-electrode problem or a global noise/movement source
that affects all electrodes simultaneously.

## Status

This is a data-quality observation at the level of raw recordings. No
inference is being made here about whether BA-2 should be included in any
analysis or how it compares to other subjects' VEPs — those decisions are
pending Dr. Besio's review.

## Cross-reference with original `Resutls Individual`

`Gel TCRE/Resutls Individual/` contains four sibling subfolders — `PSD`,
`PSD2`, `PSDMNE`, `Time Series` — each holding 51 PNGs across 23 distinct
subjects. Per-condition file naming: `<subject>_<condition>_psd.png` (or
`_tseries.png`). Conditions are `checkerboard`, `eyes_closed`, `eyes_open`.

### Subject inventory vs raw data

24 raw recordings (`*.eeg`, excluding `*Trigger*`) exist across the eight
date subfolders under `Gel TCRE/`. 23 of those have results in
`Resutls Individual/`. Difference:

- **In raw, not in results:** `BA_BrainAmp_smlMon_2-30VEP_11-8-2024`. This
  is the only subject the original team appears to have excluded from the
  per-subject analysis. There are three other near-identically-named
  files from the same date that *are* in the results
  (`BA_BrainAmp_smlMon_2-30Hz_VEP_11-8-2024-1`,
  `BA_BrainAmp_smlMon_2-30VEP-11-8-2024-1`,
  `BA_BrainAmp_smlMon_2-30VEP-11-8-2024-2`), so the exclusion is likely
  a duplicate-recording cleanup, not a data-quality call.
- **In results, not in raw:** none.

### Condition coverage per subject (10-20-2024 cohort spot-check)

| Subject              | checkerboard | eyes_closed | eyes_open |
|----------------------|:------------:|:-----------:|:---------:|
| BA-1-10-20-2024      | ✓            | ✓           | ✓         |
| BA-2-10-20-2024      | ✓            | —           | —         |
| MN-3-BrainAmp_VEP    | ✓            | ✓           | ✓         |

BA-2 has only the checkerboard condition in all four view types. The
resting-state recordings are absent for BA-2 (in both raw and results),
suggesting the BA-2 session ran only the VEP block.

### Channel naming used in the original results

All seven channels are labeled in the original PNGs as:

1. Tripolar Gel O1
2. EEG Emulator Gel O1
3. Tripolar Gel O2
4. EEG Emulator Gel O2
5. Tripolar Paste Pz
6. EEG Emulator Paste Pz
7. Normal EEG Pz

"EEG Emulator" = the inner ring used as a single-electrode (eEEG-equivalent)
signal. This matches the `fifthTCREGelProcess.py` mapping
(Ch1=outer/O1, Ch2=inner/O1, Ch3=outer/O2, Ch4=inner/O2, Ch5=outer/Pz,
Ch6=inner/Pz, Ch7=disc) and confirms that "Tripolar Gel/Paste" panels are
the computed `16·inner − outer` Laplacian.

### Visual triage — checkerboard view, three subjects across four views

**Description of what is plotted in each view type:**

- **PSD** — 7-panel subplot grid; per-channel PSD in dB/Hz on a per-channel
  axis. Each panel labels its peak frequency in red text. y-range typically
  −175 to −100 dB/Hz.
- **PSD2** — overlay plot, all 7 channels superimposed in µV²/Hz (dB), with
  a head-map inset showing 3 colored dots at the recording sites. Tripolar
  channels rendered as dashed lines.
- **PSDMNE** — same overlay format and µV²/Hz dB scaling as PSD2, with the
  MNE-style head-map inset using a slightly different topomap layout
  (sensors at black/red/green dots near the back of the head).
- **Time Series** — 7-panel subplot grid; per-channel grand-average epoch
  from −0.1 to +0.4 s. y-axis units shown as µV; numeric scale carries an
  exponent (e.g., `1e-7` for tripolar channels, `1e-6` for emulator/normal).

**BA-1 checkerboard:**

- PSD: every panel's flagged peak is at 0.59 Hz except Tripolar Paste Pz
  (1.78 Hz) and EEG Emulator Paste Pz (10.20 Hz, occupying the alpha range).
  Tripolar Gel O1/O2 panels show a steeply declining 1/f profile down to
  ~−175 dB/Hz at 30 Hz with no clear oscillatory peak. The 10 Hz alpha
  region is visible only on Paste Pz emulator (~10 Hz) and Normal EEG Pz
  (small inflection near 10 Hz).
- PSD2 / PSDMNE: 7 traces overlaid; tripolar gel channels (dashed) sit
  ~30 dB below the rest of the channel cluster, with two clusters
  visible (emulator + normal at top, tripolar at bottom). Both views are
  qualitatively similar; PSDMNE uses a different topomap renderer.
- Time Series: Tripolar Gel O1/O2 panels are at 1e-7 µV scale; only small
  late deflections are visible. Tripolar Paste Pz shows a small positive
  excursion peaking around 0.2–0.3 s. EEG Emulator Gel O1 panel (at 1e-6
  scale) shows an early positive bump around 0.1–0.2 s. Normal EEG Pz shows
  a clean positive peak around 0.15–0.2 s.

**BA-2 checkerboard:**

- PSD: every panel's labeled peak is in the 0.59–1.29 Hz range. No alpha
  peak is flagged anywhere. The Tripolar Gel O1 panel reaches −100 dB/Hz
  at the low end (higher than BA-1's −125 floor), with a flatter overall
  shape suggesting more broadband energy. Tripolar Gel O2 has a flat
  1/f-poor profile around −150 dB/Hz across all frequencies.
- PSD2 / PSDMNE: overlay traces are noisier than BA-1's, with more
  oscillatory ripples in the 5–15 Hz range. The two clusters
  (emulator/normal vs tripolar) are still visible.
- Time Series: Tripolar Gel O1 (1e-6 scale) is essentially a flat slow
  drift with no transient near 0.1 s. Tripolar Gel O2 (1e-7) is broadband
  noise. Tripolar Paste Pz shows a positive deflection peaking ~0.18 s.
  EEG Emulator Paste Pz shows a similar positive peak at ~0.2–0.25 s.
  Normal EEG Pz shows oscillations centered ~0.2 s.

**MN-3 checkerboard:**

- PSD: Tripolar Paste Pz and EEG Emulator Paste Pz both flag **1.98 Hz**
  as the peak — directly at the checkerboard reversal frequency. Tripolar
  Gel O1/O2 still peak at 0.59–0.89 Hz, but the 10 Hz alpha region shows
  visible bumps on Tripolar Gel O1, EEG Emulator Gel O1, and EEG Emulator
  Paste Pz. Cleaner per-panel curves than either BA-1 or BA-2.
- PSD2 / PSDMNE: overlay shows clean low-frequency peaks for the paste
  channels and emulator channels, with the tripolar gel channels offset
  ~30 dB below the rest as before.
- Time Series: Tripolar Gel O1 (1e-7) shows a clear positive peak at
  ~0.13 s. Tripolar Gel O2 shows a clear *negative* deflection at ~0.18 s.
  EEG Emulator Gel O1 (1e-6) shows a positive peak at ~0.13–0.15 s.
  Tripolar Paste Pz shows a slow positive build peaking near 0.2–0.3 s.
  Normal EEG Pz panel is at 1e-6 scale and shows a less defined waveform.

## BA-2 preprocessing sensitivity

To isolate which step in the Adeli/Norouzi preprocessing chain (if any) is
load-bearing for recovering a VEP from BA-2's noisy recording, three
preprocessing variants were applied to the same epoched data
(139 `Trigger,T 1` events, baseline −100 to 0 ms, tEEG reconstructed as
`16·inner − outer` from the corrected channel mapping).

| Variant | Pipeline                                                                       | Output units |
|---------|--------------------------------------------------------------------------------|--------------|
| A       | notch 60 Hz + 1–55 Hz FIR Hamming + db8 level-3 wavelet (soft, universal) + z-score | z-units      |
| B       | notch 60 Hz + 0.05–55 Hz FIR Hamming                                           | µV           |
| C       | notch 60 Hz + 0.05–55 Hz FIR Hamming + db8 level-3 wavelet (soft, universal)   | µV           |

Figure: `output/gel_vep_sanity_check/BA-2_preprocessing_sensitivity.png`

### Per-channel peak detection (search windows: N75 60–95 ms, P100 95–145 ms, N135 140–200 ms)

**tEEG @ Pz (computed 16·inner − outer):**

| Variant | N75 (ms / amp) | P100 (ms / amp) | N135 (ms / amp) | Waveform range  |
|---------|---------------:|----------------:|----------------:|-----------------|
| A       |  69 / −0.35 z  |  117 / +0.21 z  |  171 / −0.42 z  | [−0.45, +0.51] z|
| B       |  69 / −617 µV  |  117 / +404 µV  |  171 / −709 µV  | [−798, +942] µV |
| C       |  69 / −614 µV  |  117 / +403 µV  |  171 / −724 µV  | [−802, +931] µV |

**Inner ring @ Pz:**

| Variant | N75 (ms / amp) | P100 (ms / amp) | N135 (ms / amp) | Waveform range  |
|---------|---------------:|----------------:|----------------:|-----------------|
| A       |  69 / −0.37 z  |  117 / +0.23 z  |  171 / −0.41 z  | [−0.45, +0.51] z|
| B       |  69 / −41 µV   |  117 / +27 µV   |  171 / −44 µV   | [−50, +58] µV   |
| C       |  69 / −41 µV   |  117 / +27 µV   |  171 / −44 µV   | [−50, +58] µV   |

**Disc Ch7:**

| Variant | N75 (ms / amp) | P100 (ms / amp) | N135 (ms / amp) | Waveform range  |
|---------|---------------:|----------------:|----------------:|-----------------|
| A       |  85 / −0.12 z  |   99 / +0.37 z  |  190 / −0.18 z  | [−0.25, +0.37] z|
| B       |  86 / −29 µV   |   99 / +93 µV   |  190 / −41 µV   | [−69, +95] µV   |
| C       |  85 / −29 µV   |   99 / +92 µV   |  190 / −44 µV   | [−63, +92] µV   |

### What each variant produces

- **All three variants produce essentially the same waveform shape** for
  every channel. Peak latencies on the tEEG and inner-ring channels are
  identical across A/B/C (N75 = 69 ms, P100 = 117 ms, N135 = 171 ms).
  Disc latencies match within 1 ms across variants (N75 ≈ 85–86 ms,
  P100 = 99 ms, N135 = 190 ms).
- **Variant B vs C differ only in the third decimal** of amplitude on the
  inner-ring and tEEG channels — the wavelet step changed N135 on tEEG by
  ~15 µV out of a ~700 µV swing, and changed nothing perceptible on the
  inner-ring or disc.
- **Variant A** is Variant B/C divided by the per-channel standard
  deviation (z-score). The shape is preserved; amplitude is rescaled to
  ±0.5 z-units. Z-score adds no morphological information.
- **The recovered tEEG @ Pz waveform is dominated by oscillatory structure**
  (visible periodic peaks throughout the −100 to +500 ms window), not a
  clean transient with a localized P100. The N75/P100/N135 search windows
  pick out points along this oscillation that happen to fall in the
  expected latency ranges, but the wider waveform context shows similar
  excursions outside the windows.
- **The disc Ch7 channel** shows a more transient-looking morphology —
  a clear positive deflection at 99 ms preceded and followed by negativities
  — across all three variants. This is recognizable as VEP-like and is
  produced by the bandpass alone; wavelet and z-score do not change it.

### Which step appears load-bearing

For BA-2 specifically: **none** of the three preprocessing variants
isolates a clean VEP on the tEEG channel that the others miss. The
wavelet denoising step (comparing B → C) does not measurably alter the
recovered waveform on this recording. The z-score step (comparing C → A)
rescales amplitude but does not change shape or peak locations. The
oscillatory contamination present in BA-2's raw recording propagates
through all three pipelines essentially unchanged. The only channel with
a recognizable VEP shape on BA-2 is Ch7 (the disc), and that morphology
is produced by the bandpass step alone in all three variants.

This is a single-subject sensitivity test on a recording flagged as
data-quality-degraded; results may differ on cleaner recordings (e.g.,
BA-1, MN-3), where the wavelet step could plausibly have a different
effect.

### What the PSD vs PSDMNE difference suggests

PSD and PSDMNE use the same data but different rendering. The PSD folder
uses per-panel grid plots in dB/Hz on a tight per-channel axis; PSDMNE uses
an MNE-style overlay (`raw.compute_psd().plot()`) with a topomap. PSD2
matches PSDMNE in layout and scale but uses a different topomap renderer
(probably SciPy + a hand-drawn head circle vs MNE's `plot_psd` montage).
The y-axis scales differ: PSD is reported in dB/Hz with values around
−100 to −175, PSDMNE in µV²/Hz (dB) with values around −30 to +30 — a
constant offset, consistent with the same Welch PSD but different reference
power (V vs µV) and possibly different `n_fft`/window. No qualitative
difference in peak locations is visible between the three PSD views.
