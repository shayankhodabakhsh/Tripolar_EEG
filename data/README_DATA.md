# Data directory

This folder is intentionally **gitignored** because the raw EEG recordings are
access-controlled under the URI Biomedical Engineering IRB protocol.

## Expected layout for the analysis pipeline

```
data/
├── README_DATA.md              ← this file (only file in the dir that is committed)
│
├── <short-felt sessions>       ← 11-channel BrainVision recordings, root of data/
│   ├── AF felt TCRE 3-10-2026.{eeg,vhdr,vmrk}
│   ├── AH felt TCRE 2-3-2026.{eeg,vhdr,vmrk}
│   ├── ... (and matching -Triggers.{avg,vhdr,vmrk} files for VEP averages)
│
└── Long felt TCRE/             ← 11-channel BrainVision recordings, long-felt sub-cohort
    ├── HS Long Felt TCRE 3-24-2026.{eeg,vhdr,vmrk}
    ├── LS2-long-felt-TCRE-4-7-2026.{eeg,vhdr,vmrk}
    ├── TU2_long_felt_TCRE_4-7-2026.{eeg,vhdr,vmrk}
    └── ... (and -Triggers.{avg,vhdr,vmrk} VEP averages)
```

The gel-cohort recordings live in a sibling folder (`Gel TCRE/`, also
gitignored), which the analysis pipeline picks up via `GEL_DIR` in
`scripts/build_paper_stats.py`.

## How to obtain the data

Raw recordings are not distributed with the public repository. To request
access for IRB-approved collaboration, contact:

- **Shayan Khodabakhsh** — skhodabakhsh@uri.edu
- **Walter G. Besio** (PI) — University of Rhode Island, Department of
  Electrical, Computer and Biomedical Engineering

Please describe your intended use; an IRB amendment or data-use agreement
may be required.

## File format reference

All recordings are BrainVision INT16 binary at 1000 Hz. Each session has
three files: `<name>.eeg` (raw binary), `<name>.vhdr` (header with channel
labels and µV-per-bit resolution), and `<name>.vmrk` (event markers,
including stimulus and eyes-open/closed triggers). Pre-averaged VEP
responses to the checkerboard paradigm are stored in matching
`<name>-Triggers.avg` files alongside their own `.vhdr` and `.vmrk`.

## Reproducing the published figures with your own data

If you have access to the raw recordings, place them in the directory tree
shown above, then from the repository root:

```bash
pip install -e .
python scripts/build_paper_stats.py
python scripts/build_felt_vep_figure.py
python scripts/build_sensitivity_figure.py
```

The first script regenerates the comparison figures and statistical tables
in `output/three_way_comparison/` and `paper/`. The second produces the
felt-setup VEP figure in `output/felt_vep/`. The third produces the
A/B/C pipeline-variant sensitivity panel in `output/sensitivity/`.
