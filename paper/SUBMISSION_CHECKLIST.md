# Sensors submission checklist

Status of each item that gates submission to MDPI *Sensors*. Items marked
`auto` are produced by scripts in this folder; items marked `human` need a
person to fill them in.

## Auto-generated, current

- [x] §3.1 cohort and per-subject QC table — `build_paper_stats.py`
- [x] §3.2 alpha SNR (medians + BCa CI + Mann–Whitney p) — `build_paper_stats.py`
- [x] §3.3 alpha reactivity (medians + BCa CI + Berger counts) — `build_paper_stats.py`
- [x] §3.4 normalized PSD figures — `comparison_analysis.plot_three_way_comparison`
- [x] §3.5 spectrogram SSIM (medians + BCa CI + Mann–Whitney p) — `build_paper_stats.py`
- [x] §3.6 paste cross-setup bridge (Mann–Whitney p, all three endpoints) — `build_paper_stats.py`
- [x] §3.7 felt-setup VEP figure + per-subject latency/amplitude table — `build_felt_vep_figure.py`
- [x] §3.8 / Supplement S3 gel-VEP qualitative panels — already in `output/gel_vep_sanity_check/`

## Human input needed before circulating beyond the immediate team

- [ ] **Author list**: replace the four `[TBD: Author N Surname]` slots in `main.tex`. Suggested fill from email context: Behtom Adeli (gel data + original pipeline), Maryam (3D housings). Walter to confirm order.
- [ ] **ORCIDs**: replace `0000-0000-0000-0000` × 4.
- [ ] **Co-author emails** in the affiliation block.
- [ ] **CRediT contributions block** in `main.tex` (`\authorcontributions{}`).
- [ ] **`adeli2024` bibtex stub** in `refs.bib` — Behtom to confirm exact title / journal / year of the prior gel-TCRE write-up.

## Human input needed before submission

- [ ] **IRB number** and approving body — `\institutionalreview{}`.
- [ ] **Funding** sources and grant numbers — `\funding{}` (URI / NIH / NSF as applicable).
- [ ] **Acknowledgements** — `\acknowledgments{}`.
- [ ] **Repo URL** for the public code release — referenced in §S5 and `\dataavailability{}`.

## Open methodological items

- [x] **Felt-tEEG vs gel-tEEG estimator/geometry equivalence.** Resolved 2026-04-29 by W.G.B.: gel cohort already provides hardware tEEG (no software 16·in − out reconstruction required); felt and gel TCREs are the same ring geometry. Cross-construction contrast is on electrolyte and coupling, not estimator. Updated in §2.2, §4.2, abstract, and Conclusions.
- [x] **/187 scaling on stored gel tEEG channels.** Resolved 2026-05-03: B.~Adeli confirmed the factor was a transcription error and should never have been applied. Removed from `GEL_TCRE_CONFIG.amplitude_scaling`; `build_paper_stats.py`, `build_sensitivity_figure.py`, and `build_felt_vep_figure.py` re-run; main.tex, supplement.tex, and figs/ refreshed. The most-affected number is the Gel-tEEG vs Paste-tEEG SSIM contrast (now $p=0.041$, $d=-0.78$ vs prior $p=0.063$, $d=-0.70$).
- [ ] **vamp vs BrainAmp split for ADC resolution.** W.G.B. recalls two amplifier models being trialled; sessions with `vamp` in filename should carry the lower 0.0488 µV/bit value. Verify against the BA-2 / MN-3 .vhdr files and tag sessions accordingly in the loader.

## Workflow / template

- [ ] Download the official MDPI Sensors LaTeX template from <https://www.mdpi.com/authors/latex>, drop the `Definitions/` folder into `paper/`, and flip `\previewtrue` → `\previewfalse` in both `main.tex` and `supplement.tex`.
- [ ] **Figure freeze**: once cohort and stats are final, copy the PNGs from `output/three_way_comparison/`, `output/felt_vep/`, and `output/gel_vep_sanity_check/` into `paper/figs/` and switch the figure includes from the `../output/...` paths to `figs/...`. This makes the submission set reproducible without re-running notebooks.
- [ ] **Single submission ZIP**: `main.tex`, `main.pdf`, `refs.bib`, `supplement.tex`, `supplement.pdf`, `figs/`, `Definitions/`. Exclude `output/`, `data/`, `Gel TCRE/`, `venv/`.

## Outstanding paper-side TBDs (search `\TBD` in main.tex / supplement.tex)

Run `grep -n '\\TBD' paper/main.tex paper/supplement.tex` to enumerate. Current count after the latest splice: about a dozen, all in front-matter / discussion / supplement narrative. None block compile.

## Re-running the auto pipeline

```bash
source venv/bin/activate
python scripts/build_paper_stats.py
python scripts/build_felt_vep_figure.py
python scripts/build_sensitivity_figure.py
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && pdflatex supplement
```
