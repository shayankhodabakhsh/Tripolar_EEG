# TCRE Paper — Deep Review Report for Claude Code

**Manuscript:** *A Methods-Harmonized Comparison of Tripolar Concentric Ring Electrode Constructions for Non-Invasive EEG: Saline-Soaked Felt, Conductive Gel, and Paste*
**Reviewer pass:** Comprehensive substantive + methodological + structural
**Audience:** Claude Code (LaTeX/Python source repo)
**Date:** 2026-05-03

---

## How to use this document

Each issue is tagged with a **priority** and a **type**. Work top-down within each priority bucket.

| Priority | Meaning |
|----------|---------|
| **P0** | Blocks submission. Factual, ethical, or statistical correctness. |
| **P1** | Reviewer will hit hard. Strongly fix before submission. |
| **P2** | Improves quality but won't kill the paper. |
| **P3** | Polish / nice-to-have. |

| Type | Meaning |
|------|---------|
| `STATS` | Statistical method correctness |
| `SCI` | Scientific interpretation / mechanism |
| `STRUCT` | Section ordering, figure placement, redundancy |
| `WRITE` | Clarity, readability, sentence-level |
| `META` | Metadata / placeholders / TODOs |
| `REF` | References / citations |
| `FIG` | Figures and tables |
| `REPRO` | Reproducibility / methods reporting |

Each item has these fields where applicable:
- **Where:** section, subsection, table/figure, or quoted phrase
- **Problem:** what is wrong
- **Why it matters:** what a reviewer will say
- **Fix:** concrete action
- **Suggested text:** drop-in replacement (in fenced block) when applicable

---

# Part 1 — Critical (P0) Issues

These must be resolved before the manuscript leaves the lab.

## 1.1 [P0 / STATS] Wrong test family for within-setup construction comparisons

**Where:** Section 3.2 (Alpha SNR) and Section 3.3 (Reactivity).

**Problem.** You report:
- "Felt-tEEG vs Paste-tEEG **within the felt setup**: U = 13, p = 0.006, d = −1.58 (n_a = n_b = 10)"
- "Gel-tEEG vs Paste-tEEG **within the gel setup**: U = 68, p = 0.41, d = −0.44 (n_a = n_b = 13)"

These are **paired** observations: every felt subject contributes both a felt-tEEG value and a paste-tEEG value (because the paste TCRE is on the same headcap as the felt TCREs). Same for the gel cohort. Mann–Whitney U is for **independent** samples. The correct test is **Wilcoxon signed-rank** (non-parametric paired test), and the correct effect size is **matched-pairs rank-biserial correlation** or **Cohen's d_z** (paired).

**Why it matters.** This is the kind of error any methods reviewer or biostatistician will catch within 30 seconds. It also throws away statistical power: with paired data, signed-rank is more powerful than Mann–Whitney because it removes between-subject variance. The conclusions may not change, but the test must be correct.

**Fix.**
1. Replace `mannwhitneyu` with `wilcoxon` in the analysis pipeline for **within-setup, cross-construction** comparisons only.
2. Replace independent-samples Cohen's d with **d_z** (mean of paired differences / sd of paired differences) or **matched-pairs rank-biserial r**.
3. Cross-setup comparisons (Felt-tEEG vs Gel-tEEG, Felt-paste vs Gel-paste) **stay** as Mann–Whitney — those are independent.
4. Add a sentence to Section 2.6 stating which comparisons are paired and which are independent.

**Suggested edit to Section 2.6:**

```
Subject is the unit of analysis. For each metric we aggregate per subject and
per channel class (FELT_TEEG, GEL_TEEG, PASTE_TEEG, FELT_EEEG, GEL_EEEG,
PASTE_EEEG, DISC). Within-setup comparisons across constructions (e.g.
Felt-tEEG vs Paste-tEEG within the felt cohort) are paired by subject and
tested with the two-sided Wilcoxon signed-rank test, with matched-pairs
rank-biserial correlation r as the effect size. Between-setup comparisons
(e.g. Felt-tEEG vs Gel-tEEG, or paste-TCRE in felt vs gel setup) are
independent and tested with the two-sided Mann-Whitney U test, with
Cohen's d as the effect size. Central tendency is reported as the median
with bootstrap BCa 95% confidence intervals (10,000 resamples). No
parametric tests are applied to the n = 2 long-felt subgroup. No
adjustments for multiple comparisons are applied because the analysis
is descriptive and the endpoints are pre-specified.
```

**Code action for Claude Code:** Locate every call site of the form `mannwhitneyu(felt_*, paste_*)` or `mannwhitneyu(gel_*, paste_*)` where the inputs are paired by subject; switch to `scipy.stats.wilcoxon` and re-emit the results table. The relevant computation likely lives in `paper_sensors/build_paper_stats.py` (referenced under Table 2 caption).

---

## 1.2 [P0 / META] Author name placeholder

**Where:** Title block — *"Maryam [TBD: Surname]"*. Also acknowledgments — *Tugba [TBD: Surname]*, *Hunter [TBD: Surname]*, *Luci [TBD: Surname]*, *Caitlin [TBD: Surname]*, *Gabryana [TBD: Surname]*.

**Problem.** Author byline contains a TBD placeholder. This is non-negotiable to fix before any version of this paper goes to a journal, preprint server, or co-author for review.

**Fix.** Resolve all `[TBD: Surname]` instances. The author one is most urgent.

---

## 1.3 [P0 / META] All other TBD placeholders

**Where:**
- Funding section: `[TBD: Funding sources and grant numbers (URI / NIH / NSF as applicable).]`
- IRB statement: `[TBD: IRB protocol number and approving body.]`
- Data Availability: `[TBD: repo URL]`

**Fix.** Resolve all four before submission. If the IRB protocol number is not yet assigned, contact the URI IRB office; *Sensors* will not accept human-subjects work without it.

---

## 1.4 [P0 / FIG] Figures 8–12 referenced as raw paths, not embedded

**Where:** Section 3.8.

**Problem.** In the rendered PDF, Figures 8, 9, 10, 11, and 12 appear as literal text strings (`figs/gel_vep_G02.png`, etc.) rather than rendered images. Either the LaTeX `\includegraphics{}` command is broken, the files are not in the build directory, or the figure environment is malformed.

**Fix.**
1. Verify each figure file exists at the path the source references.
2. Confirm `\includegraphics{figs/gel_vep_G02.png}` (or equivalent) is wrapped in a proper `figure` environment with `\caption{...}` and `\label{...}`.
3. Rebuild and visually confirm every figure renders.

**Code action for Claude Code:** Run `ls -la figs/` and grep the `.tex` source for each `figs/gel_vep_*.png` reference; fix any broken `\includegraphics` calls.

---

# Part 2 — High-Priority (P1) Issues

These are issues a competent reviewer will raise. Fixing them strengthens the paper substantially.

## 2.1 [P1 / SCI] The headline SSIM finding has no mechanistic story

**Where:** Section 3.5 and Discussion 4.2.

**Problem.** The most novel and most quantitatively striking result in the paper is:

> Felt 0.31 < Gel 0.52 < Paste 0.71

with very large effect sizes (d = −2.52 for Felt vs Gel, d = −2.69 for Felt vs Paste). The current interpretation is one sentence:

> "felt's saline-soaked construction producing the largest spatial-filter contrast between Laplacian and conventional derivations, and paste producing the smallest."

This does not explain **why electrolyte chemistry would change the relationship between the Laplacian and the outer-ring derivation when the ring geometry is identical**. A reviewer will ask: "If saline produces the largest Laplacian-vs-conventional contrast, why doesn't it also produce the highest tEEG alpha SNR? Your felt-tEEG SNR (1.72 dB) is similar to gel-tEEG (1.43 dB) and lower than paste-tEEG (2.97 dB)."

There is also a **noise vs signal** ambiguity in SSIM you have not addressed: lower SSIM between two channels can reflect either (a) genuinely more independent neural information (good for spatial filtering) or (b) more independent noise on the two channels (bad). Right now the paper implicitly treats lower SSIM as evidence of (a), but the data are equally consistent with (b).

**Why it matters.** This is the headline finding. Without a mechanism — even a hypothesized one — the result reads as a curiosity rather than a contribution. The first methods reviewer will ask for it.

**Fix.** Add a paragraph (~150–200 words) to Section 4.2 that:
1. Lists 2–3 plausible mechanisms (effective contact area, inter-ring impedance asymmetry, independent thermal/contact noise).
2. Explicitly notes the noise-vs-signal ambiguity and what would resolve it (impedance measurements, paired SNR analysis on matched neural signal).
3. Frames the SSIM finding as a hypothesis-generating observation, not a mechanistic conclusion.

**Suggested addition to Section 4.2:**

```
A first-pass interpretation of the Felt < Gel < Paste SSIM ordering is
that saline coupling produces the largest structural contrast between
Laplacian and conventional derivations. We note three caveats. First,
SSIM is a similarity metric and does not by itself distinguish between
two channels carrying genuinely independent neural information and two
channels sharing a common neural signal but corrupted by independent
noise; lower SSIM is consistent with both. Second, the alpha-SNR
ordering (Paste > Felt ~ Gel) does not mirror the SSIM ordering, which
argues against a simple "felt is the best spatial filter" reading.
Third, plausible electrolyte-level mechanisms include: differential
spreading of saline beyond the nominal ring footprint, effectively
enlarging the outer-ring contact area; inter-ring impedance asymmetry,
which would inject construction-dependent common-mode rejection
imbalances into the on-board Laplacian; and independent contact-noise
contributions to each ring. Direct ring-by-ring impedance logging in
future cohorts would discriminate among these. We therefore frame the
SSIM ordering as a hypothesis-generating observation about electrolyte
coupling, not as a mechanistic conclusion about which construction
provides better spatial filtering.
```

---

## 2.2 [P1 / SCI] The reactivity result is buried and partially undermines the paper

**Where:** Section 3.3.

**Problem.** Felt-tEEG median Berger reactivity is **1.49**. Gel-tEEG is **3.63**. The difference is statistically significant (p = 0.017, d = −1.11). A 1.49 reactivity ratio is **weak** by the broader alpha-reactivity literature (typical eyes-closed/eyes-open ratios on posterior electrodes are 2–4× or higher). This is reported in two sentences and never discussed.

There are at least four candidate explanations and the paper picks none:
1. Real construction effect (felt physically dampens posterior alpha).
2. Felt sessions had more ADC clipping during eyes-closed (high-amplitude alpha pushes the rail), depressing measured alpha power.
3. Felt subjects had shorter or fewer eyes-closed epochs.
4. Selection / operator effect (different operator, different placement).

**Why it matters.** A reviewer will ask: "If felt-tEEG has substantially weaker Berger reactivity than gel-tEEG, on what grounds do you claim the felt construction produces a usable EEG-grade signal?" You can answer this — but you must answer it.

**Fix.** Add a paragraph at the end of Section 3.3 (or in Section 4.3 Limitations) explicitly addressing the reactivity gap. Cross-reference Table 2 to report whether the felt subjects flagged for clipping disproportionately occupy the low-reactivity tail.

**Suggested addition to Section 3.3:**

```
The Felt-tEEG reactivity median (1.49) is notably lower than the Gel-tEEG
median (3.63) and lower than typical posterior-alpha reactivity reported
in the eyes-closed/eyes-open literature [add ref]. Three non-exclusive
factors plausibly contribute: (a) ADC clipping in three of ten felt
sessions (LF02, SF01, SF02, SF07; Table 2) is concentrated on
high-amplitude eyes-closed segments and depresses measured Pa,closed; (b)
the felt cohort had shorter average eyes-closed durations than the gel
cohort (mean X s vs Y s); and (c) a residual construction effect cannot
be excluded. We retain the Felt-tEEG cohort in the main analysis because
the Berger effect (reactivity > 1) is detected in 10/10 felt subjects,
but read the absolute reactivity magnitude as confounded with acquisition
quality.
```

(Replace `X` and `Y` with actual values from your QC tables.)

---

## 2.3 [P1 / SCI] Two paste-TCRE non-responders need to be accounted for

**Where:** Section 3.3 and abstract.

**Problem.** "The Berger effect (reactivity > 1) is detected on Felt tEEG in 10/10 subjects, Gel tEEG in 13/13, and Paste tEEG in 21/23." Who are the two paste-tEEG non-responders? Did they pass QC? Are they felt-cohort or gel-cohort subjects? Is there anything anatomically or methodologically distinguishing them?

**Why it matters.** Two-of-twenty-three (~9%) non-response rate is a real number that deserves acknowledgment. Reviewers will ask, and the alternative — silently dropping them — would be worse than reporting them.

**Fix.** Add a footnote or supplementary table listing the two non-responders by code, with their per-subject paste-tEEG reactivity and any QC flags. Add one sentence to Section 3.3 acknowledging them.

**Suggested footnote:**

```
The two paste-tEEG subjects with reactivity <= 1 are [SUBJ_ID_1] (paste
reactivity 0.XX) and [SUBJ_ID_2] (paste reactivity 0.XX). Both pass
overall QC; the disc channel from these subjects also shows unusually
weak alpha reactivity (X.X and Y.Y respectively), suggesting the
attenuated Berger response is a subject-level rather than a
construction-level effect.
```

---

## 2.4 [P1 / SCI] VEP section is the weakest part of the main text

**Where:** Section 3.7 and Table 3.

**Problem.** Several issues compound:

1. **n = 2.** With one of the two flagged for clipping (LF02). Can't claim much from that.
2. **Latencies pinned at search-window boundaries.** In Table 3, five of the six rows have N75 = 50.0, P100 = 140.0, or N135 = 125.0. **These are the canonical-window edges.** When a "peak" is detected exactly at a boundary, peak-finding has failed — the function returned the within-window extremum because there was no actual local extremum inside the window. Your text acknowledges this ("latencies that land at a window boundary indicate that the canonical N75/P100/N135 morphology is approximate rather than fully resolved") but the table presents these as if they were real measurements alongside cleaner ones.
3. **LF03 P100 at 140 ms is suspicious.** Normative checkerboard-VEP P100 latency in adults is 95–115 ms. A "P100 at 140 ms" is in N135 territory and should be flagged as morphologically atypical, not reported as a P100.
4. **LF02 felt-tEEG values (–56, –23.3, –59) are not VEP amplitudes** — those are noise-dominated traces from a clipped recording. Reporting them in the table alongside actual VEPs is misleading.

**Why it matters.** The VEP section as currently written invites a reviewer to say "you have one usable VEP and you're showing it next to noise." That undermines the rest of the paper.

**Fix.** Three options, in order of strength:

**Option A (recommended):** Move all VEP material to the supplement until you have ≥4 long-felt subjects with clean, non-clipped VEPs. The main paper rests on alpha-band findings, which are stronger.

**Option B:** Keep LF03 in the main text (with appropriate caveats about latency), move LF02 to supplement, drop LF02 felt-tEEG amplitude rows entirely, and clearly mark all boundary-pinned latencies as "not resolved" rather than reporting numerical values.

**Option C:** If you must keep both subjects, restructure Table 3 to separate "morphologically resolved" peaks (only LF03 paste-tEEG) from "boundary-pinned / not resolved" rows, and report the latter as `n.r.` (not resolved) rather than as numerical latencies.

Whichever option you choose, **flag the LF03 paste-tEEG "P100 at 140 ms" as latency-atypical** and discuss in 1–2 sentences why this might happen (impedance, stimulus parameters, individual variation).

---

## 2.5 [P1 / SCI] No external benchmarking — all "physiological plausibility" claims float

**Where:** Throughout Results and Discussion.

**Problem.** You repeatedly write "physiologically plausible" without anchoring numbers. A reader cannot tell whether:
- A disc-EEG alpha SNR of 4.33 dB is normal, low, or high.
- A paste-tEEG Berger ratio of 3.43 is normal, low, or high.
- A felt-tEEG SSIM of 0.31 between Laplacian and conventional channels is qualitatively similar to or different from prior surface-Laplacian-vs-disc EEG comparisons.

**Why it matters.** Without external anchors, "physiologically plausible" is doing all the work and a reviewer will challenge it.

**Fix.** Add at least one cited numerical benchmark for each of the three primary endpoints:
1. **Alpha SNR:** Cite a paper that reports posterior alpha SNR or alpha/non-alpha power ratio on disc EEG.
2. **Berger reactivity:** Cite an eyes-closed/open meta-analysis or a representative quantification (Klimesch 1999 *Brain Res Rev*; Babiloni et al. on alpha rhythms).
3. **SSIM in EEG context:** SSIM was developed for images; cite at least one prior application to EEG spectrograms or time-frequency representations, or explicitly note this as a methodological extension.

**Suggested phrasing template:**

```
For comparison, [Author Year] report posterior-alpha SNR of X dB on
conventional disc EEG under similar paradigms; our disc median (4.33 dB
[2.85, 7.03]) sits within that range.
```

---

## 2.6 [P1 / REPRO] Cohort definitions are incomplete

### 2.6.1 "Long felt" vs "short felt" never defined

**Where:** Section 2.1, Table 1, Section 3.1, throughout.

**Problem.** The text refers to "long-felt" and "short-felt" cohorts repeatedly, but the physical difference between the two is **not defined anywhere in Methods**. Section 2.1 says only "comprising 3 long-felt and 7 short-felt sessions". The Acknowledgments mentions "the longer screw used in the felt setup", suggesting the difference is screw length, but this is not stated as the cohort-defining variable.

**Fix.** Add a sentence to Section 2.1 defining what physically distinguishes long-felt from short-felt sessions and why both are included.

**Suggested addition:**

```
Felt sessions are split into two sub-cohorts: long-felt (LF; n = 3) and
short-felt (SF; n = 7). The two sub-cohorts differ in [SCREW LENGTH /
PAD THICKNESS / SESSION DURATION — pick the actual variable]. Both
sub-cohorts are pooled for the primary analyses in Section 3; supplement
Section S6 confirms that construction-level conclusions hold within each
sub-cohort.
```

### 2.6.2 Subject demographics are entirely absent

**Where:** Section 2.1.

**Problem.** No age, sex, handedness, vision-correction status, or any other demographic information is reported. This is a hard requirement for any human-subjects EEG paper, including pilot studies. *Sensors* will require it.

**Fix.** Add a "Subject Demographics" subsection or paragraph in Section 2.1. Minimum reporting: number of subjects per cohort, age range and mean, sex distribution, handedness, normal-or-corrected-to-normal vision (relevant for VEP).

**Suggested addition:**

```
2.1.1 Subject Demographics

Felt cohort: n = 10 (X female, Y male; age [range] years, mean Z;
handedness W). Gel cohort: n = 13 (X female, Y male; age [range] years,
mean Z; handedness W). All subjects reported normal or corrected-to-
normal vision and no history of neurological disorders. Subjects were
recruited from the URI graduate community and gave written informed
consent under URI IRB protocol [TBD].
```

### 2.6.3 No skin prep / impedance reporting

**Where:** Section 2.2.

**Problem.** Skin preparation (alcohol, abrasive paste, etc.), pre-recording impedance check, and impedance thresholds (e.g., < 10 kΩ) are not described. For a paper whose central claim is that **electrolyte and coupling drive measurable signal differences**, impedance is the single most important pre-recording variable and its absence is a glaring gap.

**Fix.** Add to Section 2.2 a paragraph describing:
1. Whether and how skin was prepared.
2. Whether per-channel impedance was measured pre-recording.
3. The threshold used to accept a placement.
4. Whether impedance was logged for offline analysis (if yes, add a supplementary figure).

If impedance was not measured, **state so explicitly** and add to Limitations.

---

## 2.7 [P1 / STATS] Bootstrap resamples too few

**Where:** Section 2.6.

**Problem.** "1000 resamples" is at the low end for BCa; 10,000 is standard for stable BCa intervals (see Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993). With n as small as 10 and a heavy-tailed metric like reactivity ratio, BCa CIs from 1,000 resamples can be visibly noisy.

**Fix.** Re-run all BCa CIs at 10,000 resamples. Update the methods text. (Computational cost is trivial.)

---

## 2.8 [P1 / WRITE] Abstract has structural issues

**Where:** Abstract.

**Problem 1 — Sentence length and density.** Multiple four-and-five-clause sentences. The sentence beginning "Across constructions, alpha-band detection and reactivity are physiologically plausible..." runs 65 words and contains five distinct claims.

**Problem 2 — Methodological framing redundancy.** The final sentence ("Both cohorts use the same hardware Laplacian on TCREs of identical ring geometry, so the cross-construction contrast is on electrolyte and coupling rather than on the Laplacian estimator itself.") is repeated almost verbatim in Section 1 (last sentence of paragraph 3) and Section 4.2 (first sentence). This is the strongest single sentence in the paper, but using it three times dilutes it.

**Problem 3 — Buried headline.** The most novel quantitative finding (the Felt 0.31 < Gel 0.52 < Paste 0.71 SSIM ordering) appears at the end of a long sentence. It should lead.

**Fix.** Rewrite the abstract with: (a) shorter sentences, (b) the SSIM finding promoted to its own sentence near the front of the results section of the abstract, (c) the "same hardware Laplacian" framing said once.

**Suggested abstract rewrite (preserves all factual content):**

```
Tripolar concentric ring electrodes (TCREs) provide an on-board surface-
Laplacian derivation (tEEG) alongside a conventional outer-ring recording
(eEEG). Published TCRE studies use different electrode constructions —
saline-soaked felt, conductive gel, paste — and different processing
pipelines, which makes cross-study performance claims difficult to
interpret. We report a pilot, methods-harmonized comparison of three
TCRE constructions across two acquisition setups: a saline-soaked-felt
setup (n = 10), a conductive-gel setup (n = 13), and a paste TCRE present
in both setups that serves as a cross-setup bridge. Both setups use the
same hardware Laplacian on TCREs of identical ring geometry, so the
cross-construction contrast is on electrolyte and coupling rather than
on the Laplacian estimator itself.

All recordings are processed through a single deliberately minimal
pipeline (60 Hz notch and zero-phase FIR 0.05-55 Hz bandpass) and
evaluated with scale-invariant endpoints: alpha-band signal-to-noise
ratio, eyes-closed/eyes-open alpha reactivity, and spectrogram structural
similarity (SSIM) between paired Laplacian and conventional channels.
Visual evoked potentials are reported only for the felt setup; the gel
cohort produced unreliable evoked responses under the harmonized
pipeline and is documented qualitatively. Subject is the unit of
analysis.

Three findings. (i) Alpha-band detection and Berger reactivity are
physiologically plausible across constructions: the Berger effect is
detected on paste tEEG in 21/23, gel tEEG in 13/13, and felt tEEG in
10/10 subjects. (ii) tEEG and eEEG carry structurally distinct spectral
content rather than a simple amplitude rescaling, with per-class median
SSIM 0.31-0.71 (all below the 0.8 conventional high-similarity
threshold) and a strong construction effect (Felt 0.31 < Gel 0.52 <
Paste 0.71; Felt vs Gel U = 0, p < 0.001, d = -2.52). (iii) The paste-
TCRE cross-setup bridge yields no significant between-setup differences
on any of the three endpoints, supporting cross-setup comparability of
construction-level effects. We frame all conclusions as descriptive
given the pilot-scale cohort and offer the harmonized pipeline as a
shared infrastructure for future TCRE comparison studies.
```

---

# Part 3 — Substantive (P2) Issues

## 3.1 [P2 / STRUCT] Introduction is too thin for a methods paper

**Where:** Section 1.

**Problem.** Four short paragraphs, two citations. A methods-comparison paper benefits from a longer introduction that:
1. Reviews prior TCRE-vs-disc comparisons (Besio's later papers, Makeyev et al., Boudet et al.).
2. Reviews the surface-Laplacian validation literature more broadly (Nunez & Srinivasan; Tenke & Kayser).
3. Explicitly states which prior gaps your harmonized pipeline addresses.
4. Discusses why electrolyte/coupling deserves attention (skin–electrode interface modeling, motion artifact differences, prep-time tradeoffs).

**Fix.** Expand to ~5–6 paragraphs (~700–900 words). Add at least 6–8 references.

## 3.2 [P2 / STATS] Cross-setup bridge claim overstates power

**Where:** Section 3.6.

**Problem.** "The bridge is now reasonably powered (n = 10 paste-TCRE observations from the felt setup) and the absence of between-setup paste differences is consistent with cross-setup comparability of construction-level effects."

With n_felt = 10, n_gel = 13, the Mann–Whitney has roughly ~50–70% power to detect a Cohen's d of 0.8 at α = 0.05. The reported alpha-SNR effect is **d = +0.51** with p = 0.15 — i.e., a medium effect that is consistent with the data but not detected. Calling this "no between-setup differences" overstates what the bridge can rule out.

**Fix.** Soften the language. State the minimum effect size the bridge can detect at standard power. Change "no significant between-setup differences" to "no detected between-setup differences at this sample size" or similar.

**Suggested replacement:**

```
None of the three Mann-Whitney comparisons reach significance under the
harmonized pipeline (alpha SNR: U = 89, p = 0.15, d = +0.51; reactivity:
U = 46, p = 0.25, d = -0.39; SSIM: U = 69, p = 0.83, d = +0.01;
n_felt = 10, n_gel = 13). With these sample sizes, the bridge has
approximately 60% power to detect d = 0.8 at alpha = 0.05; smaller
between-setup effects cannot be ruled out. The point estimates do not,
however, reproduce the magnitude or direction of the cross-construction
contrasts in Section 3.5, supporting the interpretation that
construction-level effects are not driven primarily by between-setup
acquisition differences.
```

## 3.3 [P2 / WRITE] Section 3.4 (Eyes-Open vs Eyes-Closed PSD Shape) has almost no body text

**Where:** Section 3.4.

**Problem.** Two sentences. Two figures (Fig 3 and Fig 4). The reader is shown the figures but told almost nothing about what they show or why they matter.

**Fix.** Expand to ~150 words. Describe what's visible: the 8–13 Hz bump in eyes-closed conditions, any differences in the 1/f slope across constructions, the unexplained 25 Hz peak visible in disc-EEG in Figure 4 (this should be addressed — is it harmonic noise? line-related?). Tie back to alpha SNR and reactivity findings.

## 3.4 [P2 / SCI] Unexplained 25 Hz peak in disc-EEG PSD

**Where:** Figure 4.

**Problem.** Figure 4 shows a clearly visible spectral peak around 25 Hz in the **disc EEG** trace that does not appear (or is much weaker) in the tEEG traces. This is unaddressed in text.

**Why it matters.** A reviewer will spot this. If it's a contamination (e.g., line-frequency subharmonic, ambient electronics, beta-band activity), saying nothing is worse than acknowledging it.

**Fix.** Identify the source (likely either monitor refresh harmonic from the VEP stimulus, or amplifier-specific noise) and add a sentence to Section 3.4 or the Figure 4 caption.

## 3.5 [P2 / STRUCT] Discussion is short relative to the paper

**Where:** Section 4.

**Problem.** Discussion is 4 short subsections. For a paper with this much content (cross-construction comparison, cross-setup bridge, qualitative VEP analysis, three endpoints), Discussion should be ~25–35% of the manuscript body. Currently it's ~10%.

**Fix.** Expand Discussion to address:
1. Mechanism for the SSIM ordering (see 2.1).
2. Implications of the reactivity gap (see 2.2).
3. Why disc EEG outperforms all tEEG variants on raw alpha SNR (this is a counterintuitive finding worth discussing — in principle, the Laplacian should improve focal SNR, but this depends on alpha being focal vs broadly distributed).
4. Practical recommendations for which construction suits which use case (you said you'd not recommend one — fine — but discuss the tradeoffs explicitly).
5. Comparison of the harmonized pipeline against prior TCRE pipelines (Besio's group, Makeyev's group).

## 3.6 [P2 / SCI] Counterintuitive disc-EEG > tEEG alpha SNR is unaddressed

**Where:** Section 3.2 and Discussion.

**Problem.** Disc EEG median alpha SNR is **4.33 dB** — higher than every tEEG class. The whole motivation for the surface Laplacian is that it should improve focal-source SNR by suppressing volume-conducted broadband activity. If disc EEG has higher alpha SNR than tEEG across all three constructions, this either:
1. Contradicts the expected behavior of the Laplacian (worth discussing), or
2. Reflects the fact that posterior alpha is *not* a focal source — it is broadly distributed across occipital cortex, so a high-pass spatial filter rejects much of it.

The second is the conventional explanation and is well-established in the surface-Laplacian literature. **Say so.** Otherwise the data look like they undermine TCRE.

**Fix.** Add 2–3 sentences to Discussion (or end of Section 3.2) explaining that posterior alpha's broadly-distributed source structure is expected to attenuate under a Laplacian filter, and that the SNR comparison is for context rather than evidence against TCRE. Cite Tenke & Kayser or similar.

---

# Part 4 — Methodological Reporting (P2)

## 4.1 [P2 / REPRO] Recording paradigm under-specified

**Where:** Section 2.1.

**Problem.** "All sessions were recorded under a checkerboard-VEP plus eyes-open/eyes-closed paradigm." This is one sentence. Reviewers and replicators need:
- Number of eyes-open epochs, eyes-closed epochs, and durations of each.
- Order randomization or fixed?
- Checkerboard parameters: check size (degrees of visual angle), reversal rate (Hz), display, viewing distance, luminance.
- Number of VEP trials averaged.

**Fix.** Add a "2.1.2 Experimental Paradigm" subsection with these parameters.

## 4.2 [P2 / REPRO] Software / package versions not reported

**Where:** Methods, Data Availability.

**Problem.** No mention of Python version, scipy version, MNE version (if used), numpy version, or analysis script repository.

**Fix.** Add a single line in Section 2.6 listing key dependencies and versions, e.g.: "Analyses used Python 3.X, NumPy X.Y, SciPy X.Y, MNE-Python X.Y, scikit-image X.Y (for SSIM)."

## 4.3 [P2 / REPRO] SSIM parameters not fully specified

**Where:** Section 2.5.

**Problem.** "Spectrogram SSIM: structural similarity [5] between the per-pair tEEG and eEEG spectrograms (n_per_seg = 2048, f_max = 45 Hz)." Missing:
- Time-window duration and overlap of the spectrograms used as SSIM inputs.
- SSIM kernel size (default 11×11 in scikit-image).
- SSIM dynamic range parameter (`data_range`).
- Whether spectrograms were log-transformed before SSIM or used in linear power.

**Fix.** Add these parameters. SSIM on log vs linear spectrograms produces materially different numbers — this matters.

## 4.4 [P2 / REPRO] Alpha SNR neighbor bands under-specified

**Where:** Section 2.5.

**Problem.** "P_neighbor from flanking sub-bands of equal width" — but the alpha band is 8–13 Hz (5 Hz wide). What are the flanking bands? 3–8 Hz and 13–18 Hz? 5–8 Hz and 13–16 Hz? The exact choice changes the numerator/denominator ratio and thus SNR in dB.

**Fix.** State the exact frequency bounds of the two neighbor bands.

## 4.5 [P2 / REPRO] WARN-flagged subjects in main analysis

**Where:** Section 2.4 and Section 3.1.

**Problem.** WARN-flagged subjects (LF02, SF01, SF02, SF07, G05, G06, G08, G10, G13 — that's 9 of 23) are retained in the analysis cohort. This is a defensible choice, but you should:
1. State explicitly that WARN subjects are retained.
2. Show in supplement that conclusions hold when WARN subjects are excluded (sensitivity analysis).

**Fix.** Add a sentence to Section 2.4 ("WARN-flagged subjects are retained in primary analyses; supplement Section S[X] reports a PASS-only sensitivity analysis.") and add the supplementary table.

---

# Part 5 — Section-by-section detailed notes

## 5.1 Title and Author block

| Item | Note |
|------|------|
| Title | Long but acceptable for a methods paper. Could be tightened: "Methods-Harmonized Comparison of TCRE Constructions for Non-Invasive EEG: Saline-Soaked Felt, Conductive Gel, and Paste". |
| Author 3 | `Maryam [TBD: Surname]` — **MUST FIX**. |
| Affiliation | All authors listed as `1` (URI). Confirm there are no second affiliations (e.g., joint appointments, hospital affiliations for medical co-authors). |
| Corresponding author | `skhodabakhsh@uri.edu` — confirm this is the desired contact and that the field requires no additional information for *Sensors* (e.g., ORCID iDs are required for all authors at *Sensors*). |
| ORCID | Not present for any author. Required by *Sensors*. |

## 5.2 Abstract

See **2.8** above. Additional notes:
- Word count: ~360 words. *Sensors* limit is 200 words. **You will need to cut ~150 words.**

## 5.3 Keywords

Current: `tripolar concentric ring electrode; surface Laplacian; EEG; alpha rhythm; visual evoked potential; methods harmonization; electrolyte; signal quality`. Good. Consider adding `Berger effect` and `structural similarity` if the journal allows ≥10.

## 5.4 Section 1 — Introduction

See **3.1** above. Specific line-level notes:
- Paragraph 1, sentence 2: "The Laplacian acts as a spatial high-pass filter and has been shown to suppress volume conduction, sharpen focal source detection, and reduce reference-related artifact relative to disc EEG [1, 2]." This is a triple-claim sentence with two general references. Better: split into two sentences with one citation each, and add Nunez & Srinivasan or Tenke & Kayser as a third reference for the surface-Laplacian framework specifically.
- The contribution is stated only at the end. Consider stating it earlier (paragraph 2 or 3) so the reader knows what to look for.

## 5.5 Section 2 — Methods

### 2.1 Subjects and Cohort
- Define "long-felt" vs "short-felt" (see **2.6.1**).
- Add demographics (see **2.6.2**).
- Add experimental paradigm details (see **4.1**).
- Table 1: caption mentions "Auto-generated by paper_sensors/build_paper_stats.py; re-run if the cohort or QC thresholds change." This is helpful internal documentation but **probably should not appear in the published version** — move to a code comment.

### 2.2 Recording Setups
- Add skin prep and impedance protocol (see **2.6.3**).
- Mention the 1/187 hardware scaling on gel-tEEG **here**, not first in 2.5.
- "0.0488 µV/bit on a subset of gel sessions recorded with a different amplifier model" — name the model and list the specific subjects affected.
- State the disc reference electrode placement (mastoid) and ground electrode placement (currently not specified).

### 2.3 Preprocessing Pipeline
- "60 Hz IIR notch filter (Q = 30)" — specify filter type (Butterworth? notch IIR with what topology?). Q = 30 implies bandwidth = 2 Hz, which is fine for line-noise rejection.
- "Zero-phase FIR bandpass 0.05–55 Hz, Hamming window" — state filter order or transition bandwidth.
- Variant A description "Daubechies-8 level-3 wavelet" — if this was the original gel pipeline, cite the paper or thesis where it appeared.

### 2.4 Quality Control
- Make explicit that WARN subjects are kept (see **4.5**).
- Define the 5% clipping threshold rationale (why 5% and not 1% or 10%?).
- The "two simultaneous WARN flags = FAIL" rule should be explicit about which two flags can co-occur (clip and SNR are the only two soft flags listed).

### 2.5 Endpoints
- Add neighbor band frequencies (see **4.4**).
- Add SSIM parameters (see **4.3**).
- VEP "peak-to-peak amplitude" — but Table 3 reports per-component amplitudes (N75, P100, N135), not peak-to-peak. Pick one and use it consistently. Peak-to-peak is more standard for VEP magnitude reporting (e.g., N75-to-P100, P100-to-N135).

### 2.6 Statistical Analysis
- See **1.1** for the paired-vs-independent fix.
- See **2.7** for the BCa resamples.

## 5.6 Section 3 — Results

### 3.1 Cohort and QC Outcomes
- Table 2 is excellent. Two minor notes:
  - "Mean tEEG SNR (dB)" — define how this is averaged across the multiple tEEG channels per subject (mean? median? Which channels?).
  - The "Ch" column shows 11 for felt and 7 for gel, which is the channel count. Caption could note this.

### 3.2 Alpha-Band Detection (SNR)
- Statistical fix per **1.1**.
- Counterintuitive disc-EEG result needs discussion (**3.6**).

### 3.3 Eyes-Closed/Eyes-Open Reactivity
- Felt-vs-gel reactivity gap needs explanation (**2.2**).
- Two paste non-responders need acknowledgment (**2.3**).

### 3.4 PSD Shape
- Expand body text (**3.3**).
- Address 25 Hz disc-EEG peak (**3.4**).

### 3.5 SSIM
- Add mechanism paragraph (**2.1**).
- Note: text reports "Felt vs Paste at U = 9, p = 0.002, d = -2.69 (n_a = n_b = 10)" — this is a within-felt-setup comparison and should be Wilcoxon signed-rank (see **1.1**).

### 3.6 Cross-Setup Bridge
- Soften "no between-setup differences" claim (**3.2**).

### 3.7 VEPs
- Major restructuring needed (**2.4**).

### 3.8 Gel-Cohort VEP Sanity Check
- This section is honest and useful. Two suggestions:
  1. State up front (one sentence) why three sessions specifically were chosen for the qualitative figure.
  2. Move detailed gel-VEP figures (Fig 8–11) to supplement; keep one summary figure (Fig 12, the unified-pipeline overview) in the main text. This shortens the results section and keeps focus on the cross-construction story.

## 5.7 Section 4 — Discussion

See **3.5** and child issues. Specific line-level notes:
- Section 4.4 ("What This Enables") mentions internal source files: `src/eeg_analysis.py`, `src/comparison_analysis.py`. **Move these to Data Availability.** Discussion should describe scientific implications, not file paths.

## 5.8 Section 5 — Conclusions

- Mostly fine. Don't restate the result numbers a third time (they're in Abstract and Results) — instead, use Conclusions to state implications and next steps cleanly.
- "Future work will (i) grow both cohorts beyond pilot scale, especially balancing the long-felt arm, and (ii) re-collect a gel-setup VEP dataset on a single amplifier under the harmonized pipeline." — Good. Add (iii) impedance logging per ring, (iv) operator-randomization within construction.

## 5.9 References

**Current count: 5.** Methods papers in this space typically have 25–40. Add at least the following:

| # | Reference | Why |
|---|-----------|-----|
| R1 | Nunez & Srinivasan, *Electric Fields of the Brain* (2nd ed., 2006) | Definitive surface-Laplacian framework |
| R2 | Tenke & Kayser, *Clin Neurophysiol* 2012 | Surface Laplacian validation methodology |
| R3 | Makeyev et al. (any of his TCRE papers, 2016+) | TCRE Laplacian estimation literature |
| R4 | Boudet et al. on TCRE | Prior TCRE comparison study |
| R5 | Klimesch, *Brain Res Rev* 1999 | Alpha-band quantification benchmark |
| R6 | Babiloni et al. on alpha rhythms | Eyes-closed/open reactivity benchmark |
| R7 | Odom et al. or ISCEV VEP standard | VEP normative latency/amplitude ranges |
| R8 | Berger 1929 (the original) | Polite historical citation |
| R9 | At least one more recent Besio TCRE paper (2010s+) | Show TCRE field is active |
| R10 | Fitzgibbon et al. or similar on EEG quality control | QC framework grounding |

Plus journal-specific style: *Sensors* uses MDPI numbered style — confirm that reference formatting is correct (currently looks bibtex-default rather than MDPI).

## 5.10 Figures

| Figure | Issues |
|--------|--------|
| Fig 1 (Alpha SNR boxplot) | Labels are squished. Increase x-axis label font size. |
| Fig 2 (Reactivity boxplot) | Y-axis is capped with annotation — good. Confirm the 7 outliers are documented in supplement. |
| Fig 3 (PSD eyes-open vs closed) | Three subpanels are thin. Consider stacking vertically or making each subpanel larger. |
| Fig 4 (Normalized tEEG PSDs) | 25 Hz peak unexplained (see **3.4**). |
| Fig 5 (SSIM by class) | Disc EEG bar shows n=0 because disc has no paired tEEG/eEEG distinction — caption should explain why disc is shown as empty. |
| Fig 6 (Paste cross-setup bridge) | Title overlaps with axis labels at top. Re-render. |
| Fig 7 (Felt VEP grand averages) | LF02 panel shows clipped/noisy felt-tEEG; consider replacing with paste-tEEG only. |
| Figs 8–12 | **Not rendering — fix path or includegraphics command** (see **1.4**). |

## 5.11 Tables

| Table | Issues |
|-------|--------|
| Table 1 | Auto-generation comment in caption should be removed for published version. |
| Table 2 | Excellent. Minor: define "Mean tEEG SNR" averaging method. |
| Table 3 | Restructure to flag boundary-pinned latencies as `n.r.` (not resolved); separate clean from noisy rows. |

---

# Part 6 — Suggested Action Plan for Claude Code

If you give this whole report to Claude Code, here is a recommended execution order:

1. **First pass — mechanical fixes (1–2 hours).**
   - Resolve all `[TBD: ...]` placeholders that you have answers for; mark the ones you don't with explicit `% TODO(shayan):` comments.
   - Fix the broken figure includes (Section 1.4).
   - Remove "auto-generated by ..." commentary from published-version captions.
   - Update preamble for *Sensors* MDPI template if not already on it.

2. **Second pass — statistical correctness (~2 hours).**
   - Implement paired Wilcoxon signed-rank for within-setup comparisons in `paper_sensors/build_paper_stats.py`.
   - Switch BCa resamples from 1,000 to 10,000.
   - Re-emit Tables 2 and any inline numerics that change.
   - Update Methods Section 2.6 with the correct test descriptions.

3. **Third pass — content additions (~3–5 hours, requires Shayan input).**
   - Write the SSIM mechanism paragraph (2.1).
   - Write the reactivity discussion paragraph (2.2).
   - Identify the two paste non-responders and add the footnote (2.3).
   - Decide VEP scoping (2.4 — recommend Option A: move to supplement).
   - Add subject demographics (2.6.2) — needs IRB-approved data from Shayan.
   - Add skin prep / impedance protocol (2.6.3) — Shayan to supply from lab notes.
   - Add experimental paradigm details (4.1) — Shayan to supply.

4. **Fourth pass — references and benchmarking (~2 hours).**
   - Add the references in Section 5.9 of this report.
   - Add benchmark numbers from the cited references (2.5).
   - Confirm MDPI numerical citation style.

5. **Fifth pass — polish (~1–2 hours).**
   - Rewrite abstract per 2.8 and trim to 200 words.
   - Expand introduction per 3.1.
   - Expand Section 3.4 body text per 3.3.
   - Remove file paths from Discussion 4.4.
   - Final read-through for sentence-length and clause-density issues.

6. **Pre-submission checklist.**
   - All TBDs resolved (1.2, 1.3).
   - All figures render (1.4).
   - Word counts within journal limits (abstract ≤ 200, main text within journal range).
   - All authors have ORCID iDs.
   - Reference style matches journal (MDPI numbered if *Sensors*).
   - Code repo URL is live and contains a README.

---

# Part 7 — Things this report does NOT address

For transparency:

- **Cohort growth.** If you can add 5–10 more long-felt subjects before submission, do; that single change addresses 2.4 and most of 2.2 simultaneously. The methods-paper framing survives without it.
- **Re-running the gel VEP cohort.** This is mentioned in 5.9 as future work and is correct — out of scope for this revision.
- **Replication on a third construction (e.g., dry electrodes).** Out of scope; possibly a follow-up paper.
- **Formal multiple-comparisons correction.** You explicitly opt out as descriptive analysis. This is defensible for *Sensors* but a stricter reviewer might press; if so, Holm–Bonferroni on the three primary endpoints would be a small concession.

---

**End of report.**
