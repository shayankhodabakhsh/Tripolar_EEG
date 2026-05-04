# Auto-generated stats for paper/main.tex
Source: `scripts/build_paper_stats.py` on the pooled-felt cohort (felt = long + short post-QC; gel = QC-passing). Re-run to refresh.

## §3.1 Cohort post-QC
| Cohort | n loaded | Subjects |
|--|--|--|
| Felt (all) | 10 | SF01, SF02, SF04, SF03, SF05, LF01, LF02, SF06, LF03, SF07 |
| &nbsp;&nbsp;Long felt | 3 | LF01, LF02, LF03 |
| &nbsp;&nbsp;Short felt | 7 | SF01, SF02, SF04, SF03, SF05, SF06, SF07 |
| Gel       | 13  | G02, G10, G11, G05, G06, G07, G08, G13, G09, G12, G01, G03, G04 |

## §3.2–§3.5 Per-class medians (BCa 95% CI)
| Class | Alpha SNR (dB) | Reactivity (closed/open) | SSIM |
|--|--|--|--|
| Felt tEEG | 1.72 [0.93, 2.31] (n=10) | 1.49 [1.10, 1.87] (n=10) | 0.307 [0.266, 0.338] (n=10) |
| Gel tEEG | 1.43 [0.19, 3.42] (n=13) | 3.63 [1.64, 5.32] (n=13) | 0.519 [0.470, 0.694] (n=13) |
| Paste tEEG | 2.97 [2.09, 5.65] (n=23) | 3.43 [2.30, 6.88] (n=23) | 0.710 [0.619, 0.758] (n=23) |
| Felt eEEG | 1.75 [0.98, 2.35] (n=10) | 1.42 [0.70, 4.63] (n=10) | 0.307 [0.266, 0.338] (n=10) |
| Gel eEEG | 2.11 [1.03, 5.35] (n=13) | 4.72 [1.97, 12.16] (n=13) | 0.519 [0.470, 0.694] (n=13) |
| Paste eEEG | 3.24 [2.65, 6.54] (n=23) | 3.26 [2.36, 8.16] (n=23) | 0.710 [0.619, 0.758] (n=23) |
| Disc EEG | 4.33 [2.91, 6.99] (n=23) | 4.72 [2.41, 8.62] (n=23) | n/a |

## §3.3 Reactivity > 1 (Berger effect detected) per class
| Class | n with reactivity > 1 | n total |
|--|--|--|
| Felt tEEG | 10 | 10 |
| Gel tEEG | 13 | 13 |
| Paste tEEG | 21 | 23 |
| Felt eEEG | 7 | 10 |
| Gel eEEG | 11 | 13 |
| Paste eEEG | 22 | 23 |
| Disc EEG | 20 | 23 |

## §3.6 Paste-TCRE cross-setup bridge (Mann–Whitney U)
| Metric | n_felt | n_gel | U | p | Cohen d |
|--|--|--|--|--|--|
| alpha_snr | 10 | 13 | 89.0 | 0.145 | 0.51 |
| alpha_reactivity | 10 | 13 | 46.0 | 0.251 | -0.39 |
| ssim | 10 | 13 | 69.0 | 0.828 | 0.01 |

## All cross-construction comparisons
Within-setup contrasts (paired by subject) use Wilcoxon signed-rank with Cohen's $d_z$; between-setup contrasts use Mann--Whitney U with Cohen's $d$.

| Contrast | Test | n_a | n_b | stat | p | d / d_z |
|--|--|--|--|--|--|--|
| Felt_tEEG_vs_Paste_tEEG_alpha_snr | Wilcoxon signed-rank | 10 | 10 | 1.0 | 0.004 | -1.34 |
| Felt_tEEG_vs_Paste_tEEG_alpha_reactivity | Wilcoxon signed-rank | 10 | 10 | 5.0 | 0.020 | -0.58 |
| Felt_tEEG_vs_Paste_tEEG_ssim | Wilcoxon signed-rank | 10 | 10 | 1.0 | 0.004 | -1.95 |
| Gel_tEEG_vs_Paste_tEEG_alpha_snr | Wilcoxon signed-rank | 13 | 13 | 25.0 | 0.168 | -0.41 |
| Gel_tEEG_vs_Paste_tEEG_alpha_reactivity | Wilcoxon signed-rank | 13 | 13 | 23.0 | 0.127 | -0.43 |
| Gel_tEEG_vs_Paste_tEEG_ssim | Wilcoxon signed-rank | 13 | 13 | 7.0 | 0.005 | -1.03 |
| Felt_eEEG_vs_Paste_eEEG_alpha_snr | Wilcoxon signed-rank | 10 | 10 | 8.0 | 0.049 | -0.76 |
| Felt_eEEG_vs_Paste_eEEG_alpha_reactivity | Wilcoxon signed-rank | 10 | 10 | 14.0 | 0.193 | -0.38 |
| Felt_eEEG_vs_Paste_eEEG_ssim | Wilcoxon signed-rank | 10 | 10 | 1.0 | 0.004 | -1.95 |
| Gel_eEEG_vs_Paste_eEEG_alpha_snr | Wilcoxon signed-rank | 13 | 13 | 19.0 | 0.068 | -0.53 |
| Gel_eEEG_vs_Paste_eEEG_alpha_reactivity | Wilcoxon signed-rank | 13 | 13 | 32.0 | 0.376 | -0.28 |
| Gel_eEEG_vs_Paste_eEEG_ssim | Wilcoxon signed-rank | 13 | 13 | 7.0 | 0.005 | -1.03 |
| Felt_tEEG_vs_Gel_tEEG_alpha_snr | Mann-Whitney U | 10 | 13 | 68.0 | 0.877 | -0.00 |
| Felt_tEEG_vs_Gel_tEEG_alpha_reactivity | Mann-Whitney U | 10 | 13 | 26.0 | 0.017 | -1.11 |
| Felt_tEEG_vs_Gel_tEEG_ssim | Mann-Whitney U | 10 | 13 | 0.0 | 0.000 | -2.52 |
| Felt_eEEG_vs_Gel_eEEG_alpha_snr | Mann-Whitney U | 10 | 13 | 50.0 | 0.369 | -0.58 |
| Felt_eEEG_vs_Gel_eEEG_alpha_reactivity | Mann-Whitney U | 10 | 13 | 34.0 | 0.059 | -0.81 |
| Felt_eEEG_vs_Gel_eEEG_ssim | Mann-Whitney U | 10 | 13 | 0.0 | 0.000 | -2.52 |
| Paste_bridge_felt_vs_gel_alpha_snr | Mann-Whitney U | 10 | 13 | 89.0 | 0.145 | 0.51 |
| Paste_bridge_felt_vs_gel_alpha_reactivity | Mann-Whitney U | 10 | 13 | 46.0 | 0.251 | -0.39 |
| Paste_bridge_felt_vs_gel_ssim | Mann-Whitney U | 10 | 13 | 69.0 | 0.828 | 0.01 |
