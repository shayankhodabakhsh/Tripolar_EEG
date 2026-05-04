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
| Felt tEEG | 1.72 [0.93, 2.31] (n=10) | 1.49 [1.10, 1.87] (n=10) | 0.307 [0.284, 0.347] (n=10) |
| Gel tEEG | 1.43 [0.19, 3.42] (n=13) | 3.63 [1.64, 5.32] (n=13) | 0.519 [0.470, 0.694] (n=13) |
| Paste tEEG | 2.97 [1.75, 5.65] (n=23) | 3.43 [2.30, 5.79] (n=23) | 0.710 [0.628, 0.758] (n=23) |
| Felt eEEG | 1.75 [0.98, 2.35] (n=10) | 1.42 [0.70, 4.63] (n=10) | 0.307 [0.284, 0.347] (n=10) |
| Gel eEEG | 2.11 [1.03, 5.35] (n=13) | 4.72 [1.97, 12.16] (n=13) | 0.519 [0.470, 0.694] (n=13) |
| Paste eEEG | 3.24 [2.65, 6.54] (n=23) | 3.26 [2.36, 5.64] (n=23) | 0.710 [0.628, 0.758] (n=23) |
| Disc EEG | 4.33 [2.85, 7.03] (n=23) | 4.72 [2.41, 8.62] (n=23) | n/a |

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

## All Mann–Whitney comparisons
| Contrast | n_a | n_b | U | p | Cohen d |
|--|--|--|--|--|--|
| Felt_tEEG_vs_Paste_tEEG_alpha_snr | 10 | 10 | 13.0 | 0.006 | -1.58 |
| Felt_tEEG_vs_Paste_tEEG_alpha_reactivity | 10 | 10 | 27.0 | 0.089 | -0.76 |
| Felt_tEEG_vs_Paste_tEEG_ssim | 10 | 10 | 9.0 | 0.002 | -2.69 |
| Gel_tEEG_vs_Paste_tEEG_alpha_snr | 13 | 13 | 68.0 | 0.412 | -0.44 |
| Gel_tEEG_vs_Paste_tEEG_alpha_reactivity | 13 | 13 | 62.0 | 0.259 | -0.52 |
| Gel_tEEG_vs_Paste_tEEG_ssim | 13 | 13 | 49.0 | 0.073 | -0.69 |
| Felt_tEEG_vs_Gel_tEEG_alpha_snr | 10 | 13 | 68.0 | 0.877 | -0.00 |
| Felt_tEEG_vs_Gel_tEEG_alpha_reactivity | 10 | 13 | 26.0 | 0.017 | -1.11 |
| Felt_tEEG_vs_Gel_tEEG_ssim | 10 | 13 | 0.0 | 0.000 | -2.52 |
| Paste_bridge_felt_vs_gel_alpha_snr | 10 | 13 | 89.0 | 0.145 | 0.51 |
| Paste_bridge_felt_vs_gel_alpha_reactivity | 10 | 13 | 46.0 | 0.251 | -0.39 |
| Paste_bridge_felt_vs_gel_ssim | 10 | 13 | 69.0 | 0.828 | 0.01 |
