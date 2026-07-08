# Analysis Report — Refusal (qwen-2.5-3b-instruct)

- Generated: 2026-07-08T19:03:52
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.837 | [0.744, 0.879] | 32.1 |
| 1 | 0.859 | [0.784, 0.890] | 29.7 |
| 2 | 0.897 | [0.845, 0.917] | 25.5 |
| 3 | 0.874 | [0.828, 0.897] | 28.6 |
| 4 | 0.887 | [0.846, 0.904] | 27.1 |
| 5 | 0.890 | [0.848, 0.906] | 26.7 |
| 6 | 0.900 | [0.863, 0.914] | 25.2 |
| 7 | 0.904 | [0.870, 0.915] | 24.8 |
| 8 | 0.892 | [0.858, 0.901] | 26.3 |
| 9 | 0.889 | [0.854, 0.900] | 26.6 |
| 10 | 0.879 | [0.845, 0.890] | 27.9 |
| 11 | 0.883 | [0.853, 0.892] | 27.3 |
| 12 | 0.893 | [0.866, 0.898] | 26.3 |
| 13 | 0.917 | [0.891, 0.921] | 23.1 |
| 14 | 0.907 | [0.882, 0.913] | 24.5 |
| 15 | 0.907 | [0.882, 0.913] | 24.6 |
| 16 | 0.896 | [0.874, 0.901] | 26.2 |
| 17 | 0.906 | [0.887, 0.911] | 24.8 |
| 18 | 0.903 | [0.887, 0.907] | 25.2 |
| 19 | 0.905 | [0.889, 0.910] | 24.9 |
| 20 | 0.917 | [0.904, 0.921] | 23.0 |
| 21 | 0.917 | [0.903, 0.921] | 23.2 |
| 22 | 0.924 | [0.909, 0.927] | 22.2 |
| 23 | 0.929 | [0.915, 0.932] | 21.3 |
| 24 | 0.934 | [0.921, 0.938] | 20.5 |
| 25 | 0.940 | [0.928, 0.943] | 19.5 |
| 26 | 0.935 | [0.922, 0.940] | 20.2 |
| 27 | 0.934 | [0.920, 0.938] | 20.5 |
| 28 | 0.933 | [0.918, 0.937] | 20.6 |
| 29 | 0.930 | [0.914, 0.934] | 21.1 |
| 30 | 0.926 | [0.909, 0.930] | 21.7 |
| 31 | 0.922 | [0.906, 0.927] | 22.1 |
| 32 | 0.922 | [0.905, 0.926] | 22.2 |
| 33 | 0.920 | [0.903, 0.923] | 22.6 |
| 34 | 0.920 | [0.904, 0.924] | 22.5 |
| 35 | 0.920 | [0.903, 0.923] | 22.5 |

## Interpretation

- Lowest mean cosine-to-global: **0.837** at layer 0.
- Highest mean angle-to-global: **32.1°** at layer 0.
- Late-layer (final third) mean cosine: **0.928** (layers 24–35).

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction). Note: minima in early layers may reflect token/surface statistics rather than concept representations — contrast with the late-layer aggregate, where behavior is read out.

## Figures

- `heatmap_refusal_qwen-2.5-3b-instruct.pdf` / `heatmap_refusal_qwen-2.5-3b-instruct.png`
- `dispersion_profile_refusal_qwen-2.5-3b-instruct.pdf` / `dispersion_profile_refusal_qwen-2.5-3b-instruct.png`
- `transfer_matrix_refusal_qwen-2.5-3b-instruct_layer003.pdf` / `transfer_matrix_refusal_qwen-2.5-3b-instruct_layer003.png`
- `geometry_pca_refusal_qwen-2.5-3b-instruct.pdf` / `geometry_pca_refusal_qwen-2.5-3b-instruct.png`
