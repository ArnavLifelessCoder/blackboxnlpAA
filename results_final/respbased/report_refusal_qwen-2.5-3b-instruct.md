# Analysis Report — Refusal (qwen-2.5-3b-instruct)

- Generated: 2026-07-12T02:51:24
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.854 | [0.791, 0.885] | 29.9 |
| 1 | 0.887 | [0.842, 0.901] | 26.8 |
| 2 | 0.907 | [0.876, 0.914] | 24.5 |
| 3 | 0.922 | [0.898, 0.929] | 22.3 |
| 4 | 0.920 | [0.895, 0.926] | 22.5 |
| 5 | 0.927 | [0.902, 0.934] | 21.3 |
| 6 | 0.913 | [0.886, 0.922] | 23.3 |
| 7 | 0.924 | [0.898, 0.932] | 21.5 |
| 8 | 0.920 | [0.895, 0.928] | 22.0 |
| 9 | 0.916 | [0.891, 0.924] | 22.6 |
| 10 | 0.923 | [0.900, 0.930] | 21.7 |
| 11 | 0.911 | [0.887, 0.920] | 23.3 |
| 12 | 0.900 | [0.872, 0.910] | 24.7 |
| 13 | 0.902 | [0.875, 0.913] | 24.2 |
| 14 | 0.906 | [0.881, 0.916] | 23.7 |
| 15 | 0.910 | [0.886, 0.919] | 23.3 |
| 16 | 0.921 | [0.901, 0.928] | 21.9 |
| 17 | 0.923 | [0.903, 0.930] | 21.6 |
| 18 | 0.919 | [0.899, 0.926] | 22.2 |
| 19 | 0.918 | [0.898, 0.925] | 22.3 |
| 20 | 0.923 | [0.904, 0.930] | 21.5 |
| 21 | 0.924 | [0.905, 0.930] | 21.4 |
| 22 | 0.915 | [0.896, 0.922] | 22.6 |
| 23 | 0.915 | [0.896, 0.923] | 22.6 |
| 24 | 0.917 | [0.897, 0.925] | 22.3 |
| 25 | 0.912 | [0.892, 0.920] | 23.0 |
| 26 | 0.912 | [0.890, 0.920] | 23.1 |
| 27 | 0.912 | [0.892, 0.920] | 22.8 |
| 28 | 0.910 | [0.889, 0.917] | 23.1 |
| 29 | 0.912 | [0.891, 0.920] | 22.9 |
| 30 | 0.910 | [0.889, 0.918] | 23.1 |
| 31 | 0.907 | [0.886, 0.914] | 23.6 |
| 32 | 0.898 | [0.876, 0.906] | 24.7 |
| 33 | 0.890 | [0.868, 0.897] | 25.7 |
| 34 | 0.881 | [0.857, 0.889] | 26.8 |
| 35 | 0.887 | [0.863, 0.901] | 24.8 |

## Interpretation

- Lowest mean cosine-to-global: **0.854** at layer 0.
- Highest mean angle-to-global: **29.9°** at layer 0.
- Late-layer (final third) mean cosine: **0.904** (layers 24–35).

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction). Note: minima in early layers may reflect token/surface statistics rather than concept representations — contrast with the late-layer aggregate, where behavior is read out.

## Figures

- `heatmap_refusal_qwen-2.5-3b-instruct.pdf` / `heatmap_refusal_qwen-2.5-3b-instruct.png`
- `dispersion_profile_refusal_qwen-2.5-3b-instruct.pdf` / `dispersion_profile_refusal_qwen-2.5-3b-instruct.png`
- `transfer_matrix_refusal_qwen-2.5-3b-instruct_layer034.pdf` / `transfer_matrix_refusal_qwen-2.5-3b-instruct_layer034.png`
- `geometry_pca_refusal_qwen-2.5-3b-instruct.pdf` / `geometry_pca_refusal_qwen-2.5-3b-instruct.png`
