# Analysis Report — Refusal (qwen-2.5-3b)

- Generated: 2026-07-08T19:21:17
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.834 | [0.740, 0.877] | 32.3 |
| 1 | 0.858 | [0.782, 0.890] | 29.8 |
| 2 | 0.891 | [0.836, 0.913] | 26.1 |
| 3 | 0.879 | [0.830, 0.901] | 27.9 |
| 4 | 0.893 | [0.851, 0.909] | 26.2 |
| 5 | 0.893 | [0.850, 0.911] | 26.1 |
| 6 | 0.893 | [0.855, 0.909] | 25.9 |
| 7 | 0.900 | [0.867, 0.913] | 25.0 |
| 8 | 0.884 | [0.850, 0.894] | 27.2 |
| 9 | 0.880 | [0.845, 0.892] | 27.7 |
| 10 | 0.867 | [0.831, 0.881] | 29.1 |
| 11 | 0.876 | [0.843, 0.889] | 28.1 |
| 12 | 0.882 | [0.850, 0.893] | 27.5 |
| 13 | 0.902 | [0.871, 0.913] | 24.9 |
| 14 | 0.903 | [0.872, 0.914] | 24.7 |
| 15 | 0.897 | [0.866, 0.910] | 25.5 |
| 16 | 0.898 | [0.869, 0.908] | 25.6 |
| 17 | 0.904 | [0.878, 0.913] | 24.8 |
| 18 | 0.907 | [0.885, 0.914] | 24.6 |
| 19 | 0.906 | [0.883, 0.914] | 24.7 |
| 20 | 0.922 | [0.904, 0.927] | 22.4 |
| 21 | 0.926 | [0.908, 0.932] | 21.8 |
| 22 | 0.930 | [0.910, 0.936] | 21.2 |
| 23 | 0.930 | [0.911, 0.937] | 21.0 |
| 24 | 0.931 | [0.911, 0.937] | 20.9 |
| 25 | 0.933 | [0.915, 0.939] | 20.5 |
| 26 | 0.929 | [0.908, 0.937] | 21.1 |
| 27 | 0.919 | [0.896, 0.927] | 22.6 |
| 28 | 0.913 | [0.889, 0.922] | 23.4 |
| 29 | 0.909 | [0.884, 0.918] | 24.0 |
| 30 | 0.898 | [0.872, 0.909] | 25.5 |
| 31 | 0.898 | [0.872, 0.908] | 25.5 |
| 32 | 0.898 | [0.873, 0.907] | 25.6 |
| 33 | 0.898 | [0.874, 0.905] | 25.7 |
| 34 | 0.900 | [0.877, 0.906] | 25.5 |
| 35 | 0.900 | [0.876, 0.905] | 25.6 |

## Interpretation

- Lowest mean cosine-to-global: **0.834** at layer 0.
- Highest mean angle-to-global: **32.3°** at layer 0.
- Late-layer (final third) mean cosine: **0.911** (layers 24–35).

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction). Note: minima in early layers may reflect token/surface statistics rather than concept representations — contrast with the late-layer aggregate, where behavior is read out.

## Figures

- `heatmap_refusal_qwen-2.5-3b.pdf` / `heatmap_refusal_qwen-2.5-3b.png`
- `dispersion_profile_refusal_qwen-2.5-3b.pdf` / `dispersion_profile_refusal_qwen-2.5-3b.png`
- `transfer_matrix_refusal_qwen-2.5-3b_layer010.pdf` / `transfer_matrix_refusal_qwen-2.5-3b_layer010.png`
- `geometry_pca_refusal_qwen-2.5-3b.pdf` / `geometry_pca_refusal_qwen-2.5-3b.png`
