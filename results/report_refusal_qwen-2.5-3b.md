# Analysis Report — Refusal (qwen-2.5-3b)

- Generated: 2026-07-06T18:58:59
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.834 | [0.719, 0.902] | 32.3 |
| 1 | 0.858 | [0.754, 0.918] | 29.8 |
| 2 | 0.891 | [0.819, 0.934] | 26.1 |
| 3 | 0.879 | [0.814, 0.920] | 27.9 |
| 4 | 0.893 | [0.836, 0.930] | 26.2 |
| 5 | 0.893 | [0.830, 0.933] | 26.1 |
| 6 | 0.893 | [0.824, 0.935] | 25.9 |
| 7 | 0.900 | [0.834, 0.939] | 25.0 |
| 8 | 0.884 | [0.816, 0.926] | 27.2 |
| 9 | 0.880 | [0.813, 0.923] | 27.7 |
| 10 | 0.867 | [0.793, 0.914] | 29.1 |
| 11 | 0.876 | [0.806, 0.921] | 28.1 |
| 12 | 0.882 | [0.818, 0.924] | 27.5 |
| 13 | 0.902 | [0.848, 0.940] | 24.9 |
| 14 | 0.903 | [0.846, 0.942] | 24.7 |
| 15 | 0.897 | [0.835, 0.939] | 25.5 |
| 16 | 0.898 | [0.849, 0.932] | 25.6 |
| 17 | 0.904 | [0.859, 0.935] | 24.8 |
| 18 | 0.907 | [0.880, 0.933] | 24.6 |
| 19 | 0.906 | [0.876, 0.933] | 24.7 |
| 20 | 0.922 | [0.896, 0.949] | 22.4 |
| 21 | 0.926 | [0.902, 0.951] | 21.8 |
| 22 | 0.930 | [0.902, 0.953] | 21.2 |
| 23 | 0.930 | [0.896, 0.957] | 21.0 |
| 24 | 0.931 | [0.896, 0.958] | 20.9 |
| 25 | 0.933 | [0.898, 0.959] | 20.5 |
| 26 | 0.929 | [0.887, 0.956] | 21.1 |
| 27 | 0.919 | [0.874, 0.949] | 22.6 |
| 28 | 0.913 | [0.864, 0.944] | 23.4 |
| 29 | 0.909 | [0.860, 0.941] | 24.0 |
| 30 | 0.898 | [0.847, 0.932] | 25.5 |
| 31 | 0.898 | [0.851, 0.931] | 25.5 |
| 32 | 0.898 | [0.854, 0.932] | 25.6 |
| 33 | 0.898 | [0.861, 0.930] | 25.7 |
| 34 | 0.900 | [0.866, 0.929] | 25.5 |
| 35 | 0.900 | [0.872, 0.925] | 25.6 |

## Interpretation

- Lowest mean cosine-to-global: **0.834** at layer 0.
- Highest mean angle-to-global: **32.3°** at layer 0.

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction).

## Figures

- `heatmap_refusal_qwen-2.5-3b.pdf` / `heatmap_refusal_qwen-2.5-3b.png`
- `dispersion_profile_refusal_qwen-2.5-3b.pdf` / `dispersion_profile_refusal_qwen-2.5-3b.png`
- `transfer_matrix_refusal_qwen-2.5-3b_layer000.pdf` / `transfer_matrix_refusal_qwen-2.5-3b_layer000.png`
- `geometry_pca_refusal_qwen-2.5-3b.pdf` / `geometry_pca_refusal_qwen-2.5-3b.png`
