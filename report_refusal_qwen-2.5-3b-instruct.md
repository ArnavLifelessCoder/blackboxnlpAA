# Analysis Report — Refusal (qwen-2.5-3b-instruct)

- Generated: 2026-07-06T18:38:58
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.837 | [0.725, 0.904] | 32.1 |
| 1 | 0.859 | [0.758, 0.918] | 29.7 |
| 2 | 0.897 | [0.831, 0.936] | 25.5 |
| 3 | 0.874 | [0.811, 0.914] | 28.6 |
| 4 | 0.887 | [0.832, 0.923] | 27.1 |
| 5 | 0.890 | [0.834, 0.926] | 26.7 |
| 6 | 0.900 | [0.847, 0.935] | 25.2 |
| 7 | 0.904 | [0.847, 0.938] | 24.8 |
| 8 | 0.892 | [0.836, 0.929] | 26.3 |
| 9 | 0.889 | [0.826, 0.931] | 26.6 |
| 10 | 0.879 | [0.815, 0.921] | 27.9 |
| 11 | 0.883 | [0.824, 0.924] | 27.3 |
| 12 | 0.893 | [0.846, 0.928] | 26.3 |
| 13 | 0.917 | [0.880, 0.943] | 23.1 |
| 14 | 0.907 | [0.870, 0.936] | 24.5 |
| 15 | 0.907 | [0.875, 0.932] | 24.6 |
| 16 | 0.896 | [0.872, 0.919] | 26.2 |
| 17 | 0.906 | [0.884, 0.928] | 24.8 |
| 18 | 0.903 | [0.875, 0.929] | 25.2 |
| 19 | 0.905 | [0.881, 0.930] | 24.9 |
| 20 | 0.917 | [0.881, 0.947] | 23.0 |
| 21 | 0.917 | [0.889, 0.944] | 23.2 |
| 22 | 0.924 | [0.899, 0.948] | 22.2 |
| 23 | 0.929 | [0.902, 0.956] | 21.3 |
| 24 | 0.934 | [0.908, 0.960] | 20.5 |
| 25 | 0.940 | [0.914, 0.965] | 19.5 |
| 26 | 0.935 | [0.908, 0.963] | 20.2 |
| 27 | 0.934 | [0.907, 0.961] | 20.5 |
| 28 | 0.933 | [0.906, 0.960] | 20.6 |
| 29 | 0.930 | [0.901, 0.959] | 21.1 |
| 30 | 0.926 | [0.894, 0.957] | 21.7 |
| 31 | 0.922 | [0.889, 0.956] | 22.1 |
| 32 | 0.922 | [0.888, 0.955] | 22.2 |
| 33 | 0.920 | [0.886, 0.953] | 22.6 |
| 34 | 0.920 | [0.888, 0.952] | 22.5 |
| 35 | 0.920 | [0.889, 0.952] | 22.5 |

## Interpretation

- Lowest mean cosine-to-global: **0.837** at layer 0.
- Highest mean angle-to-global: **32.1°** at layer 0.

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction).

## Figures

- `heatmap_refusal_qwen-2.5-3b-instruct.pdf` / `heatmap_refusal_qwen-2.5-3b-instruct.png`
- `dispersion_profile_refusal_qwen-2.5-3b-instruct.pdf` / `dispersion_profile_refusal_qwen-2.5-3b-instruct.png`
- `transfer_matrix_refusal_qwen-2.5-3b-instruct_layer000.pdf` / `transfer_matrix_refusal_qwen-2.5-3b-instruct_layer000.png`
- `geometry_pca_refusal_qwen-2.5-3b-instruct.pdf` / `geometry_pca_refusal_qwen-2.5-3b-instruct.png`
