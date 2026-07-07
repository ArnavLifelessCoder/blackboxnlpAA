# Analysis Report — Honesty (qwen-2.5-3b)

- Generated: 2026-07-06T18:59:06
- Mode: real activations
- Domains: factual_trivia, math, politics_opinion, personal_advice
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.543 | [0.239, 0.778] | 55.5 |
| 1 | 0.536 | [0.216, 0.781] | 55.9 |
| 2 | 0.531 | [0.201, 0.783] | 56.1 |
| 3 | 0.531 | [0.205, 0.779] | 56.2 |
| 4 | 0.528 | [0.201, 0.779] | 56.5 |
| 5 | 0.522 | [0.212, 0.765] | 57.0 |
| 6 | 0.525 | [0.230, 0.758] | 57.0 |
| 7 | 0.538 | [0.253, 0.767] | 56.0 |
| 8 | 0.535 | [0.248, 0.759] | 56.3 |
| 9 | 0.534 | [0.223, 0.778] | 56.1 |
| 10 | 0.542 | [0.230, 0.787] | 55.5 |
| 11 | 0.554 | [0.258, 0.790] | 54.7 |
| 12 | 0.547 | [0.235, 0.791] | 55.1 |
| 13 | 0.554 | [0.236, 0.798] | 54.6 |
| 14 | 0.551 | [0.212, 0.808] | 54.6 |
| 15 | 0.563 | [0.216, 0.817] | 53.6 |
| 16 | 0.572 | [0.209, 0.824] | 52.8 |
| 17 | 0.583 | [0.217, 0.832] | 51.9 |
| 18 | 0.585 | [0.202, 0.839] | 51.6 |
| 19 | 0.605 | [0.248, 0.837] | 50.3 |
| 20 | 0.624 | [0.266, 0.852] | 48.7 |
| 21 | 0.623 | [0.274, 0.848] | 48.9 |
| 22 | 0.621 | [0.267, 0.849] | 48.9 |
| 23 | 0.640 | [0.320, 0.843] | 47.9 |
| 24 | 0.649 | [0.296, 0.858] | 46.7 |
| 25 | 0.664 | [0.308, 0.865] | 45.2 |
| 26 | 0.672 | [0.323, 0.871] | 44.5 |
| 27 | 0.673 | [0.316, 0.870] | 44.3 |
| 28 | 0.667 | [0.309, 0.865] | 44.9 |
| 29 | 0.667 | [0.320, 0.860] | 45.1 |
| 30 | 0.669 | [0.329, 0.858] | 45.0 |
| 31 | 0.658 | [0.312, 0.849] | 46.0 |
| 32 | 0.658 | [0.328, 0.844] | 46.2 |
| 33 | 0.668 | [0.330, 0.854] | 45.2 |
| 34 | 0.668 | [0.351, 0.843] | 45.5 |
| 35 | 0.690 | [0.431, 0.832] | 44.3 |

## Interpretation

- Lowest mean cosine-to-global: **0.522** at layer 5.
- Highest mean angle-to-global: **57.0°** at layer 5.

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction).

## Figures

- `heatmap_honesty_qwen-2.5-3b.pdf` / `heatmap_honesty_qwen-2.5-3b.png`
- `dispersion_profile_honesty_qwen-2.5-3b.pdf` / `dispersion_profile_honesty_qwen-2.5-3b.png`
- `transfer_matrix_honesty_qwen-2.5-3b_layer005.pdf` / `transfer_matrix_honesty_qwen-2.5-3b_layer005.png`
- `geometry_pca_honesty_qwen-2.5-3b.pdf` / `geometry_pca_honesty_qwen-2.5-3b.png`
