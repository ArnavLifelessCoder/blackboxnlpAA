# Analysis Report — Honesty (qwen-2.5-3b-instruct)

- Generated: 2026-07-06T18:39:05
- Mode: real activations
- Domains: factual_trivia, math, politics_opinion, personal_advice
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.542 | [0.235, 0.778] | 55.6 |
| 1 | 0.535 | [0.213, 0.782] | 55.9 |
| 2 | 0.529 | [0.194, 0.784] | 56.2 |
| 3 | 0.530 | [0.200, 0.779] | 56.3 |
| 4 | 0.526 | [0.193, 0.780] | 56.5 |
| 5 | 0.522 | [0.210, 0.766] | 57.0 |
| 6 | 0.526 | [0.232, 0.759] | 56.9 |
| 7 | 0.540 | [0.253, 0.768] | 55.9 |
| 8 | 0.534 | [0.243, 0.760] | 56.3 |
| 9 | 0.537 | [0.225, 0.781] | 55.9 |
| 10 | 0.542 | [0.229, 0.787] | 55.5 |
| 11 | 0.553 | [0.251, 0.791] | 54.8 |
| 12 | 0.550 | [0.234, 0.795] | 54.9 |
| 13 | 0.553 | [0.234, 0.797] | 54.6 |
| 14 | 0.550 | [0.208, 0.807] | 54.6 |
| 15 | 0.564 | [0.218, 0.816] | 53.5 |
| 16 | 0.573 | [0.218, 0.819] | 52.8 |
| 17 | 0.584 | [0.225, 0.827] | 51.9 |
| 18 | 0.592 | [0.221, 0.836] | 51.2 |
| 19 | 0.615 | [0.274, 0.835] | 49.7 |
| 20 | 0.641 | [0.307, 0.853] | 47.6 |
| 21 | 0.635 | [0.301, 0.849] | 48.1 |
| 22 | 0.634 | [0.298, 0.846] | 48.2 |
| 23 | 0.655 | [0.359, 0.839] | 46.9 |
| 24 | 0.673 | [0.358, 0.857] | 45.1 |
| 25 | 0.686 | [0.361, 0.868] | 43.7 |
| 26 | 0.693 | [0.369, 0.874] | 43.1 |
| 27 | 0.694 | [0.363, 0.875] | 42.8 |
| 28 | 0.684 | [0.352, 0.868] | 43.7 |
| 29 | 0.691 | [0.378, 0.863] | 43.4 |
| 30 | 0.689 | [0.374, 0.861] | 43.6 |
| 31 | 0.682 | [0.371, 0.850] | 44.4 |
| 32 | 0.683 | [0.381, 0.849] | 44.4 |
| 33 | 0.687 | [0.367, 0.860] | 43.7 |
| 34 | 0.698 | [0.407, 0.859] | 43.2 |
| 35 | 0.769 | [0.588, 0.870] | 38.1 |

## Interpretation

- Lowest mean cosine-to-global: **0.522** at layer 5.
- Highest mean angle-to-global: **57.0°** at layer 5.

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction).

## Figures

- `heatmap_honesty_qwen-2.5-3b-instruct.pdf` / `heatmap_honesty_qwen-2.5-3b-instruct.png`
- `dispersion_profile_honesty_qwen-2.5-3b-instruct.pdf` / `dispersion_profile_honesty_qwen-2.5-3b-instruct.png`
- `transfer_matrix_honesty_qwen-2.5-3b-instruct_layer005.pdf` / `transfer_matrix_honesty_qwen-2.5-3b-instruct_layer005.png`
- `geometry_pca_honesty_qwen-2.5-3b-instruct.pdf` / `geometry_pca_honesty_qwen-2.5-3b-instruct.png`
