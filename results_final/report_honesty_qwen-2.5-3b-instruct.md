# Analysis Report — Honesty (qwen-2.5-3b-instruct)

- Generated: 2026-07-08T18:54:46
- Mode: real activations
- Domains: factual_trivia, math, politics_opinion, personal_advice
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.542 | [0.481, 0.552] | 55.6 |
| 1 | 0.535 | [0.477, 0.547] | 55.9 |
| 2 | 0.529 | [0.474, 0.540] | 56.2 |
| 3 | 0.530 | [0.475, 0.540] | 56.3 |
| 4 | 0.526 | [0.475, 0.539] | 56.5 |
| 5 | 0.522 | [0.474, 0.533] | 57.0 |
| 6 | 0.526 | [0.477, 0.538] | 56.9 |
| 7 | 0.540 | [0.490, 0.549] | 55.9 |
| 8 | 0.534 | [0.486, 0.544] | 56.3 |
| 9 | 0.537 | [0.486, 0.547] | 55.9 |
| 10 | 0.542 | [0.488, 0.552] | 55.5 |
| 11 | 0.553 | [0.497, 0.562] | 54.8 |
| 12 | 0.550 | [0.496, 0.559] | 54.9 |
| 13 | 0.553 | [0.495, 0.564] | 54.6 |
| 14 | 0.550 | [0.494, 0.562] | 54.6 |
| 15 | 0.564 | [0.505, 0.574] | 53.5 |
| 16 | 0.573 | [0.515, 0.582] | 52.8 |
| 17 | 0.584 | [0.524, 0.591] | 51.9 |
| 18 | 0.592 | [0.535, 0.597] | 51.2 |
| 19 | 0.615 | [0.558, 0.616] | 49.7 |
| 20 | 0.641 | [0.579, 0.642] | 47.6 |
| 21 | 0.635 | [0.575, 0.636] | 48.1 |
| 22 | 0.634 | [0.576, 0.636] | 48.2 |
| 23 | 0.655 | [0.598, 0.651] | 46.9 |
| 24 | 0.673 | [0.617, 0.670] | 45.1 |
| 25 | 0.686 | [0.629, 0.686] | 43.7 |
| 26 | 0.693 | [0.637, 0.692] | 43.1 |
| 27 | 0.694 | [0.638, 0.695] | 42.8 |
| 28 | 0.684 | [0.628, 0.685] | 43.7 |
| 29 | 0.691 | [0.634, 0.688] | 43.4 |
| 30 | 0.689 | [0.633, 0.687] | 43.6 |
| 31 | 0.682 | [0.627, 0.679] | 44.4 |
| 32 | 0.683 | [0.627, 0.678] | 44.4 |
| 33 | 0.687 | [0.632, 0.685] | 43.7 |
| 34 | 0.698 | [0.639, 0.695] | 43.2 |
| 35 | 0.769 | [0.690, 0.770] | 38.1 |

## Interpretation

- Lowest mean cosine-to-global: **0.522** at layer 5.
- Highest mean angle-to-global: **57.0°** at layer 5.
- Late-layer (final third) mean cosine: **0.694** (layers 24–35).

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction). Note: minima in early layers may reflect token/surface statistics rather than concept representations — contrast with the late-layer aggregate, where behavior is read out.

## Figures

- `heatmap_honesty_qwen-2.5-3b-instruct.pdf` / `heatmap_honesty_qwen-2.5-3b-instruct.png`
- `dispersion_profile_honesty_qwen-2.5-3b-instruct.pdf` / `dispersion_profile_honesty_qwen-2.5-3b-instruct.png`
- `transfer_matrix_honesty_qwen-2.5-3b-instruct_layer005.pdf` / `transfer_matrix_honesty_qwen-2.5-3b-instruct_layer005.png`
- `geometry_pca_honesty_qwen-2.5-3b-instruct.pdf` / `geometry_pca_honesty_qwen-2.5-3b-instruct.png`
