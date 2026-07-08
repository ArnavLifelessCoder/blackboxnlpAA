# Analysis Report — Refusal (gemma-2-2b-it)

- Generated: 2026-06-30T08:01:56
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.693 | [0.391, 0.953] | 40.4 |
| 1 | 0.673 | [0.423, 0.908] | 43.8 |
| 2 | 0.657 | [0.418, 0.878] | 45.5 |
| 3 | 0.663 | [0.480, 0.854] | 45.7 |
| 4 | 0.651 | [0.481, 0.857] | 46.8 |
| 5 | 0.613 | [0.425, 0.832] | 49.2 |
| 6 | 0.609 | [0.427, 0.841] | 49.4 |
| 7 | 0.576 | [0.384, 0.822] | 52.2 |
| 8 | 0.575 | [0.372, 0.823] | 52.3 |
| 9 | 0.591 | [0.392, 0.838] | 51.0 |
| 10 | 0.570 | [0.361, 0.823] | 52.6 |
| 11 | 0.572 | [0.375, 0.814] | 52.7 |
| 12 | 0.559 | [0.346, 0.797] | 53.2 |
| 13 | 0.568 | [0.384, 0.786] | 53.1 |
| 14 | 0.562 | [0.362, 0.784] | 53.3 |
| 15 | 0.579 | [0.395, 0.783] | 52.8 |
| 16 | 0.582 | [0.377, 0.789] | 52.4 |
| 17 | 0.582 | [0.374, 0.794] | 52.3 |
| 18 | 0.587 | [0.421, 0.774] | 52.3 |
| 19 | 0.599 | [0.437, 0.784] | 51.6 |
| 20 | 0.614 | [0.443, 0.806] | 50.4 |
| 21 | 0.618 | [0.431, 0.818] | 50.2 |
| 22 | 0.632 | [0.436, 0.824] | 49.0 |
| 23 | 0.635 | [0.439, 0.818] | 48.5 |
| 24 | 0.638 | [0.417, 0.859] | 47.6 |
| 25 | 0.688 | [0.465, 0.910] | 43.1 |

## Interpretation

- Lowest mean cosine-to-global: **0.559** at layer 12.
- Highest mean angle-to-global: **53.3°** at layer 14.

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction).

## Figures

- `heatmap_refusal_gemma-2-2b-it.pdf` / `heatmap_refusal_gemma-2-2b-it.png`
- `dispersion_profile_refusal_gemma-2-2b-it.pdf` / `dispersion_profile_refusal_gemma-2-2b-it.png`
- `transfer_matrix_refusal_gemma-2-2b-it_layer012.pdf` / `transfer_matrix_refusal_gemma-2-2b-it_layer012.png`
- `geometry_pca_refusal_gemma-2-2b-it.pdf` / `geometry_pca_refusal_gemma-2-2b-it.png`
