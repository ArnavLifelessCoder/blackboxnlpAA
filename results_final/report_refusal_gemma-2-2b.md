# Analysis Report — Refusal (gemma-2-2b)

- Generated: 2026-06-30T09:19:30
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.685 | [0.401, 0.946] | 41.5 |
| 1 | 0.660 | [0.413, 0.890] | 45.3 |
| 2 | 0.647 | [0.401, 0.874] | 46.5 |
| 3 | 0.667 | [0.468, 0.861] | 45.5 |
| 4 | 0.676 | [0.484, 0.859] | 44.7 |
| 5 | 0.642 | [0.431, 0.852] | 47.2 |
| 6 | 0.627 | [0.396, 0.859] | 48.0 |
| 7 | 0.624 | [0.392, 0.856] | 48.2 |
| 8 | 0.614 | [0.375, 0.853] | 48.8 |
| 9 | 0.643 | [0.423, 0.855] | 46.8 |
| 10 | 0.636 | [0.401, 0.854] | 47.3 |
| 11 | 0.661 | [0.453, 0.869] | 45.7 |
| 12 | 0.651 | [0.437, 0.873] | 46.5 |
| 13 | 0.635 | [0.457, 0.843] | 48.3 |
| 14 | 0.623 | [0.452, 0.823] | 49.4 |
| 15 | 0.639 | [0.484, 0.823] | 48.3 |
| 16 | 0.620 | [0.454, 0.808] | 49.9 |
| 17 | 0.609 | [0.443, 0.805] | 50.9 |
| 18 | 0.611 | [0.449, 0.799] | 50.9 |
| 19 | 0.623 | [0.457, 0.813] | 50.0 |
| 20 | 0.629 | [0.458, 0.816] | 49.5 |
| 21 | 0.637 | [0.450, 0.817] | 48.7 |
| 22 | 0.645 | [0.446, 0.832] | 47.7 |
| 23 | 0.637 | [0.419, 0.855] | 47.9 |
| 24 | 0.626 | [0.368, 0.884] | 47.7 |
| 25 | 0.628 | [0.383, 0.873] | 47.8 |

## Interpretation

- Lowest mean cosine-to-global: **0.609** at layer 17.
- Highest mean angle-to-global: **50.9°** at layer 18.

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction).

## Figures

- `heatmap_refusal_gemma-2-2b.pdf` / `heatmap_refusal_gemma-2-2b.png`
- `dispersion_profile_refusal_gemma-2-2b.pdf` / `dispersion_profile_refusal_gemma-2-2b.png`
- `transfer_matrix_refusal_gemma-2-2b_layer017.pdf` / `transfer_matrix_refusal_gemma-2-2b_layer017.png`
- `geometry_pca_refusal_gemma-2-2b.pdf` / `geometry_pca_refusal_gemma-2-2b.png`
