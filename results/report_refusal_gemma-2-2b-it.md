# Analysis Report — Refusal (gemma-2-2b-it)

- Generated: 2026-06-30T05:40:46
- Mode: real activations
- Domains: violence, illegal_activity, medical_legal, privacy
- Layers analysed: [2, 6, 10, 14, 18, 22]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 2 | 0.803 | [0.796, 0.808] | 36.5 |
| 6 | 0.792 | [0.760, 0.825] | 37.5 |
| 10 | 0.756 | [0.749, 0.764] | 40.8 |
| 14 | 0.713 | [0.696, 0.731] | 44.5 |
| 18 | 0.677 | [0.656, 0.695] | 47.4 |
| 22 | 0.611 | [0.573, 0.635] | 52.3 |

## Interpretation

- Lowest mean cosine-to-global: **0.611** at layer 22.
- Highest mean angle-to-global: **52.3°** at layer 22.

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction).

## Figures

- `heatmap_refusal_gemma-2-2b-it.pdf` / `heatmap_refusal_gemma-2-2b-it.png`
- `dispersion_profile_refusal_gemma-2-2b-it.pdf` / `dispersion_profile_refusal_gemma-2-2b-it.png`
- `transfer_matrix_refusal_gemma-2-2b-it_layer022.pdf` / `transfer_matrix_refusal_gemma-2-2b-it_layer022.png`
- `geometry_pca_refusal_gemma-2-2b-it.pdf` / `geometry_pca_refusal_gemma-2-2b-it.png`
