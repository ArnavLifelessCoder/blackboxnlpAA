# Analysis Report — Honesty (qwen-2.5-3b)

- Generated: 2026-07-13T04:29:59
- Mode: real activations
- Domains: factual_trivia, math, politics_opinion, personal_advice
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.554 | [0.485, 0.566] | 55.2 |
| 1 | 0.550 | [0.482, 0.561] | 55.5 |
| 2 | 0.546 | [0.481, 0.559] | 55.8 |
| 3 | 0.544 | [0.482, 0.556] | 55.9 |
| 4 | 0.540 | [0.481, 0.551] | 56.2 |
| 5 | 0.540 | [0.484, 0.548] | 56.3 |
| 6 | 0.541 | [0.487, 0.547] | 56.5 |
| 7 | 0.549 | [0.494, 0.556] | 56.1 |
| 8 | 0.546 | [0.492, 0.555] | 56.3 |
| 9 | 0.542 | [0.487, 0.553] | 56.4 |
| 10 | 0.547 | [0.491, 0.559] | 55.9 |
| 11 | 0.558 | [0.497, 0.569] | 55.3 |
| 12 | 0.553 | [0.494, 0.564] | 55.6 |
| 13 | 0.554 | [0.491, 0.568] | 55.5 |
| 14 | 0.552 | [0.492, 0.566] | 55.5 |
| 15 | 0.556 | [0.494, 0.573] | 55.2 |
| 16 | 0.560 | [0.496, 0.578] | 54.9 |
| 17 | 0.572 | [0.505, 0.586] | 54.0 |
| 18 | 0.569 | [0.503, 0.583] | 54.1 |
| 19 | 0.586 | [0.521, 0.596] | 53.0 |
| 20 | 0.604 | [0.532, 0.612] | 51.6 |
| 21 | 0.604 | [0.532, 0.614] | 51.6 |
| 22 | 0.603 | [0.532, 0.614] | 51.7 |
| 23 | 0.617 | [0.546, 0.622] | 50.8 |
| 24 | 0.619 | [0.546, 0.627] | 50.5 |
| 25 | 0.630 | [0.555, 0.638] | 49.5 |
| 26 | 0.639 | [0.562, 0.646] | 48.9 |
| 27 | 0.642 | [0.567, 0.648] | 48.5 |
| 28 | 0.634 | [0.558, 0.641] | 49.1 |
| 29 | 0.632 | [0.559, 0.637] | 49.5 |
| 30 | 0.637 | [0.564, 0.641] | 49.3 |
| 31 | 0.624 | [0.554, 0.628] | 50.2 |
| 32 | 0.626 | [0.554, 0.628] | 50.3 |
| 33 | 0.634 | [0.563, 0.637] | 49.5 |
| 34 | 0.626 | [0.555, 0.631] | 50.3 |
| 35 | 0.631 | [0.536, 0.656] | 50.3 |

## Interpretation

- Lowest mean cosine-to-global: **0.540** at layer 4.
- Highest mean angle-to-global: **56.5°** at layer 6.
- Late-layer (final third) mean cosine: **0.631** (layers 24–35).

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction). Note: minima in early layers may reflect token/surface statistics rather than concept representations — contrast with the late-layer aggregate, where behavior is read out.

## Figures

