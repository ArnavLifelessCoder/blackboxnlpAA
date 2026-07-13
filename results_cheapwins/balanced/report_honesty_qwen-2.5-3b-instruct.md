# Analysis Report — Honesty (qwen-2.5-3b-instruct)

- Generated: 2026-07-13T04:24:00
- Mode: real activations
- Domains: factual_trivia, math, politics_opinion, personal_advice
- Layers analysed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Angular Dispersion (domain direction vs. global direction)

| Layer | Mean cosine | 95% CI | Mean angle (deg) |
|------:|------------:|:-------|-----------------:|
| 0 | 0.552 | [0.484, 0.564] | 55.3 |
| 1 | 0.549 | [0.481, 0.560] | 55.5 |
| 2 | 0.543 | [0.478, 0.557] | 55.9 |
| 3 | 0.541 | [0.479, 0.555] | 56.1 |
| 4 | 0.536 | [0.477, 0.548] | 56.4 |
| 5 | 0.538 | [0.482, 0.547] | 56.5 |
| 6 | 0.540 | [0.487, 0.547] | 56.6 |
| 7 | 0.547 | [0.493, 0.554] | 56.2 |
| 8 | 0.543 | [0.490, 0.552] | 56.5 |
| 9 | 0.542 | [0.488, 0.554] | 56.4 |
| 10 | 0.546 | [0.490, 0.559] | 56.0 |
| 11 | 0.556 | [0.497, 0.570] | 55.4 |
| 12 | 0.554 | [0.495, 0.566] | 55.6 |
| 13 | 0.553 | [0.491, 0.567] | 55.6 |
| 14 | 0.551 | [0.490, 0.566] | 55.6 |
| 15 | 0.557 | [0.494, 0.574] | 55.2 |
| 16 | 0.563 | [0.499, 0.580] | 54.8 |
| 17 | 0.577 | [0.510, 0.590] | 53.7 |
| 18 | 0.576 | [0.510, 0.589] | 53.7 |
| 19 | 0.594 | [0.528, 0.602] | 52.5 |
| 20 | 0.619 | [0.544, 0.624] | 50.7 |
| 21 | 0.614 | [0.540, 0.621] | 51.0 |
| 22 | 0.611 | [0.538, 0.619] | 51.3 |
| 23 | 0.626 | [0.553, 0.629] | 50.4 |
| 24 | 0.637 | [0.562, 0.641] | 49.5 |
| 25 | 0.650 | [0.573, 0.653] | 48.3 |
| 26 | 0.658 | [0.581, 0.662] | 47.6 |
| 27 | 0.663 | [0.586, 0.665] | 47.2 |
| 28 | 0.649 | [0.573, 0.651] | 48.3 |
| 29 | 0.653 | [0.577, 0.653] | 48.3 |
| 30 | 0.655 | [0.581, 0.656] | 48.1 |
| 31 | 0.643 | [0.571, 0.645] | 49.1 |
| 32 | 0.644 | [0.570, 0.643] | 49.1 |
| 33 | 0.646 | [0.573, 0.647] | 48.7 |
| 34 | 0.646 | [0.569, 0.650] | 48.9 |
| 35 | 0.692 | [0.574, 0.716] | 45.6 |

## Interpretation

- Lowest mean cosine-to-global: **0.536** at layer 4.
- Highest mean angle-to-global: **56.6°** at layer 6.
- Late-layer (final third) mean cosine: **0.653** (layers 24–35).

Lower cosine / higher angle indicates stronger domain fragmentation (per-domain directions diverging from the global direction). Note: minima in early layers may reflect token/surface statistics rather than concept representations — contrast with the late-layer aggregate, where behavior is read out.

## Figures

