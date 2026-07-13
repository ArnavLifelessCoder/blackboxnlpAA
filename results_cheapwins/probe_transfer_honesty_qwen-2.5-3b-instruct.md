# Probe transfer — honesty (qwen-2.5-3b-instruct)

## Layer 5
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.650 | 0.500 | 0.650 | 0.617 |
| math | 0.583 | 0.800 | 0.500 | 0.450 |
| personal_advice | 0.550 | 0.500 | 0.767 | 0.583 |
| politics_opinion | 0.533 | 0.600 | 0.600 | 0.683 |

summary: {'mean_within_acc': 0.7250000000000001, 'mean_cross_acc': 0.5555555555555555, 'min_cross_acc': 0.45, 'transfer_gap': 0.16944444444444462}

## Layer 18
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.800 | 0.467 | 0.717 | 0.767 |
| math | 0.550 | 0.833 | 0.500 | 0.567 |
| personal_advice | 0.700 | 0.567 | 0.717 | 0.767 |
| politics_opinion | 0.650 | 0.500 | 0.600 | 0.850 |

summary: {'mean_within_acc': 0.8, 'mean_cross_acc': 0.6124999999999999, 'min_cross_acc': 0.4666666666666667, 'transfer_gap': 0.1875000000000001}

## Layer 30
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.850 | 0.500 | 0.667 | 0.800 |
| math | 0.567 | 0.900 | 0.533 | 0.600 |
| personal_advice | 0.767 | 0.333 | 0.767 | 0.817 |
| politics_opinion | 0.750 | 0.500 | 0.667 | 0.783 |

summary: {'mean_within_acc': 0.825, 'mean_cross_acc': 0.625, 'min_cross_acc': 0.3333333333333333, 'transfer_gap': 0.19999999999999996}
