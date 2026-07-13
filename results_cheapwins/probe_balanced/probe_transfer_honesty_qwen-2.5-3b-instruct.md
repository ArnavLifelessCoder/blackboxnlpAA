# Probe transfer — honesty (qwen-2.5-3b-instruct)

## Layer 5
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.533 | 0.433 | 0.533 | 0.533 |
| math | 0.533 | 0.800 | 0.567 | 0.400 |
| personal_advice | 0.467 | 0.467 | 0.733 | 0.700 |
| politics_opinion | 0.567 | 0.433 | 0.633 | 0.600 |

summary: {'mean_within_acc': 0.6666666666666667, 'mean_cross_acc': 0.5222222222222221, 'min_cross_acc': 0.4, 'transfer_gap': 0.1444444444444446}

## Layer 30
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.800 | 0.467 | 0.667 | 0.800 |
| math | 0.600 | 0.900 | 0.500 | 0.467 |
| personal_advice | 0.767 | 0.433 | 0.833 | 0.867 |
| politics_opinion | 0.767 | 0.567 | 0.700 | 0.767 |

summary: {'mean_within_acc': 0.8250000000000001, 'mean_cross_acc': 0.6333333333333333, 'min_cross_acc': 0.43333333333333335, 'transfer_gap': 0.19166666666666676}
