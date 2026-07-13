# Probe transfer — honesty (qwen-2.5-3b)

## Layer 5
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.600 | 0.467 | 0.633 | 0.600 |
| math | 0.550 | 0.800 | 0.517 | 0.400 |
| personal_advice | 0.533 | 0.500 | 0.733 | 0.617 |
| politics_opinion | 0.517 | 0.500 | 0.600 | 0.617 |

summary: {'mean_within_acc': 0.6875, 'mean_cross_acc': 0.5361111111111111, 'min_cross_acc': 0.4, 'transfer_gap': 0.1513888888888889}

## Layer 18
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.783 | 0.500 | 0.767 | 0.800 |
| math | 0.483 | 0.733 | 0.500 | 0.467 |
| personal_advice | 0.650 | 0.633 | 0.733 | 0.783 |
| politics_opinion | 0.667 | 0.533 | 0.600 | 0.817 |

summary: {'mean_within_acc': 0.7666666666666666, 'mean_cross_acc': 0.6152777777777777, 'min_cross_acc': 0.4666666666666667, 'transfer_gap': 0.1513888888888889}

## Layer 30
| train \ test | factual_trivia | math | personal_advice | politics_opinion |
|---|---|---|---|---|
| factual_trivia | 0.833 | 0.233 | 0.700 | 0.750 |
| math | 0.517 | 0.867 | 0.483 | 0.600 |
| personal_advice | 0.800 | 0.400 | 0.783 | 0.767 |
| politics_opinion | 0.717 | 0.433 | 0.683 | 0.783 |

summary: {'mean_within_acc': 0.8166666666666667, 'mean_cross_acc': 0.5902777777777778, 'min_cross_acc': 0.23333333333333334, 'transfer_gap': 0.22638888888888886}
