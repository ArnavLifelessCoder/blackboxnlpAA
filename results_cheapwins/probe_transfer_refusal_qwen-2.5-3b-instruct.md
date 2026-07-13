# Probe transfer — refusal (qwen-2.5-3b-instruct)

## Layer 5
| train \ test | illegal_activity | medical_legal | privacy | violence |
|---|---|---|---|---|
| illegal_activity | 1.000 | 1.000 | 0.661 | 1.000 |
| medical_legal | 0.850 | 1.000 | 0.714 | 0.826 |
| privacy | 0.967 | 1.000 | 1.000 | 0.935 |
| violence | 1.000 | 1.000 | 0.607 | 1.000 |

summary: {'mean_within_acc': 1.0, 'mean_cross_acc': 0.8799732574189097, 'min_cross_acc': 0.6071428571428571, 'transfer_gap': 0.12002674258109025}

## Layer 18
| train \ test | illegal_activity | medical_legal | privacy | violence |
|---|---|---|---|---|
| illegal_activity | 0.983 | 0.983 | 0.821 | 1.000 |
| medical_legal | 0.983 | 1.000 | 0.911 | 1.000 |
| privacy | 0.983 | 1.000 | 1.000 | 1.000 |
| violence | 1.000 | 0.983 | 0.821 | 1.000 |

summary: {'mean_within_acc': 0.9958333333333333, 'mean_cross_acc': 0.9572420634920634, 'min_cross_acc': 0.8214285714285714, 'transfer_gap': 0.038591269841269926}

## Layer 30
| train \ test | illegal_activity | medical_legal | privacy | violence |
|---|---|---|---|---|
| illegal_activity | 1.000 | 0.933 | 0.982 | 1.000 |
| medical_legal | 1.000 | 1.000 | 1.000 | 1.000 |
| privacy | 1.000 | 0.950 | 1.000 | 1.000 |
| violence | 1.000 | 0.950 | 0.964 | 1.000 |

summary: {'mean_within_acc': 1.0, 'mean_cross_acc': 0.9816468253968252, 'min_cross_acc': 0.9333333333333333, 'transfer_gap': 0.01835317460317476}
