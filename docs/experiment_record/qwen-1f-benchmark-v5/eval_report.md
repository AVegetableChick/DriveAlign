# Eval v1 report

- eval_version: eval_v1 (config status: frozen)
- contract_version: v5
- evaluation_split: test
- config_sha256: `e6690f6fe950a1d1accdae3286dadbf6a1fad6cb503e3b8fe04ff22bbcf91f2d`

## Coverage & parse

- anchors_expected: 5468
- anchors_scored: 5468
- parse_ok: 5449
- parse_failed: 19
- missing_predictions: 0
- extra_predictions: 0

- parse_rate: 0.996525

Parse errors:

- json_parse_failure: 14
- schema_failure: 4
- semantic_range_failure: 1

## Object metrics

| metric | value |
|---|---|
| matched_iou_mean | 0.733641 |
| matched_iou_p50 | 0.745858 |
| matched_iou_p90 | 0.855466 |
| object_f1_macro@0.5 | 0.217269 |
| object_f1_micro@0.3 | 0.386771 |
| object_f1_micro@0.5 | 0.338360 |
| object_f1_micro@0.7 | 0.223416 |
| object_f1_per_category@0.5 | {'barrier': 0.02278481012658228, 'bicycle': 0.09388335704125178, 'bus': 0.5907780979827089, 'car': 0.4157856866109481, 'construction_vehicle': 0.11009174311926605, 'motorcycle': 0.24594594594594596, 'pedestrian': 0.07155764365739066, 'traffic_cone': 0.040520984081041975, 'trailer': 0.19014084507042253, 'truck': 0.3911966987620358} |
| object_precision_micro@0.3 | 0.476897 |
| object_precision_micro@0.5 | 0.417205 |
| object_precision_micro@0.7 | 0.275477 |
| object_recall_micro@0.3 | 0.325295 |
| object_recall_micro@0.5 | 0.284579 |
| object_recall_micro@0.7 | 0.187905 |

## Field metrics

| metric | value |
|---|---|
| motion_state_accuracy | 0.325746 |
| speed_action_f1_macro | 0.114613 |
| yield_required_f1 | 0.281170 |
| risk_factors_f1_macro | 0.088966 |
| max_items_hit_rate | 0.024959 |
| overconservative_yield_rate | 0.833889 |
| overconservative_stop_rate | 0.000582 |

## 95% CI (scene-cluster bootstrap)

| metric | point | ci_low | ci_high |
|---|---|---|---|
| max_items_hit_rate | 0.024959 | 0.016850 | 0.034369 |
| motion_state_accuracy | 0.325746 | 0.273589 | 0.376623 |
| object_f1_macro@0.5 | 0.217269 | 0.187823 | 0.243607 |
| object_f1_micro@0.5 | 0.338360 | 0.315182 | 0.361937 |
| overconservative_stop_rate | 0.000582 | 0.000000 | 0.001456 |
| overconservative_yield_rate | 0.833889 | 0.804047 | 0.862077 |
| parse_rate | 0.996525 | 0.992883 | 0.998725 |
| risk_factors_f1_macro | 0.088966 | 0.071765 | 0.106057 |
| speed_action_f1_macro | 0.114613 | 0.103244 | 0.125265 |
| yield_required_f1 | 0.281170 | 0.235971 | 0.325131 |

## Cost summary

- gpu_hours: 8.382347
- input_tokens_p50: 2335.000000
- latency_p50: 4.977101
- latency_p90: 7.870241
- peak_cuda_mem_bytes: 8522751488.000000
- throughput_anchors_per_s: 0.181201

## Visual dependence (counterfactual subset)

- blank: n_compared=197, action_flip=0.147208, output_change=1.000000
- shuffled: n_compared=195, action_flip=0.174359, output_change=1.000000
