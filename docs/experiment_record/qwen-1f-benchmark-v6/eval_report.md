# Eval v1 report

- eval_version: eval_v1 (config status: frozen)
- contract_version: v6
- evaluation_split: test
- config_sha256: `e6690f6fe950a1d1accdae3286dadbf6a1fad6cb503e3b8fe04ff22bbcf91f2d`

## Coverage & parse

- anchors_expected: 5468
- anchors_scored: 5468
- parse_ok: 5464
- parse_failed: 4
- missing_predictions: 0
- extra_predictions: 0

- parse_rate: 0.999268

Parse errors:

- schema_failure: 3
- semantic_range_failure: 1

## Object metrics

| metric | value |
|---|---|
| matched_iou_mean | 0.732395 |
| matched_iou_p50 | 0.744347 |
| matched_iou_p90 | 0.855034 |
| object_f1_macro@0.5 | 0.215986 |
| object_f1_micro@0.3 | 0.385185 |
| object_f1_micro@0.5 | 0.336178 |
| object_f1_micro@0.7 | 0.219899 |
| object_f1_per_category@0.5 | {'barrier': 0.025225225225225228, 'bicycle': 0.08284023668639053, 'bus': 0.5863284002818886, 'car': 0.4104501436628711, 'construction_vehicle': 0.12169312169312169, 'motorcycle': 0.24119241192411925, 'pedestrian': 0.0716075904045829, 'traffic_cone': 0.05106382978723405, 'trailer': 0.1768707482993197, 'truck': 0.3925905446502626} |
| object_precision_micro@0.3 | 0.460086 |
| object_precision_micro@0.5 | 0.401550 |
| object_precision_micro@0.7 | 0.262659 |
| object_recall_micro@0.3 | 0.331257 |
| object_recall_micro@0.5 | 0.289112 |
| object_recall_micro@0.7 | 0.189112 |

## Field metrics

| metric | value |
|---|---|
| motion_state_accuracy | 0.313285 |
| speed_action_f1_macro | 0.162243 |
| yield_required_f1 | 0.044944 |
| risk_factors_f1_macro | 0.088237 |
| max_items_hit_rate | 0.039531 |
| overconservative_yield_rate | 0.021258 |
| overconservative_stop_rate | 0.000000 |

## 95% CI (scene-cluster bootstrap)

| metric | point | ci_low | ci_high |
|---|---|---|---|
| max_items_hit_rate | 0.039531 | 0.027185 | 0.054675 |
| motion_state_accuracy | 0.313285 | 0.266218 | 0.361262 |
| object_f1_macro@0.5 | 0.215986 | 0.186694 | 0.242835 |
| object_f1_micro@0.5 | 0.336178 | 0.313755 | 0.359223 |
| overconservative_stop_rate | 0.000000 | 0.000000 | 0.000000 |
| overconservative_yield_rate | 0.021258 | 0.013856 | 0.029273 |
| parse_rate | 0.999268 | 0.998534 | 0.999818 |
| risk_factors_f1_macro | 0.088237 | 0.069326 | 0.106417 |
| speed_action_f1_macro | 0.162243 | 0.148878 | 0.176227 |
| yield_required_f1 | 0.044944 | 0.022366 | 0.071622 |

## Cost summary

- gpu_hours: 6.904192
- input_tokens_p50: 2313.000000
- latency_p50: 3.978527
- latency_p90: 6.660792
- peak_cuda_mem_bytes: 8509661696.000000
- throughput_anchors_per_s: 0.219995

## Visual dependence (counterfactual subset)

- blank: n_compared=200, action_flip=0.065000, output_change=1.000000
- shuffled: n_compared=200, action_flip=0.050000, output_change=1.000000
