# Eval v3 report

- eval_version: eval_v3 (config status: frozen)
- contract_version: v7
- evaluation_split: test
- config_sha256: `fae363adc8dd085d90e87709fcb9e2c19f31afa86c64b4c00a39474b65e7c7cf`

## Coverage & parse

- anchors_expected: 5468
- parse_ok: 5465
- parse_failed: 3

- parse_rate: 0.999451

Parse errors:

- json_parse_failure: 1
- semantic_range_failure: 2

## Object metrics

| metric | value |
|---|---|
| matched_iou_mean | 0.786404 |
| matched_iou_p50 | 0.805970 |
| matched_iou_p90 | 0.914919 |
| matched_pair_count | 10091 |
| object_empty_hallucination_rate | 0.056890 |
| object_empty_miss_rate | 0.093496 |
| object_f1_macro@0.5 | 0.442337 |
| object_f1_micro@0.3 | 0.642087 |
| object_f1_micro@0.5 | 0.606449 |
| object_f1_micro@0.7 | 0.472610 |
| object_precision_micro@0.3 | 0.720966 |
| object_precision_micro@0.5 | 0.680950 |
| object_precision_micro@0.7 | 0.530670 |
| object_recall_micro@0.3 | 0.578765 |
| object_recall_micro@0.5 | 0.546641 |
| object_recall_micro@0.7 | 0.426002 |

## BEV localization errors (matched pairs)

| metric | value |
|---|---|
| bev_dx_mean | 1.475572 |
| bev_dx_mean_bucket_0_10 | 0.529175 |
| bev_dx_mean_bucket_10_25 | 1.187090 |
| bev_dx_mean_bucket_25_40 | 2.267955 |
| bev_dx_mean_bucket_40p | 5.159108 |
| bev_dx_p50 | 0.900000 |
| bev_dx_p50_bucket_0_10 | 0.300000 |
| bev_dx_p50_bucket_10_25 | 0.900000 |
| bev_dx_p50_bucket_25_40 | 1.800000 |
| bev_dx_p50_bucket_40p | 5.000000 |
| bev_dx_p90 | 3.500000 |
| bev_dx_p90_bucket_0_10 | 1.200000 |
| bev_dx_p90_bucket_10_25 | 2.600000 |
| bev_dx_p90_bucket_25_40 | 4.800000 |
| bev_dx_p90_bucket_40p | 9.100000 |
| bev_dxy_mean | 1.716729 |
| bev_dxy_mean_bucket_0_10 | 0.651020 |
| bev_dxy_mean_bucket_10_25 | 1.401045 |
| bev_dxy_mean_bucket_25_40 | 2.620989 |
| bev_dxy_mean_bucket_40p | 5.565187 |
| bev_dxy_p50 | 1.140175 |
| bev_dxy_p50_bucket_0_10 | 0.447214 |
| bev_dxy_p50_bucket_10_25 | 1.077033 |
| bev_dxy_p50_bucket_25_40 | 2.088061 |
| bev_dxy_p50_bucket_40p | 5.186521 |
| bev_dxy_p90 | 3.889730 |
| bev_dxy_p90_bucket_0_10 | 1.403567 |
| bev_dxy_p90_bucket_10_25 | 2.927456 |
| bev_dxy_p90_bucket_25_40 | 5.315073 |
| bev_dxy_p90_bucket_40p | 10.128813 |
| bev_dy_mean | 0.637905 |
| bev_dy_mean_bucket_0_10 | 0.269207 |
| bev_dy_mean_bucket_10_25 | 0.547090 |
| bev_dy_mean_bucket_25_40 | 0.947301 |
| bev_dy_mean_bucket_40p | 1.661710 |
| bev_dy_p50 | 0.400000 |
| bev_dy_p50_bucket_0_10 | 0.200000 |
| bev_dy_p50_bucket_10_25 | 0.400000 |
| bev_dy_p50_bucket_25_40 | 0.600000 |
| bev_dy_p50_bucket_40p | 1.200000 |
| bev_dy_p90 | 1.500000 |
| bev_dy_p90_bucket_0_10 | 0.600000 |
| bev_dy_p90_bucket_10_25 | 1.200000 |
| bev_dy_p90_bucket_25_40 | 2.100000 |
| bev_dy_p90_bucket_40p | 4.400000 |
| bev_signed_dx_neg_mean | -1.684844 |
| bev_signed_dx_pos_mean | 1.327616 |
| bev_signed_dy_neg_mean | -0.731507 |
| bev_signed_dy_pos_mean | 0.657461 |

## Cardinality histograms (parse-ok anchors)

- n_gt_objects: 0:791, 1:878, 2:757, 3:697, 4:560, 5:501, 6:380, 7:256, 8:645
- n_pred_objects: 0:1183, 1:875, 2:1074, 3:635, 4:451, 5:386, 6:286, 7:159, 8:416
- n_gt_risk_factors: 0:3475, 1:1435, 2:482, 3:65, 4:8, 5:0, 6:0, 7:0, 8:0
- n_pred_risk_factors: 0:3572, 1:1769, 2:122, 3:0, 4:2, 5:0, 6:0, 7:0, 8:0

## Empty-direction counts (parse-ok anchors)

- object: gt_empty_pred_empty=746, gt_empty_pred_nonempty=45, gt_nonempty_pred_empty=437, gt_nonempty_pred_nonempty=4237
- risk: gt_empty_pred_empty=2927, gt_empty_pred_nonempty=548, gt_nonempty_pred_empty=645, gt_nonempty_pred_nonempty=1345

## Field metrics

| metric | value |
|---|---|
| motion_state_f1_macro | 0.673448 |
| speed_action_f1_macro | 0.572517 |
| yield_required_f1 | 0.700807 |
| yield_miss_rate | 0.280259 |
| yield_overconservative_rate | 0.113809 |
| risk_factors_f1_macro | 0.379918 |
| stop_overconservative_rate | 0.045231 |
| stop_miss_rate | 0.524306 |
| max_items_hit_rate | 0.076121 |

## Risk attribution (conditional, matched-bridge)

| metric | value |
|---|---|
| risk_attribution_f1_micro | 0.600817 |
| risk_attribution_f1_macro | 0.702583 |

| factor | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|
| lead_vehicle_braking | 15 | 1 | 2 | 0.937500 | 0.882353 | 0.909091 |
| object_near | 48 | 16 | 31 | 0.750000 | 0.607595 | 0.671329 |
| oncoming_traffic | 133 | 13 | 20 | 0.910959 | 0.869281 | 0.889632 |
| stationary_obstacle | 785 | 536 | 665 | 0.594247 | 0.541379 | 0.566582 |
| vehicle_crossing | 77 | 38 | 59 | 0.669565 | 0.566176 | 0.613546 |
| vru_crossing | 119 | 121 | 62 | 0.495833 | 0.657459 | 0.565321 |

## Per-class F1

| block | class | f1 |
|---|---|---|
| motion_state_f1_per_class | crossing | 0.649722 |
| motion_state_f1_per_class | oncoming | 0.552610 |
| motion_state_f1_per_class | same_direction | 0.656830 |
| motion_state_f1_per_class | stationary | 0.834632 |
| object_f1_per_category@0.5 | barrier | 0.441030 |
| object_f1_per_category@0.5 | bicycle | 0.264574 |
| object_f1_per_category@0.5 | bus | 0.764378 |
| object_f1_per_category@0.5 | car | 0.706504 |
| object_f1_per_category@0.5 | construction_vehicle | 0.218750 |
| object_f1_per_category@0.5 | motorcycle | 0.390777 |
| object_f1_per_category@0.5 | pedestrian | 0.500307 |
| object_f1_per_category@0.5 | traffic_cone | 0.441434 |
| object_f1_per_category@0.5 | trailer | 0.164580 |
| object_f1_per_category@0.5 | truck | 0.531032 |
| risk_factors_f1_per_class | lead_vehicle_braking | 0.096677 |
| risk_factors_f1_per_class | object_near | 0.300716 |
| risk_factors_f1_per_class | oncoming_traffic | 0.452713 |
| risk_factors_f1_per_class | stationary_obstacle | 0.616420 |
| risk_factors_f1_per_class | vehicle_crossing | 0.293103 |
| risk_factors_f1_per_class | vru_crossing | 0.519878 |
| speed_action_per_class_f1 | ACCELERATE | 0.471961 |
| speed_action_per_class_f1 | DECELERATE | 0.186857 |
| speed_action_per_class_f1 | KEEP_SPEED | 0.769260 |
| speed_action_per_class_f1 | STOP | 0.861990 |

## 95% CI (scene-cluster bootstrap)

| metric | point | ci_low | ci_high |
|---|---|---|---|
| bev_dx_mean | 1.475572 | 1.383526 | 1.576760 |
| bev_dxy_mean | 1.716729 | 1.611373 | 1.835329 |
| bev_dy_mean | 0.637905 | 0.579384 | 0.712046 |
| max_items_hit_rate | 0.076121 | 0.056217 | 0.097228 |
| motion_state_f1_macro | 0.673448 | 0.622760 | 0.712479 |
| object_empty_hallucination_rate | 0.056890 | 0.027225 | 0.095709 |
| object_empty_miss_rate | 0.093496 | 0.071732 | 0.118188 |
| object_f1_macro@0.5 | 0.442337 | 0.405113 | 0.471102 |
| object_f1_micro@0.5 | 0.606449 | 0.584239 | 0.627768 |
| object_precision_micro@0.3 | 0.720966 | 0.693996 | 0.748639 |
| object_precision_micro@0.5 | 0.680950 | 0.653810 | 0.708867 |
| object_precision_micro@0.7 | 0.530670 | 0.501351 | 0.558854 |
| object_recall_micro@0.3 | 0.578765 | 0.555815 | 0.601493 |
| object_recall_micro@0.5 | 0.546641 | 0.523251 | 0.570233 |
| object_recall_micro@0.7 | 0.426002 | 0.401540 | 0.449998 |
| parse_rate | 0.999451 | 0.998720 | 1.000000 |
| risk_attribution_f1_macro | 0.702583 | 0.656708 | 0.756431 |
| risk_attribution_f1_micro | 0.600817 | 0.556548 | 0.648503 |
| risk_empty_hallucination_rate | 0.157698 | 0.122831 | 0.195446 |
| risk_empty_miss_rate | 0.324121 | 0.276967 | 0.377625 |
| risk_factors_f1_macro | 0.379918 | 0.333003 | 0.417492 |
| speed_action_f1_macro | 0.572517 | 0.550789 | 0.594343 |
| stop_miss_rate | 0.524306 | 0.445772 | 0.612732 |
| stop_overconservative_rate | 0.045231 | 0.033578 | 0.058361 |
| yield_miss_rate | 0.280259 | 0.230417 | 0.336670 |
| yield_overconservative_rate | 0.113809 | 0.088217 | 0.142071 |
| yield_required_f1 | 0.700807 | 0.656162 | 0.741479 |

Per-class CI blocks:

| block | class | point | ci_low | ci_high |
|---|---|---|---|---|
| motion_state_f1_per_class | crossing | 0.649722 | 0.494565 | 0.760729 |
| motion_state_f1_per_class | oncoming | 0.552610 | 0.484192 | 0.613398 |
| motion_state_f1_per_class | same_direction | 0.656830 | 0.592502 | 0.715816 |
| motion_state_f1_per_class | stationary | 0.834632 | 0.801052 | 0.864474 |
| object_f1_per_category@0.5 | barrier | 0.441030 | 0.382259 | 0.490068 |
| object_f1_per_category@0.5 | bicycle | 0.264574 | 0.172909 | 0.348162 |
| object_f1_per_category@0.5 | bus | 0.764378 | 0.670999 | 0.838081 |
| object_f1_per_category@0.5 | car | 0.706504 | 0.685184 | 0.727750 |
| object_f1_per_category@0.5 | construction_vehicle | 0.218750 | 0.013590 | 0.420757 |
| object_f1_per_category@0.5 | motorcycle | 0.390777 | 0.281728 | 0.510853 |
| object_f1_per_category@0.5 | pedestrian | 0.500307 | 0.450411 | 0.544591 |
| object_f1_per_category@0.5 | traffic_cone | 0.441434 | 0.387654 | 0.495215 |
| object_f1_per_category@0.5 | trailer | 0.164580 | 0.054185 | 0.240642 |
| object_f1_per_category@0.5 | truck | 0.531032 | 0.474978 | 0.581235 |
| risk_factors_f1_per_class | lead_vehicle_braking | 0.096677 | 0.054054 | 0.133333 |
| risk_factors_f1_per_class | object_near | 0.300716 | 0.183656 | 0.399229 |
| risk_factors_f1_per_class | oncoming_traffic | 0.452713 | 0.368157 | 0.524751 |
| risk_factors_f1_per_class | stationary_obstacle | 0.616420 | 0.537630 | 0.680271 |
| risk_factors_f1_per_class | vehicle_crossing | 0.293103 | 0.188283 | 0.395483 |
| risk_factors_f1_per_class | vru_crossing | 0.519878 | 0.351644 | 0.622407 |
| speed_action_per_class_f1 | ACCELERATE | 0.471961 | 0.430812 | 0.513101 |
| speed_action_per_class_f1 | DECELERATE | 0.186857 | 0.138631 | 0.232845 |
| speed_action_per_class_f1 | KEEP_SPEED | 0.769260 | 0.733813 | 0.804102 |
| speed_action_per_class_f1 | STOP | 0.861990 | 0.814931 | 0.897763 |

## Denominators

- anchors_expected: 5468
- parse_ok: 5465
- matched_pairs: 10091
- motion_pairs: 10091
- bev_pairs: 10091
- yield_counts: {"fn": 389, "fp": 464, "tn": 3613, "tp": 999}
- stop_counts: {"stop_fn": 1057, "stop_fp": 156, "stop_gt_keep": 3449, "stop_gt_stop": 2016}
- micro_counts: {"0.3": [10684, 4135, 7776], "0.5": [10091, 4728, 8369], "0.7": [7864, 6955, 10596]}

## Cost summary

- gpu_hours: 4.077503
- input_tokens_p50: 1082.000000
- latency_p50: 3.079706
- latency_p90: 3.277147
- peak_cuda_mem_bytes: 14037792256.000000
- throughput_anchors_per_s: 0.372505

## Visual dependence (counterfactual subset)

- blank: n_compared=200, action_flip=0.155000, output_change=0.795000
- shuffled: n_compared=200, action_flip=0.125000, output_change=0.790000
