# Eval v3 report

- eval_version: eval_v3 (config status: frozen)
- contract_version: v7
- evaluation_split: test
- config_sha256: `fae363adc8dd085d90e87709fcb9e2c19f31afa86c64b4c00a39474b65e7c7cf`

## Coverage & parse

- anchors_expected: 5468
- parse_ok: 5011
- parse_failed: 457

- parse_rate: 0.916423

Parse errors:

- json_parse_failure: 298
- schema_failure: 155
- semantic_range_failure: 4

## Object metrics

| metric | value |
|---|---|
| matched_iou_mean | 0.736527 |
| matched_iou_p50 | 0.747153 |
| matched_iou_p90 | 0.862998 |
| matched_pair_count | 5757 |
| object_empty_hallucination_rate | 1.000000 |
| object_empty_miss_rate | 0.000000 |
| object_f1_macro@0.5 | 0.243426 |
| object_f1_micro@0.3 | 0.420464 |
| object_f1_micro@0.5 | 0.376398 |
| object_f1_micro@0.7 | 0.247793 |
| object_precision_micro@0.3 | 0.435410 |
| object_precision_micro@0.5 | 0.389777 |
| object_precision_micro@0.7 | 0.256601 |
| object_recall_micro@0.3 | 0.406511 |
| object_recall_micro@0.5 | 0.363906 |
| object_recall_micro@0.7 | 0.239570 |

## BEV localization errors (matched pairs)

| metric | value |
|---|---|
| bev_dx_mean | 24.309884 |
| bev_dx_mean_bucket_0_10 | 8.216788 |
| bev_dx_mean_bucket_10_25 | 18.811198 |
| bev_dx_mean_bucket_25_40 | 34.085434 |
| bev_dx_mean_bucket_40p | 43.567202 |
| bev_dx_p50 | 23.000000 |
| bev_dx_p50_bucket_0_10 | 8.600000 |
| bev_dx_p50_bucket_10_25 | 18.000000 |
| bev_dx_p50_bucket_25_40 | 33.000000 |
| bev_dx_p50_bucket_40p | 42.800000 |
| bev_dx_p90 | 40.500000 |
| bev_dx_p90_bucket_0_10 | 10.700000 |
| bev_dx_p90_bucket_10_25 | 25.200000 |
| bev_dx_p90_bucket_25_40 | 40.300000 |
| bev_dx_p90_bucket_40p | 49.100000 |
| bev_dxy_mean | 24.992906 |
| bev_dxy_mean_bucket_0_10 | 8.618080 |
| bev_dxy_mean_bucket_10_25 | 19.421412 |
| bev_dxy_mean_bucket_25_40 | 34.947165 |
| bev_dxy_mean_bucket_40p | 44.415835 |
| bev_dxy_p50 | 23.621177 |
| bev_dxy_p50_bucket_0_10 | 8.909544 |
| bev_dxy_p50_bucket_10_25 | 18.505405 |
| bev_dxy_p50_bucket_25_40 | 33.733663 |
| bev_dxy_p50_bucket_40p | 43.537340 |
| bev_dxy_p90 | 41.305883 |
| bev_dxy_p90_bucket_0_10 | 10.900000 |
| bev_dxy_p90_bucket_10_25 | 25.735966 |
| bev_dxy_p90_bucket_25_40 | 41.013900 |
| bev_dxy_p90_bucket_40p | 49.100407 |
| bev_dy_mean | 4.516206 |
| bev_dy_mean_bucket_0_10 | 1.888929 |
| bev_dy_mean_bucket_10_25 | 3.871713 |
| bev_dy_mean_bucket_25_40 | 6.017355 |
| bev_dy_mean_bucket_40p | 6.592431 |
| bev_dy_p50 | 3.300000 |
| bev_dy_p50_bucket_0_10 | 1.900000 |
| bev_dy_p50_bucket_10_25 | 3.200000 |
| bev_dy_p50_bucket_25_40 | 4.900000 |
| bev_dy_p50_bucket_40p | 5.300000 |
| bev_dy_p90 | 10.600000 |
| bev_dy_p90_bucket_0_10 | 3.500000 |
| bev_dy_p90_bucket_10_25 | 7.600000 |
| bev_dy_p90_bucket_25_40 | 12.700000 |
| bev_dy_p90_bucket_40p | 14.600000 |
| bev_signed_dx_neg_mean | -24.228803 |
| bev_signed_dx_pos_mean | 73.330000 |
| bev_signed_dy_neg_mean | -5.236746 |
| bev_signed_dy_pos_mean | 4.158776 |

## Cardinality histograms (parse-ok anchors)

- n_gt_objects: 0:788, 1:870, 2:734, 3:655, 4:505, 5:442, 6:319, 7:211, 8:487
- n_pred_objects: 0:0, 1:895, 2:2130, 3:645, 4:566, 5:157, 6:121, 7:71, 8:426
- n_gt_risk_factors: 0:3271, 1:1277, 2:404, 3:53, 4:6, 5:0, 6:0, 7:0, 8:0
- n_pred_risk_factors: 0:0, 1:3841, 2:1151, 3:14, 4:4, 5:1, 6:0, 7:0, 8:0

## Empty-direction counts (parse-ok anchors)

- object: gt_empty_pred_empty=0, gt_empty_pred_nonempty=788, gt_nonempty_pred_empty=0, gt_nonempty_pred_nonempty=4223
- risk: gt_empty_pred_empty=0, gt_empty_pred_nonempty=3271, gt_nonempty_pred_empty=0, gt_nonempty_pred_nonempty=1740

## Field metrics

| metric | value |
|---|---|
| motion_state_f1_macro | 0.289597 |
| speed_action_f1_macro | 0.144259 |
| yield_required_f1 | 0.368996 |
| yield_miss_rate | 0.311328 |
| yield_overconservative_rate | 0.662791 |
| risk_factors_f1_macro | 0.106376 |
| stop_overconservative_rate | 0.000000 |
| stop_miss_rate | 0.997265 |
| max_items_hit_rate | 0.085013 |

## Risk attribution (conditional, matched-bridge)

| metric | value |
|---|---|
| risk_attribution_f1_micro | 0.211604 |
| risk_attribution_f1_macro | 0.214421 |

| factor | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|
| lead_vehicle_braking | 24 | 49 | 24 | 0.328767 | 0.500000 | 0.396694 |
| object_near | 6 | 181 | 115 | 0.032086 | 0.049587 | 0.038961 |
| oncoming_traffic | 60 | 352 | 145 | 0.145631 | 0.292683 | 0.194489 |
| stationary_obstacle | 0 | 1 | 1 | 0.000000 | 0.000000 | 0.000000 |
| vehicle_crossing | 26 | 67 | 19 | 0.279570 | 0.577778 | 0.376812 |
| vru_crossing | 39 | 105 | 96 | 0.270833 | 0.288889 | 0.279570 |

## Per-class F1

| block | class | f1 |
|---|---|---|
| motion_state_f1_per_class | crossing | 0.211382 |
| motion_state_f1_per_class | oncoming | 0.211444 |
| motion_state_f1_per_class | same_direction | 0.463059 |
| motion_state_f1_per_class | stationary | 0.272503 |
| object_f1_per_category@0.5 | barrier | 0.030769 |
| object_f1_per_category@0.5 | bicycle | 0.134653 |
| object_f1_per_category@0.5 | bus | 0.600712 |
| object_f1_per_category@0.5 | car | 0.450563 |
| object_f1_per_category@0.5 | construction_vehicle | 0.128696 |
| object_f1_per_category@0.5 | motorcycle | 0.210361 |
| object_f1_per_category@0.5 | pedestrian | 0.127694 |
| object_f1_per_category@0.5 | traffic_cone | 0.118196 |
| object_f1_per_category@0.5 | trailer | 0.205128 |
| object_f1_per_category@0.5 | truck | 0.427485 |
| risk_factors_f1_per_class | lead_vehicle_braking | 0.120261 |
| risk_factors_f1_per_class | object_near | 0.112480 |
| risk_factors_f1_per_class | oncoming_traffic | 0.132265 |
| risk_factors_f1_per_class | stationary_obstacle | 0.001850 |
| risk_factors_f1_per_class | vehicle_crossing | 0.061750 |
| risk_factors_f1_per_class | vru_crossing | 0.209651 |
| speed_action_per_class_f1 | ACCELERATE | 0.000000 |
| speed_action_per_class_f1 | DECELERATE | 0.273355 |
| speed_action_per_class_f1 | KEEP_SPEED | 0.299564 |
| speed_action_per_class_f1 | STOP | 0.004115 |

## 95% CI (scene-cluster bootstrap)

| metric | point | ci_low | ci_high |
|---|---|---|---|
| bev_dx_mean | 24.309884 | 23.029790 | 25.645801 |
| bev_dxy_mean | 24.992906 | 23.690243 | 26.352720 |
| bev_dy_mean | 4.516206 | 4.187016 | 4.880570 |
| max_items_hit_rate | 0.085013 | 0.061934 | 0.112545 |
| motion_state_f1_macro | 0.289597 | 0.234887 | 0.334862 |
| object_empty_hallucination_rate | 1.000000 | 1.000000 | 1.000000 |
| object_empty_miss_rate | 0.000000 | 0.000000 | 0.000000 |
| object_f1_macro@0.5 | 0.243426 | 0.210596 | 0.272013 |
| object_f1_micro@0.5 | 0.376398 | 0.350657 | 0.401410 |
| object_precision_micro@0.3 | 0.435410 | 0.403524 | 0.467407 |
| object_precision_micro@0.5 | 0.389777 | 0.358932 | 0.421216 |
| object_precision_micro@0.7 | 0.256601 | 0.232097 | 0.281866 |
| object_recall_micro@0.3 | 0.406511 | 0.379656 | 0.434597 |
| object_recall_micro@0.5 | 0.363906 | 0.338397 | 0.390289 |
| object_recall_micro@0.7 | 0.239570 | 0.217245 | 0.263573 |
| parse_rate | 0.916423 | 0.896308 | 0.935801 |
| risk_attribution_f1_macro | 0.214421 | 0.175089 | 0.291878 |
| risk_attribution_f1_micro | 0.211604 | 0.164915 | 0.254216 |
| risk_empty_hallucination_rate | 1.000000 | 1.000000 | 1.000000 |
| risk_empty_miss_rate | 0.000000 | 0.000000 | 0.000000 |
| risk_factors_f1_macro | 0.106376 | 0.085323 | 0.127492 |
| speed_action_f1_macro | 0.144259 | 0.131918 | 0.156573 |
| stop_miss_rate | 0.997265 | 0.993917 | 0.999488 |
| stop_overconservative_rate | 0.000000 | 0.000000 | 0.000000 |
| yield_miss_rate | 0.311328 | 0.262891 | 0.359609 |
| yield_overconservative_rate | 0.662791 | 0.623951 | 0.700645 |
| yield_required_f1 | 0.368996 | 0.322565 | 0.415859 |

Per-class CI blocks:

| block | class | point | ci_low | ci_high |
|---|---|---|---|---|
| motion_state_f1_per_class | crossing | 0.211382 | 0.038012 | 0.369911 |
| motion_state_f1_per_class | oncoming | 0.211444 | 0.171249 | 0.253939 |
| motion_state_f1_per_class | same_direction | 0.463059 | 0.396795 | 0.523408 |
| motion_state_f1_per_class | stationary | 0.272503 | 0.218160 | 0.335847 |
| object_f1_per_category@0.5 | barrier | 0.030769 | 0.011475 | 0.054294 |
| object_f1_per_category@0.5 | bicycle | 0.134653 | 0.065474 | 0.199652 |
| object_f1_per_category@0.5 | bus | 0.600712 | 0.491869 | 0.686002 |
| object_f1_per_category@0.5 | car | 0.450563 | 0.419822 | 0.483548 |
| object_f1_per_category@0.5 | construction_vehicle | 0.128696 | 0.006705 | 0.310778 |
| object_f1_per_category@0.5 | motorcycle | 0.210361 | 0.118389 | 0.314162 |
| object_f1_per_category@0.5 | pedestrian | 0.127694 | 0.072957 | 0.183984 |
| object_f1_per_category@0.5 | traffic_cone | 0.118196 | 0.022221 | 0.233343 |
| object_f1_per_category@0.5 | trailer | 0.205128 | 0.066667 | 0.301999 |
| object_f1_per_category@0.5 | truck | 0.427485 | 0.355907 | 0.500605 |
| risk_factors_f1_per_class | lead_vehicle_braking | 0.120261 | 0.057065 | 0.185923 |
| risk_factors_f1_per_class | object_near | 0.112480 | 0.070810 | 0.156627 |
| risk_factors_f1_per_class | oncoming_traffic | 0.132265 | 0.103778 | 0.161291 |
| risk_factors_f1_per_class | stationary_obstacle | 0.001850 | 0.000000 | 0.005923 |
| risk_factors_f1_per_class | vehicle_crossing | 0.061750 | 0.036018 | 0.089112 |
| risk_factors_f1_per_class | vru_crossing | 0.209651 | 0.117647 | 0.307997 |
| speed_action_per_class_f1 | ACCELERATE | 0.000000 | 0.000000 | 0.000000 |
| speed_action_per_class_f1 | DECELERATE | 0.273355 | 0.234031 | 0.315011 |
| speed_action_per_class_f1 | KEEP_SPEED | 0.299564 | 0.265140 | 0.333435 |
| speed_action_per_class_f1 | STOP | 0.004115 | 0.000000 | 0.010841 |

## Denominators

- anchors_expected: 5468
- parse_ok: 5011
- matched_pairs: 5757
- motion_pairs: 5757
- bev_pairs: 5757
- yield_counts: {"fn": 382, "fp": 2508, "tn": 1276, "tp": 845}
- stop_counts: {"stop_fn": 1823, "stop_fp": 0, "stop_gt_keep": 3183, "stop_gt_stop": 1828}
- micro_counts: {"0.3": [6431, 8339, 9389], "0.5": [5757, 9013, 10063], "0.7": [3790, 10980, 12030]}

## Cost summary

- gpu_hours: 5.122805
- input_tokens_p50: 1082.000000
- latency_p50: 3.878153
- latency_p90: 3.975950
- peak_cuda_mem_bytes: 13918057984.000000
- throughput_anchors_per_s: 0.296496

## Visual dependence (counterfactual subset)

- blank: n_compared=86, action_flip=0.697674, output_change=1.000000
- shuffled: n_compared=165, action_flip=0.551515, output_change=1.000000
