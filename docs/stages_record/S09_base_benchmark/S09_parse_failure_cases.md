# S09 失败案例目录：1F 全量 parse 失败 19 例（逐例详单）

> 来源：`runs/S09_base_benchmark/predictions_full.jsonl`（test 5,468 anchors，2026-10-04 全量跑）。
> 口径：`parse_ok=false` 的全部 19 例，按错误类别分组逐例结构化落档；raw_text 原文全量保留（含截断断点原文）。
> 总览：json_parse_failure 14（其中 13 例 output_tokens 触顶 512 即截断型）、schema_failure 4、semantic_range_failure 1、generation_failure 0。

## 汇总表

| # | sample_token | 错误类别 | output_tokens | objects_started | 特征 |
|---|---|---|---|---|---|
| 1 | `06be0e3b665c…` | json_parse_failure | 512 | 9 | 截断（触顶512） |
| 2 | `29ba61593299…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 3 | `38a28a3aaf26…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 4 | `6832e7176213…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 5 | `6bfd42cf0aba…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 6 | `6d9984d09d52…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 7 | `700c1a25559b…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 8 | `747aa46b9a46…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 9 | `7fafcee5a8d1…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 10 | `88a6b5e35eb4…` | json_parse_failure | 475 | 8 | 短输出未触顶 |
| 11 | `88bbcc1cfea4…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 12 | `a98fba72bde9…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 13 | `b6b0d9f2f2e1…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 14 | `f5193e87e045…` | json_parse_failure | 512 | 8 | 截断（触顶512） |
| 15 | `01cf25932e6a…` | schema_failure | 115 | 1 | 短输出未触顶 |
| 16 | `3f6ce131c557…` | schema_failure | 130 | 1 | 短输出未触顶 |
| 17 | `7c21c98a10a0…` | schema_failure | 116 | 1 | 短输出未触顶 |
| 18 | `d04c38eb3667…` | schema_failure | 116 | 1 | 短输出未触顶 |
| 19 | `a572dd2e95e9…` | semantic_range_failure | 309 | 5 | 短输出未触顶 |

## json_parse_failure（14 例）

### Case 1: json_parse_failure @ `06be0e3b665c44fa8d17d9f4770bdf9c`

- sample_token: `06be0e3b665c44fa8d17d9f4770bdf9c`
- scene_token: `325cef682f064c55a255f2625c533b75`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 9
- manifest request_hash: `78cc6ce9dade54a5588618830adb90e38d8ac24390fab7cf46914cffb84b7dec`
- 重算 request_hash: `78cc6ce9dade54a5588618830adb90e38d8ac24390fab7cf46914cffb84b7dec`（match=True）
- record_hash: `e5c946fcd0681c1713088c500a495c4368d3ac43b17160081f150d5612d133b0`
- error: `json_parse_failure` — Raw text is not valid JSON: Expecting ',' delimiter: line 53 column 25 (char 1385) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [0, 453, 527, 681],
      "motion_state": "same_direction"
    },
    {
      "category": "truck",
      "bbox_2d": [418, 452, 665, 543],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1214, 470, 1595, 660],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1220, 470, 1595, 660],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1220, 470, 1595, 660],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1220, 470, 1595, 660],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1220, 470, 1595, 660],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1220, 470, 1595, 660],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1220, 470, 1595, 660],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "oncoming_traffic"
  ],
  "reasoning": "The image shows multiple cars on the road, some moving in the same direction as the ego vehicle. There is also oncoming traffic that could pose a risk if the ego vehicle does not maintain its current speed.",
  "yield_required": true
````

### Case 2: json_parse_failure @ `29ba615932994396a7ee5b465f9ea5e9`

- sample_token: `29ba615932994396a7ee5b465f9ea5e9`
- scene_token: `40209c4e465d4b4e8341ebd52be0d842`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `9fdb6f0defd0abbe531929a854d4183af4cc51e3e107a3edc932ec8f6599e06e`
- 重算 request_hash: `9fdb6f0defd0abbe531929a854d4183af4cc51e3e107a3edc932ec8f6599e06e`（match=True）
- record_hash: `35c30bb5fad33001a0b95e2a1a2e65552009c8341f89fa34d37d04f1ec064dba`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 53 column 16 (char 1181) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [819, 435, 876, 497],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [900, 413, 930, 441],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1006, 420, 1036, 463],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1036, 424, 1070, 468],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1070, 426, 1110, 483],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1110, 426, 1146, 472],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1146, 426, 1186, 474],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1186, 426, 1226, 476],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple cars and pedestrians. The lead vehicle is braking, which could indicate a sudden stop or change in direction. There is also a pedestrian crossing the road, which adds to the risk factors. The traffic is congested, and
````

### Case 3: json_parse_failure @ `38a28a3aaf2647f2a8c0e90e31267bf8`

- sample_token: `38a28a3aaf2647f2a8c0e90e31267bf8`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `9b031101f91ea9c4b2cd3e07c65f5dc195cef47ebbe3bdc8501a102abbfa1aab`
- 重算 request_hash: `9b031101f91ea9c4b2cd3e07c65f5dc195cef47ebbe3bdc8501a102abbfa1aab`（match=True）
- record_hash: `fe04e3b82f5e7e30b4cb506a8e818d0545db8a36cb01c109e5e6844e2575bbc2`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 53 column 16 (char 1182) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [741, 482, 785, 514],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [974, 478, 1000, 500],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1000, 478, 1028, 500],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1028, 478, 1056, 500],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1056, 478, 1084, 500],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1084, 478, 1112, 500],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1112, 478, 1140, 500],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1140, 478, 1168, 500],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple vehicles and pedestrians. The risk factors include a pedestrian crossing, vehicles merging, lead vehicle braking, corridor conflict, stationary obstacles, congestion, and oncoming traffic. The ego speed is low, indicating that the vehicle is
````

### Case 4: json_parse_failure @ `6832e717621341568c759151b5974512`

- sample_token: `6832e717621341568c759151b5974512`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `08bbf377aa9868b9a04cc2ff45adf548796b6866f8678ec3c0ac16395a746a55`
- 重算 request_hash: `08bbf377aa9868b9a04cc2ff45adf548796b6866f8678ec3c0ac16395a746a55`（match=True）
- record_hash: `fc504d40058ffaf248692e86ee4f6fe15d2cd4f59849a1a7040697ecbb0cdf83`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 53 column 16 (char 1178) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [435, 457, 584, 536],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [614, 447, 752, 542],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [895, 436, 961, 487],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [978, 427, 1010, 456],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1018, 424, 1038, 444],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1028, 421, 1041, 433],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1038, 420, 1050, 432],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1041, 419, 1053, 431],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple cars and pedestrians. The lead vehicle is braking, which could indicate a sudden stop or change in direction. There is also a pedestrian crossing the road, which requires caution. The traffic is congested, making it difficult to navigate through the
````

### Case 5: json_parse_failure @ `6bfd42cf0aba4f1a94ec11fa43e2dd92`

- sample_token: `6bfd42cf0aba4f1a94ec11fa43e2dd92`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `275e43b95c3f9406b01c9ec3fe02a93db4f557061dd4d8ab77a2a11a712b1d4e`
- 重算 request_hash: `275e43b95c3f9406b01c9ec3fe02a93db4f557061dd4d8ab77a2a11a712b1d4e`（match=True）
- record_hash: `0d0bf85ababb27255f92cf1160f64275e3d10a7f0936eb8590e7aee85ef08a13`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 55 column 19 (char 1469) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [0, 514, 367, 749],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [252, 485, 432, 588],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [884, 460, 997, 558],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1022, 463, 1109, 530],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1138, 462, 1221, 530],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1221, 442, 1266, 497],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1266, 442, 1318, 497],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1318, 442, 1371, 497],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple vehicles and pedestrians. The lead vehicle is braking, which could indicate a potential conflict. There is also a pedestrian crossing the road, increasing the risk of a collision.",
  "yield_required": true,
  "speed_action": "DECELER
````

### Case 6: json_parse_failure @ `6d9984d09d52479e837da2fd09e192cc`

- sample_token: `6d9984d09d52479e837da2fd09e192cc`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `239013b2c900b44d24b5c7be6e56c4b1d990a46901e1ff8208584308673312fb`
- 重算 request_hash: `239013b2c900b44d24b5c7be6e56c4b1d990a46901e1ff8208584308673312fb`（match=True）
- record_hash: `0bf85854d800644c93b07570eda70658949bfe8f51ee8a998c9031a5c5947c22`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 53 column 16 (char 1191) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [737, 485, 785, 523],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1068, 475, 1186, 568],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1148, 483, 1321, 602],
      "motion_state": "same_direction"
    },
    {
      "category": "truck",
      "bbox_2d": [428, 460, 580, 578],
      "motion_state": "same_direction"
    },
    {
      "category": "bicycle",
      "bbox_2d": [268, 522, 297, 618],
      "motion_state": "same_direction"
    },
    {
      "category": "bicycle",
      "bbox_2d": [192, 548, 227, 633],
      "motion_state": "same_direction"
    },
    {
      "category": "bicycle",
      "bbox_2d": [148, 562, 180, 650],
      "motion_state": "same_direction"
    },
    {
      "category": "bicycle",
      "bbox_2d": [115, 566, 148, 656],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple vehicles and pedestrians. The lead vehicle is braking, which could indicate a potential collision risk. Additionally, there is a pedestrian crossing the road, increasing the risk of a pedestrian accident. The ego vehicle should decelerate to avoid any potential collisions
````

### Case 7: json_parse_failure @ `700c1a25559b4433be532de3475e58a9`

- sample_token: `700c1a25559b4433be532de3475e58a9`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `333bc77f32b5bdb9f5bab7aaa83a472b60b6562059339148d7fd076986f30606`
- 重算 request_hash: `333bc77f32b5bdb9f5bab7aaa83a472b60b6562059339148d7fd076986f30606`（match=True）
- record_hash: `a171db882885fa1332c3eb5872d0f0b0794b26a6efffd8f59e292dc42ad2fc05`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 53 column 16 (char 1180) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [594, 471, 750, 578],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [855, 459, 911, 503],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [972, 460, 1039, 511],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1080, 459, 1124, 494],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1022, 448, 1040, 460],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1018, 448, 1022, 452],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1008, 448, 1018, 452],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1000, 448, 1008, 452],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple vehicles and pedestrians. The lead vehicle is braking, which could indicate a sudden stop or change in direction. There is also a pedestrian crossing the road, which adds to the risk factors. The traffic is congested, and vehicles
````

### Case 8: json_parse_failure @ `747aa46b9a4641fe90db05d97db2acea`

- sample_token: `747aa46b9a4641fe90db05d97db2acea`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `09b9d8a7864d596fd38751fa5a9dc5a706dbca322717bb0b8164db26862cc264`
- 重算 request_hash: `09b9d8a7864d596fd38751fa5a9dc5a706dbca322717bb0b8164db26862cc264`（match=True）
- record_hash: `4625567349e32679d3fbdec0205cf9e656c2b47eda7a153565c7c7daa1968bc6`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 53 column 16 (char 1173) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [335, 460, 619, 637],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [609, 452, 709, 511],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [799, 439, 871, 494],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [950, 439, 1017, 490],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1030, 438, 1088, 479],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [864, 426, 902, 455],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [968, 415, 990, 438],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [998, 415, 1012, 430],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple vehicles and pedestrians. The lead vehicle is braking, which could indicate a sudden stop or change in direction. There is also a pedestrian crossing the road, which adds to the risk factors. The traffic is congested, and vehicles are merging, making it crucial to
````

### Case 9: json_parse_failure @ `7fafcee5a8d1453397d168c3f9568564`

- sample_token: `7fafcee5a8d1453397d168c3f9568564`
- scene_token: `dce6f3f2bf6b4859abcf3268581969d3`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `0da030d07318f0a7ea5dc38a13cc714665e0d86c9fe7674273de526e8f764d72`
- 重算 request_hash: `0da030d07318f0a7ea5dc38a13cc714665e0d86c9fe7674273de526e8f764d72`（match=True）
- record_hash: `ee2d0a0778291450d5b2f3900a4311f900ebd702c477f2cb0bc8f278dcf45709`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 55 column 3 (char 1471) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [640, 471, 675, 506],
      "motion_state": "same_direction"
    },
    {
      "category": "truck",
      "bbox_2d": [158, 477, 382, 584],
      "motion_state": "same_direction"
    },
    {
      "category": "construction_vehicle",
      "bbox_2d": [0, 480, 164, 552],
      "motion_state": "stationary"
    },
    {
      "category": "bus",
      "bbox_2d": [622, 428, 675, 480],
      "motion_state": "same_direction"
    },
    {
      "category": "trailer",
      "bbox_2d": [1000, 456, 1097, 536],
      "motion_state": "same_direction"
    },
    {
      "category": "barrier",
      "bbox_2d": [0, 480, 164, 552],
      "motion_state": "stationary"
    },
    {
      "category": "motorcycle",
      "bbox_2d": [1000, 456, 1097, 536],
      "motion_state": "same_direction"
    },
    {
      "category": "bicycle",
      "bbox_2d": [1000, 456, 1097, 536],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The road is wet, and there are multiple vehicles and barriers on the left side of the road. The lead vehicle is braking, which could cause a sudden stop. There is also a pedestrian crossing the road, which increases the risk of accidents.",
  "yield_required": true,
  "speed
````

### Case 10: json_parse_failure @ `88a6b5e35eb44deba6b1065d74b0a3c4`

- sample_token: `88a6b5e35eb44deba6b1065d74b0a3c4`
- scene_token: `01452fbfbf4543af8acdfd3e8a1ee806`
- output_tokens: 475
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `ba306da3a0a9dca63703383f5426efd917dc8c929075f0562b64a35d4cc86391`
- 重算 request_hash: `ba306da3a0a9dca63703383f5426efd917dc8c929075f0562b64a35d4cc86391`（match=True）
- record_hash: `5ba4d2e0e08e5157160b6ca66d78d0761c74bfa177be378217301d89f9ad3dbb`
- error: `json_parse_failure` — Raw text is not valid JSON: Expecting property name enclosed in double quotes: line 40 column 42 (char 928)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [461, 500, 620, 597],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [401, 490, 502, 562],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [908, 492, 1012, 573],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [605, 480, 687, 551],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [667, 474, 711, 518],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [704, 468, 738, 504],
      "motion_state": "same_direction"
    },
    {
      "category": "construction_vehicle",
      "bbox_2d": [740, 449, 763, 480],
      "motion_state": "stationary"
    },
    {
      "category": "barrier",
      "bbox_2d": [1398, 522, 1538, 644], // Assuming this is a barrier based on the context
      "motion_state": "stationary"
    }
  ],
  "risk_factors": [
    "corridor_conflict"
  ],
  "reasoning": "There is a potential corridor conflict due to the presence of a construction vehicle and barriers blocking part of the road. The ego vehicle should be cautious and maintain a safe distance.",
  "yield_required": true,
  "speed_action": "DECELERATE"
}
````

### Case 11: json_parse_failure @ `88bbcc1cfea447b2a3d6ca38f5c12174`

- sample_token: `88bbcc1cfea447b2a3d6ca38f5c12174`
- scene_token: `3a2d9bf6115f40898005d1c1df2b7282`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `0aa7beef50d8431cada421945b48409d842d2f555bc99b21096797b1f1846db2`
- 重算 request_hash: `0aa7beef50d8431cada421945b48409d842d2f555bc99b21096797b1f1846db2`（match=True）
- record_hash: `8b798d00be4c6ebb9b0990f494a0655228b80e47aee7eb98bc1052a61ef694e7`
- error: `json_parse_failure` — Raw text is not valid JSON: Expecting value: line 54 column 20 (char 1557) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [324, 482, 556, 619],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [227, 484, 440, 626],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [560, 463, 648, 555],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [610, 462, 658, 540],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [658, 462, 710, 522],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [710, 462, 757, 522],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [757, 462, 808, 519],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [808, 462, 848, 519],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy urban street with multiple vehicles and pedestrians. The presence of a construction vehicle and barriers suggests potential obstructions or changes in traffic flow. The risk factors include a pedestrian crossing, vehicles merging, lead vehicle braking, corridor conflict, stationary obstacles, congestion, and oncoming traffic.",
  "yield_required":
````

### Case 12: json_parse_failure @ `a98fba72bde9433fb882032d18aedb2e`

- sample_token: `a98fba72bde9433fb882032d18aedb2e`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `66f7f253fe6f8df81bd24d2221a42070a89120c292701a57b2ed392a2100798a`
- 重算 request_hash: `66f7f253fe6f8df81bd24d2221a42070a89120c292701a57b2ed392a2100798a`（match=True）
- record_hash: `f13883922dcc8ca23229de6dca17e8eb7029dbe5b07b92aecc097fbbe89a9585`
- error: `json_parse_failure` — Raw text is not valid JSON: Expecting property name enclosed in double quotes: line 54 column 1 (char 1471) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [493, 476, 604, 558],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [851, 475, 912, 526],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [989, 485, 1106, 559],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1058, 473, 1106, 507],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1004, 473, 1058, 507],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1058, 473, 1096, 507],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1106, 473, 1140, 507],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1140, 473, 1173, 507],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "There are multiple vehicles in the same direction as the ego vehicle, which may indicate a potential merging or congestion. Additionally, there is a pedestrian crossing the path, which could pose a risk. The lead vehicle appears to be braking, which might affect the ego vehicle's speed.",

````

### Case 13: json_parse_failure @ `b6b0d9f2f2e14a3aaa2c8aedeb1edb69`

- sample_token: `b6b0d9f2f2e14a3aaa2c8aedeb1edb69`
- scene_token: `fcbccedd61424f1b85dcbf8f897f9754`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `01cc0de4dc6ddb0e7b3ffe57213c92e9484a4a26dc6d3a1081507a2e24945e6a`
- 重算 request_hash: `01cc0de4dc6ddb0e7b3ffe57213c92e9484a4a26dc6d3a1081507a2e24945e6a`（match=True）
- record_hash: `32bef6f5b460c424570d0e17e8087117fd0b99c0782ff74d335744e3216d0182`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 55 column 3 (char 1465) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "car",
      "bbox_2d": [324, 470, 583, 566],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [856, 462, 917, 511],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [648, 468, 734, 528],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [934, 474, 1053, 562],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [956, 458, 1004, 498],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1004, 458, 1024, 476],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1024, 458, 1040, 476],
      "motion_state": "same_direction"
    },
    {
      "category": "car",
      "bbox_2d": [1040, 458, 1056, 476],
      "motion_state": "same_direction"
    }
  ],
  "risk_factors": [
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic"
  ],
  "reasoning": "The ego vehicle is approaching a busy intersection with multiple vehicles and pedestrians. The lead vehicle is braking, which could indicate a sudden stop or change in direction. There is also a pedestrian crossing the road, which adds to the risk factors.",
  "yield_required": true,
  "speed_action
````

### Case 14: json_parse_failure @ `f5193e87e04547808e974d88fed8159e`

- sample_token: `f5193e87e04547808e974d88fed8159e`
- scene_token: `2eb4d7f00e584a548aa0b899638bfb0a`
- output_tokens: 512（触顶 512，截断型失败）
- objects_started（raw_text 中 category 出现次数）: 8
- manifest request_hash: `19f46a1b6e5c7819aa160acdf73132547d491997e0ce1710b0bbfa62304e0349`
- 重算 request_hash: `19f46a1b6e5c7819aa160acdf73132547d491997e0ce1710b0bbfa62304e0349`（match=True）
- record_hash: `79378b444ab8be0b5020c5e229d40070f72d2f863ea88e3a4998e88d8448ed1f`
- error: `json_parse_failure` — Raw text is not valid JSON: Unterminated string starting at: line 53 column 18 (char 1415) (raw text does not end with '}': output looks truncated, check the generation token budget)
- extracted_from_fences: False

raw_text 原文：

````text
{
    "critical_objects": [
        {
            "category": "car",
            "bbox_2d": [693, 484, 765, 544],
            "motion_state": "same_direction"
        },
        {
            "category": "car",
            "bbox_2d": [808, 488, 883, 548],
            "motion_state": "same_direction"
        },
        {
            "category": "car",
            "bbox_2d": [751, 494, 778, 529],
            "motion_state": "same_direction"
        },
        {
            "category": "car",
            "bbox_2d": [539, 504, 591, 541],
            "motion_state": "same_direction"
        },
        {
            "category": "car",
            "bbox_2d": [513, 506, 542, 540],
            "motion_state": "same_direction"
        },
        {
            "category": "car",
            "bbox_2d": [572, 504, 600, 536],
            "motion_state": "same_direction"
        },
        {
            "category": "car",
            "bbox_2d": [542, 506, 572, 540],
            "motion_state": "same_direction"
        },
        {
            "category": "car",
            "bbox_2d": [442, 502, 520, 552],
            "motion_state": "same_direction"
        }
    ],
    "risk_factors": [
        "pedestrian_crossing",
        "vehicle_merging",
        "lead_vehicle_braking",
        "corridor_conflict",
        "stationary_obstacle",
        "congestion",
        "oncoming_traffic"
    ],
    "reasoning": "The ego vehicle is approaching a busy intersection with multiple vehicles and pedestrians. The lead vehicle is braking, which could indicate a sudden stop or change in direction. There is also a pedestrian crossing the road, which adds to the risk factors. The traffic is congested, and vehicles are merging, making it important to maintain a safe speed
````


## schema_failure（4 例）

### Case 15: schema_failure @ `01cf25932e6a449e9398b0a0219de47e`

- sample_token: `01cf25932e6a449e9398b0a0219de47e`
- scene_token: `e8099a6136804f3bb9b38ff94d98eb64`
- output_tokens: 115
- objects_started（raw_text 中 category 出现次数）: 1
- manifest request_hash: `e957d6e685059265fd1fb17d817375bfc2e8b3e68b5dbab57ddf5b862c74bd2e`
- 重算 request_hash: `e957d6e685059265fd1fb17d817375bfc2e8b3e68b5dbab57ddf5b862c74bd2e`（match=True）
- record_hash: `74e14bc0e901393d79f24ebcaeab33f9b9c8ddf1d6646cef39e1c8ec20e6b307`
- error: `schema_failure` — 'construction_vehicle' is not one of ['pedestrian_crossing', 'vehicle_merging', 'lead_vehicle_braking', 'corridor_conflict', 'stationary_obstacle', 'congestion', 'oncoming_traffic']
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "construction_vehicle",
      "bbox_2d": [530, 461, 748, 894],
      "motion_state": "oncoming"
    }
  ],
  "risk_factors": [
    "construction_vehicle"
  ],
  "reasoning": "There is a construction vehicle on the road, which may be obstructing the path and requires caution.",
  "yield_required": true,
  "speed_action": "DECELERATE"
}
````

### Case 16: schema_failure @ `3f6ce131c557499d847d012cb9a555db`

- sample_token: `3f6ce131c557499d847d012cb9a555db`
- scene_token: `c92fdd793fbb4401b12782a9b8d4a499`
- output_tokens: 130
- objects_started（raw_text 中 category 出现次数）: 1
- manifest request_hash: `8e04ed2ad1786be32b44b8e57af12e7b11993d309a5c935c0ef2392c9c33a4d5`
- 重算 request_hash: `8e04ed2ad1786be32b44b8e57af12e7b11993d309a5c935c0ef2392c9c33a4d5`（match=True）
- record_hash: `92b270fd56cf0b0112f18045d863a7db51d255b7baa4570e23801a1e22add7fd`
- error: `schema_failure` — 'motion_state' is a required property
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "traffic_cone",
      "bbox_2d": [617, 496, 644, 550],
      "bbox_2d": [638, 496, 656, 537]
    }
  ],
  "risk_factors": [
    "pedestrian_crossing"
  ],
  "reasoning": "There is a pedestrian crossing the ego path, which requires caution.",
  "yield_required": true,
  "speed_action": "DECELERATE"
}
````

### Case 17: schema_failure @ `7c21c98a10a04997975b8d527e15bc81`

- sample_token: `7c21c98a10a04997975b8d527e15bc81`
- scene_token: `f97bf749746c4c3a8ad9f1c11eab6444`
- output_tokens: 116
- objects_started（raw_text 中 category 出现次数）: 1
- manifest request_hash: `7bafa6eef80c5e8673393657c09ff958a443133f0a2021b15e671742a6276714`
- 重算 request_hash: `7bafa6eef80c5e8673393657c09ff958a443133f0a2021b15e671742a6276714`（match=True）
- record_hash: `4f5fbe9f07b2a1806ceffaf6106ca644e97d9bed0aa6008109fbaa332257b330`
- error: `schema_failure` — 'construction_vehicle' is not one of ['pedestrian_crossing', 'vehicle_merging', 'lead_vehicle_braking', 'corridor_conflict', 'stationary_obstacle', 'congestion', 'oncoming_traffic']
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "construction_vehicle",
      "bbox_2d": [470, 439, 600, 600],
      "motion_state": "oncoming"
    }
  ],
  "risk_factors": [
    "construction_vehicle"
  ],
  "reasoning": "A construction vehicle is approaching from the oncoming side, which could pose a risk to the ego vehicle.",
  "yield_required": true,
  "speed_action": "DECELERATE"
}
````

### Case 18: schema_failure @ `d04c38eb3667435cbcadc46ee8eae374`

- sample_token: `d04c38eb3667435cbcadc46ee8eae374`
- scene_token: `265f002f02d447ad9074813292eef75e`
- output_tokens: 116
- objects_started（raw_text 中 category 出现次数）: 1
- manifest request_hash: `d5ed261dd41b2039bda14fbb3583c48fa5cadd4c07a36220af0992a29ffe53da`
- 重算 request_hash: `d5ed261dd41b2039bda14fbb3583c48fa5cadd4c07a36220af0992a29ffe53da`（match=True）
- record_hash: `32b46c2e12ada2a69c936abe8253f127177073e99acf35311f8639e2603f3d07`
- error: `schema_failure` — 'construction_vehicle' is not one of ['pedestrian_crossing', 'vehicle_merging', 'lead_vehicle_braking', 'corridor_conflict', 'stationary_obstacle', 'congestion', 'oncoming_traffic']
- extracted_from_fences: False

raw_text 原文：

````text
{
  "critical_objects": [
    {
      "category": "construction_vehicle",
      "bbox_2d": [134, 458, 290, 536],
      "motion_state": "oncoming"
    }
  ],
  "risk_factors": [
    "construction_vehicle"
  ],
  "reasoning": "A construction vehicle is approaching from the oncoming side, which could pose a risk to the ego vehicle.",
  "yield_required": true,
  "speed_action": "DECELERATE"
}
````


## semantic_range_failure（1 例）

### Case 19: semantic_range_failure @ `a572dd2e95e94e4db66bb5dbfaf870b5`

- sample_token: `a572dd2e95e94e4db66bb5dbfaf870b5`
- scene_token: `6f83169d067343658251f72e1dd17dbc`
- output_tokens: 309
- objects_started（raw_text 中 category 出现次数）: 5
- manifest request_hash: `2049e424dc98018a9b2af0985514276a807f3474c4cedf3395ef46719800e69c`
- 重算 request_hash: `2049e424dc98018a9b2af0985514276a807f3474c4cedf3395ef46719800e69c`（match=True）
- record_hash: `5c0f6def2fd807eeedc7f70ecb317c9dcfc615dcb70b7fbda615ab1d142907a9`
- error: `semantic_range_failure` — bbox_2d corners are unordered: got [526.0, 410.0, 523.0, 417.0]
- extracted_from_fences: False

raw_text 原文：

````text
{
    "critical_objects": [
        {
            "category": "pedestrian",
            "bbox_2d": [526, 410, 523, 417],
            "motion_state": "crossing"
        },
        {
            "category": "pedestrian",
            "bbox_2d": [139, 448, 212, 726],
            "motion_state": "crossing"
        },
        {
            "category": "pedestrian",
            "bbox_2d": [1, 449, 178, 784],
            "motion_state": "crossing"
        },
        {
            "category": "pedestrian",
            "bbox_2d": [802, 457, 908, 710],
            "motion_state": "crossing"
        },
        {
            "category": "pedestrian",
            "bbox_2d": [1125, 407, 1300, 778],
            "motion_state": "crossing"
        }
    ],
    "risk_factors": [
        "pedestrian_crossing"
    ],
    "reasoning": "There are pedestrians crossing the street, which increases the risk of collision.",
    "yield_required": true,
    "speed_action": "DECELERATE"
}
````


## 错误分析（按错误原因）

19 例 parse 失败共归为五类原因，逐类给出机制说明与真实输出示例（示例均为 raw_text 原文摘录，格式保持原样）。

### 原因一：max_new_tokens=512 预算截断 —— 13 例（json_parse_failure @ 512）

JSON 未闭合即触顶：7 例断在 reasoning 字符串中间（`Unterminated string`），6 例断在尾部结构字段附近、距闭合仅数个 token。共性根因是退化冗长输出——先堆叠 8 个 critical_objects（成功输出均值 2.3）再进入长 reasoning；其中 `06be0e3b…` 堆了 9 个对象，即使完成也会被 maxItems=8 拒绝。另有 1 例 `6a2b60e5…` 恰在 512 处完美闭合解析成功（预算边界的幸运拟合，压在边界上的实际是 14 条）。

**处置裁定**：max_new_tokens=512 保持冻结值不动——预注册纪律（跑完 test 后不回头调参）；失败量级 0.24% 不动任何指标；"能解析"不等于"值得解析"。提额至 768 留给未来配置代际；S12 SFT 后截断率预期自然下降，若仍高再评估。

示例（`6bfd42cf…`，原文末尾，断在枚举值中间、差 2 个 token 闭合）：

```text
  "yield_required": true,
  "speed_action": "DECELER
```

### 原因二：JSON 非法行内注释 —— 1 例（`88a6b5e35eb4…`，475 tok）

完整闭合后 EOS 停止，但 Base 在 bbox 值后写了一条 JS 风格注释——JSON 不允许注释，解析器直接拒绝。与 token 预算无关，是模型试图内联解释的有趣行为样本。

```text
"bbox_2d": [1398, 522, 1538, 644], // Assuming this is a barrier based on the context
```

### 原因三：类别词写入 risk_factors 枚举 —— 3 例（`01cf…` / `7c21…` / `d04c…`，115–116 tok）

同一模式复现三次：把类别名 `construction_vehicle` 当作风险因子写进 `risk_factors`（类别→风险字段混淆），三例的对象均为 construction_vehicle + `motion_state: oncoming`。

```text
"critical_objects": [ { "category": "construction_vehicle", "bbox_2d": [530, 461, 748, 894], "motion_state": "oncoming" } ],
"risk_factors": [ "construction_vehicle" ]
```

### 原因四：缺 motion_state 且重复 bbox_2d 键 —— 1 例（`3f6ce131…`，130 tok）

单个对象里写了两条 `bbox_2d`（像是想给一个物体两个框）并因此丢了必填的 `motion_state`，schema 以 required property 缺失拒绝。

```text
{ "category": "traffic_cone", "bbox_2d": [617, 496, 644, 550], "bbox_2d": [638, 496, 656, 537] }
```

### 原因五：bbox 角点逆序 —— 1 例（`a572dd2e…`，309 tok）

`bbox_2d` 违反 x1<x2 的语义范围约束（x1=526 > x2=523），`semantic_range_failure`。

```text
{ "category": "pedestrian", "bbox_2d": [526, 410, 523, 417], "motion_state": "crossing" }
```

### 对象重复的交叉验证（2026-10-04 用户假设，数据核实）

13 条截断案例的对象重叠检测：2 条明确重复循环（`06be0e3b…` 同一 car 框逐字重复 9 次、21 对 IoU=1.00；`7fafcee5…` 同一像素框重复 4 次且换 4 个类别标签）、3 条轻度重叠（maxIoU 0.52–0.79）、9 条完全无重叠。对照：成功输出 10,980 对象中 IoU≥0.8 重复对 64 个（31 帧，0.57%），失败帧重复密度高约 40 倍但全部来自上述 2 帧。结论：重复循环是退化输出的极端形态而非截断的普遍直接原因；一对一贪心匹配自动惩罚两类行为（重复预测计 FP、"同框换标"因跨类别禁配计 FP+FN），无需专门处理。

### 评测侧处理

全部 19 例由批处理降级记录、未中止运行（generation_failure = 0）；Step 5 评测按 parse_ok 分母排除本目录所列 anchors，`parse_error_counts` 键逐类落报告。
