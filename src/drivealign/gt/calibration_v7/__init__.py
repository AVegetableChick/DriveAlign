"""S13 v7 标定扫描包（Stage A 中间量提取 + Stage B 网格扫描）。

- :mod:`drivealign.gt.calibration_v7.extract` — Stage A：参数无关中间量
  一次性提取（含同步执行的 G3 闸门：v6 公式重建 7 因子 + yield 与
  dataset_v4 冻结 GT 逐 anchor 100% 对比）。
- :mod:`drivealign.gt.calibration_v7.sweep` — Stage B：B0–B5 扫描块
  （全部为中间量上的向量化比较，零 nuscenes 依赖）。

设计基准：docs/stages_record/S13_contract_and_eval_revision/
S13_v7_calibration_plan.md（train-only 纪律、必报项映射、终选准则）。
"""
