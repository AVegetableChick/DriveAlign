"""S12 Step 1 单测：正式冻结配置与 smoke / base 基准的**面 parity**。

覆盖 §3 Step 1 的 `test_sft_config_parity`：`configs/sft/sft_1f.yaml` 与
`configs/sft/sft_smoke_1f.yaml` 的 **face / DUMP / LoRA pattern / max_pixels /
生成参数**逐值相同（只允许训练规模类字段不同），防止"正式配置悄悄换了 face"。

追加一条同源检查（同一份 parity 担心，落在 benchmark 侧）：`sft_1f_v6.yaml`
与 `base_1f_v6.yaml` 的面字段逐值相同，且其 `max_pixels` 与训练侧**逐值相同**
（A1：train/eval 同分辨率口径）。

启动命令：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_config_parity.py -q``
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]     # DriveAlign/
SMOKE_SFT = REPO / "configs" / "sft" / "sft_smoke_1f.yaml"
FULL_SFT = REPO / "configs" / "sft" / "sft_1f.yaml"
BASE_BENCH = REPO / "configs" / "benchmark" / "base_1f_v6.yaml"
SFT_BENCH = REPO / "configs" / "benchmark" / "sft_1f_v6.yaml"

#: 训练侧分辨率口径（A1，2026-10-06 用户纠正后冻结）。
EXPECTED_MAX_PIXELS = 360000

#: 必须**逐值相同**的训练配置字段（面 + 目标序列化 + 模型装载 + LoRA 形状/范围）。
FACE_FIELDS = ("contract_version", "input_policy", "seed", "target_dump", "runtime")
MODEL_FIELDS = ("path", "revision", "dtype", "device_map", "attn_implementation")
LORA_FIELDS = ("r", "alpha", "dropout", "target_modules")
#: 训练规模类字段（允许不同）：lr / epochs / save_* 与 batch 等。


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def smoke() -> dict:
    return _load(SMOKE_SFT)


@pytest.fixture(scope="module")
def full() -> dict:
    return _load(FULL_SFT)


class TestSftConfigParity:
    def test_both_configs_frozen(self, smoke, full):
        assert smoke["status"] == "frozen"
        assert full["status"] == "frozen"

    def test_face_fields_identical(self, smoke, full):
        for field in FACE_FIELDS:
            assert full[field] == smoke[field], f"face field drifted: {field}"
        # v6 面（训练面 == 评测面）是本阶段的硬前提。
        assert full["contract_version"] == "v6"
        assert full["input_policy"] == "1F"

    def test_model_and_processor_identical(self, smoke, full):
        for field in MODEL_FIELDS:
            assert full["model"][field] == smoke["model"][field], f"model.{field} drifted"
        assert full["model"]["processor"] == smoke["model"]["processor"]
        assert full["model"]["processor"]["max_pixels"] == EXPECTED_MAX_PIXELS

    def test_lora_identical_and_path_qualified(self, smoke, full):
        for field in LORA_FIELDS:
            assert full["lora"][field] == smoke["lora"][field], f"lora.{field} drifted"
        # B7：必须是**路径限定正则字符串**（str），不能是裸后缀 list。
        target = full["lora"]["target_modules"]
        assert isinstance(target, str)
        assert target.startswith("model\\.layers\\.")
        assert "visual" not in target

    def test_generation_identical(self, smoke, full):
        assert full["generation"] == smoke["generation"]
        # S09 冻结解码口径。
        assert full["generation"]["do_sample"] is False
        assert full["generation"]["temperature"] == 0.0
        assert full["generation"]["max_new_tokens"] == 512

    def test_optimization_fields_identical_except_lr(self, smoke, full):
        """优化器/调度/batch 口径不变，只有 lr 与训练规模按 D/B/H 拍板变化。"""
        for field in (
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "optimizer",
            "lr_scheduler_type",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "gradient_checkpointing",
        ):
            assert full["training"][field] == smoke["training"][field], f"training.{field} drifted"

    def test_full_training_scale_is_the_frozen_values(self, full):
        """§2.5 / §6 B·D·H 的四个拍板值（冻结后不得就地改写）。"""
        assert full["training"]["learning_rate"] == 5.0e-5
        assert full["training"]["num_train_epochs"] == 1
        assert full["training"]["save_steps"] == 500
        assert full["training"]["save_total_limit"] is None

    def test_no_stage_section_can_override_epochs(self, full):
        """sft_train 的 seg 缺省是 `train32`；正式配置不得留阶段覆盖把 1 epoch 改回 2。"""
        assert full["training"]["num_train_epochs"] == 1
        for seg in ("train32", "overfit"):
            override = full.get(seg, {}).get("training", {}).get("num_train_epochs")
            assert override in (None, 1), f"{seg} override would change epochs to {override}"


class TestBenchmarkConfigParity:
    @pytest.fixture(scope="module")
    def base_bench(self) -> dict:
        return _load(BASE_BENCH)

    @pytest.fixture(scope="module")
    def sft_bench(self) -> dict:
        return _load(SFT_BENCH)

    def test_face_fields_identical(self, base_bench, sft_bench):
        assert sft_bench["status"] == "frozen"
        assert base_bench["status"] == "frozen"
        for field in (
            "split",
            "input_policy",
            "contract_version",
            "image_transform",
            "generation",
            "speed_line_precision",
            "seed",
            "runtime",
        ):
            assert sft_bench[field] == base_bench[field], f"benchmark face field drifted: {field}"

    def test_model_matches_base_besides_max_pixels(self, base_bench, sft_bench):
        for field in MODEL_FIELDS:
            assert sft_bench["model"][field] == base_bench["model"][field], f"model.{field} drifted"
        # processor 以 base 为基准逐键比对（SFT 侧只允许**新增** max_pixels）。
        base_processor = base_bench["model"]["processor"]
        for key, value in base_processor.items():
            assert sft_bench["model"]["processor"][key] == value, f"processor.{key} drifted"

    def test_max_pixels_matches_training_side(self, sft_bench, full):
        """A1：train 与 eval 两侧必须逐值同一个 max_pixels。"""
        assert sft_bench["model"]["processor"]["max_pixels"] == EXPECTED_MAX_PIXELS
        assert sft_bench["model"]["processor"]["max_pixels"] == full["model"]["processor"]["max_pixels"]

    def test_adapter_path_present_and_base_untouched(self, base_bench, sft_bench):
        assert isinstance(sft_bench["adapter_path"], str)
        assert sft_bench["adapter_path"]
        # base 基准配置不得带 adapter（否则会污染 S09 v6 面读数的可复现性）。
        assert "adapter_path" not in base_bench


class TestSftV7ConfigParity:
    """S13 Step 6：v7 三件 run config 的面 parity + V7-A2 口径 pin。

    覆盖 `base_1f_v7.yaml`（benchmark base）/ `sft_1f_v7.yaml`（训练）/
    `benchmark/sft_1f_v7.yaml`（SFT 评测）：代际（contract v7）、V7-A2 生成
    预算 700、紧凑发射（target_dump ↔ targets.py DUMP 同源）、训练/优化面与
    v6 冻结件逐值同（v7 只换代际与发射形态）、A1 train/eval 同分辨率。

    注意：`sft_1f_v7.yaml` 已于 2026-10-15 GPU 训练前冻结（sha 登记决策台账 §9）；
    `benchmark/sft_1f_v7.yaml` 已于 2026-10-15 SFT 评测开跑前冻结（best = 训练末点
    adapter，见 best_freeze.json 溯源）。
    """

    @pytest.fixture(scope="class")
    def v7_sft(self) -> dict:
        return _load(REPO / "configs" / "sft" / "sft_1f_v7.yaml")

    @pytest.fixture(scope="class")
    def v7_base_bench(self) -> dict:
        return _load(REPO / "configs" / "benchmark" / "base_1f_v7.yaml")

    @pytest.fixture(scope="class")
    def v7_sft_bench(self) -> dict:
        return _load(REPO / "configs" / "benchmark" / "sft_1f_v7.yaml")

    def test_status_gate_alignment(self, v7_sft, v7_base_bench, v7_sft_bench):
        """CLI 强制 status=frozen（load_config fail-fast）：base_1f_v7 已冻结
        开跑；SFT 训练件/评测件均已冻结（2026-10-15 训练前/评测前）。"""
        assert v7_base_bench["status"] == "frozen"
        assert v7_sft["status"] == "frozen"
        assert v7_sft_bench["status"] == "frozen"

    def test_v7_face_and_v7a2_generation(self, v7_sft, v7_base_bench, v7_sft_bench):
        for cfg in (v7_sft, v7_base_bench, v7_sft_bench):
            assert cfg["contract_version"] == "v7"
            assert cfg["input_policy"] == "1F"
            # S09 冻结解码口径（greedy），V7-A2 统一预算 700。
            assert cfg["generation"]["do_sample"] is False
            assert cfg["generation"]["temperature"] == 0.0
            assert cfg["generation"]["max_new_tokens"] == 700
        # base 基准不得带 adapter（与 v6 同款纪律）。
        assert "adapter_path" not in v7_base_bench

    def test_target_dump_matches_targets_py(self, v7_sft):
        """训练 config 的 target_dump 必须与 targets.py 的 DUMP 同源
        （模型发射形态 = SFT 监督文本形态，V7-A2）。"""
        from drivealign.sft.targets import DUMP

        assert v7_sft["target_dump"] == {
            "ensure_ascii": DUMP["ensure_ascii"],
            "separators": list(DUMP["separators"]),
        }
        # V7-A2：紧凑发射（替代 v6 的 [", ", ": "]）。
        assert v7_sft["target_dump"]["separators"] == [",", ":"]

    def test_training_fields_unchanged_from_v6(self, v7_sft, full):
        """训练规模 / LoRA / 优化面与 `sft_1f.yaml`（frozen）逐值同。"""
        for field in LORA_FIELDS:
            assert v7_sft["lora"][field] == full["lora"][field], f"lora.{field} drifted"
        for field in (
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "num_train_epochs",
            "optimizer",
            "lr_scheduler_type",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "gradient_checkpointing",
            "save_steps",
            "save_total_limit",
        ):
            assert v7_sft["training"][field] == full["training"][field], f"training.{field} drifted"
        assert v7_sft["model"]["processor"]["max_pixels"] == EXPECTED_MAX_PIXELS
        assert v7_sft["seed"] == full["seed"]
        # 路径限定正则纪律（B7）沿用。
        assert isinstance(v7_sft["lora"]["target_modules"], str)
        assert "visual" not in v7_sft["lora"]["target_modules"]

    def test_v7_bench_parity(self, v7_base_bench, v7_sft_bench, v7_sft):
        for field in (
            "split",
            "input_policy",
            "contract_version",
            "image_transform",
            "generation",
            "speed_line_precision",
            "seed",
            "runtime",
        ):
            assert v7_sft_bench[field] == v7_base_bench[field], f"benchmark face field drifted: {field}"
        base_processor = v7_base_bench["model"]["processor"]
        for key, value in base_processor.items():
            assert v7_sft_bench["model"]["processor"][key] == value, f"processor.{key} drifted"
        # A1：train 与 eval 两侧必须逐值同一个 max_pixels。
        assert v7_sft_bench["model"]["processor"]["max_pixels"] == v7_sft["model"]["processor"]["max_pixels"]
        assert isinstance(v7_sft_bench["adapter_path"], str)
        assert v7_sft_bench["adapter_path"]
