"""S12 Step 3 单测：`cli/sft_train.py` 的训练规模覆盖开关（CPU，不加载模型）。

只钉住一条口径——**没给开关 = 行为不变**：`_build_params_override` 在全为缺省时
必须返回 None，`make_trainer` 才不会被写进任何 override（否则正式全量训练会
悄悄偏离冻结配置 `sft_1f.yaml`）。给了开关则逐键进 override，供 `--max-steps`
的诊断 smoke 与 Step 4 的限步验证使用。

启动命令：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_train_cli.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.cli.sft_train import _build_params_override, parse_args

BASE_ARGV = [
    "--config", "c.yaml",
    "--samples", "s.jsonl",
    "--out", "o",
    "--dataroot", "d",
]


class TestParseArgsDefaults:
    def test_scale_switches_default_to_none(self):
        """四个规模开关缺省都是 None（不传 = 完全走冻结配置）。"""
        args = parse_args(BASE_ARGV)
        assert args.epochs is None
        assert args.max_steps is None
        assert args.save_steps is None
        assert args.resume_from_checkpoint is None
        assert args.overfit is False

    def test_max_steps_parses_int(self):
        args = parse_args([*BASE_ARGV, "--max-steps", "30"])
        assert args.max_steps == 30


class TestBuildParamsOverride:
    def test_all_default_returns_none(self):
        """核心口径：全缺省 → None（`make_trainer` 不写任何 override）。"""
        assert _build_params_override(parse_args(BASE_ARGV)) is None

    def test_max_steps_only(self):
        args = parse_args([*BASE_ARGV, "--max-steps", "30"])
        assert _build_params_override(args) == {"max_steps": 30}

    def test_multiple_switches(self):
        args = parse_args([*BASE_ARGV, "--epochs", "1", "--max-steps", "30", "--save-steps", "500"])
        assert _build_params_override(args) == {
            "num_train_epochs": 1,
            "max_steps": 30,
            "save_steps": 500,
        }

    @pytest.mark.parametrize("flag", ["--epochs", "--max-steps", "--save-steps"])
    def test_zero_is_respected_not_treated_as_missing(self, flag):
        """0 也是"显式给了"——判据是 `is not None`，不是真值判断。"""
        args = parse_args([*BASE_ARGV, flag, "0"])
        key = {"--epochs": "num_train_epochs", "--max-steps": "max_steps", "--save-steps": "save_steps"}[flag]
        assert _build_params_override(args) == {key: 0}
