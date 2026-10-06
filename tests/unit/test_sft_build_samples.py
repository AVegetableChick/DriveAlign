"""S12 Step 2 单测：`cli/sft_build_samples.py serialize` 的 `--subsets` 过滤。

背景
----
`select` 子命令固定写出 train32 / overfit128 / train_full 三份 token 列表（三个
子集共用一条字典序选取律，见 `_select_train_anchors`），而 `serialize` 缺省按
"目录里实际存在什么"决定打什么。S12 的正式训练集只有 train_full，所以新增
`--subsets` 把产出收敛到指定子集，且必须**严格向后兼容** S11：

- 不带 `--subsets` → 行为与 S11 一字不差（G4 复算依赖它）；
- `cli/sft_closeout.py` 的 `disk` 子命令以 `argparse.Namespace` 复用 `cmd_serialize`
  （该 Namespace 没有 `subsets` 字段）→ 也必须退化为 S11 行为。

这里不触碰真实数据集：`load_record` / `build_training_sample` 被替换成假实现，
被测的是**子集选择与产出收敛**这一段逻辑（真实面 parity 由
`build_training_sample` 的 ValueError 断言与 G3 复核负责）。

Run:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest DriveAlign/tests/unit/test_sft_build_samples.py -q``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drivealign.cli import sft_build_samples

TOKENS = {
    "train32": ["tok_a", "tok_b"],
    "overfit128": ["tok_a", "tok_b", "tok_c"],
    "train_full": ["tok_a", "tok_b", "tok_c", "tok_d"],
}


# ---------------------------------------------------------------------------
# 测试夹具：假 manifest / 假 token 列表 / 假 record 装配
# ---------------------------------------------------------------------------


def _write_manifest(root: Path) -> Path:
    """写一份最小 anchor manifest：token -> split/shard/line/hash 字段齐备。"""
    anchors = {}
    for name, tokens in TOKENS.items():
        for token in tokens:
            anchors[token] = {
                "split": "train" if name == "train_full" else "train",
                "shard": "shard.jsonl",
                "line": 0,
                "record_hash": f"rh_{token}",
                # 假 hash：真实 parity 由 build_training_sample 断言，此处只求字段齐备
                "request_hash_1f": f"rq_{token}",
                "anchor_image_relpath": f"images/{token}.jpg",
            }
    path = root / "anchor_policy_manifest.json"
    path.write_text(
        json.dumps({"contract_version": "v6", "anchors": anchors}), encoding="utf-8"
    )
    return path


def _write_token_lists(subsets_dir: Path, names: list[str]) -> None:
    subsets_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (subsets_dir / f"{name}_tokens.json").write_text(
            json.dumps(TOKENS[name]), encoding="utf-8"
        )


@pytest.fixture(autouse=True)
def fake_record_stack(monkeypatch):
    """把真实数据读取与真实面重建换成假实现（本文件只测选择逻辑）。"""

    def fake_load_record(_dataset_root, _manifest_like, token):
        return SimpleNamespace(sample_token=token)

    def fake_build_training_sample(record, entry):
        token = record.sample_token
        return SimpleNamespace(
            sample_token=token,
            scene_token=f"scene_{token}",
            split=str(entry["split"]),
            request_hash_1f=str(entry["request_hash_1f"]),
            manifest_request_hash=str(entry["request_hash_1f"]),
            record_hash=str(entry["record_hash"]),
            sample_sha256=f"sha_{token}",
            prompt=f"prompt {token}",
            image_relpath=str(entry["anchor_image_relpath"]),
            target={},
            target_text="{}",
            ego_speed_mps=1.0,
        )

    monkeypatch.setattr(sft_build_samples, "load_record", fake_load_record)
    monkeypatch.setattr(
        sft_build_samples, "build_training_sample", fake_build_training_sample
    )


def _namespace(root: Path, subsets_dir: Path, out_dir: Path, **extra) -> argparse.Namespace:
    """构造 cmd_serialize 的入参；`subsets` 字段按需给（默认不出现，模拟 S11/sft_closeout）。"""
    base = {
        "dataset_root": str(root / "dataset_v4"),
        "anchor_manifest": str(root / "anchor_policy_manifest.json"),
        "subsets_dir": str(subsets_dir),
        "out_dir": str(out_dir),
    }
    base.update(extra)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# 向后兼容：缺省行为 == S11
# ---------------------------------------------------------------------------


class TestDefaultBehaviourUnchanged:
    def test_serializes_every_present_subset(self, tmp_path):
        """不带 --subsets：目录里有几份 token 列表就打几份（S11 行为）。"""
        subsets_dir = tmp_path / "subsets"
        out_dir = tmp_path / "out"
        _write_manifest(tmp_path)
        _write_token_lists(subsets_dir, ["train32", "overfit128", "train_full"])

        assert sft_build_samples.cmd_serialize(_namespace(tmp_path, subsets_dir, out_dir)) == 0

        assert sorted(p.name for p in out_dir.glob("*.jsonl")) == [
            "overfit128.jsonl",
            "train32.jsonl",
            "train_full.jsonl",
        ]
        manifest = json.loads((out_dir / "samples_manifest.json").read_text(encoding="utf-8"))
        assert sorted(manifest["artifacts"]) == ["overfit128", "train32", "train_full"]

    def test_namespace_without_subsets_field_works(self, tmp_path):
        """`sft_closeout.cmd_disk` 的 Namespace 没有 `subsets` 字段，必须不炸。"""
        subsets_dir = tmp_path / "subsets"
        out_dir = tmp_path / "out"
        _write_manifest(tmp_path)
        _write_token_lists(subsets_dir, ["train32"])

        args = _namespace(tmp_path, subsets_dir, out_dir)  # 无 subsets 字段
        assert not hasattr(args, "subsets")
        assert sft_build_samples.cmd_serialize(args) == 0
        assert [p.name for p in out_dir.glob("*.jsonl")] == ["train32.jsonl"]

    def test_no_token_lists_at_all_raises(self, tmp_path):
        """一份都没有时报错，避免"静默产出空 artifact"。"""
        subsets_dir = tmp_path / "subsets"
        subsets_dir.mkdir()
        _write_manifest(tmp_path)

        with pytest.raises(FileNotFoundError, match="no token subset files found"):
            sft_build_samples.cmd_serialize(
                _namespace(tmp_path, subsets_dir, tmp_path / "out")
            )


# ---------------------------------------------------------------------------
# S12：--subsets 收敛产出
# ---------------------------------------------------------------------------


class TestSubsetsFilter:
    def test_only_requested_subset_is_serialized(self, tmp_path):
        """`--subsets train_full`：S12 冻结目录只含 train_full，不掺 S11 smoke 子集。"""
        subsets_dir = tmp_path / "subsets"
        out_dir = tmp_path / "out"
        _write_manifest(tmp_path)
        _write_token_lists(subsets_dir, ["train32", "overfit128", "train_full"])

        assert (
            sft_build_samples.cmd_serialize(
                _namespace(tmp_path, subsets_dir, out_dir, subsets=["train_full"])
            )
            == 0
        )

        assert [p.name for p in out_dir.glob("*.jsonl")] == ["train_full.jsonl"]
        manifest = json.loads((out_dir / "samples_manifest.json").read_text(encoding="utf-8"))
        assert sorted(manifest["artifacts"]) == ["train_full"]
        assert manifest["artifacts"]["train_full"]["count"] == len(TOKENS["train_full"])
        # 行数 == 该子集的 token 数（逐 token 一行）
        assert len(
            (out_dir / "train_full.jsonl").read_text(encoding="utf-8").splitlines()
        ) == len(TOKENS["train_full"])

    def test_multiple_subsets_in_argv_order(self, tmp_path):
        """可以一次要多个，产出顺序按 argv 给定顺序（此处仅验证集合正确）。"""
        subsets_dir = tmp_path / "subsets"
        out_dir = tmp_path / "out"
        _write_manifest(tmp_path)
        _write_token_lists(subsets_dir, ["train32", "overfit128", "train_full"])

        sft_build_samples.cmd_serialize(
            _namespace(tmp_path, subsets_dir, out_dir, subsets=["train_full", "train32"])
        )
        assert sorted(p.name for p in out_dir.glob("*.jsonl")) == [
            "train32.jsonl",
            "train_full.jsonl",
        ]

    def test_unknown_subset_name_raises(self, tmp_path):
        """拼错子集名必须报错，不能静默产出 0 份。"""
        subsets_dir = tmp_path / "subsets"
        _write_manifest(tmp_path)
        _write_token_lists(subsets_dir, ["train_full"])

        with pytest.raises(ValueError, match="unknown subset name"):
            sft_build_samples.cmd_serialize(
                _namespace(tmp_path, subsets_dir, tmp_path / "out", subsets=["trainfull"])
            )

    def test_requested_but_absent_subset_raises(self, tmp_path):
        """点名要一个目录里不存在的子集 → FileNotFoundError（不静默跳过）。"""
        subsets_dir = tmp_path / "subsets"
        _write_manifest(tmp_path)
        _write_token_lists(subsets_dir, ["train_full"])

        with pytest.raises(FileNotFoundError, match="not found"):
            sft_build_samples.cmd_serialize(
                _namespace(tmp_path, subsets_dir, tmp_path / "out", subsets=["overfit128"])
            )


# ---------------------------------------------------------------------------
# CLI 解析：--subsets 确实接到了 argparse 上
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_subsets_is_optional_and_variadic(self):
        common = [
            "--dataset-root",
            "d",
            "serialize",
            "--anchor-manifest",
            "m.json",
            "--subsets-dir",
            "s",
            "--out-dir",
            "o",
        ]
        assert sft_build_samples.parse_args(common).subsets is None
        parsed = sft_build_samples.parse_args(common + ["--subsets", "train_full"])
        assert parsed.subsets == ["train_full"]
        parsed_two = sft_build_samples.parse_args(
            common + ["--subsets", "train_full", "train32"]
        )
        assert parsed_two.subsets == ["train_full", "train32"]
