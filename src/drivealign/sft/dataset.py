"""record → 训练样本（§2.3 面 parity + §3 Step 1 `sft/dataset.py`）。

职责
----
单条 shard record 转成一个可训练样本，并钉死三个关键属性：

1. **读取**：沿用 `cli/base_benchmark.load_record` 的语义——按 manifest 的
   `shard`/`line` 逐行读，并校验 `record_hash`（保证读到的确实是冻结的 v4 记录）。
2. **面 parity（§2.3，训练面 == 评测面）**：用 `serialize(model_inputs, ONE_FRAME)`
   以 v6 面重建请求，断言其 `canonical_hash()` 等于 manifest 里的 `request_hash_1f`。
   这条是 S08 的 parity 纪律：SFT 样本的"user 面文本"必须与评测时恰好一致，唯一
   差异只允许出现在 assistant 目标段（§2.3 明文）。
3. **样本级 token 编码（§2.2）**：把"user 前景（含图像 token）→ assistant 目标段"
   编码成 `input_ids` / `labels`。监督位 = 目标段 + `<|im_end|>`，**全监督、无 mask**；
   user 前缀全部 `-100`。

为什么"样本级编码"，而不是"collator 一次性编码"（§2.4 分工）
-----------------------------------------------------------
这里的 `encode_chat` 负责**单样本**的 processor 调用（产出 `input_ids`、
`pixel_values`、`image_grid_thw` 与 `labels`），collator 只负责**批内融合**（文本右
pad、图像沿 patch 维拼接）。这样单样本逻辑可独立单测，且未来 4F 分支只需扩展本
函数的图像列表。

为什么 `request_hash` 用 `serialize()` 而不是 record 自身字段
------------------------------------------------------------
record 的 `model_inputs` 与 prompt 无关；请求 hash 追踪的是"模型实际上看到什么"。
`serialize()` 总是以 `model_face_version(DEFAULT_CONTRACT_VERSION)=="v6"` 盖章
（DEFAULT 驱动），因此用它重建的 hash 才是与 v6-face manifest 对齐的 hash。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from drivealign.records.record import DriveAlignRecord
from drivealign.records.serializer import InputPolicy, serialize
from drivealign.sft.targets import as_target, serialize_target

#: 1F 监督段的结束 token 在 Qwen2.5 风格聊天模板中的名字。
IM_END_TOKEN = "<|im_end|>"


@dataclass(frozen=True)
class TrainingSample:
    """一条冻结的、可直接进 collator 的训练样本（文本侧 + 追踪字段）。

    字段
    ----
    sample_token : 锚点样本 token（= manifest 键，也是批次内的唯一 id）。
    scene_token / split : 追踪用，来自 manifest。
    prompt : v6 面 user 文本（含因果 ego-speed 行），由 `serialize` 产出。
    image_relpath : 锚点图像相对 dataroot 的路径（用 manifest 的 `anchor_image_relpath`，
        与评测时图像解析一致）。
    request_hash_1f : `serialize` 重建的 v6 面请求 hash。
    manifest_request_hash : manifest 里登记的 `request_hash_1f`（parity 断言对象）。
    record_hash / target : 追踪用（target 为四字段 dict）。
    target_text : 整串 dump 的目标文本（§2.2 `full`），assistant 监督段的正文。
    sample_sha256 : 下方 `_sample_canonical` 的 sha256（逐样本确定性钉子）。

    ``frozen=True`` 表示构建后不可变：样本在 Step 2 冻结即不再改动。
    """

    sample_token: str
    scene_token: str
    split: str
    prompt: str
    image_relpath: str
    request_hash_1f: str
    manifest_request_hash: str
    record_hash: str
    target: Mapping[str, Any]
    target_text: str
    sample_sha256: str

    # 供文档/调试用的宿主构造参数（不影响 hash 面）：编码时需要的字段。
    ego_speed_mps: float
    frame_tokens: tuple[str, ...] = ()
    time_offsets_s: tuple[float, ...] = ()


def _sample_canonical(sample_fields: Mapping[str, Any]) -> bytes:
    """样本的可哈希规范字节：请求 hash + 目标文本 + 追踪 ids。

    只取**训练行为相关**的稳定字段（不取易变的 wall-time 等），保证：
    相同 record + 相同面 → 相同字节 → 相同 sha256（确定性）。这是 Step 2 冻
    结 artifact 里逐样本 sha 的来源。
    """
    payload = {
        "request_hash_1f": sample_fields["request_hash_1f"],
        "manifest_request_hash": sample_fields["manifest_request_hash"],
        "record_hash": sample_fields["record_hash"],
        "target_text": sample_fields["target_text"],
        "sample_token": sample_fields["sample_token"],
        "image_relpath": sample_fields["image_relpath"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return encoded.encode("utf-8")


def build_training_sample(
    record: DriveAlignRecord,
    entry: Mapping[str, Any],
    *,
    policy: InputPolicy = InputPolicy.ONE_FRAME,
) -> TrainingSample:
    """从一条 record + 一个 manifest 条目构建一条训练样本。

    参数
    ----
    record : 已从 shard 读入且 `record_hash` 校验通过（沿 base_benchmark 语义）。
    entry  : manifest `anchors[token]` 条目（含 split/scene/request_hash_1f/
             record_hash/anchor_image_relpath）。

    抛出
    ----
    ValueError
        面 parity 断言失败（`serialize` 重建的请求 hash ≠ manifest 值）。
        这"拒绝带错误面跑训练"，杜绝静默用错面。

    NOTE: 本函数**不校验** `record_hash`（那一步在读取方做），只负责 parity 与目标。
    """
    # (1) 以 v6 面重建请求（policy=1F，§2.3 训练面 == 评测面）。
    model_request = serialize(record.model_inputs, policy)
    request_hash_1f = model_request.canonical_hash()
    manifest_request_hash = str(entry["request_hash_1f"])

    # (2) 面 parity 断言：重建值必须等于 manifest 值，否则拒绝该样本。
    if request_hash_1f != manifest_request_hash:
        raise ValueError(
            f"sft face parity failed for {record.sample_token}: recomputed "
            f"{request_hash_1f} != manifest {manifest_request_hash} "
            f"(wrong contract face for training?)"
        )

    # (3) 四字段目标（§2.2）：显式列举 + 整串 dump。reasoning 不进训练目标。
    target = as_target(record.training_targets.expected_output)
    target_text = serialize_target(target)

    # (4) 组装并计算逐样本 sha256。
    fields = {
        "request_hash_1f": request_hash_1f,
        "manifest_request_hash": manifest_request_hash,
        "record_hash": str(entry["record_hash"]),
        "target_text": target_text,
        "sample_token": record.sample_token,
        "image_relpath": str(entry["anchor_image_relpath"]),
    }
    sample_sha256 = hashlib.sha256(_sample_canonical(fields)).hexdigest()

    return TrainingSample(
        sample_token=record.sample_token,
        scene_token=str(entry["scene_token"]),
        split=str(entry["split"]),
        prompt=model_request.prompt,
        image_relpath=str(entry["anchor_image_relpath"]),
        request_hash_1f=request_hash_1f,
        manifest_request_hash=manifest_request_hash,
        record_hash=str(entry["record_hash"]),
        target=target,
        target_text=target_text,
        sample_sha256=sample_sha256,
        ego_speed_mps=float(record.model_inputs.ego_speed_mps),
        frame_tokens=tuple(model_request.frame_tokens),
        time_offsets_s=tuple(model_request.time_offsets_s),
    )


def encode_chat(
    processor: Any,
    *,
    prompt: str,
    image_path: str | Path,
    target_text: str,
) -> dict[str, Any]:
    """把单样本编码成训练所需张量（§2.2 监督段 + §2.4 单样本分工）。

    实现要点（与 `inference/base_runner.generate_one` 同源，§2.3 parity）：
      1. 构造 1F user 消息（image + prompt），`apply_chat_template` 加生成前缀
         （wait：训练要的是**完整上下文**，非"仅生成前缀"——见下 NOTE）。
      2. `process_vision_info` 取图像输入，`processor(text, images, ...)` 得
         `input_ids`（含图像占位 token）与 `pixel_values`/`image_grid_thw`。
      3. 监督目标 = 目标文本 token 化（`add_special_tokens=False`），再补一个
         IM_END；`labels` 前缀全 -100、监督段为完整目标 ids。

    NOTE（重要歧义澄清）
    ---------------- 
    训练样本的 assistant 段必须出现在上下文里，因此**不能**用
    `add_generation_prompt=True`（那是评测生成时把"assistant 开头"留给模型续写）。
    这里我们手工把目标文本编码进 `input_ids`，用 `labels` 标记监督位，等价于
    "user 已答完、且答案为该目标文本"。若误用 `add_generation_prompt=True`，
    会漏掉 assistant 标记导致目标位错位——故此处显式注释澄清，不做裁剪。

    返回 dict：`input_ids`(list[int])、`labels`(list[int])、`pixel_values`、
    `image_grid_thw`、`im_end_id`。
    """
    from qwen_vl_utils import process_vision_info

    # (1) 1F user 消息：唯一一帧图像 + v6 prompt 文本。
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).expanduser())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # (2) 完整聊天模板（不裁剪、不加生成提示）：user + 待填空的 assistant 上下文。
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_ids: list[int] = inputs["input_ids"][0].tolist()

    # (2b) 图像张量：pixel_values / image_grid_thw（collator 做批内拼接）。
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("processor returned no pixel_values/image_grid_thw; bad image?")

    # (3) 监督段：目标文本 token（不加特殊符）+ 一个 IM_END。
    ids_target = processor.tokenizer(
        target_text, add_special_tokens=False
    )["input_ids"]
    im_end_id = processor.tokenizer.convert_tokens_to_ids(IM_END_TOKEN)

    full_ids = input_ids + ids_target + [im_end_id]
    # 标签：user 前缀全 -100（不监督）；target + im_end 全监督（§2.2 无 mask 位）。
    labels = [-100] * len(input_ids) + ids_target + [im_end_id]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "im_end_id": im_end_id,
        "target_text": target_text,
    }