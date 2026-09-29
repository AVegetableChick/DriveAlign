"""Stage 05: AutoVLA 上游 loader/collator Reference Smoke 检查入口。

原样驱动 pinned checkout 的 SFTDataset / DataCollator（不修改上游代码），
将其指向 Stage 05 preprocessing 产物目录，检查首条、末条、随机样本的
消息结构、四帧顺序、像素预算、目标张量与 collator 标签掩码。

示例启动命令:
    conda activate autovla_codeclean
    cd /root/autodl-tmp/drivealign_workspace/DriveAlign
    python src/drivealign/cli/autovla_loader_smoke.py \
        --data-dir ../artifacts/s05_autovla_prep_smoke/train \
        --out ../artifacts/s05_autovla_prep_smoke/loader_smoke_report.json
"""

import os
import sys
import json
import random
import hashlib
import argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORKSPACE = os.path.dirname(REPO_ROOT)
AUTOVLA_ROOT = os.path.join(WORKSPACE, "third_party", "AutoVLA")
sys.path.insert(0, AUTOVLA_ROOT)
sys.path.insert(0, os.path.join(AUTOVLA_ROOT, "navsim"))

import yaml  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
import qwen_vl_utils  # noqa: E402
from dataset_utils.sft_dataset import SFTDataset, DataCollator  # noqa: E402

ASSISTANT_IDS = [151644, 77091]  # <|im_start|>assistant


def find_sublist(hay, needle):
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if list(hay[i:i + n]) == list(needle):
            return i
    return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="preprocessing 产物目录（*.json）")
    parser.add_argument("--out", required=True, help="报告 JSON 输出路径")
    parser.add_argument("--config", default=os.path.join(AUTOVLA_ROOT, "config/training/qwen2.5-vl-3B-mix-sft.yaml"))
    parser.add_argument("--num-random", type=int, default=1)
    args = parser.parse_args()

    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": str(detail)[:400]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        return ok

    # ---- 上游配置：仅覆盖数据路径与本地绝对路径 ----
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_ROOT, "Qwen2.5-VL-3B-Instruct")
    config["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_ROOT, "codebook_cache/agent_vocab.pkl")
    data_dir = os.path.abspath(args.data_dir)
    n_json = len([f for f in os.listdir(data_dir) if f.endswith(".json")])
    config["data"]["train"]["json_dataset_path"] = [data_dir]
    config["data"]["train"]["sensor_data_path"] = [None]

    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)

    # ---- 被动 spy：记录 loader 实际传给 process_vision_info 的 video 路径 ----
    captured_videos = []
    _orig_pvi = qwen_vl_utils.process_vision_info
    import dataset_utils.sft_dataset as _sft_mod

    def _spy(messages, **kw):
        for msg in messages:
            content = msg.get("content") or []
            if isinstance(content, dict):
                content = [content]
            for c in content:
                if isinstance(c, dict) and c.get("type") == "video" and isinstance(c.get("video"), list):
                    captured_videos.append(list(c["video"]))
        return _orig_pvi(messages, **kw)

    _sft_mod.process_vision_info = _spy

    dataset = SFTDataset(config["data"]["train"], config["model"], processor,
                         using_cot=config["model"]["use_cot"])
    check("dataset_len_equals_json_count", len(dataset) == n_json,
          f"len={len(dataset)} n_json={n_json}")

    indices = [0, len(dataset) - 1]
    rng = random.Random(42)
    for _ in range(args.num_random):
        indices.insert(1, rng.randrange(len(dataset)))

    sample_reports = []
    for idx in indices:
        rep = {"idx": idx}
        item = dataset[idx]
        with open(item["data_path"]) as f:
            scene = json.load(f)
        rep["token"] = scene["token"]

        # 1) video spy: 3 相机 × 4 帧，路径与 JSON 字段逐一对齐（顺序敏感）
        expected = {
            "front": scene["front_camera_paths"],
            "front_left": scene["front_left_camera_paths"],
            "front_right": scene["front_right_camera_paths"],
        }
        order_ok = len(captured_videos) == 3
        if order_ok:
            for vid, (cam, paths) in zip(captured_videos, expected.items()):
                stripped = [u.replace("file://", "") for u in vid]
                if stripped != paths:
                    order_ok = False
                    rep["mismatch_cam"] = cam
        check(f"sample[{idx}] video_paths_3x4_order", order_ok, f"captured={len(captured_videos)}")

        # 2) 像素与帧数
        videos = item["video_inputs"]
        frames_info = []
        budget_ok = len(videos) == 3
        for vid in videos:
            if len(vid) != 4:
                budget_ok = False
            for im in vid:
                w, h = im.size
                frames_info.append([h, w])
                if h * w > 100352 * 1.06 or h % 28 != 0 or w % 28 != 0:
                    budget_ok = False
        rep["video_frames_hw"] = frames_info
        check(f"sample[{idx}] videos_3x4_pixel_budget", budget_ok, str(frames_info[:2]))

        # 3) 文本契约
        text = item["text"]
        rep["text_md5"] = hashlib.md5(text.encode()).hexdigest()
        check(f"sample[{idx}] text_3_video_placeholders", text.count("<|video_pad|>") == 3)
        check(f"sample[{idx}] text_state_injection",
              f"{scene['velocity']:.3f}" in text and f"{scene['acceleration']:.3f}" in text
              and scene["instruction"].lower() in text)
        check(f"sample[{idx}] nocot_answer_format",
              (not item["has_cot"]) and "The final output action is:" in text
              and "straightforward scenario" in text)

        # 4) 目标张量
        rep["gt_action_shape"] = list(item["gt_action"].shape)
        rep["gt_pos_raw_shape"] = list(item["gt_trajectory"].shape)
        check(f"sample[{idx}] gt_action_shape", tuple(item["gt_action"].shape)[1] == 10,
              str(rep["gt_action_shape"]))
        sample_reports.append(rep)
        captured_videos.clear()

    # ---- DataCollator 小批量 ----
    feats = [dataset[0], dataset[1]]
    collator = DataCollator(processor=processor, ignore_index=-100,
                            assistant_id=list(ASSISTANT_IDS))
    batch = collator(feats)
    ids = batch["input_ids"]
    labels = batch["labels"]
    mask_ok = ids.shape == labels.shape
    for i in range(ids.shape[0]):
        start = find_sublist(ids[i], ASSISTANT_IDS)
        if start < 0 or not (labels[i, :start] == -100).all():
            mask_ok = False
    check("collator_labels_masked_before_assistant", mask_ok,
          f"input_ids={tuple(ids.shape)}")
    check("collator_gt_stacked",
          tuple(batch["gt_action"].shape)[0] == 2 and tuple(batch["gt_trajectory"].shape)[0] == 2,
          f"gt_action={tuple(batch['gt_action'].shape)} gt_traj={tuple(batch['gt_trajectory'].shape)}")
    if "pixel_values_videos" in batch:
        pv = batch["pixel_values_videos"]
        h, w = frames_info[0] if frames_info else (224, 420)
        expected_rows = 2 * 3 * (4 // 2) * (h // 14) * (w // 14)  # batch×video×temporal×patch
        check("collator_video_tensor",
              pv.dim() == 2 and pv.shape[0] == expected_rows and pv.shape[1] == 2 * 3 * 14 * 14,
              f"shape={tuple(pv.shape)} expected_rows={expected_rows}")

    overall = all(c["passed"] for c in checks)
    report = {
        "data_dir": data_dir,
        "config": os.path.basename(args.config),
        "n_dataset": len(dataset),
        "overall": "PASS" if overall else "FAIL",
        "samples": sample_reports,
        "checks": checks,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nOVERALL: {report['overall']}  -> report: {args.out}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
