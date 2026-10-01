"""Stage 07 scene split: stratified dictionary-order train/val + frozen test.

Purpose:
    Split nuScenes v1.0-trainval scenes into train / validation / test scene
    manifests (decision D4): the 700 devkit train scenes are grouped into
    (location x time_of_day) buckets -- time_of_day derives from the first
    keyframe UTC hour in [06:00, 18:00) -> "day", else "night" -- and within
    each bucket scene tokens are ordered lexicographically; the last
    ``val_fraction`` of every bucket becomes validation, the first 90% train.
    The 150 devkit val scenes become the test manifest unchanged (the official
    nuScenes test split is a blind set without ground truth, decision D4).
    No sample-level work happens here: Stage 07 builds records only after the
    scene manifests are frozen.

    Outputs:
    - data assets (decision D3) under ``--out`` (default ``data/dataset_v1``):
      ``train_scene_manifest.json`` / ``val_...`` / ``test_...`` with sorted
      scene tokens, bucket provenance, rule snapshot and manifest sha256.
    - run artifacts under ``--reports`` (default ``runs/S07_dataset/reports``):
      ``split_distribution_report.json`` + ``.md`` comparing location,
      time_of_day and scene-description keyword distributions per split.

Example launch command (loads ~10-15 GB RAM, run inside tmux):
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    tmux new-session -d -s s07_split \
    'PYTHONPATH=DriveAlign/src python -m drivealign.dataset.split \
    2>&1 | tee runs/S07_dataset/split.log'``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as trainval_train_names
from nuscenes.utils.splits import val as trainval_val_names

from drivealign.contracts.versions import DEFAULT_CONTRACT_VERSION
from drivealign.data.nuscenes_io import load_nuscenes

DEFAULT_DATAROOT = "data/nuscenes/trainval"
DEFAULT_VERSION = "v1.0-trainval"
DEFAULT_OUT = "data/dataset_v1"
DEFAULT_REPORTS = "runs/S07_dataset/reports"
DEFAULT_VAL_FRACTION = 0.1
TRAINVAL_TRAIN_SCENES = 700
TRAINVAL_VAL_SCENES = 150

# Minimal English stopword set for scene-description keyword statistics.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or
    that the their there this to with with-area without""".split()
)

# Scene-description keywords ranked for the report (top N overall).
_REPORT_TOP_KEYWORDS = 30


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[3]  # .../DriveAlign
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _scene_samples(nusc: NuScenes, scene: dict) -> List[str]:
    """Ordered keyframe sample tokens of one scene (oldest -> newest)."""
    tokens = []
    current = scene["first_sample_token"]
    while current:
        tokens.append(current)
        current = nusc.get("sample", current)["next"]
    return tokens


def _time_of_day(nusc: NuScenes, scene: dict) -> str:
    """Day/night from the first keyframe UTC hour: [06, 18) -> day (D4)."""
    first = nusc.get("sample", scene["first_sample_token"])
    hour = datetime.fromtimestamp(first["timestamp"] / 1e6, tz=timezone.utc).hour
    return "day" if 6 <= hour < 18 else "night"


def _bucket_key(nusc: NuScenes, scene: dict) -> Tuple[str, str]:
    location = nusc.get("log", scene["log_token"])["location"]
    return (location, _time_of_day(nusc, scene))


def _half_up(fraction: float, n: int) -> int:
    """Deterministic half-up rounding of fraction * n."""
    return int(fraction * n + 0.5)


def stratified_split(
    nusc: NuScenes, val_fraction: float = DEFAULT_VAL_FRACTION
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """Assign the 700 devkit-train scenes into train/val by (D4).

    Buckets are (location, time_of_day); within a bucket scene tokens are
    sorted lexicographically and the last ``val_fraction`` share becomes val.
    Returns ({"train": [...], "val": [...], "test": [...]}, bucket_table)
    with all token lists sorted; bucket_table rows carry per-bucket counts.
    """
    names_to_scene = {s["name"]: s for s in nusc.scene}
    missing = sorted(set(trainval_train_names) - set(names_to_scene))
    if missing:
        raise ValueError(f"Devkit train names missing in metadata: {missing}")

    buckets: Dict[Tuple[str, str], List[str]] = {}
    for name in trainval_train_names:
        scene = names_to_scene[name]
        buckets.setdefault(_bucket_key(nusc, scene), []).append(scene["token"])

    train_tokens: List[str] = []
    val_tokens: List[str] = []
    bucket_table: List[dict] = []
    for location, tod in sorted(buckets):  # dictionary order of bucket keys
        tokens = sorted(buckets[(location, tod)])
        n_val = _half_up(val_fraction, len(tokens))
        bucket_table.append(
            {
                "location": location,
                "time_of_day": tod,
                "scene_count": len(tokens),
                "train_count": len(tokens) - n_val,
                "val_count": n_val,
            }
        )
        train_tokens.extend(tokens[: len(tokens) - n_val])
        val_tokens.extend(tokens[len(tokens) - n_val :])

    test_tokens = sorted(
        names_to_scene[name]["token"] for name in set(trainval_val_names)
    )
    return {
        "train": sorted(train_tokens),
        "val": sorted(val_tokens),
        "test": test_tokens,
    }, bucket_table


def _keyword_counts(nusc: NuScenes, tokens: List[str]) -> Counter:
    counts: Counter = Counter()
    for token in tokens:
        text = nusc.get("scene", token).get("description", "")
        words = re.findall(r"[a-z][a-z'-]*", text.lower())
        counts.update(w for w in words if w not in _STOPWORDS)
    return counts


def distribution_stats(
    nusc: NuScenes, split_tokens: Dict[str, List[str]]
) -> dict:
    """Location / time_of_day / description-keyword stats per split."""
    stats: Dict[str, dict] = {}
    keyword_counts: Dict[str, Counter] = {}
    for split, tokens in split_tokens.items():
        locations: Counter = Counter()
        tods: Counter = Counter()
        n_keyframes = 0
        for token in tokens:
            scene = nusc.get("scene", token)
            location, tod = _bucket_key(nusc, scene)
            locations[location] += 1
            tods[tod] += 1
            n_keyframes += len(_scene_samples(nusc, scene))
        keyword_counts[split] = _keyword_counts(nusc, tokens)
        n = len(tokens)
        stats[split] = {
            "scene_count": n,
            "keyframe_count": n_keyframes,
            "locations": dict(locations),
            "time_of_day": dict(tods),
        }

    # Keyword comparison table over the union of per-split top terms.
    union = Counter()
    for counts in keyword_counts.values():
        union.update(counts)
    top = [w for w, _ in union.most_common(_REPORT_TOP_KEYWORDS)]
    total_scenes = sum(s["scene_count"] for s in stats.values())
    keywords = []
    for word in top:
        row = {"keyword": word}
        for split, counts in keyword_counts.items():
            count = counts.get(word, 0)
            scenes = stats[split]["scene_count"]
            row[f"{split}_count"] = count
            row[f"{split}_per_1k_scenes"] = round(count / scenes * 1000, 1)
        row["total_count"] = union.get(word, 0)
        row["total_per_1k_scenes"] = round(
            union.get(word, 0) / total_scenes * 1000, 1
        )
        keywords.append(row)
    return {"per_split": stats, "keywords": keywords}


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def build_scene_manifest(
    split: str,
    tokens: List[str],
    bucket_table: List[dict],
    rule: dict,
    builder_commit: str,
) -> dict:
    """Frozen scene manifest: sorted tokens + provenance + sha256."""
    payload = {
        "manifest_type": "scene_manifest",
        "contract_version": DEFAULT_CONTRACT_VERSION,
        "split": split,
        "scene_tokens": tokens,
        "scene_count": len(tokens),
        "buckets": bucket_table if split in ("train", "val") else [],
        "rule": rule,
        "builder_commit": builder_commit,
        "source_split": (
            "nuScenes trainval devkit train name list"
            if split in ("train", "val")
            else "nuScenes trainval devkit val name list (full, unfiltered)"
        ),
    }
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return {**payload, "manifest_sha256": digest}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _markdown_report(
    split_tokens: Dict[str, List[str]],
    bucket_table: List[dict],
    stats: dict,
    rule: dict,
) -> str:
    lines = ["# S07 scene split distribution report", ""]
    lines.append("| split | scenes | keyframes |")
    lines.append("|---|---|---|")
    for split in ("train", "val", "test"):
        s = stats["per_split"][split]
        lines.append(
            f"| {split} | {s['scene_count']} | {s['keyframe_count']} |"
        )
    lines += ["", "## Train-split buckets (location x time_of_day, D4)", ""]
    lines.append("| location | tod | scenes | train | val |")
    lines.append("|---|---|---|---|---|")
    for row in bucket_table:
        lines.append(
            f"| {row['location']} | {row['time_of_day']} | "
            f"{row['scene_count']} | {row['train_count']} | {row['val_count']} |"
        )
    for dim, title in (("locations", "Location"), ("time_of_day", "Time of day")):
        lines += ["", f"## {title} distribution", ""]
        header = "| " + dim[:-1].replace("_", " ") + " | " + " | ".join(
            f"{s} count" for s in split_tokens
        ) + " | " + " | ".join(f"{s} %" for s in split_tokens) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (1 + 2 * len(split_tokens)))
        values = sorted({v for s in stats["per_split"].values() for v in s[dim]})
        for value in values:
            cells = []
            for split in split_tokens:
                count = stats["per_split"][split][dim].get(value, 0)
                cells.append(str(count))
                cells.append(
                    f"{count / stats['per_split'][split]['scene_count'] * 100:.1f}"
                )
            lines.append(f"| {value} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## Scene-description keywords (top {len(stats['keywords'])} overall, "
        "report only, not a filter)",
        "",
        "| keyword | "
        + " | ".join(f"{s} count" for s in split_tokens)
        + " | "
        + " | ".join(f"{s} /1k" for s in split_tokens)
        + " | total /1k |",
    ]
    lines.append("|" + "---|" * (2 + 2 * len(split_tokens)))
    for row in stats["keywords"]:
        cells = [row["keyword"]]
        for split in split_tokens:
            cells.append(str(row[f"{split}_count"]))
        for split in split_tokens:
            cells.append(str(row[f"{split}_per_1k_scenes"]))
        cells.append(str(row["total_per_1k_scenes"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Rule: {rule}", ""]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path(DEFAULT_DATAROOT))
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--reports", type=Path, default=Path(DEFAULT_REPORTS))
    parser.add_argument(
        "--val-fraction", type=float, default=DEFAULT_VAL_FRACTION
    )
    parser.add_argument(
        "--builder-commit",
        default=None,
        help="Git commit recorded in manifests; defaults to `git rev-parse HEAD`.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.out.expanduser().mkdir(parents=True, exist_ok=True)
    args.reports.expanduser().mkdir(parents=True, exist_ok=True)
    builder_commit = args.builder_commit or _git_commit()
    rule = {
        "val_fraction": args.val_fraction,
        "stratification": "(log.location x time_of_day)",
        "time_of_day_rule": "first keyframe UTC hour in [06, 18) -> day else night",
        "order": "lexicographic scene_token; first (1 - val_fraction) -> train, last -> val",
        "test_rule": "nuScenes trainval devkit val name list, full, unfiltered",
        "version": args.version,
    }

    nusc = load_nuscenes(args.dataroot.expanduser(), args.version)
    split_tokens, bucket_table = stratified_split(nusc, args.val_fraction)

    # Scene-level gates before anything is frozen.
    s_train, s_val, s_test = (
        set(split_tokens["train"]),
        set(split_tokens["val"]),
        set(split_tokens["test"]),
    )
    gates = {
        "train_val_disjoint": not (s_train & s_val),
        "test_disjoint": not (s_test & (s_train | s_val)),
        "train_plus_val_is_700": len(s_train) + len(s_val)
        == TRAINVAL_TRAIN_SCENES,
        "test_is_150": len(s_test) == TRAINVAL_VAL_SCENES,
        "val_nonempty": len(s_val) > 0,
        "assignment_matches_devkit": {
            nusc.get("scene", t)["name"] for t in s_train | s_val
        }
        == set(trainval_train_names)
        and {nusc.get("scene", t)["name"] for t in s_test}
        == set(trainval_val_names),
    }

    stats = distribution_stats(nusc, split_tokens)

    if all(gates.values()):
        for split in ("train", "val", "test"):
            manifest = build_scene_manifest(
                split, split_tokens[split], bucket_table, rule, builder_commit
            )
            _write_json(args.out / f"{split}_scene_manifest.json", manifest)

        report_md = _markdown_report(split_tokens, bucket_table, stats, rule)
        (args.reports / "split_distribution_report.md").write_text(
            report_md, encoding="utf-8"
        )
        report_payload = {
            "rule": rule,
            "gates": gates,
            "bucket_table": bucket_table,
            **stats,
        }
        _write_json(args.reports / "split_distribution_report.json", report_payload)

    for split in ("train", "val", "test"):
        s = stats["per_split"][split]
        print(
            f"{split}: {s['scene_count']} scenes, {s['keyframe_count']} keyframes"
        )
    print(
        "train/val buckets: "
        + ", ".join(
            f"{r['location']}/{r['time_of_day']}={r['val_count']}/{r['scene_count']}"
            for r in bucket_table
        )
    )
    if all(gates.values()):
        print(f"PASS: scene-level gates ok, manifests -> {args.out}")
        return 0
    print(f"FAIL: gates={gates}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
