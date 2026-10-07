"""S12 Step 6b：批量（batch>1）结构化推理 runner（batched_v1 path）。

纪律边界
--------
本模块属于"推理基建 v2"实验 path（S09 备案第 1 路线：batch 化优先）。
冻结面（``inference.base_runner.generate_one``、``inference.structured_runner``、
``cli.base_benchmark``）**一行不改**；批量 path 只复用它们的公开函数，其余
新逻辑（左 padding 组批、EOS 截断计数、telemetry 摊平口径）全部收敛在本文件，
保证 diff 可审计。

与 bs=1 冻结面的语义对齐
------------------------
1. prompt 构造、严格解析、错误分类逐字段沿用冻结面（``structured_runner``），
   v6 面 prompt/hash 守卫在 CLI 层逐样本执行，与本文件无关；
2. ``output_tokens`` 按每样本第一个 EOS 截断计数——bs=1 的 ``generate`` 恰好
   在 EOS 处返回（计数含 EOS），批内先结束的样本会以 pad 陪跑到批最长，
   截断后两者口径一致；
3. 文本解码沿用 ``skip_special_tokens=True``：批内 pad 与 EOS 都会被剥掉，
   截断只影响 ``output_tokens`` 计数，不影响 ``text``；
4. 整批生成失败（典型：OOM）时整批回退到冻结面单样本路径
   ``run_structured_inference``，S03 冻结语义"一个样本失败绝不中止整个 run"
   在批量 path 下保持不变。

与 bs=1 的已知差异（预注册登记）
--------------------------------
- bf16 下批归约顺序不同，greedy 解码存在偶发 token 翻转的可能；
  parity gate（CPU 层对两份 JSONL 做 diff）延后执行，产出物不变。
- ``preprocess_seconds`` / ``generation_seconds`` 为批级耗时按批大小摊平，
  不再是单样本真实 latency；新增 ``batch_size`` 字段记录批大小。
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import torch

from drivealign.contracts.output import (
    OutputContractError,
    OutputErrorCategory,
    ParseResult,
    parse_structured_output,
)
from drivealign.contracts.prompt import build_structured_prompt
from drivealign.inference.base_runner import GenerationResult
from drivealign.inference.model_loader import LoadedModel
from drivealign.inference.structured_runner import (
    OK_STATUS,
    SampleRecord,
    SampleRequest,
    run_structured_inference,
)


def _chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """按顺序切批：确定性、不重排（输入近等长，padding 浪费可忽略）。

    S09 画像里 input_tokens p50 = p90 ≈ 2335（同一 face 下 prompt 只差 speed
    行、图像同分辨率），因此不做长度分组——顺序切批让 resume/日志顺序与
    anchor 顺序天然一致。
    """
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _eos_id_set(loaded: LoadedModel) -> set[int]:
    """收集 generate 实际使用的 EOS id（``generation_config`` 里是 int 或 list）。"""
    eos = loaded.model.generation_config.eos_token_id
    if eos is None:
        eos = loaded.model.config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {int(eos)}
    return {int(value) for value in eos}


def _first_eos_index(token_ids: Sequence[int], eos_ids: set[int]) -> int | None:
    """返回该行生成序列中第一个 EOS 的下标；无 EOS 返回 None。"""
    for index, token in enumerate(token_ids):
        if int(token) in eos_ids:
            return index
    return None


def _per_sample_output_tokens(
    generated_ids: torch.Tensor, eos_ids: set[int]
) -> list[int]:
    """逐样本统计"到第一个 EOS（含）为止"的输出 token 数。

    bs=1 的 generate 恰好在 EOS 处返回（``output_tokens`` 含 EOS），批内先结束
    的样本后面是 pad，本函数把两者对齐到同一口径；无 EOS（跑满
    ``max_new_tokens``）则计满整行。
    """
    lengths: list[int] = []
    for row in generated_ids.tolist():
        index = _first_eos_index(row, eos_ids)
        lengths.append(index + 1 if index is not None else len(row))
    return lengths


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def generate_batch(
    loaded: LoadedModel,
    images: Sequence[str],
    prompts: Sequence[str],
    *,
    generation_config: Mapping[str, Any] | None = None,
) -> list[GenerationResult]:
    """对等长的 (image, prompt) 列表做一次批式 greedy 生成。

    这是 batched_v1 的最小核心：一个 batch 一次 ``process_vision_info`` +
    一次 ``processor(...)``（左 padding）+ 一次 ``model.generate``。
    单批内逐样本 telemetry 按批摊平（见模块 docstring 的已知差异）。
    """
    if len(images) != len(prompts):
        raise ValueError(
            f"images/prompts length mismatch: {len(images)} != {len(prompts)}"
        )
    if not images:
        raise ValueError("generate_batch requires a non-empty batch")

    from qwen_vl_utils import process_vision_info

    preprocess_started = perf_counter()
    # 消息构造与 generate_one 逐字段一致，只是列表化。
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }
        for image, prompt in zip(images, prompts)
    ]
    texts = [
        # 注意：apply_chat_template 要求**消息列表**（generate_one 传的正是
        # [message]）；传单个 dict 会被 jinja 按 key 迭代，prompt 塌缩成
        # generation prompt（~1 token），图像 pad 全丢 → "tokens: 0" 报错。
        loaded.processor.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)

    # generate 要求左 padding（decoder 从序列末尾续写）；FA2 兼容左 padding。
    # 每次调用显式设置，避免依赖调用方的 tokenizer 全局状态。
    loaded.processor.tokenizer.padding_side = "left"
    inputs = loaded.processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    preprocess_seconds = perf_counter() - preprocess_started

    device = _model_device(loaded.model)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    # 逐样本真实输入长度（attention_mask 才含 pad 后的真实 token 数）。
    per_sample_input_tokens = [
        int(mask) for mask in inputs["attention_mask"].sum(dim=1).tolist()
    ]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    generation_started = perf_counter()
    generate_kwargs = dict(generation_config or {})
    generate_kwargs["do_sample"] = False
    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **inputs,
            **generate_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = perf_counter() - generation_started

    # 左 padding 下所有行等长，切掉 prompt 段即纯生成段；行尾是先结束样本的
    # pad，batch_decode(skip_special_tokens=True) 会剥掉，不污染 text。
    prompt_length = int(inputs["input_ids"].shape[1])
    generated_slice = generated_ids[:, prompt_length:]
    per_sample_output_tokens = _per_sample_output_tokens(
        generated_slice, _eos_id_set(loaded)
    )
    generated_texts = loaded.processor.batch_decode(
        generated_slice,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    batch_size = len(images)
    peak_memory = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    )
    # telemetry 批级摊平：时间字段不再是单样本 latency（预注册已知差异）。
    results: list[GenerationResult] = []
    for index in range(batch_size):
        results.append(
            GenerationResult(
                text=generated_texts[index],
                input_tokens=per_sample_input_tokens[index],
                output_tokens=per_sample_output_tokens[index],
                preprocess_seconds=preprocess_seconds / batch_size,
                generation_seconds=generation_seconds / batch_size,
                peak_cuda_memory_bytes=peak_memory,
            )
        )
    return results


def _telemetry_to_dict(generation: GenerationResult) -> dict[str, Any]:
    """与 structured_runner 同字段，另加 batch_size（摊平口径的显式标记）。"""
    return {
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "preprocess_seconds": generation.preprocess_seconds,
        "generation_seconds": generation.generation_seconds,
        "peak_cuda_memory_bytes": generation.peak_cuda_memory_bytes,
        "batch_size": getattr(generation, "batch_size", 1),
    }


def _batch_size_on_result(generation: GenerationResult, batch_size: int) -> GenerationResult:
    """给 GenerationResult 附带批大小（dataclass 无此字段，用属性注入）。"""
    object.__setattr__(generation, "batch_size", batch_size)
    return generation


def _record_from_generation(
    request: SampleRequest,
    prompt: str,
    generation: GenerationResult,
) -> SampleRecord:
    """把一条 GenerationResult 走严格解析并组装 SampleRecord。

    与 ``structured_runner.run_structured_inference`` 的解析/组 record 部分逐
    字段一致（冻结面不改，故在此镜像），telemetry 换成批口径。
    """
    result: ParseResult = parse_structured_output(
        generation.text,
        contract_version=request.contract_version,
        image_size=request.image_size,
    )
    status = OK_STATUS if result.ok else result.errors[0].category.value
    return SampleRecord(
        sample_id=request.sample_id,
        image=str(request.image),
        prompt=prompt,
        status=status,
        parse_ok=result.ok,
        raw_text=result.raw_text,
        output=result.output,
        errors=result.errors,
        generation_error=None,
        telemetry=_telemetry_to_dict(generation),
        available_speed=request.available_speed,
        extracted_from_fences=result.extracted_from_fences,
        contract_version=result.contract_version,
        normalized_terms=result.normalized_terms,
        unmapped_terms=result.unmapped_terms,
        derived_positions=result.derived_positions,
    )


def _generation_failure_record(
    request: SampleRequest,
    prompt: str,
    message: str,
) -> SampleRecord:
    """单样本 generation_failure 记录（S03 冻结分类，镜像冻结面构造）。"""
    return SampleRecord(
        sample_id=request.sample_id,
        image=str(request.image),
        prompt=prompt,
        status=OutputErrorCategory.GENERATION_FAILURE.value,
        parse_ok=False,
        raw_text=None,
        output=None,
        errors=(
            OutputContractError(
                category=OutputErrorCategory.GENERATION_FAILURE,
                message=message,
            ),
        ),
        generation_error=message,
        telemetry=None,
        available_speed=request.available_speed,
    )


def run_structured_inference_batched(
    loaded: LoadedModel,
    requests: Sequence[SampleRequest],
    *,
    generation_config: Mapping[str, Any] | None = None,
    batch_size: int = 8,
) -> list[SampleRecord]:
    """批量结构化推理：顺序切批 → 批式生成 → 严格解析，返回与输入等长对齐。

    容错语义与冻结面一致：任何单样本/单批失败都降级为 generation_failure
    记录，绝不中止整个 run；整批失败（典型 OOM）回退到冻结面单样本路径
    ``run_structured_inference`` 逐样本重试。
    """
    records: list[SampleRecord] = []
    for batch in _chunked(list(requests), batch_size):
        prompts = [
            build_structured_prompt(
                request.available_speed, version=request.contract_version
            )
            for request in batch
        ]
        try:
            generations = generate_batch(
                loaded,
                [str(request.image) for request in batch],
                prompts,
                generation_config=generation_config,
            )
        except Exception as exc:  # 整批失败 → 冻结面单样本路径兜底
            # 回退必须留痕：静默吞异常会把批量 path 的故障伪装成 bs=1 慢跑
            # （S12 首次 smoke 即栽在这里）。
            print(
                f"[batched_runner] batch generation failed "
                f"(batch_size={len(batch)}), falling back to per-sample: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            for request, prompt in zip(batch, prompts):
                try:
                    records.append(
                        run_structured_inference(
                            loaded,
                            request,
                            generation_config=generation_config,
                        )
                    )
                except Exception as single_exc:  # 防御：兜底也不允许中止
                    records.append(
                        _generation_failure_record(
                            request,
                            prompt,
                            f"{type(single_exc).__name__}: {single_exc}",
                        )
                    )
            continue

        for request, prompt, generation in zip(batch, prompts, generations):
            _batch_size_on_result(generation, len(batch))
            try:
                records.append(_record_from_generation(request, prompt, generation))
            except Exception as exc:  # 防御：解析异常也不允许中止
                message = f"{type(exc).__name__}: {exc}"
                records.append(
                    _generation_failure_record(request, prompt, message)
                )
    return records
