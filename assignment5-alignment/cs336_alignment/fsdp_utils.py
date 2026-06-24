import contextlib
import os
from functools import partial
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


DEFAULT_TRANSFORMER_LAYER_NAMES = {
    "Qwen2DecoderLayer",
    "Qwen3DecoderLayer",
    "LlamaDecoderLayer",
    "MistralDecoderLayer",
    "GemmaDecoderLayer",
    "GPTNeoXLayer",
    "PhiDecoderLayer",
}


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not is_distributed():
        dist.init_process_group(backend=backend)

    return local_rank, rank, world_size


def destroy_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def maybe_no_sync(model: torch.nn.Module, enabled: bool):
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return contextlib.nullcontext()


def reduce_scalar(value: float, device: torch.device, op: str = "mean") -> float:
    if not is_distributed():
        return float(value)

    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor /= get_world_size()
    return tensor.item()


def reduce_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    return {key: reduce_scalar(value, device) for key, value in metrics.items()}


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def build_mixed_precision(dtype_name: str) -> MixedPrecision:
    dtype = _resolve_torch_dtype(dtype_name)
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def _resolve_sharding_strategy(strategy_name: str) -> ShardingStrategy:
    mapping = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    normalized = strategy_name.lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported FSDP sharding strategy: {strategy_name}")
    return mapping[normalized]


def _find_transformer_layer_classes(model: torch.nn.Module, config: Dict[str, Any]) -> tuple[type, ...]:
    fsdp_cfg = config.get("fsdp", {})
    explicit_names = fsdp_cfg.get("transformer_layer_cls_names")
    if explicit_names is None:
        layer_names = DEFAULT_TRANSFORMER_LAYER_NAMES
    else:
        layer_names = set(explicit_names)

    classes = []
    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name in layer_names or class_name.endswith("DecoderLayer"):
            classes.append(module.__class__)

    unique_classes = []
    seen = set()
    for cls in classes:
        if cls not in seen:
            unique_classes.append(cls)
            seen.add(cls)
    return tuple(unique_classes)


def build_auto_wrap_policy(model: torch.nn.Module, config: Dict[str, Any]):
    fsdp_cfg = config.get("fsdp", {})
    transformer_layer_classes = _find_transformer_layer_classes(model, config)
    if transformer_layer_classes:
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_layer_classes),
        )

    min_num_params = int(fsdp_cfg.get("min_num_params", 10_000_000))
    return partial(size_based_auto_wrap_policy, min_num_params=min_num_params)


def wrap_model_with_fsdp(model: torch.nn.Module, config: Dict[str, Any], local_rank: int) -> FSDP:
    fsdp_cfg = config.get("fsdp", {})
    runtime_cfg = config.get("runtime", {})
    dtype_name = config["model"].get("dtype", "bfloat16")

    wrapped_model = FSDP(
        model,
        auto_wrap_policy=build_auto_wrap_policy(model, config),
        mixed_precision=build_mixed_precision(dtype_name),
        sharding_strategy=_resolve_sharding_strategy(fsdp_cfg.get("sharding_strategy", "full_shard")),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=bool(fsdp_cfg.get("use_orig_params", True)),
        limit_all_gathers=bool(fsdp_cfg.get("limit_all_gathers", True)),
        sync_module_states=bool(fsdp_cfg.get("sync_module_states", get_world_size() > 1)),
        device_id=torch.device("cuda", local_rank) if torch.cuda.is_available() else None,
        cpu_offload=CPUOffload(offload_params=bool(fsdp_cfg.get("cpu_offload", False))),
        forward_prefetch=bool(fsdp_cfg.get("forward_prefetch", False)),
    )

    rank0_print(
        "FSDP 配置:"
        f" sharding={fsdp_cfg.get('sharding_strategy', 'full_shard')},"
        f" use_orig_params={fsdp_cfg.get('use_orig_params', True)},"
        f" cuda_visible_devices={runtime_cfg.get('cuda_visible_devices')}"
    )
    return wrapped_model


def get_wrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, FSDP) else model


def save_fsdp_model(
    model: FSDP,
    tokenizer,
    output_dir: str,
    safe_serialization: bool = True,
) -> None:
    state_dict = gather_full_state_dict(model)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        unwrapped_model = get_wrapped_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=safe_serialization,
        )
        tokenizer.save_pretrained(output_dir)

    barrier()


def gather_full_state_dict(model: FSDP) -> Dict[str, Any] | None:
    """Materialize a full CPU state dict on rank 0 for save/eval handoff."""
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()
    barrier()
    return state_dict if is_main_process() else None
