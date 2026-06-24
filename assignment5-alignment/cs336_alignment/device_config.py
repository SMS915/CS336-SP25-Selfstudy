import os
from typing import Any, Dict

import torch


AUTO_DEVICE_MAP_MODES = {"auto", "balanced", "balanced_low_0", "sequential"}


def get_runtime_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    if not config:
        return {}
    return config.get("runtime", {})


def build_runtime_wrapper_from_flat_config(flat_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {}
    for key in [
        "device",
        "cuda_visible_devices",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "seed",
        "enforce_eager",
        "enable_prefix_caching",
        "swap_space",
        "cpu_offload_gb",
    ]:
        if key in flat_config and flat_config[key] is not None:
            runtime[key] = flat_config[key]

    model: Dict[str, Any] = {}
    if "dtype" in flat_config and flat_config["dtype"] is not None:
        model["dtype"] = flat_config["dtype"]

    return {"runtime": runtime, "model": model}


def apply_runtime_environment(config: Dict[str, Any] | None, verbose: bool = True) -> Dict[str, Any]:
    runtime = get_runtime_config(config)
    cuda_visible_devices = runtime.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        if verbose:
            print(f"Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    return runtime


def resolve_torch_device(config: Dict[str, Any] | None, default: str = "cuda") -> torch.device:
    runtime = get_runtime_config(config)
    requested_device = runtime.get("device")
    if requested_device:
        return torch.device(requested_device)

    if default == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_max_memory_map(config: Dict[str, Any] | None) -> Dict[int, str] | None:
    runtime = get_runtime_config(config)
    max_memory_gb_per_gpu = runtime.get("max_memory_gb_per_gpu")
    if max_memory_gb_per_gpu is None or not torch.cuda.is_available():
        return None

    max_memory = {}
    for gpu_idx in range(torch.cuda.device_count()):
        max_memory[gpu_idx] = f"{max_memory_gb_per_gpu}GiB"
    return max_memory


def _normalize_device_map(device_map: Any) -> Any:
    if device_map is None:
        return None
    if isinstance(device_map, str):
        normalized = device_map.strip()
        if normalized.lower() == "none":
            return None
        if normalized in AUTO_DEVICE_MAP_MODES:
            return normalized
        return {"": normalized}
    if isinstance(device_map, int):
        return {"": device_map}
    return device_map


def build_model_load_kwargs(
    config: Dict[str, Any],
    default_device_map: Any = None,
) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    runtime = get_runtime_config(config)

    kwargs: Dict[str, Any] = {
        "torch_dtype": getattr(torch, model_cfg["dtype"]),
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
    }

    attn_implementation = model_cfg.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    device_map = runtime.get("hf_device_map", model_cfg.get("device_map", default_device_map))
    normalized_device_map = _normalize_device_map(device_map)
    if normalized_device_map is not None:
        kwargs["device_map"] = normalized_device_map

    max_memory = build_max_memory_map(config)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    return kwargs


def get_model_primary_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_autocast_device_type(device: torch.device | str) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def get_vllm_load_kwargs(
    config: Dict[str, Any],
    default_gpu_memory_utilization: float,
) -> Dict[str, Any]:
    runtime = get_runtime_config(config)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    kwargs: Dict[str, Any] = {
        "dtype": model_cfg.get("dtype", "auto"),
        "tensor_parallel_size": int(runtime.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": runtime.get(
            "gpu_memory_utilization",
            training_cfg.get("gpu_memory_utilization", default_gpu_memory_utilization),
        ),
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
    }

    if "seed" in runtime:
        kwargs["seed"] = int(runtime["seed"])

    if "max_seq_length" in data_cfg:
        kwargs["max_model_len"] = data_cfg["max_seq_length"]

    for key in ["enforce_eager", "enable_prefix_caching", "swap_space", "cpu_offload_gb"]:
        if key in runtime:
            kwargs[key] = runtime[key]
        elif key in training_cfg:
            kwargs[key] = training_cfg[key]

    return kwargs
