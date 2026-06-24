import os
import sys

import yaml


def _read_cuda_visible_devices_from_config(config_path: str | None) -> str | None:
    if not config_path or not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return None

    runtime = config.get("runtime", {})
    if isinstance(runtime, dict) and runtime.get("cuda_visible_devices") is not None:
        return str(runtime["cuda_visible_devices"])

    if config.get("cuda_visible_devices") is not None:
        return str(config["cuda_visible_devices"])

    return None


def _parse_bootstrap_args(argv: list[str]) -> tuple[str | None, str | None]:
    cli_cuda_visible_devices = None
    config_path = None

    for i, arg in enumerate(argv):
        if arg == "--cuda_visible_devices" and i + 1 < len(argv):
            cli_cuda_visible_devices = argv[i + 1]
        elif arg.startswith("--cuda_visible_devices="):
            cli_cuda_visible_devices = arg.split("=", 1)[1]
        elif arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]

    return cli_cuda_visible_devices, config_path


def bootstrap_cuda_visible_devices(default_config_path: str | None = None) -> None:
    cli_cuda_visible_devices, config_path = _parse_bootstrap_args(sys.argv[1:])
    existing_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cli_cuda_visible_devices is not None:
        chosen_cuda_visible_devices = str(cli_cuda_visible_devices)
        source = "命令行参数"
    elif existing_cuda_visible_devices is not None:
        print(
            f"Bootstrap 保留已有 CUDA_VISIBLE_DEVICES={existing_cuda_visible_devices}",
            file=sys.stderr,
        )
        return
    else:
        chosen_config_path = config_path or default_config_path
        chosen_cuda_visible_devices = _read_cuda_visible_devices_from_config(chosen_config_path)
        source = f"配置文件 {chosen_config_path}" if chosen_config_path else "配置文件"

    if chosen_cuda_visible_devices is None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = chosen_cuda_visible_devices
    print(
        "Bootstrap CUDA_VISIBLE_DEVICES="
        f"{chosen_cuda_visible_devices} (来源: {source}; 注意: 逻辑 cuda:0 会映射到这里指定的第一张物理卡)",
        file=sys.stderr,
    )
