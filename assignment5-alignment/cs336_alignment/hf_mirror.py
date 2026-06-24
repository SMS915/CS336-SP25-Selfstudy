import os
from typing import Any

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_MAX_WORKERS = 1


def configure_hf_mirror(verbose: bool = True) -> str:
    """Configure a mirror-first Hugging Face setup for flaky server networks."""
    endpoint = os.environ.get("CS336_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT")
    if not endpoint:
        endpoint = DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = endpoint

    disable_xet = os.environ.get("CS336_HF_DISABLE_XET") or os.environ.get("HF_HUB_DISABLE_XET") or "1"
    enable_hf_transfer = (
        os.environ.get("CS336_HF_ENABLE_HF_TRANSFER")
        or os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        or "0"
    )

    os.environ["HF_HUB_DISABLE_XET"] = disable_xet
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = enable_hf_transfer

    if verbose:
        print(
            "Using Hugging Face endpoint: "
            f"{endpoint} (HF_HUB_DISABLE_XET={disable_xet}, HF_HUB_ENABLE_HF_TRANSFER={enable_hf_transfer})"
        )
    return endpoint


def get_hf_endpoint() -> str:
    """Return the effective Hugging Face endpoint after mirror configuration."""
    return os.environ.get("CS336_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT


def get_hf_snapshot_download_kwargs() -> dict[str, Any]:
    """Return conservative download settings that are friendlier to unstable networks."""
    configure_hf_mirror(verbose=False)
    max_workers = int(os.environ.get("CS336_HF_MAX_WORKERS", str(DEFAULT_HF_MAX_WORKERS)))
    return {
        "endpoint": get_hf_endpoint(),
        "max_workers": max_workers,
    }
