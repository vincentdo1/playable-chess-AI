"""Preflight checks for the chess training pipeline."""

from __future__ import annotations

import os
import sys


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main() -> int:
    require_cuda = _env_flag("REQUIRE_CUDA", default=False)

    print("=== Training environment check ===", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"Python version   : {sys.version.split()[0]}", flush=True)

    try:
        import torch
    except ModuleNotFoundError:
        print(
            "ERROR: PyTorch is not installed in this Python environment.",
            flush=True,
        )
        print(
            "Install CUDA PyTorch, for example:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cu124",
            flush=True,
        )
        return 2

    print(f"PyTorch version  : {torch.__version__}", flush=True)
    print(f"torch CUDA build : {torch.version.cuda}", flush=True)
    print(f"CUDA available   : {torch.cuda.is_available()}", flush=True)
    print(f"CUDA devices     : {torch.cuda.device_count()}", flush=True)

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        gib = props.total_memory / (1024 ** 3)
        print(f"GPU              : {props.name} ({gib:.1f} GiB VRAM)", flush=True)
        print(f"cuDNN            : {torch.backends.cudnn.version()}", flush=True)
    elif require_cuda:
        print(
            "ERROR: REQUIRE_CUDA=1, but this PyTorch install cannot use CUDA.",
            flush=True,
        )
        print(
            "You probably installed the CPU-only torch wheel or are running a "
            "different Python than the one you configured for training.",
            flush=True,
        )
        return 3
    else:
        print(
            "WARNING: CUDA is unavailable; full training will be very slow.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
