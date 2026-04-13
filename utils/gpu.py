"""
utils/gpu.py — GPU detection, VRAM monitoring, dtype helpers.
Optimised for RTX 3060 (12GB VRAM) + CUDA 13.1.
"""

import torch
import numpy as np
from utils.logger import get_logger

logger = get_logger()

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info() -> dict:
    """Return a dict of GPU properties."""
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    props = torch.cuda.get_device_properties(0)
    return {
        "device":       "cuda",
        "name":         props.name,
        "vram_gb":      round(props.total_memory / 1024 ** 3, 1),
        "cuda_cap":     f"{props.major}.{props.minor}",
        "sm_count":     props.multi_processor_count,
    }


def log_device_info() -> None:
    info = device_info()
    if info["device"] == "cuda":
        logger.info(
            f"GPU: {info['name']} | "
            f"VRAM: {info['vram_gb']}GB | "
            f"CUDA cap: {info['cuda_cap']} | "
            f"SMs: {info['sm_count']}"
        )
    else:
        logger.warning("No CUDA device found — running on CPU.")


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

def vram_used_gb() -> float:
    """Return currently allocated VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated(0) / 1024 ** 3, 2)


def vram_reserved_gb() -> float:
    """Return reserved (cached) VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_reserved(0) / 1024 ** 3, 2)


def vram_free_gb() -> float:
    """Return free VRAM in GB via pynvml (most accurate)."""
    if not NVML_OK:
        return 0.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return round(info.free / 1024 ** 3, 2)


def log_vram() -> None:
    logger.debug(
        f"VRAM — used: {vram_used_gb()}GB | "
        f"reserved: {vram_reserved_gb()}GB | "
        f"free: {vram_free_gb()}GB"
    )


def clear_vram() -> None:
    """Free PyTorch VRAM cache — call between model training runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("VRAM cache cleared.")


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to the project device."""
    return tensor.to(get_device())


def numpy_to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to float32 CUDA tensor."""
    return torch.tensor(arr, dtype=dtype).to(get_device())


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Safely detach and convert tensor to numpy."""
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Batch size advisor
# ---------------------------------------------------------------------------

def suggest_batch_size(
    seq_len: int,
    n_features: int,
    dtype_bytes: int = 4,       # float32
    vram_fraction: float = 0.7, # use 70% of VRAM for data
    vram_gb: float = 12.0,
) -> int:
    """
    Estimate a safe batch size given sequence length and feature count.
    Leaves headroom for model weights and gradients.
    """
    vram_bytes = vram_gb * (1024 ** 3) * vram_fraction
    bytes_per_sample = seq_len * n_features * dtype_bytes
    batch = int(vram_bytes // bytes_per_sample)
    # Round down to nearest power of 2
    batch = 2 ** int(np.log2(max(batch, 1)))
    batch = max(1, min(batch, 4096))
    logger.debug(f"Suggested batch size: {batch} (seq={seq_len}, feats={n_features})")
    return batch