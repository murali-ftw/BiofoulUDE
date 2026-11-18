"""
Device selection. Order: env var ML_DEVICE; device recorded by `python -m ml.bench_device` in
ml/artifacts/device.json; "mps" if available; else "cpu".

MPS has no float64, so MPS runs use float32. MPS results are not bit-reproducible; all parity,
gradcheck and determinism checks run on CPU in float64.
"""

import json
from pathlib import Path

import ml  # noqa: F401  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch is imported)
import torch

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEVICE_FILE = ARTIFACTS_DIR / "device.json"


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def get_device() -> torch.device:
    env = __import__("os").environ.get("ML_DEVICE")
    if env:
        return torch.device(env)
    if DEVICE_FILE.exists():
        try:
            name = json.loads(DEVICE_FILE.read_text())["device"]
            if name != "mps" or _mps_available():
                return torch.device(name)
        except (KeyError, ValueError, OSError):
            pass
    return torch.device("mps" if _mps_available() else "cpu")


def default_dtype(device, prefer_float64: bool = False) -> torch.dtype:
    """float32 on MPS (no float64 support); on CPU float64 only if requested."""
    device = torch.device(device)
    if device.type == "cpu" and prefer_float64:
        return torch.float64
    return torch.float32


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
