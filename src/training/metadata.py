"""
Run metadata — hardware environment and timing, captured once per training
run and saved alongside model checkpoints (results/mlp/run_metadata.json) so
the scale and hardware a run trained on is reproducible from disk, without
re-parsing logs.
"""

from __future__ import annotations

import json
import os
import platform
import socket
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch


def collect_hardware_info(device: torch.device) -> dict:
    """Snapshot of the hardware/software environment a run executed on."""
    info = {
        "device": device.type,
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    elif device.type == "mps":
        info["gpu_name"] = "Apple MPS"
    return info


def save_run_metadata(
    path: Path,
    hardware: dict,
    cfg,
    fold_timings: list[dict],
    total_wall_time_s: float,
) -> None:
    """Write hardware + config + per-fold timing to `path` as JSON."""
    total_cells = sum(f["total_train_cells"] for f in fold_timings)
    total_train_time = sum(f["train_time_s"] for f in fold_timings)
    metadata = {
        "hardware": hardware,
        "config": asdict(cfg) if is_dataclass(cfg) else dict(cfg),
        "timing": {
            "total_wall_time_s": round(total_wall_time_s, 1),
            "total_train_time_s": round(total_train_time, 1),
            "total_cells_trained": total_cells,
            "avg_cells_per_s": round(total_cells / total_train_time, 1) if total_train_time else None,
            "per_fold": fold_timings,
        },
    }
    path.write_text(json.dumps(metadata, indent=2, default=str))
