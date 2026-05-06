from pathlib import Path
from typing import Any

import torch


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    class_names: list[str],
    image_size: int,
    val_acc: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "val_acc": val_acc,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"{path} is not a checkpoint saved by train.py.")
    return checkpoint
