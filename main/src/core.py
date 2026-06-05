import random
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights, resnet50


CONFIG_FILES = ("default.yaml", "dataset.yaml", "model.yaml")
CLASS_NAMES = ["A", "B", "C", "Five", "Point", "V"]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") and current_list_key:
            config[current_list_key].append(_parse_scalar(line[4:]))
            continue

        current_list_key = None
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid config line in {path}: {line}")
        key = key.strip()
        value = value.strip()
        if value:
            config[key] = _parse_scalar(value)
        else:
            config[key] = []
            current_list_key = key

    return config


def load_project_config(root: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    config_dir = project_root / "configs"

    config: dict[str, Any] = {}
    for file_name in CONFIG_FILES:
        path = config_dir / file_name
        if path.exists():
            config.update(_load_simple_yaml(path))
    return config


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


class HandPostureCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(
    num_classes: int = 6,
    pretrained: bool = True,
    pretrained_weights_path: str | Path | None = None,
) -> nn.Module:
    local_weights_path = Path(pretrained_weights_path) if pretrained_weights_path else None
    if pretrained and local_weights_path and local_weights_path.exists():
        model = resnet50(weights=None)
        state_dict = torch.load(local_weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class TrainTransform:
    def __init__(self, image_size: int = 64) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ensure_rgb(image).resize((self.image_size + 8, self.image_size + 8))
        left = random.randint(0, 8)
        top = random.randint(0, 8)
        image = image.crop((left, top, left + self.image_size, top + self.image_size))
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        return image_to_tensor(image)


class EvalTransform:
    def __init__(self, image_size: int = 64) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ensure_rgb(image).resize((self.image_size, self.image_size))
        return image_to_tensor(image)


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "P" and "transparency" in image.info:
        return image.convert("RGBA").convert("RGB")
    return image.convert("RGB")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array).permute(2, 0, 1)


class HandPostureDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        class_names: Sequence[str] = CLASS_NAMES,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.glob("*.png")):
                samples.append((path, self.class_to_idx[class_name]))
        if not samples:
            raise RuntimeError(f"No PNG images found under {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image) if self.transform else image_to_tensor(ensure_rgb(image))
        return tensor, label


def stratified_split_indices(
    samples: Sequence[tuple[Path, int]],
    num_classes: int,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    indices_by_class: dict[int, list[int]] = {i: [] for i in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        indices_by_class[label].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in indices_by_class.values():
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices
