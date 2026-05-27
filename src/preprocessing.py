import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


CLASS_NAMES = ["A", "B", "C", "Five", "Point", "V"]


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
