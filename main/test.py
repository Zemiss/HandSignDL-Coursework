import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from core import (  # noqa: E402
    CLASS_NAMES,
    build_model,
    ensure_rgb,
    get_device,
    image_to_tensor,
    load_checkpoint,
    load_project_config,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    config = load_project_config(ROOT)
    paths = config.get("paths", {})
    data = config.get("data", {})
    train_cfg = config.get("train", {})
    parser = argparse.ArgumentParser(description="Predict all images in test_images by default.")
    parser.add_argument(
        "--test_data_dir",
        "--image_dir",
        dest="image_dir",
        default=paths.get("test_data_dir", "./test_images"),
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--input_model_path",
        "--model_path",
        dest="model_path",
        default=paths.get("input_model_path", paths.get("model_path", "./model/best_model.pth")),
        help="Path to a trained .pth model.",
    )
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", default=train_cfg.get("device", "auto"), help="auto, cpu, cuda, or cuda:0.")
    return parser.parse_args()


def collect_image_paths(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def build_tta_tensors(image: Image.Image, image_size: int) -> list[torch.Tensor]:
    # Mirror the training crop geometry and average over a few stable views.
    expanded_size = image_size + 8
    image = ensure_rgb(image).resize((expanded_size, expanded_size))
    crop_offsets = [
        (0, 0),
        (8, 0),
        (0, 8),
        (8, 8),
        (4, 4),
    ]

    tensors: list[torch.Tensor] = []
    for left, top in crop_offsets:
        crop = image.crop((left, top, left + image_size, top + image_size))
        tensors.append(image_to_tensor(crop))
        tensors.append(image_to_tensor(crop.transpose(Image.FLIP_LEFT_RIGHT)))
    return tensors


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    tensors = build_tta_tensors(image, image_size)
    batch = torch.stack(tensors).to(device)
    logits = model(batch)
    return logits.mean(dim=0, keepdim=True)


def main() -> None:
    args = parse_args()
    config = load_project_config(ROOT)
    data = config.get("data", {})
    device = get_device(args.device)
    print(f"loading model: {args.model_path}")
    checkpoint = load_checkpoint(args.model_path, device)

    class_names = checkpoint.get("class_names", data.get("class_names", CLASS_NAMES))
    image_size = args.image_size or checkpoint.get("image_size", data.get("image_size", 64))

    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_paths = collect_image_paths(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No supported images found in {args.image_dir}")

    print(f"predicting {len(image_paths)} images from {args.image_dir}")
    with torch.no_grad():
        for image_path in image_paths:
            with Image.open(image_path) as image:
                logits = predict_image(model, image, image_size, device)
            pred = int(logits.argmax(dim=1).item())
            print(f"{image_path.name}: {class_names[pred]}")


if __name__ == "__main__":
    main()
