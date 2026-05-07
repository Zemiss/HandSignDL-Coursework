import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_posture_recognition import CLASS_NAMES, EvalTransform, build_model, get_device, load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict all PNG images in a folder.")
    parser.add_argument("--image_dir", required=True, help="Directory containing PNG images.")
    parser.add_argument("--model_path", required=True, help="Path to a trained .pth model.")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.model_path, device)

    class_names = checkpoint.get("class_names", CLASS_NAMES)
    image_size = args.image_size or checkpoint.get("image_size", 64)
    transform = EvalTransform(image_size)

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_paths = sorted(Path(args.image_dir).glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {args.image_dir}")

    with torch.no_grad():
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = int(logits.argmax(dim=1).item())
            print(f"{image_path.name}: {class_names[pred]}")


if __name__ == "__main__":
    main()

