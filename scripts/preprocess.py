import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_posture_recognition import CLASS_NAMES, HandPostureDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset layout and count samples.")
    parser.add_argument("--data_dir", default="./data/Hand_Posture_Hard_Stu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = HandPostureDataset(args.data_dir)
    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in dataset.samples:
        counts[CLASS_NAMES[label]] += 1

    print(f"data_dir: {Path(args.data_dir).resolve()}")
    for name in CLASS_NAMES:
        print(f"{name}: {counts[name]}")


if __name__ == "__main__":
    main()

