import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_posture_recognition.model import build_model


class TestModel(unittest.TestCase):
    def test_output_shape(self) -> None:
        model = build_model(num_classes=6)
        x = torch.randn(2, 3, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
