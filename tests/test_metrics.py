import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_posture_recognition.metrics import accuracy_from_logits


class TestMetrics(unittest.TestCase):
    def test_accuracy_from_logits(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        labels = torch.tensor([1, 0])
        self.assertEqual(accuracy_from_logits(logits, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
