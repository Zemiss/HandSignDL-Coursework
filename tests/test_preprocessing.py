import unittest
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_posture_recognition.preprocessing import EvalTransform


class TestPreprocessing(unittest.TestCase):
    def test_eval_transform(self) -> None:
        image = Image.new("RGB", (80, 80), color=(128, 64, 32))
        transform = EvalTransform(64)
        tensor = transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))


if __name__ == "__main__":
    unittest.main()
