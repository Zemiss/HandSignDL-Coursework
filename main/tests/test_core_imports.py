import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class CoreImportTests(unittest.TestCase):
    def test_core_exports_training_and_prediction_helpers(self) -> None:
        from core import (  # noqa: PLC0415
            CLASS_NAMES,
            EvalTransform,
            HandPostureDataset,
            TrainTransform,
            build_model,
            get_device,
            load_checkpoint,
            load_project_config,
            save_checkpoint,
            stratified_split_indices,
        )

        self.assertEqual(CLASS_NAMES, ["A", "B", "C", "Five", "Point", "V"])
        self.assertTrue(callable(build_model))
        self.assertTrue(callable(load_project_config))
        self.assertTrue(callable(get_device))
        self.assertTrue(callable(save_checkpoint))
        self.assertTrue(callable(load_checkpoint))
        self.assertTrue(callable(stratified_split_indices))
        self.assertIsNotNone(TrainTransform)
        self.assertIsNotNone(EvalTransform)
        self.assertIsNotNone(HandPostureDataset)


if __name__ == "__main__":
    unittest.main()
