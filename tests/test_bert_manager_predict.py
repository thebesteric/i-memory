import os
import unittest

from src.ai.model.bert_manager import BertIncrModel, BertManager, LabelBranchConfig, LabelConfig


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "models",
    "google-bert",
    "bert-base-multilingual-cased",
    "params",
    "checkpoint.pth",
)


class TestBertManagerPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_CHECKPOINT_PATH):
            raise unittest.SkipTest(f"Real checkpoint not found: {REAL_CHECKPOINT_PATH}")

        cls.manager = BertManager(
            model_name_or_path="google-bert/bert-base-multilingual-cased",
        )
        cls.model = BertIncrModel(
            bert_manager=cls.manager,
            in_features=768,
            out_features_config=LabelConfig(
                branches={
                    "primary": LabelBranchConfig(type="single", num_classes=5),
                    "additional": LabelBranchConfig(type="multi", num_classes=5),
                }
            ),
        )
        cls.checkpoint_path = REAL_CHECKPOINT_PATH
        cls.text = "今天完成了一个重要任务，进展顺利。"

    def test_predict_mixed_single_and_multi_returns_structured_results(self):
        result = self.manager.predict(
            text=self.text,
            bert_incr_model=self.model,
            checkpoint_path=self.checkpoint_path,
            strict=False,
            return_probabilities=True,
        )

        self.assertEqual(result["checkpoint"]["path"], REAL_CHECKPOINT_PATH)
        self.assertEqual(result["checkpoint"]["type"], "full_checkpoint")
        self.assertIsInstance(result["checkpoint"]["epoch"], int)

        primary = result["predictions"]["primary"]
        self.assertEqual(primary["type"], "single")
        self.assertIsInstance(primary["pred"], int)
        self.assertTrue(0.0 <= primary["confidence"] <= 1.0)
        self.assertEqual(len(primary["probabilities"]), 5)
        self.assertAlmostEqual(sum(primary["probabilities"]), 1.0, places=6)

        additional = result["predictions"]["additional"]
        self.assertEqual(additional["type"], "multi")
        self.assertEqual(len(additional["pred"]), 5)
        self.assertTrue(all(v in (0, 1) for v in additional["pred"]))
        self.assertEqual(len(additional["probabilities"]), 5)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in additional["probabilities"]))

    def test_predict_multi_threshold_boundary_uses_greater_equal(self):
        baseline = self.manager.predict(
            text=self.text,
            bert_incr_model=self.model,
            checkpoint_path=self.checkpoint_path,
            strict=False,
            return_probabilities=True,
        )
        multi_probs = baseline["predictions"]["additional"]["probabilities"]

        target_idx = next((i for i, p in enumerate(multi_probs) if p < 1.0), None)
        if target_idx is None:
            self.skipTest("No probability below 1.0 found to validate threshold boundary")
        threshold = float(multi_probs[target_idx])

        result_eq = self.manager.predict(
            text=self.text,
            bert_incr_model=self.model,
            checkpoint_path=self.checkpoint_path,
            strict=False,
            multi_label_threshold=threshold,
            return_probabilities=False,
        )
        result_high = self.manager.predict(
            text=self.text,
            bert_incr_model=self.model,
            checkpoint_path=self.checkpoint_path,
            strict=False,
            multi_label_threshold=min(threshold + 1e-6, 1.0),
            return_probabilities=False,
        )

        self.assertEqual(result_eq["predictions"]["additional"]["pred"][target_idx], 1)
        self.assertEqual(result_high["predictions"]["additional"]["pred"][target_idx], 0)
        self.assertNotIn("probabilities", result_eq["predictions"]["primary"])
        self.assertNotIn("probabilities", result_eq["predictions"]["additional"])

    def test_predict_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, r"multi_label_threshold must be in \[0, 1\]"):
            self.manager.predict(
                text="invalid threshold",
                bert_incr_model=self.model,
                checkpoint_path=self.checkpoint_path,
                strict=False,
                multi_label_threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

