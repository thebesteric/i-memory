import os
import unittest

from src.ai.model.bert_manager import BertIncrModel, BertManager, LabelBranchConfig, LabelConfig


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "bert",
    "checkpoint",
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
                    "labels": LabelBranchConfig(type="multi", num_classes=5),
                }
            ),
        )
        cls.checkpoint_path = REAL_CHECKPOINT_PATH
        cls.text = "旧毛衣袖口磨出毛边，套上时熟悉的柔软裹住手腕，像被时光温柔地抱了一下。"
        # {
        #     "text": "旧毛衣袖口磨出毛边，套上时熟悉的柔软裹住手腕，像被时光温柔地抱了一下。",
        #     "primary": 3,
        #     "labels": [1,0,0,1,1]
        #   },

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

        labels = result["predictions"]["labels"]
        self.assertEqual(labels["type"], "multi")
        self.assertEqual(len(labels["pred"]), 5)
        self.assertTrue(all(v in (0, 1) for v in labels["pred"]))
        self.assertEqual(len(labels["probabilities"]), 5)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in labels["probabilities"]))

    def test_predict_multi_threshold_boundary_uses_greater_equal(self):
        baseline = self.manager.predict(
            text=self.text,
            bert_incr_model=self.model,
            checkpoint_path=self.checkpoint_path,
            strict=False,
            return_probabilities=True,
        )
        multi_probs = baseline["predictions"]["labels"]["probabilities"]

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

        self.assertEqual(result_eq["predictions"]["labels"]["pred"][target_idx], 1)
        self.assertEqual(result_high["predictions"]["labels"]["pred"][target_idx], 0)
        self.assertNotIn("probabilities", result_eq["predictions"]["primary"])
        self.assertNotIn("probabilities", result_eq["predictions"]["labels"])

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

