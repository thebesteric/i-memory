import os
import os
import tempfile
import unittest
from typing import cast
from unittest.mock import patch

import torch

from src.ai.model.bert_manager import BertManager, LabelConfig, LabelBranchConfig, BertDataset, BertIncrModel


class DummyDataset:
    def __init__(self, length: int = 1):
        self.length = length

    def __len__(self):
        return self.length


class DummyIncrModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze_bert = True
        self.proj = torch.nn.Linear(4, 2)
        self.out_features_config = LabelConfig(
            branches={
                "primary": LabelBranchConfig(type="single", num_classes=2),
            }
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        batch_size = input_ids.shape[0]
        dummy_features = torch.zeros(batch_size, 4, device=input_ids.device)
        return [self.proj(dummy_features)]


class DummyMultiBranchIncrModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze_bert = True
        self.primary_proj = torch.nn.Linear(4, 2)
        self.additional_proj = torch.nn.Linear(4, 3)
        self.out_features_config = LabelConfig(
            branches={
                "primary": LabelBranchConfig(type="single", num_classes=2),
                "additional": LabelBranchConfig(type="multi", num_classes=3),
            }
        )

        # 固定输出，让测试结果可预期
        with torch.no_grad():
            self.primary_proj.weight.zero_()
            self.primary_proj.bias.copy_(torch.tensor([2.0, -2.0]))
            self.additional_proj.weight.zero_()
            self.additional_proj.bias.copy_(torch.tensor([2.0, -2.0, 2.0]))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        batch_size = input_ids.shape[0]
        dummy_features = torch.zeros(batch_size, 4, device=input_ids.device)
        return [self.primary_proj(dummy_features), self.additional_proj(dummy_features)]


class DummyPrimaryMultiIncrModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze_bert = True
        self.primary_proj = torch.nn.Linear(4, 3)
        self.out_features_config = LabelConfig(
            branches={
                "primary": LabelBranchConfig(type="multi", num_classes=3),
            }
        )

        with torch.no_grad():
            self.primary_proj.weight.zero_()
            self.primary_proj.bias.copy_(torch.tensor([2.0, -2.0, 2.0]))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        batch_size = input_ids.shape[0]
        dummy_features = torch.zeros(batch_size, 4, device=input_ids.device)
        return [self.primary_proj(dummy_features)]


class TestBertManagerTrain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        print(f"temp_dir: {self.temp_dir}")
        self.addCleanup(self.temp_dir.cleanup)
        self.manager = BertManager(
            model_name_or_path="google-bert/bert-base-multilingual-cased",
            cache_dir=self.temp_dir.name,
        )
        self.model = cast(BertIncrModel, cast(torch.nn.Module, DummyIncrModel()))
        self.dataset = cast(BertDataset, cast(object, DummyDataset(length=1)))
        self.label_fields = ["primary"]

    def test_train_starts_fresh_without_checkpoint(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        with patch.object(self.manager, "_create_data_loader", side_effect=[[], None]):
            result = self.manager.train(
                bert_incr_model=self.model,
                train_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=1,
                label_fields=self.label_fields,
                optimizer=optimizer,
            )

        self.assertIsNone(result["checkpoint_path"])
        self.assertIsNone(result["checkpoint_epoch"])
        self.assertEqual(result["start_epoch"], 1)
        self.assertEqual(result["target_epochs"], 1)
        self.assertIsNone(result["load_result"])

    def test_train_uses_shuffle_false_for_validation_loader(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        with patch.object(self.manager, "_create_data_loader", side_effect=[[], []]) as mocked_create_loader:
            self.manager.train(
                bert_incr_model=self.model,
                train_dataset=self.dataset,
                valid_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=1,
                label_fields=self.label_fields,
                optimizer=optimizer,
            )

        self.assertEqual(mocked_create_loader.call_count, 2)
        self.assertTrue(mocked_create_loader.call_args_list[0].kwargs["shuffle"])
        self.assertFalse(mocked_create_loader.call_args_list[1].kwargs["shuffle"])

    def test_train_keeps_only_latest_periodic_checkpoints_when_limited(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        batch = (
            torch.ones((1, 4), dtype=torch.long),
            torch.ones((1, 4), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )

        with patch.object(self.manager, "_create_data_loader", side_effect=[[batch], None]):
            self.manager.train(
                bert_incr_model=self.model,
                train_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=30,
                label_fields=self.label_fields,
                optimizer=optimizer,
                periodic_checkpoint_max_keep=2,
            )

        periodic_files = sorted(
            f for f in os.listdir(self.temp_dir.name)
            if f.startswith("periodic_epoch_") and f.endswith(".pth")
        )
        self.assertEqual(periodic_files, ["periodic_epoch_20.pth", "periodic_epoch_30.pth"])

    def test_train_rejects_invalid_periodic_checkpoint_max_keep(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        with self.assertRaisesRegex(ValueError, "periodic_checkpoint_max_keep must be >= 1 or None"):
            self.manager.train(
                bert_incr_model=self.model,
                train_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=1,
                label_fields=self.label_fields,
                optimizer=optimizer,
                periodic_checkpoint_max_keep=0,
            )

    def test_train_resumes_when_checkpoint_path_is_provided(self):
        load_result = {
            "checkpoint_type": "full_checkpoint",
            "path": "/tmp/mock_checkpoint.pth",
            "epoch": 3,
            "has_optimizer_state": True,
            "optimizer_loaded": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }

        with patch.object(self.manager, "_create_data_loader", side_effect=[[], None]), patch.object(
            self.manager,
            "load_checkpoint",
            return_value=load_result,
        ) as mocked_load_checkpoint:
            result = self.manager.train(
                bert_incr_model=self.model,
                train_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=5,
                label_fields=self.label_fields,
                checkpoint_path=load_result["path"],
            )

        self.assertEqual(result["checkpoint_path"], load_result["path"])
        self.assertEqual(result["checkpoint_epoch"], 3)
        self.assertEqual(result["start_epoch"], 4)
        self.assertEqual(result["target_epochs"], 5)
        self.assertEqual(result["load_result"], load_result)
        self.assertEqual(mocked_load_checkpoint.call_count, 1)
        self.assertIsNotNone(mocked_load_checkpoint.call_args.kwargs["optimizer"])

    def test_train_rejects_start_epoch_when_checkpoint_path_is_provided(self):
        with patch.object(self.manager, "load_checkpoint") as mocked_load_checkpoint:
            with self.assertRaisesRegex(ValueError, "start_epoch cannot be used together with checkpoint_path"):
                self.manager.train(
                    bert_incr_model=self.model,
                    train_dataset=self.dataset,
                    params_save_path=self.temp_dir.name,
                    epochs=5,
                    label_fields=self.label_fields,
                    checkpoint_path="/tmp/mock_checkpoint.pth",
                    start_epoch=2,
                )

        mocked_load_checkpoint.assert_not_called()

    def test_train_rejects_resume_when_no_epochs_left(self):
        with patch.object(
            self.manager,
            "load_checkpoint",
            return_value={
                "checkpoint_type": "full_checkpoint",
                "path": "/tmp/mock_checkpoint.pth",
                "epoch": 5,
                "has_optimizer_state": True,
                "optimizer_loaded": True,
                "missing_keys": [],
                "unexpected_keys": [],
            },
        ):
            with self.assertRaisesRegex(ValueError, r"start_epoch \(6\) cannot be greater than epochs \(5\)"):
                self.manager.train(
                    bert_incr_model=self.model,
                    train_dataset=self.dataset,
                    params_save_path=self.temp_dir.name,
                    epochs=5,
                    label_fields=self.label_fields,
                    checkpoint_path="/tmp/mock_checkpoint.pth",
                )

    def test_resume_train_delegates_to_train_with_checkpoint_path(self):
        expected = {"ok": True}

        with patch.object(self.manager, "train", return_value=expected) as mocked_train:
            result = self.manager.resume_train(
                bert_incr_model=self.model,
                train_dataset=self.dataset,
                checkpoint_path="/tmp/mock_checkpoint.pth",
                params_save_path=self.temp_dir.name,
                epochs=5,
                label_fields=self.label_fields,
            )

        self.assertEqual(result, expected)
        self.assertEqual(mocked_train.call_count, 1)
        self.assertEqual(mocked_train.call_args.kwargs["checkpoint_path"], "/tmp/mock_checkpoint.pth")

    def test_train_returns_additional_branch_accuracy_metrics(self):
        model = cast(BertIncrModel, cast(torch.nn.Module, DummyMultiBranchIncrModel()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # 预置旧 best 文件，验证新 best 保存后会被清理
        stale_best_paths = [
            os.path.join(self.temp_dir.name, "best_bert_epoch_99_acc_0.1000.pth"),
            os.path.join(self.temp_dir.name, "best_bert_epoch_100_acc_0.2000.pth"),
        ]
        for stale_path in stale_best_paths:
            with open(stale_path, "wb") as f:
                f.write(b"stale")

        input_ids = torch.ones((2, 4), dtype=torch.long)
        attention_mask = torch.ones((2, 4), dtype=torch.long)
        token_type_ids = torch.zeros((2, 4), dtype=torch.long)
        primary_labels = torch.tensor([0, 0], dtype=torch.long)
        additional_labels = torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.long)
        batch = (input_ids, attention_mask, token_type_ids, primary_labels, additional_labels)

        with patch.object(self.manager, "_create_data_loader", side_effect=[[batch], [batch]]):
            result = self.manager.train(
                bert_incr_model=model,
                train_dataset=self.dataset,
                valid_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=1,
                label_fields=["primary", "additional"],
                loss_weights=[0.7, 0.3],
                optimizer=optimizer,
            )

        self.assertIn("additional_acc", result["final_train_metrics"])
        self.assertIn("additional_label_acc", result["final_train_metrics"])
        self.assertIn("additional_acc", result["final_valid_metrics"])
        self.assertIn("additional_label_acc", result["final_valid_metrics"])
        self.assertAlmostEqual(result["final_train_metrics"]["additional_acc"], 1.0, places=4)
        self.assertAlmostEqual(result["final_train_metrics"]["additional_label_acc"], 1.0, places=4)
        self.assertAlmostEqual(result["final_valid_metrics"]["additional_acc"], 1.0, places=4)
        self.assertAlmostEqual(result["final_valid_metrics"]["additional_label_acc"], 1.0, places=4)

        best_files = [f for f in os.listdir(self.temp_dir.name) if f.startswith("best_bert_epoch_") and f.endswith(".pth")]
        self.assertEqual(len(best_files), 1)
        self.assertTrue(best_files[0].startswith("best_bert_epoch_1_acc_"))

    def test_train_main_acc_is_sample_weighted_across_uneven_batches(self):
        model = cast(BertIncrModel, cast(torch.nn.Module, DummyIncrModel()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with torch.no_grad():
            model.proj.weight.zero_()
            model.proj.bias.copy_(torch.tensor([2.0, -2.0]))

        # batch1: 1 个样本全错（acc=0.0）
        b1 = (
            torch.ones((1, 4), dtype=torch.long),
            torch.ones((1, 4), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )
        # batch2: 3 个样本全对（acc=1.0）
        b2 = (
            torch.ones((3, 4), dtype=torch.long),
            torch.ones((3, 4), dtype=torch.long),
            torch.zeros((3, 4), dtype=torch.long),
            torch.tensor([0, 0, 0], dtype=torch.long),
        )

        with patch.object(self.manager, "_create_data_loader", side_effect=[[b1, b2], None]):
            result = self.manager.train(
                bert_incr_model=model,
                train_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=1,
                label_fields=["primary"],
                optimizer=optimizer,
            )

        # 样本加权后应为 (0 + 3) / 4 = 0.75，而不是批次平均 0.5
        self.assertAlmostEqual(result["final_train_metrics"]["main_acc"], 0.75, places=4)
        self.assertAlmostEqual(result["final_train_metrics"]["primary_acc"], 0.75, places=4)

    def test_train_supports_multi_primary_main_acc(self):
        model = cast(BertIncrModel, cast(torch.nn.Module, DummyPrimaryMultiIncrModel()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        input_ids = torch.ones((2, 4), dtype=torch.long)
        attention_mask = torch.ones((2, 4), dtype=torch.long)
        token_type_ids = torch.zeros((2, 4), dtype=torch.long)
        # 第一条 exact-match 正确，第二条错误 => main_acc=0.5，label_acc=5/6
        primary_labels = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.long)
        batch = (input_ids, attention_mask, token_type_ids, primary_labels)

        with patch.object(self.manager, "_create_data_loader", side_effect=[[batch], None]):
            result = self.manager.train(
                bert_incr_model=model,
                train_dataset=self.dataset,
                params_save_path=self.temp_dir.name,
                epochs=1,
                label_fields=["primary"],
                optimizer=optimizer,
            )

        self.assertAlmostEqual(result["final_train_metrics"]["main_acc"], 0.5, places=4)
        self.assertAlmostEqual(result["final_train_metrics"]["primary_acc"], 0.5, places=4)
        self.assertAlmostEqual(result["final_train_metrics"]["primary_label_acc"], 5 / 6, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

