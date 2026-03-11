import json
import unittest

import torch

from src.ai.model.bert_manager import BertManager


class TestBertManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manager = BertManager(model_name_or_path="google-bert/bert-base-multilingual-cased")
        cls.model, cls.tokenizer = cls.manager.load_model()
        print(f"Download and load success: {cls.manager.model_name_or_path}")
        print(f"cache_dir={cls.manager.cache_dir}")
        print(f"model_info={cls.manager.model_info.model_dump()}")
        print(f"model_class={cls.model.__class__.__name__}")
        print(f"tokenizer_class={cls.tokenizer.__class__.__name__}")

    def test_query(self):
        text = "今天天气不错，适合测试模型是否可用。"
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        print("OK")
        print("last_hidden_state shape:", tuple(outputs.last_hidden_state.shape))

    def test_dataset_csv(self):
        dataset_path = "/Users/wangweijun/llm/datasets/ChnSentiCorp/csv_files"
        dataset = self.manager.load_dataset(dataset_path, load_type="csv", split="train")
        print(f"Dataset loaded from CSV: {dataset}")
        print(f"Dataset length: {len(dataset)}")
        print(f"First sample: {dataset[0]}")

    def test_dataset_json(self):
        dataset_path = "/Users/wangweijun/PycharmProjects/i-memory/assets/datasets"
        dataset = self.manager.load_dataset(dataset_path, load_type="json", split="train")
        print(f"Dataset loaded from JSON: {dataset}")
        print(f"Dataset length: {len(dataset)}")
        print(f"First sample: {dataset[0]}")

        m: dict[int, list] = {}
        for d in dataset:
            primary = d["primary"]
            l = m.get(primary)
            if l is None:
                l = []
                m[primary] = l
            l.append(d)

        # 输出每个 primary 的样本数量
        for primary, items in m.items():
            print(f"Primary: {primary}, Count: {len(items)}")

        count = 0
        for primary, items in m.items():
            for item in items:
                # 找到所有 additional 全部为 [1,1,1,1,1] 的数据
                additional = item["additional"]
                if additional == [1, 1, 1, 1, 1]:
                    print(f"{primary}: {item}")
                    count += 1
        print(f"Total samples with additional [1,1,1,1,1]: {count}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
