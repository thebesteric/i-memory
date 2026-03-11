import argparse

from src.ai.model.bert_manager import BertManager, BertIncrModel, LabelConfig, LabelBranchConfig

if __name__ == '__main__':

    # 从命令行获取 model_name_or_path
    parser = argparse.ArgumentParser(description='BERT 模型训练脚本 - 从命令行指定模型路径/名称')
    parser.add_argument(
        '--dataset_path',  # 参数名（命令行使用 --xxx 形式）
        type=str,  # 参数类型
        required=True,  # 设置为必选参数
        help='数据集路径'
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path or "/Users/wangweijun/PycharmProjects/i-memory/assets/datasets"

    manager = BertManager(model_name_or_path="google-bert/bert-base-multilingual-cased")

    train_dataset = manager.load_dataset(dataset_path, load_type="json", split="train")
    valid_dataset = manager.load_dataset(dataset_path, load_type="json", split="validation")

    bert_incr_model = BertIncrModel(
        bert_manager=manager,
        in_features=768,
        out_features_config=LabelConfig(
            branches={
                "primary": LabelBranchConfig(type="single", num_classes=5),
                "additional": LabelBranchConfig(type="multi", num_classes=5)
            }
        )
    )

    manager.train(
        bert_incr_model=bert_incr_model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=100,
        epochs=1000,
        text_field="text",
        label_fields=["primary", "additional"],
        loss_weights=[0.7, 0.3],
    )