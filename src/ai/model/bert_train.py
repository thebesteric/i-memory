from src.ai.model.bert_manager import BertManager, BertIncrModel, LabelConfig, LabelBranchConfig

if __name__ == '__main__':
    manager = BertManager(model_name_or_path="google-bert/bert-base-multilingual-cased")

    dataset_path = "/Users/wangweijun/PycharmProjects/i-memory/assets/datasets"
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