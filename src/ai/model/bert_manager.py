import json
import os
from typing import Literal, Union, Any

import pyrootutils
import torch
from agile.utils import LogHelper
from datasets import load_dataset, load_from_disk, IterableDataset, IterableDatasetDict, DatasetDict, Dataset, IterableColumn
from pydantic import BaseModel, Field, field_validator
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, PreTrainedConfig

logger = LogHelper.get_logger()

DatasetType = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


class BertDataset(TorchDataset):
    """
    数据集类，负责加载和提供数据集样本
    """

    def __init__(
            self,
            *,
            dataset_path: str,
            load_type: Literal["csv", "json", "arrow"] = "csv",
            split: Literal["train", "test", "validation"] = "train"
    ):
        """
        初始化数据集
        :param dataset_path: 数据集路径，可以是 CSV/JSON 文件夹路径或 Arrow 格式数据集的磁盘路径
        :param load_type: 数据集类型，可选值为 "csv"、"json" 或 "arrow"
        :param split: 数据集划分，可选值为 "train"、"test" 或 "validation"
        """
        self.dataset_path = dataset_path
        self.load_type = load_type
        self.split = split
        self.dataset: DatasetType = self._load_dataset()

    def _load_dataset(self) -> DatasetType:
        # 加载数据集
        if self.load_type == "csv":
            # 从 CSV 加载数据集
            csv_file_path = f"{self.dataset_path}/{self.split}.csv"
            dataset = load_dataset("csv", data_files=csv_file_path, split=self.split)
        elif self.load_type == "json":
            # 从 JSON 加载数据集
            json_file_path = f"{self.dataset_path}/{self.split}.json"
            with open(json_file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                # 重置文件指针
                f.seek(0)
                # 判断是单行 JSONL（每行一个样本）还是数组 JSON
                if first_line.startswith("["):
                    # 数组 JSON 格式：[{"text": "...", "label": ...}, ...]
                    data = json.load(f)
                    dataset = Dataset.from_list(data)
                else:
                    # 单行 JSONL 格式：每行一个 JSON 对象
                    dataset = load_dataset(
                        "json",
                        data_files=json_file_path,
                        split=self.split
                    )
        elif self.load_type == "arrow":
            # 加载 Arrow 格式（从磁盘加载已保存的 Dataset）
            dataset = load_from_disk(self.dataset_path)[self.split]
        else:
            raise ValueError(f"不支持的加载类型：{self.load_type}，仅支持 csv/json/arrow")

        return dataset

    def __len__(self) -> int:
        """
        返回数据集长度
        :return: 数据集样本数量
        """
        return len(self.dataset)

    def __getitem__(self, index) -> Dataset | IterableColumn:
        """
        获取数据集的某一个样本
        :param index: 样本索引
        :return: 数据集样本，类型为 Dataset 或 IterableColumn，具体取决于数据集的类型
        """
        data = self.dataset[index]
        return data


class BertModelInfo(BaseModel):
    model_name_or_path: str = Field(..., description="模型名称或本地路径")
    cache_dir: str = Field(..., description="模型缓存目录")
    device: str = Field(default=None, description="模型运行设备")
    hidden_size: int = Field(default=None, description="隐藏层大小")
    num_hidden_layers: int = Field(default=None, description="隐藏层数量")
    num_attention_heads: int = Field(default=None, description="注意力头数量")
    intermediate_size: int = Field(default=None, description="中间层大小")
    vocab_size: int = Field(default=None, description="词汇表大小")
    max_position_embeddings: int = Field(default=None, description="最大位置嵌入数量")

    def __init__(self, model_name_or_path: str, cache_dir: str, device: str, config: PreTrainedConfig):
        super().__init__(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir
        )
        self.device = device or None
        self.hidden_size = int(getattr(config, "hidden_size", 0)) or None
        self.num_hidden_layers = getattr(config, "num_hidden_layers", None)
        self.num_attention_heads = getattr(config, "num_attention_heads", None)
        self.intermediate_size = getattr(config, "intermediate_size", None)
        self.vocab_size = getattr(config, "vocab_size", None)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", None)


class BertManager:

    def __init__(self,
                 model_name_or_path: str | os.PathLike[str] = "google-bert/bert-base-multilingual-cased",
                 cache_dir: str | os.PathLike[str] | None = None):
        """
        模型管理器，负责按需加载和管理 BERT 模型及其分词器
        :param model_name_or_path: BERT 模型名称或本地路径
        :param cache_dir: 模型缓存目录，默认使用 <project_root>/assets/models/<model_name_or_path>
        """
        self.model_name_or_path = os.fspath(model_name_or_path)
        project_root = pyrootutils.find_root()
        self.cache_dir = os.fspath(cache_dir) if cache_dir else os.path.join(project_root, "assets", "models", self.model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型和分词器实例，初始为 None，按需加载
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

        # 初始化时检查模型是否已本地存在
        self._is_model_local = self._check_model_files_exist()

        logger.info(f"BERT model initialized: {self.model_name_or_path}, "
                    f"cache_dir={self.cache_dir}, device={self.device}, is_model_local={self._is_model_local}")

    def _check_model_files_exist(self) -> bool:
        """
        检查模型文件是否已存在于本地缓存目录
        :return: 存在返回 True，否则返回 False
        """
        return os.path.exists(self.cache_dir)

    def load_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._model is None:
            try:
                self._model = AutoModel.from_pretrained(
                    self.model_name_or_path,
                    cache_dir=self.cache_dir,
                    local_files_only=self._is_model_local,
                ).to(self.device)
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch is required to load transformer models. "
                    "Install it in this project environment with: pip install torch"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load model '{self.model_name_or_path}' from path '{self.cache_dir}'. "
                    f"Error: {exc}"
                ) from exc
            logger.info(f"Model {self.model_name_or_path} loaded successfully")

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                local_files_only=self._is_model_local,
            )
            logger.info(f"Model {self.model_name_or_path} tokenizer loaded successfully")

        return self._model, self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        return self._model or self.load_model()[0]

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer or self.load_model()[1]

    @property
    def model_info(self) -> BertModelInfo:
        """
        返回模型的基本信息
        :return: BertModelInfo 实例，包含模型名称、缓存目录、设备和配置等信息
        """
        config = self.model.config
        return BertModelInfo(
            model_name_or_path=self.model_name_or_path,
            cache_dir=self.cache_dir,
            device=str(self.device),
            config=config
        )

    @staticmethod
    def load_dataset(
            dataset_path: str,
            load_type: Literal["csv", "json", "arrow"] = "csv",
            split: Literal["train", "test", "validation"] = "train"
    ) -> BertDataset:
        """
        负责加载数据集，返回 BertDataset 实例
        :param dataset_path: 数据集路径，可以是 CSV/JSON 文件夹路径或 Arrow 格式数据集的磁盘路径
        :param load_type: 数据集类型，可选值为 "csv"、"json" 或 "arrow"
        :param split: 数据集划分，可选值为 "train"、"test" 或 "validation"
        :return: BertDataset 实例
        """
        return BertDataset(dataset_path=dataset_path, load_type=load_type, split=split)

    def train(self,
              bert_incr_model: "BertIncrModel",
              train_dataset: BertDataset,
              *,
              text_field: str = "text",
              label_fields: list[str] = None,
              loss_weights: list[float] = None,
              params_save_path: str = None,
              valid_dataset: BertDataset = None,
              batch_size: int = 100,
              epochs: int = 10000,
              lr: float = 1e-3,
              patience: int = 10,
              drop_last: bool = False,
              start_epoch: int | None = None,
              optimizer: torch.optim.Optimizer | None = None,
              checkpoint_path: str | None = None,
              load_optimizer: bool = True,
              strict: bool = True,
              map_location: str | torch.device | None = None) -> dict[str, Any]:
        """
        增量训练。默认从头开始；当传入 checkpoint_path 时自动加载 checkpoint 并续训。
        :param bert_incr_model: BertIncrModel 实例，包含 BERT 模型和增量训练的分支配置
        :param train_dataset: BertDataset 实例，训练数据集
        :param params_save_path: 模型参数保存路径，默认为模型缓存目录下的 "params" 文件夹
        :param valid_dataset: BertDataset 实例，验证数据集
        :param batch_size: 批次大小
        :param epochs: 训练轮数
        :param lr: 学习率
        :param patience: 提前停止的耐心值
        :param text_field: 文本字段名称
        :param label_fields: 标签字段名称列表
        :param loss_weights: 损失权重列表
        :param drop_last: 是否丢弃最后一个不完整的批次
        :param start_epoch: 训练起始轮次（手动控制时使用），默认从 1 开始
        :param optimizer: 可选外部优化器（用于断点续训恢复状态）
        :param checkpoint_path: 可选 checkpoint 路径；传入后自动从下一轮继续训练
        :param load_optimizer: 续训时是否恢复优化器状态
        :param strict: 加载 checkpoint 时是否严格匹配模型权重 key
        :param map_location: checkpoint 加载设备
        :return: 训练元信息（包含是否续训、起始 epoch、checkpoint 加载结果等）
        """

        if not bert_incr_model:
            raise ValueError("Model for incremental training is required")

        if not label_fields:
            raise ValueError("At least one label field is required for training")

        # 校验训练数据集是否为空
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty, Please check your dataset path or load_type.")

        # 默认损失权重：等权，loss_weights 的长度必须与 label_fields 的长度一致
        loss_weights = loss_weights or [1.0 / len(label_fields)] * len(label_fields)
        if len(loss_weights) != len(label_fields):
            raise ValueError(f"loss_weights ({len(loss_weights)}) must match the length of label_fields ({len(label_fields)})")

        # 检查 label_fields 中是否有重复的字段名
        if len(set(label_fields)) != len(label_fields):
            raise ValueError(f"label_fields contains duplicate names: {label_fields}")

        # 续训时 start_epoch 会根据 checkpoint 计算出 start_epoch，无需外部传入
        if checkpoint_path and start_epoch is not None:
            raise ValueError(
                "start_epoch cannot be used together with checkpoint_path; "
                "when checkpoint_path is provided, start_epoch is derived from the checkpoint automatically"
            )

        # 定义优化器（续训时可复用外部传入的优化器状态）
        optimizer = optimizer or AdamW(bert_incr_model.parameters(), lr=lr)

        load_result = None
        checkpoint_epoch = None
        if checkpoint_path:
            load_result = self.load_checkpoint(
                bert_incr_model=bert_incr_model,
                checkpoint_path=checkpoint_path,
                optimizer=optimizer,
                load_optimizer=load_optimizer,
                strict=strict,
                map_location=map_location,
            )
            # 读取 checkpoint 的当前 epoch
            checkpoint_epoch = int(load_result.get("epoch") or 0)
            if load_result["epoch"] is None:
                logger.warning(
                    f"[RESUME] checkpoint={checkpoint_path}, epoch_metadata=missing, resume_from_epoch=1"
                )

            # 根据 checkpoint 计算出续训的起始 start_epoch，从 checkpoint_epoch 的下一轮开始续训
            start_epoch = checkpoint_epoch + 1
            logger.info(
                f"[RESUME] checkpoint={checkpoint_path}, checkpoint_epoch={checkpoint_epoch}, "
                f"resume_range={start_epoch}-{epochs}, load_optimizer={load_optimizer}"
            )

        # 初始化 start_epoch（当没有 checkpoint_path 时，默认为 1）
        start_epoch = start_epoch or 1
        if start_epoch < 1:
            raise ValueError(f"start_epoch must be >= 1, got {start_epoch}")

        if start_epoch > epochs:
            raise ValueError(f"start_epoch ({start_epoch}) cannot be greater than epochs ({epochs})")

        # 检查 label_fields 与 configured_label_names 是否一致
        configured_label_names = bert_incr_model.out_features_config.label_names
        if set(label_fields) != set(configured_label_names):
            raise ValueError(
                f"label_fields {label_fields} must match model branches {configured_label_names}"
            )

        # 按 label_fields 构建分支配置，确保标签、输出、损失函数都按名称对齐
        branch_meta: list[tuple[str, torch.nn.Module, float, LabelBranchConfig]] = []
        for idx, label_name in enumerate(label_fields):
            branch_cfg = bert_incr_model.out_features_config.get_branch_config(label_name)
            if branch_cfg.type == "single":
                loss_fn = torch.nn.CrossEntropyLoss()
            elif branch_cfg.type == "multi":
                loss_fn = torch.nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unsupported branch type '{branch_cfg.type}' for label '{label_name}'")
            branch_meta.append((label_name, loss_fn, loss_weights[idx], branch_cfg))

        # 主标签名称，默认为 label_fields 的第一个字段
        main_label_name = label_fields[0]

        # 构建训练参数保存路径，默认为模型缓存目录下的 "params" 文件夹
        params_save_path = params_save_path or os.path.join(self.cache_dir, "params")
        os.makedirs(params_save_path, exist_ok=True)

        logger.info(
            "[CONFIG]\n"
            f"  runtime: model={self.model_name_or_path}, device={self.device}, freeze_bert={bert_incr_model.freeze_bert}\n"
            f"  train: batch_size={batch_size}, epochs={epochs}, lr={lr}, drop_last={drop_last}\n"
            f"  labels: text_field={text_field}, label_fields={label_fields}, loss_weights={loss_weights}\n"
            f"  dataset: train_size={len(train_dataset)}, valid_size={len(valid_dataset) if valid_dataset else 'N/A'}"
        )

        # 创建训练数据集
        train_loader = self._create_data_loader(
            train_dataset,
            text_field=text_field,
            label_fields=label_fields,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
        # 创建验证数据集
        valid_loader = self._create_data_loader(
            valid_dataset,
            text_field=text_field,
            label_fields=label_fields,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
        )
        # 定义学习率调度器，根据验证准确率调整学习率，验证准确率连续 2 个 epoch 没有提升，则将学习率降低到原来的 0.5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=2,
            factor=0.5
        )

        # 初始化验证最佳准确率
        best_val_acc = 0.0
        # 早停计数器，记录验证准确率连续没有提升的 epoch 数
        early_stop_count = 0
        # 显式设置训练模式
        bert_incr_model.train()
        # 记录最终完成训练的 epoch（用于训练结束后的兜底保存）
        final_epoch = 0
        final_train_metrics: dict[str, float] = {}
        final_valid_metrics: dict[str, float] = {}
        # 开始训练（默认从 1 开始；续训时可指定 start_epoch）
        for epoch in range(start_epoch, epochs + 1):
            # 防止验证阶段切换到 eval 后影响后续 epoch
            bert_incr_model.train()
            # 训练总损失
            train_total_loss = 0.0
            # 主标签准确率
            train_main_correct = 0.0
            train_main_total = 0.0
            # 每个分支的正确数/总数累加（用于计算加权平均准确率）
            train_branch_correct_sums = {label_name: 0.0 for label_name, _, _, _ in branch_meta}
            train_branch_total_sums = {label_name: 0.0 for label_name, _, _, _ in branch_meta}
            # 仅多标签分支的标签位正确数/总标签位累加
            train_branch_label_correct_sums = {label_name: 0.0 for label_name, _, _, branch_cfg in branch_meta if branch_cfg.type == "multi"}
            train_branch_label_total_sums = {label_name: 0.0 for label_name, _, _, branch_cfg in branch_meta if branch_cfg.type == "multi"}
            # 训练批次计数器
            train_steps = 0

            # 加载训练集
            for i, batch in enumerate(train_loader):
                # 拆分 拆分 batch：input_ids(0), attention_mask(1), token_type_ids(2), 主标签(3), 辅标签(4), ...
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                token_type_ids = batch[2].to(self.device)
                labels = [label.to(self.device) for label in batch[3:]]

                # 前向计算；将数据输入模型，得到输出
                logits_list = bert_incr_model(input_ids, attention_mask, token_type_ids)
                if len(logits_list) != len(configured_label_names):
                    raise ValueError(
                        f"Number of model output branches ({len(logits_list)}) must match configured branches ({len(configured_label_names)})"
                    )
                logits_by_name = dict(zip(configured_label_names, logits_list))

                # 计算多标签损失（加权求和）
                loss = torch.zeros((), device=self.device)
                for branch_idx, (label_name, loss_fn, weight, branch_cfg) in enumerate(branch_meta):
                    logits = logits_by_name[label_name]
                    label = labels[branch_idx]
                    if branch_cfg.type == "single":
                        label = label.long()
                    else:
                        label = label.float()
                    branch_loss = loss_fn(logits, label)
                    loss = loss + weight * branch_loss

                # 根据误差优化参数
                # 所有参数的梯度清零
                optimizer.zero_grad()
                # 执行反向传播（求导），计算损失函数相对于模型所有可训练参数的梯度
                loss.backward()
                # 根据计算得到的梯度，使用优化算法来自动更新模型的参数
                optimizer.step()

                # 累计指标（主标签准确率）
                train_total_loss += loss.item()
                train_steps += 1

                # 计算各分支准确率（single=分类准确率，multi=样本级 exact-match）
                train_batch_branch_acc: dict[str, float] = {}
                train_batch_branch_label_acc: dict[str, float] = {}
                main_batch_acc = 0.0
                for branch_idx, (label_name, _, _, branch_cfg) in enumerate(branch_meta):
                    branch_metrics = self._compute_branch_metrics(
                        logits=logits_by_name[label_name],
                        labels=labels[branch_idx],
                        branch_cfg=branch_cfg,
                    )
                    train_branch_correct_sums[label_name] += branch_metrics["correct"]
                    train_branch_total_sums[label_name] += branch_metrics["total"]
                    train_batch_branch_acc[label_name] = branch_metrics["acc"]
                    if "label_acc" in branch_metrics:
                        train_branch_label_correct_sums[label_name] += branch_metrics["label_correct"]
                        train_branch_label_total_sums[label_name] += branch_metrics["label_total"]
                        train_batch_branch_label_acc[label_name] = branch_metrics["label_acc"]
                    if label_name == main_label_name:
                        main_batch_acc = branch_metrics["acc"]
                        train_main_correct += branch_metrics["correct"]
                        train_main_total += branch_metrics["total"]

                # 每隔 5 个批次，输出训练信息
                if i % 5 == 0:
                    extra_acc_logs = ", ".join(
                        f"{name}_acc={acc:.4f}"
                        for name, acc in train_batch_branch_acc.items()
                        if name != main_label_name
                    )
                    extra_label_acc_logs = ", ".join(
                        f"{name}_label_acc={acc:.4f}"
                        for name, acc in train_batch_branch_label_acc.items()
                    )
                    logger.info(
                        f"[TRAIN] epoch={epoch}/{epochs}, batch_size={batch_size}, step={i}, "
                        f"loss={loss.item():.4f}, main_acc={main_batch_acc:.4f}"
                        f"{(', ' + extra_acc_logs) if extra_acc_logs else ''}"
                        f"{(', ' + extra_label_acc_logs) if extra_label_acc_logs else ''}"
                    )

            # 训练集平均指标
            if train_steps == 0:
                logger.warning(
                    f"[TRAIN] epoch={epoch}/{epochs} no_batches, batch_size={batch_size}, drop_last={drop_last}"
                )
                continue
            avg_train_loss = train_total_loss / train_steps
            avg_train_acc = (train_main_correct / train_main_total) if train_main_total > 0 else 0.0
            avg_train_branch_accs = {
                label_name: (train_branch_correct_sums[label_name] / train_branch_total_sums[label_name])
                if train_branch_total_sums[label_name] > 0 else 0.0
                for label_name in train_branch_correct_sums.keys()
            }
            avg_train_branch_label_accs = {
                label_name: (train_branch_label_correct_sums[label_name] / train_branch_label_total_sums[label_name])
                if train_branch_label_total_sums[label_name] > 0 else 0.0
                for label_name in train_branch_label_correct_sums.keys()
            }
            final_train_metrics = {
                "loss": avg_train_loss,
                "main_acc": avg_train_acc,
                **{f"{name}_acc": acc for name, acc in avg_train_branch_accs.items()},
                **{f"{name}_label_acc": acc for name, acc in avg_train_branch_label_accs.items()},
            }
            train_extra_acc_summary = ", ".join(
                f"avg_{name}_acc={acc:.4f}"
                for name, acc in avg_train_branch_accs.items()
                if name != main_label_name
            )
            train_extra_label_acc_summary = ", ".join(
                f"avg_{name}_label_acc={acc:.4f}"
                for name, acc in avg_train_branch_label_accs.items()
            )
            logger.info(
                f"[TRAIN] epoch={epoch}/{epochs}, summary, "
                f"avg_loss={avg_train_loss:.4f}, avg_main_acc={avg_train_acc:.4f}, "
                f"steps={train_steps}{(', ' + train_extra_acc_summary) if train_extra_acc_summary else ''}"
                f"{(', ' + train_extra_label_acc_summary) if train_extra_label_acc_summary else ''}"
            )
            final_epoch = epoch

            # 判断是否存在验证集，如果存在，则进行验证，否则跳过验证步骤
            if not valid_loader:
                # 每训练 10 个 epoch，保存一次参数
                if epoch % 10 == 0:
                    periodic_file_param_save_path = os.path.join(params_save_path, f"periodic_epoch_{epoch}.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": bert_incr_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, periodic_file_param_save_path)
                    logger.info(
                        f"[SAVE] epoch={epoch}/{epochs}, kind=periodic, path={periodic_file_param_save_path}"
                    )
            else:
                # 验证模型，判断模型是否过拟合
                val_total_loss = 0.0
                val_main_correct = 0.0
                val_main_total = 0.0
                val_branch_correct_sums = {label_name: 0.0 for label_name, _, _, _ in branch_meta}
                val_branch_total_sums = {label_name: 0.0 for label_name, _, _, _ in branch_meta}
                val_branch_label_correct_sums = {label_name: 0.0 for label_name, _, _, branch_cfg in branch_meta if branch_cfg.type == "multi"}
                val_branch_label_total_sums = {label_name: 0.0 for label_name, _, _, branch_cfg in branch_meta if branch_cfg.type == "multi"}
                val_steps = 0
                # 设置为评估模式
                bert_incr_model.eval()
                # 验证的时候，不需要模型参与训练，所以不需要梯度
                with torch.no_grad():
                    # 加载验证集
                    for i, batch in enumerate(valid_loader):
                        # 拆分 拆分 batch：input_ids(0), attention_mask(1), token_type_ids(2), 主标签(3), 辉标签(4), ...
                        input_ids = batch[0].to(self.device)
                        attention_mask = batch[1].to(self.device)
                        token_type_ids = batch[2].to(self.device)
                        labels = [label.to(self.device) for label in batch[3:]]

                        # 前向计算；将数据输入模型，得到输出
                        logits_list = bert_incr_model(input_ids, attention_mask, token_type_ids)
                        if len(logits_list) != len(configured_label_names):
                            raise ValueError(
                                f"Number of model output branches ({len(logits_list)}) must match configured branches ({len(configured_label_names)})"
                            )
                        logits_by_name = dict(zip(configured_label_names, logits_list))

                        # 计算多标签损失（加权求和）
                        loss = torch.zeros((), device=self.device)
                        for branch_idx, (label_name, loss_fn, weight, branch_cfg) in enumerate(branch_meta):
                            logits = logits_by_name[label_name]
                            label = labels[branch_idx]
                            if branch_cfg.type == "single":
                                label = label.long()
                            else:
                                label = label.float()
                            branch_loss = loss_fn(logits, label)
                            loss = loss + weight * branch_loss

                        # 累计指标
                        val_total_loss += loss.item()
                        val_steps += 1
                        # 计算各分支准确率（single=分类准确率，multi=样本级 exact-match）
                        val_batch_branch_acc: dict[str, float] = {}
                        val_batch_branch_label_acc: dict[str, float] = {}
                        main_batch_acc = 0.0
                        for branch_idx, (label_name, _, _, branch_cfg) in enumerate(branch_meta):
                            branch_metrics = self._compute_branch_metrics(
                                logits=logits_by_name[label_name],
                                labels=labels[branch_idx],
                                branch_cfg=branch_cfg,
                            )
                            val_branch_correct_sums[label_name] += branch_metrics["correct"]
                            val_branch_total_sums[label_name] += branch_metrics["total"]
                            val_batch_branch_acc[label_name] = branch_metrics["acc"]
                            if "label_acc" in branch_metrics:
                                val_branch_label_correct_sums[label_name] += branch_metrics["label_correct"]
                                val_branch_label_total_sums[label_name] += branch_metrics["label_total"]
                                val_batch_branch_label_acc[label_name] = branch_metrics["label_acc"]
                            if label_name == main_label_name:
                                main_batch_acc = branch_metrics["acc"]
                                val_main_correct += branch_metrics["correct"]
                                val_main_total += branch_metrics["total"]

                        # 每隔 5 个批次，输出验证信息
                        if i % 5 == 0:
                            extra_acc_logs = ", ".join(
                                f"{name}_acc={acc:.4f}"
                                for name, acc in val_batch_branch_acc.items()
                                if name != main_label_name
                            )
                            extra_label_acc_logs = ", ".join(
                                f"{name}_label_acc={acc:.4f}"
                                for name, acc in val_batch_branch_label_acc.items()
                            )
                            logger.info(
                                f"[VALID] epoch={epoch}/{epochs}, step={i}, "
                                f"loss={loss.item():.4f}, main_acc={main_batch_acc:.4f}"
                                f"{(', ' + extra_acc_logs) if extra_acc_logs else ''}"
                                f"{(', ' + extra_label_acc_logs) if extra_label_acc_logs else ''}"
                            )

                    if val_steps == 0:
                        logger.warning(
                            f"[VALID][epoch={epoch}/{epochs}] no_batches, batch_size={batch_size}, drop_last={drop_last}"
                        )
                        continue
                    # 计算验证的平均损失
                    avg_val_loss = val_total_loss / val_steps
                    # 计算验证的平均精度
                    avg_val_acc = (val_main_correct / val_main_total) if val_main_total > 0 else 0.0
                    avg_val_branch_accs = {
                        label_name: (val_branch_correct_sums[label_name] / val_branch_total_sums[label_name])
                        if val_branch_total_sums[label_name] > 0 else 0.0
                        for label_name in val_branch_correct_sums.keys()
                    }
                    avg_val_branch_label_accs = {
                        label_name: (val_branch_label_correct_sums[label_name] / val_branch_label_total_sums[label_name])
                        if val_branch_label_total_sums[label_name] > 0 else 0.0
                        for label_name in val_branch_label_correct_sums.keys()
                    }
                    final_valid_metrics = {
                        "loss": avg_val_loss,
                        "main_acc": avg_val_acc,
                        **{f"{name}_acc": acc for name, acc in avg_val_branch_accs.items()},
                        **{f"{name}_label_acc": acc for name, acc in avg_val_branch_label_accs.items()},
                    }
                    valid_extra_acc_summary = ", ".join(
                        f"avg_{name}_acc={acc:.4f}"
                        for name, acc in avg_val_branch_accs.items()
                        if name != main_label_name
                    )
                    valid_extra_label_acc_summary = ", ".join(
                        f"avg_{name}_label_acc={acc:.4f}"
                        for name, acc in avg_val_branch_label_accs.items()
                    )

                    logger.info(
                        f"[VALID] epoch={epoch}/{epochs}, summary, "
                        f"avg_loss={avg_val_loss:.4f}, avg_main_acc={avg_val_acc:.4f}, "
                        f"steps={val_steps}{(', ' + valid_extra_acc_summary) if valid_extra_acc_summary else ''}"
                        f"{(', ' + valid_extra_label_acc_summary) if valid_extra_label_acc_summary else ''}"
                    )

                    # 学习率调度
                    scheduler.step(avg_val_acc)

                    # 根据验证准确率，保存最优参数
                    if avg_val_acc > best_val_acc:
                        # 把最优的参数保存下来，就是为了方式过拟合，因为一旦过拟合是无法回退的，如果沒有保存，那么只有重新训练
                        # 这就是为什么要保存最优参数的原因
                        best_val_acc = avg_val_acc
                        best_file_params_save_path = os.path.join(params_save_path, f"best_bert_epoch_{epoch}_acc_{best_val_acc:.4f}.pth")
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": bert_incr_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }, best_file_params_save_path)
                        early_stop_count = 0
                        logger.info(
                            f"[SAVE] epoch={epoch}/{epochs}, kind=best, acc={best_val_acc:.4f}, "
                            f"path={best_file_params_save_path}"
                        )
                    else:
                        early_stop_count += 1
                        if early_stop_count >= patience:
                            # 早停触发，停止训练
                            logger.info(
                                f"[EARLY_STOP] epoch={epoch}/{epochs}, patience={patience}, "
                                f"best_val_acc={best_val_acc:.4f}"
                            )
                            break

        # 训练结束后的统一兜底：保存最后一轮参数（避免早停或轮次过小无法保存的情况）
        if final_epoch > 0:
            last_file_params_save_path = os.path.join(params_save_path, f"last_bert_epoch_{final_epoch}.pth")
            torch.save({
                "epoch": final_epoch,
                "model_state_dict": bert_incr_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, last_file_params_save_path)
            logger.info(
                f"[SAVE] epoch={final_epoch}/{epochs}, kind=last_fallback, path={last_file_params_save_path}"
            )

        return {
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": checkpoint_epoch,
            "start_epoch": start_epoch,
            "target_epochs": epochs,
            "load_result": load_result,
            "final_train_metrics": final_train_metrics,
            "final_valid_metrics": final_valid_metrics,
        }

    @staticmethod
    def _compute_branch_metrics(
            *,
            logits: torch.Tensor,
            labels: torch.Tensor,
            branch_cfg: "LabelBranchConfig") -> dict[str, float | int]:
        if branch_cfg.type == "single":
            preds = logits.argmax(dim=1)
            targets = labels.long()
            correct = (preds == targets).sum().item()
            total = targets.numel()
            return {
                "acc": (correct / total) if total > 0 else 0.0,
                "correct": correct,
                "total": total,
            }

        preds = (torch.sigmoid(logits) >= 0.5).long()
        targets = labels.long()
        sample_correct = preds.eq(targets).all(dim=1).sum().item()
        sample_total = targets.size(0)
        label_correct = (preds == targets).sum().item()
        label_total = targets.numel()
        # 多标签按样本级 exact-match 统计，只有整行完全命中才记为正确
        return {
            "acc": (sample_correct / sample_total) if sample_total > 0 else 0.0,
            "correct": sample_correct,
            "total": sample_total,
            "label_acc": (label_correct / label_total) if label_total > 0 else 0.0,
            "label_correct": label_correct,
            "label_total": label_total,
        }

    def resume_train(self,
                     bert_incr_model: "BertIncrModel",
                     train_dataset: BertDataset,
                     checkpoint_path: str,
                     *,
                     params_save_path: str = None,
                     valid_dataset: BertDataset = None,
                     batch_size: int = 100,
                     epochs: int = 10000,
                     lr: float = 1e-3,
                     patience: int = 10,
                     text_field: str = "text",
                     label_fields: list[str] = None,
                     loss_weights: list[float] = None,
                     drop_last: bool = False,
                     load_optimizer: bool = True,
                     strict: bool = True,
                     map_location: str | torch.device | None = None) -> dict[str, Any]:
        """
        断点续训入口：先加载 checkpoint，再从下一轮继续训练。
        :param bert_incr_model: BertIncrModel 实例
        :param train_dataset: 训练数据集
        :param checkpoint_path: checkpoint 文件路径
        :param params_save_path: 参数保存路径
        :param valid_dataset: 验证数据集
        :param batch_size: 批次大小
        :param epochs: 目标总轮次（例如 checkpoint 的 epoch=3，epochs=10，则从 4 训到 10）
        :param lr: 学习率（当不恢复 optimizer 时生效）
        :param patience: 早停耐心值
        :param text_field: 文本字段
        :param label_fields: 标签字段
        :param loss_weights: 损失权重
        :param drop_last: 是否丢弃最后一个不完整批次
        :param load_optimizer: 是否恢复优化器状态
        :param strict: 是否严格匹配模型权重 key
        :param map_location: checkpoint 加载设备
        :return: 续训结果信息
        """
        return self.train(
            bert_incr_model=bert_incr_model,
            train_dataset=train_dataset,
            params_save_path=params_save_path,
            valid_dataset=valid_dataset,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            text_field=text_field,
            label_fields=label_fields,
            loss_weights=loss_weights,
            drop_last=drop_last,
            optimizer=None,
            checkpoint_path=checkpoint_path,
            load_optimizer=load_optimizer,
            strict=strict,
            map_location=map_location,
        )

    def load_checkpoint(self,
                        bert_incr_model: "BertIncrModel",
                        checkpoint_path: str,
                        *,
                        optimizer: torch.optim.Optimizer | None = None,
                        load_optimizer: bool = False,
                        strict: bool = True,
                        map_location: str | torch.device | None = None) -> dict[str, Any]:
        """
        加载 checkpoint（兼容完整 checkpoint 字典或纯 model state_dict）。
        :param bert_incr_model: BertIncrModel 实例
        :param checkpoint_path: checkpoint 文件路径
        :param optimizer: 可选优化器实例（用于续训时恢复优化器状态）
        :param load_optimizer: 是否尝试恢复优化器状态
        :param strict: 是否严格匹配模型权重 key
        :param map_location: torch.load 的 map_location，默认使用当前设备
        :return: 加载结果信息
        """
        if not bert_incr_model:
            raise ValueError("Model for loading checkpoint is required")

        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        if load_optimizer and optimizer is None:
            raise ValueError("optimizer is required when load_optimizer=True")

        effective_map_location = map_location or self.device
        checkpoint = torch.load(checkpoint_path, map_location=effective_map_location)

        checkpoint_type = "full_checkpoint" if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else "state_dict"
        model_state_dict = checkpoint["model_state_dict"] if checkpoint_type == "full_checkpoint" else checkpoint
        load_result = bert_incr_model.load_state_dict(model_state_dict, strict=strict)
        bert_incr_model.to(self.device)

        has_optimizer_state = isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint
        optimizer_loaded = False
        if load_optimizer:
            if has_optimizer_state:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                optimizer_loaded = True
            else:
                logger.warning(
                    f"[LOAD] checkpoint={checkpoint_path}, optimizer_state_dict=missing, optimizer_restored=False"
                )

        result = {
            "checkpoint_type": checkpoint_type,
            "path": checkpoint_path,
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "has_optimizer_state": has_optimizer_state,
            "optimizer_loaded": optimizer_loaded,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

        logger.info(
            f"[LOAD] path={checkpoint_path}, type={checkpoint_type}, strict={strict}, "
            f"epoch={result['epoch']}, optimizer_loaded={optimizer_loaded}, "
            f"missing_keys={len(result['missing_keys'])}, unexpected_keys={len(result['unexpected_keys'])}"
        )

        return result

    def _create_data_loader(self,
                            dataset: BertDataset,
                            *,
                            text_field: str = "text",
                            label_fields: list[str] = None,
                            return_raw: bool = False,
                            batch_size: int = 100,
                            shuffle: bool = True,
                            drop_last: bool = False) -> DataLoader | None:
        """
        创建数据加载器，返回 DataLoader 实例
        :param dataset: BertDataset 实例，提供数据集样本
        :param batch_size: 批次大小
        :param text_field: 用于 BERT 编码的文本字段名，默认为 "text"
        :param label_fields: 标签字段名称列表，默认为 None 不返回标签
        :param return_raw: 是否返回原始样本字典（调试用）
        :param shuffle: 是否打乱数据，训练集建议 True，验证集建议 False
        :param drop_last: 是否舍弃最后一个批次，防止形状出错
        :return: DataLoader 实例
        """
        if not dataset:
            return None

        # 封装字段配置，传递给 collate_fn
        collate_kwargs = {
            "text_field": text_field,
            "label_fields": label_fields or [],
            "return_raw": return_raw
        }

        # 偏函数传递配置（避免 lambda 导致多进程问题）
        def collate_fn_wrapper(batch):
            return self._collate_fn(batch, **collate_kwargs)

        return DataLoader(
            # 指定数据集
            dataset=dataset,
            # 批次越大，显存占用越大，训练速度越快
            batch_size=batch_size,
            # 打乱数据
            shuffle=shuffle,
            # 舍弃最后一个批次，防止形状出错
            # 比如：数据总共有 1000 条，批次大小为 100，那么最后一个批次就只有 100 条，形状就不会出错
            # 比如：数据总共有 1003 条，批次大小为 100，那么最后一个批次就只有 3 条，形状就会出错
            # 因为数据是被打乱了，训练轮数也不止一轮，所以舍弃的数据，一定有概率会被学到
            drop_last=drop_last,
            # 加载的数据进行编码
            collate_fn=collate_fn_wrapper
        ) if dataset else None

    def _collate_fn(
            self,
            batch: list[dict[str, Any]],
            text_field: str = "text",
            label_fields: list[str] = None,
            return_raw: bool = False
    ):
        """
        数据编码函数，将原始样本批次转换为模型输入格式
        :param batch: 批次样本（字典列表）
        :param text_field: 用于 BERT 编码的文本字段名，默认为 "text"
        :param label_fields: 标签字段名称列表，默认为 None 不返回标签
        :param return_raw: 是否返回原始样本字典（调试用）
        :return: 元组 (input_ids, attention_mask, token_type_ids, *labels, [raw_batch])
        """
        # 提取文本
        try:
            sentences = [sample[text_field] for sample in batch]
        except KeyError as e:
            raise KeyError(f"The text field '{text_field}' is missing from the sample. Please check the dataset format.") from e

        # 编码
        data = self.tokenizer(
            # 要编码的文本数据
            sentences,
            # 是否加入特殊字符
            add_special_tokens=True,
            # 表示编码后的最大长度，它的上限是 tokenizer_config.json 中的 model_max_length 的值
            max_length=512,
            # 是否切断文本，以适应文本最大的输入长度，即：长了就截断
            truncation=True,
            # 一律补 0 到 max_length，即：短了就补 0
            padding="max_length",
            # 编码后返回的类型
            # 可选：tf、pt、np，None
            # tf：返回 TensorFlow 的张量 Tensor
            # pt：返回 PyTorch 的张量 torch.Tensor
            # np：返回 Numpy 的数组 ndarray
            # None：返回 Python 的列表 list
            return_tensors="pt",
            # 返回 attentions_mask
            return_attention_mask=True,
            # 返回 token_type_ids
            return_token_type_ids=True,
            # 返回 special_tokens
            return_special_tokens_mask=True,
            # 返回编码后的序列长度
            return_length=True,
        )

        # 编码后的文本数据
        input_ids = data["input_ids"]
        # attention_mask：注意力掩码，标识哪些位置是有意义的，有意义的事 1，哪些位置是填充的，填充的是 0
        attention_mask = data["attention_mask"]
        # token_type_ids：第一个句子和特殊符号的位置是 0，第二个句子的位置是 1，只针对上下文的编码
        token_type_ids = data["token_type_ids"]

        # 3. 提取并转换标签（可选，支持多标签、多维度）
        labels = []
        for label_field in label_fields or []:
            try:
                # 提取该字段的所有值
                label_vals = [sample[label_field] for sample in batch]
                # 统一转换为张量（兼容标量、列表、数组）
                # 自动判断类型：整数标签用 LongTensor，浮点用 FloatTensor
                if isinstance(label_vals[0], (int, bool)):
                    label_tensor = torch.LongTensor(label_vals)
                elif isinstance(label_vals[0], float):
                    label_tensor = torch.FloatTensor(label_vals)
                elif isinstance(label_vals[0], (list, tuple)):
                    # 多维度标签：如 additional=[1, 0, 0]
                    label_tensor = torch.tensor(label_vals, dtype=torch.long if isinstance(label_vals[0][0], int) else torch.float)
                else:
                    raise TypeError(f"Unsupported label type: {type(label_vals[0])}")
                labels.append(label_tensor)
            except KeyError as e:
                raise KeyError(f"The label field '{label_field}' is missing from the sample. Please check the dataset format.") from e

        # 构造返回值（基础编码结果 + 所有标签 + 可选原始样本）
        result = [input_ids, attention_mask, token_type_ids] + labels
        # 调试时返回原始样本
        if return_raw:
            result.append(batch)

        return tuple(result)


class LabelBranchConfig(BaseModel):
    """
    单个标签分支的配置
    """
    type: Literal["single", "multi"] = Field(..., description="标签类型：single（单标签）/multi（多标签）")
    num_classes: int = Field(..., ge=1, description="类别数（单标签）或维度数（多标签），必须大于等于 1")

    @classmethod
    @field_validator("num_classes")
    def validate_num_classes(cls, v):
        """额外校验：确保类别数是正整数"""
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"num_classes 必须是 ≥1 的整数，当前值: {v}")
        return v


class LabelConfig(BaseModel):
    """
    整体标签配置
    """
    branches: dict[str, LabelBranchConfig] = Field(..., description="标签分支配置字典")

    @property
    def label_names(self) -> list[str]:
        """
        便捷属性：获取所有标签分支名称
        :return:
        """
        return list(self.branches.keys())

    def get_branch_config(self, label_name: str) -> LabelBranchConfig:
        """获取指定分支的配置"""
        if label_name not in self.branches:
            raise KeyError(f"标签分支 {label_name} 不存在，已配置的分支: {self.label_names}")
        return self.branches[label_name]


class BertIncrModel(torch.nn.Module):
    """
    定义下游任务（增量模型）
    """

    def __init__(self,
                 *,
                 bert_manager: BertManager,
                 in_features=768,
                 out_features_config: LabelConfig,
                 fc_layer_num: int = 1,
                 dropout_prob: float = 0.1,
                 hidden_dim: int = 128,
                 freeze_bert: bool = True):
        """
        初始化增量模型
        :param bert_manager: 预训练模型管理器实例，提供预训练模型和分词器
        :param in_features: 输入特征维度，默认为 768，对应 BERT 模型的隐藏层大小
        :param out_features_config: 输出特征配置，使用 LabelConfig 定义多个标签分支的类型和类别数
        :param fc_layer_num: 全连接层（Fully Connected Layer）的数量，默认为 1
        :param dropout_prob: 训练时随机丢弃的神经元比例，默认为 0.1，范围应在 [0, 1] 之间
        :param hidden_dim: 全连接层的隐藏层维度，默认为 128，可以根据需要调整
        :param freeze_bert: 是否冻结 BERT 模型的参数，默认为 True，冻结后 BERT 模型不参与训练，仅增量模型参与训练
        """
        super().__init__()
        self.bert_manager = bert_manager
        self.in_features = in_features
        self.out_features_config = out_features_config
        self.fc_layer_num = fc_layer_num
        self.dropout = torch.nn.Dropout(p=dropout_prob if 0 <= dropout_prob <= 1 else 0.1)
        # 限制最小隐藏层维度，避免过小导致拟合不足
        self.hidden_dim = max(hidden_dim, 16)
        self.device = bert_manager.device

        self.freeze_bert = freeze_bert
        # 可选：另一种冻结方式（更灵活，支持部分微调）
        if self.freeze_bert:
            for param in self.bert_manager.model.parameters():
                param.requires_grad = False

        # 动态构建各标签分支的全连接层
        self.fc_branches = torch.nn.ModuleDict()
        # 分层次设置维度（in → hidden → out）
        for label_name, branch_config in self.out_features_config.branches.items():
            fc_layers = torch.nn.ModuleList()

            # 计算隐藏层数量（至少 0 层，如果 fc_layer_num=1，则没有隐藏层，直接从输入到输出）
            num_hidden_layers = max(fc_layer_num - 1, 0)

            # 1、构建隐藏层（in → hidden → hidden -> ...）
            for i in range(num_hidden_layers):
                in_dim = in_features if i == 0 else hidden_dim
                fc_layers.append(torch.nn.Linear(in_dim, hidden_dim))
                # 激活函数
                fc_layers.append(torch.nn.ReLU())
                # 增加 dropout 防止过拟合
                fc_layers.append(self.dropout)

            # 2、构建输出层（hidden → num_classes）
            final_in_dim = in_features if num_hidden_layers == 0 else hidden_dim
            fc_layers.append(torch.nn.Linear(final_in_dim, branch_config.num_classes))

            # 将该分支的层列表封装为 Sequential 模块，并添加到 ModuleDict 中
            self.fc_branches[label_name] = fc_layers

        # 模型移至设备
        self.to(self.device)

    # 使用模型处理数据，执行前向计算
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> list[torch.Tensor]:
        # 根据 freeze_bert 控制是否冻结 BERT 梯度
        context = torch.no_grad() if self.freeze_bert else torch.enable_grad()
        with context:
            # 目前为止 Transformer 模型都是沿用了 RNN 数据的模式，数据是 NSV 模式，N 表示批次，S 表示序列长度，V 表示数据特征
            out = self.bert_manager.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # 增量模型参与训练，取 NSV 的 V，即数据特征
        # out.last_hidden_state[:, 0]：提取每个批次中第一个 token（[CLS] 标记）的隐藏状态，它通常用于表示整个句子的语义
        cls_feature = out.last_hidden_state[:, 0]
        cls_feature = self.dropout(cls_feature)

        # 逐分支计算 logits
        logits_list = []
        for label_name in self.out_features_config.label_names:
            fc_layers = self.fc_branches[label_name]
            branch_out = cls_feature
            for layer in fc_layers.children():
                branch_out = layer(branch_out)
            logits_list.append(branch_out)

        return logits_list


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
        epochs=1,
        text_field="text",
        label_fields=["primary", "additional"],
        loss_weights=[0.7, 0.3],
    )
