# SyntheticAdapter vs Sector-Aware Model 评测说明

本评测用于快速比较两种向量方案在同一批检索样本上的效果：

- `synthetic`：`SyntheticAdapter._gen_syn_emb`
- `sector_model`：当前 `src/memory/embed.py` 的 sector-aware 文本注入方案

评测脚本：`src/ops/eval_embedding_retrieval.py`

## 指标

- `Recall@1 / @3 / @5 / @10`
- `MRR`
- `latency_ms`（单次评测总耗时，仅用于粗略对比）

## 快速运行

```zsh
cd /Users/wangweijun/PycharmProjects/i-memory
python -u -m src.ops.eval_embedding_retrieval --backends synthetic sector_model --output logs/eval/synthetic_vs_sector.json
```

同时输出 Markdown 报告：

```zsh
cd /Users/wangweijun/PycharmProjects/i-memory
python -u -m src.ops.eval_embedding_retrieval --backends synthetic sector_model --output logs/eval/synthetic_vs_sector.json --output-md logs/eval/synthetic_vs_sector.md
```

仅评测 `SyntheticAdapter`：

```zsh
cd /Users/wangweijun/PycharmProjects/i-memory
python -u -m src.ops.eval_embedding_retrieval --backends synthetic --output logs/eval/synthetic_only.json
```

## 使用外部业务样本

脚本支持通过 `--dataset` 读取 JSON 数据集（示例：`assets/eval/min_business_eval.json`）。

如果你希望直接从数据库 `memories` 表生成数据集，可先执行：

```zsh
cd /Users/wangweijun/PycharmProjects/i-memory
python -u -m src.ops.export_eval_dataset --mode answer_only --output assets/eval/dataset_answer_only.json
```

说明：导出器会使用 `qa_pair_id` 将 `qa_role='human'` 作为 query，`qa_role='assistant'` 作为 `gold_ids`。

```zsh
cd /Users/wangweijun/PycharmProjects/i-memory
python -u -m src.ops.eval_embedding_retrieval \
  --backends synthetic sector_model \
  --dataset assets/eval/dataset_answer_only.json \
  --k-values 1,3,5,10 \
  --top-k 10 \
  --output logs/eval/business_eval_answer_only.json \
  --output-md logs/eval/business_eval_answer_only.md
```

`export_eval_dataset` 支持两种模式：

- `answer_only`（默认）：候选记忆仅保留 assistant，更适合问答检索评测。
- `legacy_all`：候选记忆保留全量 memories（历史兼容口径）。

### 数据集格式

```json
{
  "memories": [
	{"id": "m1", "text": "...", "sector": "emotional"}
  ],
  "queries": [
	{"id": "q1", "text": "...", "sector": "emotional", "gold_ids": ["m1"]}
  ]
}
```

要求：

- `queries[].gold_ids` 必须都能在 `memories[].id` 中找到。
- `sector` 建议使用项目已定义扇区：`semantic/procedural/episodic/emotional/reflective`。

## 结果解读建议

- 先看 `Recall@1` 与 `MRR`：反映头部命中质量。
- 再看 `Recall@5/@10`：反映召回覆盖能力。
- 若 `synthetic` 在头部指标更高，说明规则化扇区分化对当前样本更有效。
- 若 `sector_model` 更高，说明预训练语义空间对该任务更有优势。

## 注意

- 当前脚本使用内置小样本（`DEFAULT_MEMORIES` / `DEFAULT_QUERIES`），主要用于快速对比与回归。
- 若要做正式结论，建议替换成业务真实 query + gold 相关记忆集。

