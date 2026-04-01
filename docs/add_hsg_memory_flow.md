# add_hsg_memory 流程说明

本文说明 `src/memory/hsg.py` 中 `add_hsg_memory(...)` 的执行流程，并解释每一步的作用。

## 1. 方法定位

`add_hsg_memory` 是 HSG 写入主入口，负责：
- 记忆去重（高相似时不重复写入）
- 新记忆写入（结构化字段 + 摘要 + 扇区）
- 向量生成与落库（多扇区向量 + 均值向量 + 可选压缩向量）
- 图关系构建（Waypoint）
- 用户摘要联动更新

---

## 2. 输入与输出

### 输入参数
- `content: str`：原始记忆文本
- `tags: List[str] | None`：标签
- `metadata: Any`：扩展元数据（函数内会补充统计字段）
- `user_identity: IMemoryUserIdentity`：用户身份（`user_key/tenant_key/project_key`）
- `qa_role: QARole | None`：可选 `human` / `assistant`

### 返回结果（两类）
1. **去重命中**：返回已有记忆 `id` 等信息，`deduplicated=True`
2. **新建成功**：返回新记忆 `id/content/sector/chunks/salience/qa 信息`

---

## 3. 主流程（按执行顺序）

### Step 1: 入参校验（QA 角色）
- 操作：校验 `qa_role` 只能是 `human` 或 `assistant`
- 作用：避免非法角色进入后续配对和写库逻辑

### Step 2: 自动补齐 QA 配对标识
- 操作：调用 `_resolve_auto_qa_linking(user_identity, qa_role)`
- 作用：
  - `human`：生成新的 `qa_pair_id`
  - `assistant`：尝试复用最近一条未配对 `human` 的 `qa_pair_id`
  - 非 QA 场景：返回 `None`

### Step 3: 生成当前内容向量
- 操作：获取嵌入模型并执行 `embed_model.embed(content)`
- 作用：后续用于与历史记忆做相似度比对（去重判断）

### Step 4: 查询同用户历史记忆（带租户/项目过滤）
- 操作：动态拼 SQL（`user_key` 必选，`tenant_key/project_key` 可选），并关联 `vectors` 表
- 作用：将候选范围限制在同身份域，避免跨用户误判相似

### Step 5: 计算最高相似候选
- 操作：遍历候选向量，逐条计算 `similarity`，保留最优项
- 作用：找到“最可能重复”的历史记忆

### Step 6: 去重短路分支（高相似）
- 条件：`best_similarity >= env.SIMILARITY_THRESHOLD`
- 操作：
  - 不新建记忆
  - 提升已有记忆 `salience`（封顶 1.0）
  - 更新 `last_seen_at/updated_at`
- 作用：减少重复写入，保留记忆热度与时效

### Step 7: 用户存在性保障
- 操作：`get_user` 查询用户，不存在则 `add_user`
- 作用：保证记忆挂载主体存在，避免后续关联数据缺失

### Step 8: 内容切分与元数据补全
- 操作：`chunk_text(content)` 得到 `chunks` 和 `total_token`，并补充 `metadata` 统计字段
- 作用：
  - 判断是否分块写向量（长文本）
  - 在元数据中保存文本规模信息（字符数/估算 token）

### Step 9: 扇区分类 + 分段轮转
- 操作：
  - 调用分类器得到 `primary + additional sectors`
  - 查询当前 `segment` 容量，超阈值则切换到新 segment
- 作用：
  - 给记忆分配语义组织维度（HSG 的分层基础）
  - 控制单 segment 规模，避免过度集中

### Step 10: 摘要提取与主记录写入
- 操作：
  - `ExtractEssence(...).extract()` 生成摘要
  - 计算初始显著性 `init_sal`，计算规则：基础分为 0.4，每多一个辅扇区，显著性加 0.1，最终显著性限定在 0.0~1.0 之间
  - 调用 `mem_ops.ins_mem(...)` 写入 `memories`
- 作用：
  - 用摘要替代长原文，提高存储与后续检索效率
  - 记录完整业务字段（扇区、标签、meta、QA 字段等）

### Step 11: 多扇区嵌入并写入向量存储
- 操作：
  - `embed_multi_sector(...)` 生成各扇区向量
  - 遍历执行 `vector_store.store_vector(...)`
- 作用：支持多维召回，不把记忆语义压缩为单一向量

### Step 12: 计算并保存均值向量（可选压缩）
- 操作：
  - 计算 `mean_vec` 并写回 `mean_dim/mean_vec`
  - 维度大于 128 时，执行压缩并写 `compressed_vec`
- 作用：
  - 提供统一全局表征，便于后续融合计算
  - 降低高维存储成本

### Step 13: 建立图关系与用户摘要更新
- 操作：
  - `waypoints.create_single_waypoint(...)`
  - 有 `user_id` 时触发 `update_user_summary(...)`
- 作用：
  - 把新记忆接入记忆图，支持后续图扩展检索
  - 维持用户长期画像/摘要的新鲜度

### Step 14: 返回结果
- 操作：返回新记忆核心字段（含 `qa_role/qa_pair_id`）
- 作用：让上层调用方可直接拿到可展示、可追踪的写入结果

---

## 4. 分支总览

- **分支 A（去重命中）**：更新旧记录热度并直接返回
- **分支 B（新建记录）**：执行分类、入库、向量化、建图、摘要更新后返回

---

## 5. 关键设计意图

- **防重复**：通过相似度阈值避免近重复记忆膨胀
- **多扇区组织**：同一记忆可投影到多个语义扇区，增强召回弹性
- **图增强能力**：写入阶段即建立 waypoint，为查询阶段图扩展做准备
- **可演化性**：显著性、摘要、用户总结会随使用持续更新

---

## 6. 注意点（阅读代码时建议关注）

- `metadata` 在函数中会被直接 `update`，调用方应确保其为可变字典对象
- 去重阈值以 `env.SIMILARITY_THRESHOLD` 为准，不要仅依赖注释文字
- `except Exception as e: raise e` 为透传异常，问题定位依赖上层日志与调用链

