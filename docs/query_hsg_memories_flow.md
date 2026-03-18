# query_hsg_memories 流程说明

本文说明 `src/memory/hsg.py` 中 `query_hsg_memories(...)` 的执行链路，并解释每一步的作用。

## 1. 方法定位

`query_hsg_memories` 是 HSG 检索主入口，负责：
- 查询缓存命中与短路返回
- 查询文本分类与多扇区向量召回
- 低置信场景下的图扩展补召回
- 多信号混合评分与排序
- 命中后强化与衰减联动

---

## 2. 输入与输出

### 输入参数
- `query: str`：用户查询文本
- `top_k: int = 10`：期望返回条数
- `filters: IMemoryFilters = None`：过滤条件、检索模式、调试开关等

### 输出结果
- `List[IMemoryItemInfo]`：按最终得分排序后的记忆列表（最多 `effective_k`）

---

## 3. 主流程（按执行顺序）

### Step 1: 初始化与并发计数
- 操作：记录起始时间、`decay.inc_q()` 增加查询计数、补齐默认 `filters`
- 作用：
  - 用于统计检索耗时
  - 为衰减/强化策略提供并发状态
  - 保证后续逻辑有稳定默认配置

### Step 2: 构建缓存键并尝试命中
- 操作：基于 `query + top_k + filters.model_dump_json()` 构建 `cache_key`，查询 `MEMORIES_CACHE`
- 作用：
  - 命中时直接返回，降低延迟与数据库/向量库压力
  - 适合短时间内重复查询场景

### Step 3: 查询分类与检索扇区确定
- 操作：
  - `get_sector_classifier().classify(content=query)` 获取主扇区
  - 提取 `query_tokens`
  - 确定检索扇区集合（`filters.sectors` 或默认全扇区）
- 作用：
  - 给检索提供语义方向
  - 为后续动态权重与扇区惩罚提供上下文

### Step 4: 生成各扇区查询向量
- 操作：调用 `embed_query_for_all_sectors(query, sectors)`
- 作用：
  - 统一构建召回向量输入
  - 当前实现中 `sector` 不参与 embedding 计算，因此复用同一向量映射到所有扇区

### Step 5: 动态权重设定
- 操作：根据查询主扇区构建 `weight`（语义/情感/程序/时序/反思维度权重）
- 作用：
  - 强化与查询类型更相关的语义维度
  - 降低不相关维度的噪声影响

### Step 6: 多扇区向量召回
- 操作：对每个扇区执行 `vector_store.search(query_vector, sector, top_k * 3, filters)`
- 作用：
  - 先做宽召回（每扇区 `top_k * 3`）为后续重排保留候选余量
  - 获得候选 ID 与基础相似度分布

### Step 7: 评估召回置信度并自适应扩展
- 操作：
  - 统计所有候选相似度均值 `avg_sim`
  - 计算 `adapt_exp = ceil(0.3 * top_k * (1 - avg_sim))`
  - 得到 `effective_k = top_k + adapt_exp`
- 作用：
  - 相似度低时自动增加候选规模，减少漏召回
  - 相似度高时保持检索效率

### Step 8: 低置信触发 Waypoint 图扩展
- 条件：`avg_sim < 0.55`
- 操作：`waypoints.expand_via_waypoints(list(ids), effective_k * 2)`，并把扩展节点加入候选集
- 作用：
  - 用图关系补足“语义向量召回不到但关联强”的记忆
  - 提高复杂查询和跨语义联想场景的召回率

### Step 9: 拉取候选记忆内容
- 操作：`dml_ops.find_mem(ids)` 获取候选详情
- 作用：为后续过滤、打分、结果构建提供结构化数据

### Step 10: 预计算关键词重叠分
- 操作：对候选执行 `compute_keyword_overlap(query_tokens, mem["content"])`，存入 `kw_scores`
- 作用：
  - 作为混合评分的一个轻量语义信号
  - 先算一遍避免重复计算

### Step 11: 候选过滤与单条综合评分
- 操作（逐条记忆）：
  - 基础过滤：`min_salience`、用户身份（`user/tenant/project`）
  - 多向量融合：`calc_multi_vec_fusion_score(...)`
  - 跨扇区共振：`calc_cross_sector_resonance_score(...)`
  - 最佳相似度回填：从召回列表中取该 ID 的最大相似度
  - 扇区惩罚：`SECTOR_RELATIONSHIPS` 不匹配时降权
  - 时间衰减与时效分：`decay.calc_decay`、`decay.calc_recency_score_decay`
  - 标签与 token 匹配：`compute_token_overlap` + `compute_tag_match_score`
  - 最终融合：`compute_hybrid_score(...)`
- 作用：把“向量相似、关键词、图关系、时效、标签”等多维信号融合为可排序分数

### Step 12: 组装结果项
- 操作：构建 `IMemoryItemInfo`，`filters.debug=True` 时补充 `IMemoryItemDebugInfo`
- 作用：
  - 形成统一返回结构
  - 支持线上调参与问题排查

### Step 13: QA 模式结果提升
- 条件：`filters.query_mode in ("qa", "prefer")`
- 操作：调用 `_promote_qa_assistant_answer(res_list, query_classify.primary)`
- 作用：
  - 若命中高分 `human` 记忆，优先提升其配对 `assistant` 回答
  - 让问答场景输出更贴近“问题-答案”形式

### Step 14: 排序与截断
- 操作：按 `score` 降序，取 `res_list[:effective_k]`
- 作用：输出动态规模下的最优候选

### Step 15: 命中后强化与关联传播
- 操作：对每个命中项执行：
  - `apply_retrieval_trace_reinforcement_to_memory` 提升自身显著性
  - 更新 `last_seen_at`
  - 若存在路径，查询 waypoint 邻接并执行 `propagate_associative_reinforcement_to_linked_nodes`
  - 对关联节点按时间衰减因子计算增益并更新显著性
  - 调用 `decay.on_query_hit` 记录命中事件
- 作用：
  - 建立“用得越多越容易再被召回”的记忆自增强闭环
  - 通过图传播强化相关记忆网络

### Step 16: 查询结果缓存并返回
- 操作：`MEMORIES_CACHE.set(cache_key, effective_k_list)`，记录日志并返回
- 作用：
  - 降低重复查询成本
  - 完成本次检索闭环

### Step 17: finally 收尾
- 操作：`decay.dec_q()`
- 作用：无论成功或失败都正确回收查询计数，避免并发状态漂移

---

## 4. 关键机制速览

- **缓存机制**：60s 内同请求参数可直接命中，优先提升响应速度。
- **双阶段召回**：先向量召回，再在低置信时引入图扩展补召回。
- **混合重排**：融合相似度、关键词、标签、时效、图权重，避免单一向量分主导。
- **检索反哺写侧**：命中后会强化显著性并传播到关联节点，系统具备自演化能力。

---

## 5. 常见关注点

- `effective_k` 可能大于 `top_k`，这是低置信场景下的自适应扩展策略。
- 缓存键包含 `filters` 序列化结果，过滤条件变化会导致缓存失效。
- QA 提升逻辑依赖 `qa_pair_id`，写入端需保证配对字段完整。
- 用户身份过滤是强约束，不匹配的候选会在重排阶段直接剔除。

