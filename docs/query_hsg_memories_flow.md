# `query_hsg_memories` 查询流程详解

本文基于 `src/services/memory/hsg.py` 中的 `query_hsg_memories` 实现，按“输入 -> 召回 -> 排序 -> 后处理 -> 返回”的顺序解释完整检索路径。

---

## 1. 函数定位与目标

函数签名：

```python
async def query_hsg_memories(query: str, top_k: int = 10, filters: IMemoryFilters | None = None) -> IMemorySearchResult
```

目标：对用户输入 `query` 执行多路召回与混合打分，返回 `IMemorySearchResult`，其中包含：

- 记忆列表 `memories`
- 可选用户画像 `user_profile`

它不是单一“向量搜索”，而是一个**融合检索管线**：

- BM25 关键词召回
- 多扇区向量召回
- 会话摘要召回
- Waypoint 图结构扩展
- 关系图（topic/fact/entity）扩展
- 多因素混合评分与 QA 配对提升

---

## 2. 入参与配置预处理

### 2.1 基础初始化

函数开头会：

1. 记录起始时间 `start_q`
2. `decay.inc_q()` 增加并发查询计数
3. 规范化 `filters` 与 `filters.config`

### 2.2 Graph 策略模式切换

`filters.config.graph.type` 支持快捷策略：

- `recall` -> `IMemoryGraphConfig.recall_first()`（偏召回）
- `precision` -> `IMemoryGraphConfig.precision_first()`（偏精度）

### 2.3 用户合法性与用户加载

- `user_identity.check_legality()` 做身份合法性校验
- `user_repo.get_user(..., using_cache=True)` 获取用户
- 用户不存在则抛 `UserNotFoundError`

---

## 3. 一级短路：查询缓存

在真正检索前，先检查 `MEMORIES_CACHE`：

- key: `f"{query}:{top_k}:{filters.model_dump_json()}"`
- value 若是 `IMemorySearchResult`，直接返回
- 若缓存类型异常（不是结果对象），打 warning 并忽略

这一步避免重复查询的全链路开销。

---

## 4. 查询理解与检索准备

### 4.1 查询分类

调用 `sector_classifier.classify(content=query)` 得到：

- 主扇区 `query_classify.primary`
- 可用于扇区权重动态调整

### 4.2 查询 token 化

`canonical_token_set(query)` 得到规范化 token 集，用于关键词/标签重叠评分。

### 4.3 检索扇区确定

- 优先使用 `filters.sectors`
- 否则默认使用全部 `SECTOR_CONFIGS.keys()`
- 若仍为空，兜底 `['semantic']`

### 4.4 多扇区查询向量

`embed_query_for_all_sectors(query, sectors)` 批量生成每个 sector 的查询向量。

### 4.5 动态扇区权重

根据查询主扇区计算：

```python
dynamic_sector_weights = get_dynamic_sector_weights(primary_sector=query_classify.primary)
```

该权重后续用于多向量融合分数计算。

---

## 5. 多路召回阶段（候选集构建）

函数核心是把多个来源的候选 ID 合并到 `ids` 集合。

### 5.1 路径 A：BM25 召回（可选）

触发条件：`filters.config.bm25_enable == True`

流程：

1. 拉取用户最近记忆（最多 2000 条）
2. `bm25_searcher.search(query, docs=..., top_k=top_k * 3)`
3. 得到 `bm25_ids`

> 作用：补足“语义向量召回漏掉但词面命中很强”的候选。

### 5.2 路径 B：向量召回（主干）

对每个 sector 执行：

1. 取该 sector 查询向量
2. `vector_store.search(user, query_vector, sector, top_k * 3)`
3. 收集 `vector_search_ids` 与 `sector_result[sector]`

之后把所有 vector 结果 ID 加入总候选 `ids`。

### 5.3 自适应规模 `effective_k`

计算向量结果平均相似度 `avg_sim`：

- `adapt_exp = ceil(0.3 * top_k * (1 - avg_sim))`
- `effective_k = top_k + adapt_exp`

含义：

- `avg_sim` 低 -> 扩展量变大 -> 拉高召回广度
- `avg_sim` 高 -> 扩展量变小 -> 保持精度

### 5.4 路径 C：会话摘要召回（L2）

调用 `session_ops.session_search(user, query, top_k=effective_k)`，取所有 `dialogue_ids`，并入 `ids`。

### 5.5 路径 D：Waypoint 扩展召回（按需）

触发条件：`avg_sim < 0.55`

执行 `waypoints.expand_via_waypoints(list(ids), effective_k * 2)`，将扩展出的 `waypoint_expanded_ids` 加入候选。

> 这是“低置信时补召回”的结构化扩展机制。

### 5.6 路径 E：图扩展召回（可选）

触发条件：`graph_cfg.enable == True`

通过 `graph_search.expand_candidate_ids_via_graph(...)` 基于 topic/fact/entity 图关系扩展候选，并记录：

- `graph_expanded_ids`
- `graph_candidate_scores`（后续用于 bonus）

---

## 6. 候选打分阶段（精排）

### 6.1 候选详情加载

`mem_ops.find_mem_by_ids(list(ids))` 拉取候选记忆实体。

### 6.2 来源标记与关键词分

先构建：

- `kw_scores[mem_id] = compute_keyword_overlap(...) * 0.15`
- `mem_from[mem_id]` 标记首个命中来源（bm25/vector/session/waypoint/graph）

### 6.3 单条记忆评分流程

每条候选 `mem` 依次处理：

1. **过滤**
   - `filters.min_salience` 不满足则跳过
   - `mem.user_id != user.id` 跳过（强隔离）

2. **语义相关性核心分**
   - `mvf = calc_multi_vec_fusion_score(...)`：多扇区向量加权余弦
   - `csr = calc_cross_sector_resonance_score(...)`：跨扇区共振分
   - `best_sim = max(csr, 各 sector 向量命中相似度)`

3. **扇区惩罚**
   - 同扇区无惩罚
   - 异扇区按 `SECTOR_RELATIONSHIPS[query_sec][mem_sec]` 降权
   - 得到 `sim_adjust = best_sim * penalty`

4. **结构与时效特征**
   - waypoint 权重 `waypoint_weight`
   - 记忆衰减显著性 `salience = decay.calc_decay(...)`
   - token 重叠 `token_overlap`
   - 时效分 `rec_sc = decay.calc_recency_score_decay(...)`
   - 标签匹配 `tag_match_score`
   - 图扩展加成 `graph_bonus = min(0.12, graph_score * 0.12)`

5. **混合得分融合**

   ```python
   final_score = compute_hybrid_score(
       sim=sim_adjust,
       tok_ov=token_overlap,
       wp_wt=waypoint_weight,
       rec_sc=rec_sc,
       kw_score=kw_scores.get(mid, 0),
       tag_match=tag_match_score,
   )
   final_score = clamp(final_score + graph_bonus, 0.0, 1.0)
   ```

6. **结果封装**
   - 构建 `IMemoryItemInfo`
   - metadata 注入 `from/graph_score/graph_bonus/type`
   - `filters.config.debug` 时附加 `IMemoryItemDebugInfo`

---

## 7. QA 模式后处理

当 `filters.query_mode` 为 `qa` 或 `prefer` 时，执行 `_promote_qa_assistant_answer`：

1. 找分数最高的 `human` 记忆
2. 用 `qa_pair_id` 查最新 `assistant` 回答
3. 若 assistant 已在结果中：提升分数到至少 `best_human.score + 0.2`
4. 若不在：动态插入结果集

用途：在问答场景下，把“回答”提到前面，降低只返回问题本身的概率。

---

## 8. 排序、强化与结果组装

### 8.1 排序与截断

- 对 `res_list` 按 `score` 降序
- 取前 `effective_k` 作为 `effective_k_list`

### 8.2 异步强化

后台任务：`asyncio.create_task(reinforce_memories(effective_k_list))`

会对命中记忆及关联节点进行强化（显著性更新、last_seen 更新、动态衰减模块 hit 记录）。

### 8.3 用户画像

若 `filters.config.user_profile_enable`：

- 调 `user_profile_ops.get_user_profile(user, query_cache=True)`

### 8.4 Session 级结果融合与去重

调用 `extract_sessions_if_necessary(...)`：

- 可将 session 摘要封装为 `session:*` 类型记忆
- 可按配置移除被 session 覆盖的原始对话记忆（去重）

### 8.5 缓存并返回

- 构建 `IMemorySearchResult(user_profile=..., memories=...)`
- 写入 `MEMORIES_CACHE`
- 返回结果

`finally` 中无论成功与否都会执行 `decay.dec_q()`，保证查询计数平衡。

---

## 9. 流程图（文字版）

```text
query_hsg_memories
  -> 参数/用户校验
  -> 缓存命中? 是: 直接返回
  -> 查询分类 + token + 多扇区向量 + 动态权重
  -> 多路召回(BM25/向量/session/waypoint/graph)合并ID
  -> 加载候选记忆
  -> 单条混合打分(语义+关键词+时效+标签+结构+图加成)
  -> QA模式提升(可选)
  -> 排序并截断(effective_k)
  -> 异步强化命中记忆
  -> 融合用户画像/会话摘要(可选)
  -> 写缓存并返回
```

---

## 10. 关键设计特点

- **召回层面**：多路并行思想，避免“单检索器偏见”。
- **排序层面**：融合语义、时效、关键词、图结构、标签等多信号。
- **自适应层面**：用 `avg_sim` 自动调节召回规模与扩展策略。
- **在线学习层面**：命中后异步强化，使系统具有“用后即学”的动态性。
- **工程层面**：查询缓存 + 背景任务，兼顾时延和效果。

---

## 11. 可关注的调优点（实践建议）

1. `avg_sim < 0.55` 的 waypoint 触发阈值可按业务集群分布调整。
2. `compute_hybrid_score` 各权重应结合离线评测集做网格搜索。
3. `graph_bonus` 当前上限 `0.12`，可按误召回率观察后再放宽。
4. `effective_k` 的扩展公式可加上下界/上界，减少极端 query 波动。
5. 若高并发下命中强化任务积压，可为 `reinforce_memories` 增加队列或限流。

