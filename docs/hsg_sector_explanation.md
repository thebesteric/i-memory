# HSG 记忆分扇区详解（为什么要分？怎么发挥作用？）

> 面向 i-memory 项目，结合 `src/memory/hsg.py` 主流程说明。

## 一句话结论

**记忆分扇区（sector）的核心价值是：把“同一条内容的不同语义维度”拆开建索引与打分，减少语义串扰，让<font color="red">写入更结构化</font>、<font color="red">检索更精准</font>、<font color="red">跨主题联想更可控</font>、<font color="red">遗忘与强化更符合记忆类型</font>。**

---

## 1. 总体目标：解决“单向量记忆”的三类痛点

如果系统只为每条记忆保存一个向量，常见问题是：

1. **写入混杂**：一段话里可能同时包含事实、情绪、步骤，单向量难以表达层次。  
2. **检索跑偏**：文本表面相似但意图不同时，容易召回“看起来像、其实不对路”的结果。  
3. **演化粗糙**：不同类型记忆（事件/情绪/流程）本应有不同衰减节奏，却被同一规则处理。

HSG 的分扇区机制就是为这三点服务。

---

## 2. 写入时更“有结构”

对应主入口：`add_hsg_memory(...)`（`src/memory/hsg.py`）

### 2.1 写入流程里扇区怎么参与

在 `add_hsg_memory` 里，关键步骤是：

1. `sector_classifier.classify(content=content, metadata=metadata)` 先得到：
   - `primary`（主扇区）
   - `additional`（辅扇区列表）
2. `all_secs = [primary] + additional` 形成该记忆的扇区视图。
3. `mem_ops.ins_mem(...)` 把 `primary_sector` 和 `sectors` 持久化到数据库。
4. `embed_multi_sector(mid, content, all_secs, chunks...)` 为每个扇区生成向量。
5. `vector_store.store_vector(mid, sector, vector, ...)` 分扇区存入向量库。
6. `calc_mean_vec(...)` 生成均值向量，补充全局表示（并可压缩存储）。

这意味着：**一条记忆不再只有一个“扁平向量”，而是“主扇区 + 多扇区向量 + 全局均值向量”的组合表达。**

### 2.2 直观例子：为什么叫“更有结构”

输入内容：

> “今天跟老板复盘项目后有点焦虑（情绪），决定先把接口重构分三步做（流程），并在周五前给出结果（事件）。”

无扇区时：
- 只有一个向量，后续查询“焦虑原因”或“重构步骤”都可能被同一向量牵着走。

有扇区时：
- `primary` 可能是 `procedural`，`additional` 包含 `emotional`、`episodic`；
- 向量库中会有多条 `mid` 相同但 `sector` 不同的向量记录；
- 后续检索“我最近焦虑什么”会优先匹配 emotional 维度，“接口怎么做”会优先 procedural 维度。

---

## 3. 检索时更“对路”

对应主入口：`query_hsg_memories(...)`（`src/memory/hsg.py`）

### 3.1 检索流程里扇区怎么参与

`query_hsg_memories` 中，扇区参与点非常多：

1. **先分查询意图扇区**：`sector_classifier.classify(content=query)`。
2. **多扇区向量检索**：对每个 sector 分别 `vector_store.search(...)`。
3. **按查询类型动态调维度权重**：
   - `semantic_dimension_weight`
   - `emotional_dimension_weight`
   - `procedural_dimension_weight`
   - `episodic_dimension_weight`
   - `reflective_dimension_weight`
4. **多向量融合**：`calc_multi_vec_fusion_score(...)`。
5. **扇区不匹配惩罚**：`SECTOR_RELATIONSHIPS`（`src/core/constants.py`）控制跨扇区加减权。
6. **最终混合评分**：结合相似度、token overlap、waypoint、时效、标签等。

### 3.2 直观例子：为什么叫“更对路”

查询：

> “我最近为什么总是焦虑？”

系统行为（理想路径）：

1. 查询被分类为 `emotional`。
2. emotional 维度权重提高（代码中 `1.5`），其他维度相对降低。
3. 候选里即使有一条“Redis 配置步骤”文本相似度不错，也会因扇区关系和惩罚不占优。
4. 最终更可能返回“复盘后压力大”“担心截止日期”等情绪相关记忆。

效果：**不只是“字面像”，而是“问题类型也匹配”。**

---

## 4. 跨扇区有“关系感”

对应逻辑：
- `SECTOR_RELATIONSHIPS`（扇区关系矩阵）
- `calc_cross_sector_resonance_score(...)`（`src/ops/dynamic_memory.py`）

### 4.1 不是“硬隔离”，而是“有关系的软连接”

分扇区并不代表互相看不见。
系统会根据扇区关系做“软连接”：

- 关系近：降低惩罚或给共振加分；
- 关系远：保留一定召回机会，但降低排序优先级。

这能避免两种极端：

1. 完全不跨扇区（漏掉有价值的联想）；
2. 完全无边界（跨域噪声过大）。

### 4.2 直观例子：为什么叫“有关系感”

查询：

> “我为什么拖延这次重构？”

候选记忆可能有：

- A：`procedural`（重构步骤卡在第二步）
- B：`emotional`（担心改坏线上，压力大）
- C：`semantic`（某技术名词解释）

有扇区关系时：
- A、B 可能互相支撑（流程问题 + 情绪阻碍）；
- C 可能被保留但排序靠后。

结果：回答更像真实原因链，而不是只给“术语说明”。

---

## 5. 衰减与强化：按扇区“区别对待”

关键逻辑：
- 写入时保存 `decay_lambda`（来自 `SECTOR_CONFIGS[primary]`）
- 查询时 `decay.calc_decay(primary_sector, salience, days)`
- 命中后强化：
  - `apply_retrieval_trace_reinforcement_to_memory(...)`
  - `propagate_associative_reinforcement_to_linked_nodes(...)`
  - `decay.on_query_hit(...)`

### 5.1 为什么要按扇区差异化衰减

不同记忆天生寿命不同：

- `episodic`（事件）可能时效性更强；
- `procedural`（流程）常常寿命更长；
- `emotional`（情绪）可能受近期触发影响更明显。

按扇区设置 `decay_lambda`，能让“该快忘的快忘、该长期保留的慢忘”。

### 5.2 直观例子

- 一条“今天午饭吃了什么”（事件）几周后价值很低；
- 一条“线上故障排查 SOP”（流程）半年后依然常用；

如果统一衰减，两者会被同样速度削弱，不合理。分扇区后可分别调。

---

## 6. 调参建议（从保守到激进）

### 6.1 第一优先级：先看召回是否“偏题”

重点参数/位置：

- `SECTOR_RELATIONSHIPS`（`src/core/constants.py`）
- `calc_multi_vec_fusion_score` 里的维度权重
- `compute_hybrid_score` 的综合权重

建议：

1. 若“总偏技术，不懂情绪诉求”：提高 emotional 相关权重，或降低 emotional->semantic 的关系强度惩罚。  
2. 若“总是过度情绪化，缺事实步骤”：提高 procedural/semantic 权重。  
3. 若“跨域噪声多”：收紧 `SECTOR_RELATIONSHIPS` 中弱相关扇区的得分。

### 6.2 第二优先级：再看时效性是否合理

- 按扇区调 `decay_lambda`；
- 观察常用记忆是否被过快衰减；
- 命中强化后是否出现“马太效应”（总是同一批记忆霸榜）。

---

## 7. 常见误区

1. **误区：分扇区就是“多加几个标签”**  
   事实：这里不是静态标签，而是“多向量索引 + 关系惩罚 + 动态融合评分”。

2. **误区：扇区越多越好**  
   事实：扇区过细会导致样本稀疏、维护复杂、调参成本飙升。

3. **误区：扇区能完全替代关键词/时效/图关系**  
   事实：本实现是混合评分体系，扇区只是一条关键轴，不是唯一轴。

4. **误区：跨扇区召回都是噪声**  
   事实：适度跨扇区可提高解释完整性，关键是关系矩阵与惩罚力度。

---

## 8. FAQ

### Q1：查询向量是按 sector 分别计算的吗？
A：是。`embed_query_for_all_sectors` 会并发调用 `embed(query, sector)`，为每个扇区生成独立查询向量；`embed` 内部会把 sector 信息注入文本后再进行向量化。因此各扇区向量不再是同一份副本。

### Q2：既然已有 `primary_sector`，为什么还要 `additional sectors`？
A：现实内容往往多维。`primary` 用于主导衰减与主语义归类，`additional` 保留辅语义入口，防止检索信息损失。

### Q3：为什么还要 `mean_vec` / `compressed_vec`？
A：用于提供统一全局向量表示与存储优化，既支撑关联构建（如 waypoint），又控制存储与计算成本。

### Q4：扇区会不会导致查全率下降？
A：如果关系矩阵过于严格，确实可能。当前实现通过多扇区召回 + 跨扇区关系 + waypoint 扩展来平衡查准率与查全率。

---

## 9. 最小心智模型（便于和团队沟通）

可以把 HSG 分扇区理解成三层：

1. **分层写入**：同一记忆按不同语义维度建索引。  
2. **意图对齐检索**：先识别你在问什么，再选择相关维度排序。  
3. **持续演化**：命中会强化，长时间不用会衰减，而且不同扇区节奏不同。

这三层叠加，才构成“更像长期记忆系统，而不是向量搜索表”的体验。

