# Waypoint 查询与创建机制说明

本文档用于说明项目中 Waypoint 的作用、查询路径、创建时机与数据结构。

## 1. Waypoint 是什么

在本项目中，Waypoint 可以理解为“记忆之间的有向关联边”，用于把离散的记忆节点连接成可遍历的关系网络。

- 边模型：`src_id -> dst_id`
- 权重：`weight`（0~1，越大表示关联越强）
- 主要用途：低置信检索时做扩展召回、命中后做关联强化传播

相关表定义见：`src/infra/db/orm_models.py` 中 `Waypoints`。

---

## 2. 数据模型与存储

`waypoints` 表字段：

- `src_id`：源记忆 ID（主键之一，FK -> `memories.id`）
- `dst_id`：目标记忆 ID（主键之一，FK -> `memories.id`）
- `user_id`：用户 ID
- `weight`：边权重
- `created_at` / `updated_at`：时间戳

索引：

- `idx_waypoints_src`
- `idx_waypoints_dst`

代码位置：`src/infra/db/orm_models.py`。

---

## 3. 查询阶段：Waypoint 如何参与检索

Waypoint 在 `query_hsg_memories` 中属于**第 4 路召回扩展**，不是单独查询入口。

代码位置：`src/services/memory/hsg.py`。

### 3.1 触发条件

只有在向量召回平均相似度较低时才触发：

- 条件：`avg_sim < 0.55`
- 调用：`waypoints.expand_via_waypoints(list(ids), effective_k * 2)`

这表示系统在“低置信”场景下启用关系图扩展，补足纯向量召回可能遗漏的候选。

### 3.2 扩展算法（`expand_via_waypoints`）

代码位置：`src/services/memory/waypoints.py`。

核心机制：

1. 以当前候选 ID 集合作为起点（初始权重 1.0）
2. 按 `src_id` 查询邻居 `dst_id`
3. 使用权重递推：

   `exp_wt = cur.weight * edge_weight * 0.8`

4. 剪枝：
   - `exp_wt < 0.1` 不继续
   - 达到 `max_expansion` 停止
5. 去重：访问过的节点不重复扩展
6. 输出 `Expansion(id, weight, path)`，其中 `path` 记录扩展路径

### 3.3 对最终排序的影响

Waypoint 扩展命中的候选会带入 `waypoint_weight`，进入混合评分：

- `compute_hybrid_score(..., wp_wt=waypoint_weight, ...)`
- 默认权重占比见 `src/domain/memory/scoring.py`：`waypoint = 0.15`

因此 Waypoint 既影响“候选覆盖率”，也影响“最终排序分数”。

---

## 4. 写入阶段：Waypoint 如何创建

Waypoint 的创建有两条主路径。

### 4.1 普通记忆新增（自动建边）

调用链：

- `add_hsg_memory` -> `waypoints.create_single_waypoint(...)`
- 代码位置：`src/services/memory/hsg.py`

`create_single_waypoint` 行为：

1. 获取当前用户历史记忆（最多 1000 条）
2. 读取历史记忆 `mean_vec`，与新记忆均值向量做余弦相似
3. 选取最相似记忆 `best`
4. upsert 一条边：
   - `src_id = new_id`
   - `dst_id = best`（若无候选则自环 `new_id -> new_id`）
   - `weight = best_sim`（无候选时为 `1.0`）

代码位置：`src/services/memory/waypoints.py`。

### 4.2 文档分段导入（root-child 显式建边）

调用链：

- `services/memory/ingest.py` 中 root-child 模式
- 每个子段写入后执行：`waypoints.link(user_identity, root_id, child_id, i)`

`link` 行为：

- 直接创建 `root_id -> child_id`
- `weight` 当前固定为 `1.0`

用于显式表达“根记忆与子段”的结构关系。

---

## 5. 命中后强化：Waypoint 的二次作用

在 `reinforce_memories` 中，命中记忆会触发关联传播：

1. 先强化命中节点自身显著性
2. 读取 `mem_ops.get_waypoints_by_src(hit_id)` 的邻接边
3. 调用 `propagate_associative_reinforcement_to_linked_nodes(...)`
4. 按边权将强化量传播到关联节点，再结合时间衰减更新显著性

相关代码：

- `src/services/memory/hsg.py`（`reinforce_memories`）
- `src/services/memory/dynamic_memory.py`

这使系统具备“检索-强化-再检索”的动态学习闭环。

---

## 6. 机制意义（为什么要有 Waypoint）

1. **补召回**：在低置信查询中，提高覆盖率，减少漏召回。
2. **结构化记忆网络**：不只依赖向量空间邻近，还保留显式语义关联。
3. **可解释性增强**：`Expansion.path` 可说明候选是如何被扩展出来的。
4. **动态学习能力**：命中后可沿关联传播强化，提升长期检索质量。

简化理解：

- 向量召回解决“像不像”
- Waypoint 解决“有没有关系链能到达”

---

## 7. 当前实现注意点（维护时可重点关注）

1. `create_single_waypoint` 仅扫描最多 1000 条历史记忆，是性能与全局最优的折中。
2. `link(..., idx)` 中 `idx` 当前未入库，可作为后续排序/权重扩展位。
3. Waypoint 扩展是有向的（`src -> dst`），调试时需注意方向性。
4. `avg_sim < 0.55`、`0.8` hop 衰减、`0.1` 剪枝阈值都是关键调参点。

---

## 8. 关键代码索引

- 查询入口：`src/services/memory/hsg.py` (`query_hsg_memories`)
- 查询扩展：`src/services/memory/waypoints.py` (`expand_via_waypoints`)
- 写入建边：`src/services/memory/waypoints.py` (`create_single_waypoint`, `link`)
- 导入建边：`src/services/memory/ingest.py`
- 表结构：`src/infra/db/orm_models.py` (`Waypoints`)
- 仓储访问：`src/infra/db/repos/memory_repo.py` (`get_waypoints_by_src`)
- 混合评分：`src/domain/memory/scoring.py`
- 关联强化：`src/services/memory/dynamic_memory.py`

---

## 9. 文字时序图

```text
写入新记忆
  -> 计算 mean_vec
  -> create_single_waypoint(new_id -> best_match)

查询 query_hsg_memories
  -> BM25 + 向量 + session 初始召回
  -> avg_sim 低于阈值时，expand_via_waypoints
  -> 扩展候选进入混合评分（含 waypoint_weight）
  -> 返回结果

查询命中后
  -> reinforce_memories
  -> 读取 src 的 waypoint 邻居
  -> 向关联节点传播强化
```

