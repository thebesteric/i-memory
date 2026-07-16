# i-memory 当前能力清单（基于代码实现）

> 生成日期：2026-07-07

## 1. 记忆存储与检索能力

- 记忆新增：支持文本内容入库，附带 `tags`、`metadata`、`qa_role`（human/assistant）等信息。  
  证据：`src/services/i_memory.py`、`src/interfaces/api/routes/memory_router.py`
- 记忆搜索：支持按 query + filter 检索，返回结构化结果。  
  证据：`src/services/i_memory.py`、`src/services/memory/hsg.py`、`src/interfaces/api/routes/memory_router.py`
- 记忆历史分页：支持按用户拉取历史记录、分页与排序。  
  证据：`src/services/i_memory.py`、`src/interfaces/api/routes/memory_router.py`
- 单条读取/删除、用户级清空。  
  证据：`src/services/i_memory.py`、`src/interfaces/api/routes/memory_router.py`

## 2. HSG（分层语义图）核心能力

- 五类记忆扇区建模：情景、语义、程序、情绪、反思。  
  证据：`src/services/memory/hsg.py`（扇区配置与写入/查询逻辑）
- 混合召回与重排：向量相似度、关键词、标签、路标、时间等多因子融合评分。  
  证据：`src/services/memory/hsg.py`（query + hybrid score）
- 路标/关联扩展召回：支持基于关系的扩散与增强。  
  证据：`src/services/memory/hsg.py`
- QA 配对模式：支持 `query_mode`（prefer/qa/vector）与问答对关联检索。  
  证据：`README.md`（QA 参数说明）、`src/services/memory/hsg.py`、`src/interfaces/api/routes/memory_router.py`

## 3. 记忆生命周期能力（遗忘与强化）

- 衰减机制：实现基于时间和显著性的记忆衰减（艾宾浩斯思路）。  
  证据：`domain/memory/decay_policy.py`、`src/services/memory/hsg.py`
- 检索强化：查询命中的记忆可被强化，影响后续排序与保留。  
  证据：`src/services/memory/hsg.py`
- 定时衰减任务：后台周期执行记忆衰减。  
  证据：`src/infra/scheduler/jobs.py`

## 4. 知识图谱构建与查询能力

- 事实抽取：从对话/记忆中抽取事实、实体、关系。  
  证据：`src/services/graph/fact_extractor.py`
- 实体规范化：对实体做 canonical 合并，减少重复实体。  
  证据：`src/services/graph/entity_canonicalize.py`
- 图谱查询：支持事实列表、事实-实体、实体关系、实体话题、话题记忆查询。  
  证据：`src/interfaces/api/routes/graph_router.py`
- 图探索聚合：支持以 canonical/fact/topic 为种子，返回前端可渲染 `nodes/edges`。  
  证据：`src/interfaces/api/routes/graph_router.py`（`/graph/explore`）
- 图构建任务：支持周期图构建与每日强制图构建。  
  证据：`src/infra/scheduler/jobs.py`、`src/services/graph/graph_builder.py`

## 5. 用户画像与会话能力

- 用户画像生成：从用户历史记忆提取偏好/特征，支持查询画像。  
  证据：`src/services/profile/user_profile_extractor.py`、`src/interfaces/api/routes/memory_router.py`（`/memory/user_profile`）
- 会话聚合/总结：按会话维度进行构建与总结（定时任务触发）。  
  证据：`src/services/session/session_extractor.py`、`src/infra/scheduler/jobs.py`

## 6. 分类与 AI 能力

- BERT 扇区分类：本地 BERT 管理器支持预测/训练，用于扇区判定。  
  证据：`src/infra/ai/classifier/bert_manager.py`、`tests/test_bert_manager_predict.py`、`tests/test_bert_manager_train.py`
- 多 LLM 接入：支持 OpenAI/Gemini/DashScope 等模型注册与调用。  
  证据：`src/infra/ai/*/registrars/`、`README.md`
- 嵌入向量流程：支持 embedding + 向量检索路径。  
  证据：`src/services/memory/*`、`src/infra/ai/embedding/*`

## 7. 对外 API 能力（FastAPI）

基础前缀：`/imemory`

- 内存管理接口：`/memory/add`、`/memory/search`、`/memory/history`、`/memory/get/{memory_id}`、`/memory/delete`、`/memory/clear`、`/memory/user_profile`、`/memory/canonical_relations`  
  证据：`src/interfaces/api/routes/memory_router.py`
- 图谱接口：`/graph/facts`、`/graph/fact/entities`、`/graph/entity/relations`、`/graph/entity/topics`、`/graph/topic/memories`、`/graph/explore`  
  证据：`src/interfaces/api/routes/graph_router.py`
- 后台与鉴权接口：包含手动触发图构建、手动触发定时任务、用户注册等能力。  
  证据：`src/interfaces/api/routes/backend_router.py`、`src/interfaces/api/routes/auth_router.py`

## 8. 多租户与数据安全能力

- 多租户隔离：租户/项目/用户三级身份。  
  证据：`domain/memory/models.py`、`README.md`
- 字段级加密：关键内容支持加密存储/读取。  
  证据：`src/services/commons/encrypt_service.py`、`src/infra/db/repos/user_repo.py`

## 9. 任务调度与运维能力

- APScheduler 定时任务：interval + cron 双模式，支持任务开关、并发限制、补偿策略。  
  证据：`src/infra/scheduler/jobs.py`
- 图构建后台 worker：支持多 worker 消费队列。  
  证据：`src/infra/scheduler/jobs.py`
- 启停脚本：`run.sh` 支持 `start|stop|restart|status|logs`。  
  证据：`run.sh`

## 10. 存储与基础设施集成

- PostgreSQL + pgvector：作为主要结构化与向量数据存储路径。  
  证据：`README.md`、`src/infra/db/*`
- Redis：用于缓存/辅助能力。  
  证据：`docker/docker-redis.sh`、`README.md`
- Milvus：提供可选向量存储支持（通过配置开启）。  
  证据：`src/services/i_memory.py`（`env.VECTOR_MILVUS_SUPPORT`）、`docker/docker-compose-milvus.yml`


