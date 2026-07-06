---
title: 默认模块
language_tabs:
  - shell: Shell
  - http: HTTP
  - javascript: JavaScript
  - ruby: Ruby
  - python: Python
  - php: PHP
  - java: Java
  - go: Go
toc_footers: []
includes: []
search: true
code_clipboard: true
highlight_theme: darkula
headingLevel: 2
generator: "@tarslib/widdershins v4.0.30"

---

# 默认模块

Base URLs:

# Authentication

# Default

## GET 健康检查

GET /health

> 返回示例

> 200 Response

```json
{"code":200,"message":"OK","data":null,"http_status":200,"succeed":true,"track_id":null}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 记忆-添加

POST /memory/add

> Body 请求参数

```json
{
  "content": "string",
  "user_identity": {
    "user_key": "string",
    "project_key": "string",
    "tenant_key": "string"
  },
  "tags": [
    "string"
  ],
  "metadata": {},
  "qa_role": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» content|body|string| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» project_key|body|string| 否 |none|
|»» tenant_key|body|string| 否 |none|
|» tags|body|[string]| 否 |none|
|» metadata|body|object| 否 |none|
|» qa_role|body|string| 是 |human/assistant/null|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 记忆-删除

POST /memory/delete

> Body 请求参数

```json
[
  "string"
]
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|array[string]| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 记忆-清空

POST /memory/clear

> Body 请求参数

```json
{
  "user_id": "string",
  "tenant_id": "string",
  "project_id": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_id|body|string| 是 |none|
|» tenant_id|body|string| 否 |none|
|» project_id|body|string| 否 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 记忆-查询

POST /memory/search

> Body 请求参数

```json
{
  "query": "string",
  "limit": 0,
  "filters": {
    "user_identity": {
      "user_key": "string",
      "tenant_key": "string",
      "project_key": "string"
    },
    "sectors": [
      "string"
    ],
    "min_salience": 0,
    "config": {
      "debug": true,
      "bm25_enable": true,
      "user_profile_enable": true,
      "session_summary_enable": true,
      "session_dedup_enable": true,
      "graph": {
        "enable": true,
        "type": "string",
        "max_hops": 0,
        "hop_decay": 0,
        "per_hop_limit": 0,
        "min_walk_score": 0,
        "min_relation_confidence": 0
      }
    }
  }
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 否 |none|
|» query|body|string| 是 |none|
|» limit|body|integer| 是 |none|
|» filters|body|object| 是 |none|
|»» user_identity|body|object| 是 |none|
|»»» user_key|body|string| 是 |none|
|»»» tenant_key|body|string| 否 |none|
|»»» project_key|body|string| 否 |none|
|»» sectors|body|[string]| 是 |none|
|»» min_salience|body|number| 是 |none|
|»» config|body|object| 是 |none|
|»»» debug|body|boolean| 是 |none|
|»»» bm25_enable|body|boolean| 是 |none|
|»»» user_profile_enable|body|boolean| 是 |none|
|»»» session_summary_enable|body|boolean| 是 |none|
|»»» session_dedup_enable|body|boolean| 是 |none|
|»»» graph|body|object| 是 |none|
|»»»» enable|body|boolean| 是 |none|
|»»»» type|body|string| 是 |recall、precision、custom|
|»»»» max_hops|body|integer| 是 |none|
|»»»» hop_decay|body|number| 是 |none|
|»»»» per_hop_limit|body|integer| 是 |none|
|»»»» min_walk_score|body|number| 是 |none|
|»»»» min_relation_confidence|body|number| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## GET 记忆-获取

GET /memory/get/{memory_id}

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|memory_id|path|string| 是 |none|

> 返回示例

> 200 Response

```json
{"detail":"Not Found"}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 记忆-历史

POST /memory/history

> Body 请求参数

```json
{
  "user_identity": {
    "user_id": "string",
    "tenant_id": "string",
    "project_id": "string"
  },
  "current": 0,
  "size": 0,
  "sort_order": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_id|body|string| 是 |none|
|»» tenant_id|body|string| 否 |none|
|»» project_id|body|string| 否 |none|
|» current|body|number| 是 |none|
|» size|body|number| 是 |none|
|» sort_order|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 记忆-用户画像

POST /memory/user_profile

> Body 请求参数

```json
{
  "user_key": "string",
  "tenant_key": "string",
  "project_key": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_key|body|string| 是 |none|
|» tenant_key|body|string| 是 |none|
|» project_key|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 记忆-查询 canonical 实体关系边

POST /memory/canonical_relations

> Body 请求参数

```json
{
  "user_identity": {
    "user_key": "string",
    "tenant_key": "string",
    "project_key": "string"
  },
  "canonical_id": "string",
  "limit": 0
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» tenant_key|body|string| 是 |none|
|»» project_key|body|string| 是 |none|
|» canonical_id|body|string| 是 |none|
|» limit|body|integer| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 图-查询用户关联事实

POST /graph/facts

> Body 请求参数

```json
{
  "user_identity": {
    "user_key": "string",
    "tenant_key": "string",
    "project_key": "string"
  },
  "current": 0,
  "size": 0,
  "filters": {
    "topic_id": "string",
    "fact_kind": "string",
    "min_confidence": "string",
    "max_confidence": "string",
    "keyword": "string"
  }
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» tenant_key|body|string| 是 |none|
|»» project_key|body|string| 是 |none|
|» current|body|integer| 是 |none|
|» size|body|integer| 是 |none|
|» filters|body|object¦null| 否 |none|
|»» topic_id|body|string¦null| 否 |none|
|»» fact_kind|body|string¦null| 否 |none|
|»» min_confidence|body|string¦null| 否 |none|
|»» max_confidence|body|string¦null| 否 |none|
|»» keyword|body|string¦null| 否 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 图-查询事实关联的规范化实体

POST /graph/fact/entities

> Body 请求参数

```json
{
  "user_identity": {
    "user_key": "string",
    "tenant_key": "string",
    "project_key": "string"
  },
  "current": 0,
  "size": 0,
  "fact_id": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» tenant_key|body|string| 是 |none|
|»» project_key|body|string| 是 |none|
|» current|body|integer¦null| 否 |none|
|» size|body|integer¦null| 否 |none|
|» fact_id|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 图-查询实体关系

POST /graph/entity/relations

> Body 请求参数

```json
{
  "user_identity": {
    "user_key": "string",
    "tenant_key": "string",
    "project_key": "string"
  },
  "current": 0,
  "size": 0,
  "canonical_id": "string",
  "filters": {
    "fact_id": "string",
    "edge_relations": [
      "string"
    ],
    "infer_sources": [
      "string"
    ],
    "min_confidence": 0,
    "max_confidence": 0,
    "related_canonical_id": "string"
  }
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» tenant_key|body|string| 是 |none|
|»» project_key|body|string| 是 |none|
|» current|body|integer¦null| 否 |none|
|» size|body|integer¦null| 否 |none|
|» canonical_id|body|string| 是 |规范化实体 ID|
|» filters|body|object¦null| 否 |none|
|»» fact_id|body|string¦null| 否 |按证据 fact_id 过滤|
|»» edge_relations|body|[string]¦null| 否 |按边关系|
|»» infer_sources|body|[string]¦null| 否 |按推断来源过滤|
|»» min_confidence|body|number¦null| 否 |最小置信度|
|»» max_confidence|body|number¦null| 否 |最大置信度|
|»» related_canonical_id|body|string¦null| 否 |按关联实体 ID 过滤|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 图-查询实体关联话题

POST /graph/entity/topics

> Body 请求参数

```json
{
  "user_identity": {
    "user_key": "string",
    "tenant_key": "string",
    "project_key": "string"
  },
  "current": 0,
  "size": 0,
  "canonical_id": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» tenant_key|body|string| 是 |none|
|»» project_key|body|string| 是 |none|
|» current|body|integer¦null| 否 |none|
|» size|body|integer¦null| 否 |none|
|» canonical_id|body|string| 是 |规范化实体 ID|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 图-查询话题关联记忆

POST /graph/topic/memories

> Body 请求参数

```json
{
  "user_identity": {
    "user_key": "string",
    "tenant_key": "string",
    "project_key": "string"
  },
  "current": 0,
  "size": 0,
  "topic_id": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» tenant_key|body|string| 是 |none|
|»» project_key|body|string| 是 |none|
|» current|body|integer¦null| 否 |none|
|» size|body|integer¦null| 否 |none|
|» topic_id|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 图-图探索聚合查询

POST /graph/explore

> Body 请求参数

```json
{
  "user_identity": {
    "user_key": "string",
    "tenant_key": "string",
    "project_key": "string"
  },
  "current": 0,
  "size": 0,
  "seed_type": "string",
  "seed_id": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_identity|body|object| 是 |none|
|»» user_key|body|string| 是 |none|
|»» tenant_key|body|string| 是 |none|
|»» project_key|body|string| 是 |none|
|» current|body|integer¦null| 否 |none|
|» size|body|integer¦null| 否 |none|
|» seed_type|body|string| 是 |"canonical", "fact", "topic"|
|» seed_id|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 认证-注册用户

POST /auth/register

> Body 请求参数

```json
{
  "user_key": "string",
  "project_key": "string",
  "tenant_key": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_key|body|string| 是 |none|
|» project_key|body|string| 是 |none|
|» tenant_key|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 后台-构建记忆图

POST /backend/build-graph

> Body 请求参数

```json
{
  "user_key": "string",
  "project_key": "string",
  "tenant_key": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_key|body|string| 是 |none|
|» project_key|body|string| 是 |none|
|» tenant_key|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 后台-构建用户画像

POST /backend/build-user-profile

> Body 请求参数

```json
{
  "user_key": "string",
  "project_key": "string",
  "tenant_key": "string"
}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|
|» user_key|body|string| 是 |none|
|» project_key|body|string| 是 |none|
|» tenant_key|body|string| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## GET 后台-任务列表

GET /backend/jobs

> Body 请求参数

```json
{}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 后台-任务触发

POST /backend/jobs/{job_id}/trigger

> Body 请求参数

```json
{}
```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|job_id|path|string| 是 |session_build、user_profile、graph_build|
|body|body|object| 是 |none|

> 返回示例

> 200 Response

```json
{}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

# 数据模型

