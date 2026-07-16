from typing import Dict, List

from agile.web import PagingResponse
from agile.web.common_result import gen_response_model, R
from fastapi import APIRouter
from fastapi.params import Path, Body

from services.i_memory import IMemory
from services.graph import graph_ops
from domain.memory.models import IMemoryUserIdentity, IMemorySearchResult
from services.profile import user_profile_ops
from services.commons.user_access import get_user_for_access
from domain.profile.models import UserProfile
from shared.utils.json_utils import coerce_json_field
from interfaces.api.schemas.web_models import AddMemoryRequest, SearchMemoryRequest, HistoryMemoryRequest, \
    CanonicalRelationsRequest

from typing import Any

router = APIRouter(prefix="/memory", tags=["memory"])

mem = IMemory()


@router.post(
    "/add",
    summary="添加记忆内容",
    response_model=gen_response_model(
        "AddResponse",
        data_type=list[dict[str, Any]],
        data_desc="新增记忆的详细信息列表",
    ),
    description=(
        "向记忆库添加记忆内容。content 支持单条 {role: text} 或多条 [{role: text}, ...]，"
        "role 仅允许 human / assistant。每条消息独立存储，按顺序返回结果列表。"
    )
)
async def add(req: AddMemoryRequest):
    """
    添加记忆内容（支持成对批量添加）
    :param req: 添加记忆请求模型
    :return: 各条记忆的添加结果列表
    """
    results = []
    for item in req.content:
        for role, text in item.items():
            result = await mem.add(
                text,
                user_identity=req.user_identity,
                meta=req.metadata or {},
                tags=req.tags or [],
                role=role,
            )
            results.append(result)
    return R.success(data=results)


@router.post(
    "/search",
    summary="搜索相关记忆内容",
    response_model=gen_response_model(
        "SearchResponse",
        data_type=IMemorySearchResult,
        data_desc="匹配的记忆内容列表",
    ),
    description="根据查询关键词和可选过滤条件，检索相关记忆内容。支持分页和多条件过滤。"
)
async def search(req: SearchMemoryRequest):
    """
    搜索相关记忆内容
    :param req: 搜索记忆请求模型
    :return: 搜索结果列表
    """
    results: IMemorySearchResult = await mem.search(
        query=req.query,
        limit=req.limit,
        filters=req.filters
    )
    return R.success(data=results)


@router.post(
    "/history",
    summary="获取用户的历史记忆内容",
    response_model=gen_response_model(
        "HistoryResponse",
        data_type=PagingResponse,
        data_desc="分页的记忆内容列表",
    ),
    description="分页获取指定用户的历史记忆内容，按时间倒序排列。"
)
async def history(req: HistoryMemoryRequest):
    """
    获取用户的历史记忆内容
    :param req: 历史记忆请求模型
    :return: 历史记忆列表
    """
    results: PagingResponse = await mem.history(
        user_identity=req.user_identity,
        current=req.current,
        size=req.size,
        sort_order=req.sort_order
    )
    results.records = _handle_memories(results.records if results.records else [])
    return R.success(data=results)


@router.get(
    "/get/{memory_id}",
    summary="获取指定 ID 的记忆内容",
    response_model=gen_response_model(
        "GetResponse",
        data_type=dict[str, Any] | None,
        data_desc="记忆内容详情",
    ),
    description="根据记忆 ID 获取详细内容。"
)
async def get(memory_id: str = Path(..., description="记忆 ID")):
    """
    获取指定 ID 的记忆内容
    :param memory_id: 记忆 ID
    :return: 记忆内容
    """
    row = await mem.get(memory_id)
    rows = _handle_memories([row] if row else [])
    return R.success(data=rows[0] if rows else None)


@router.post(
    "/delete",
    summary="删除指定 ID 的记忆内容",
    response_model=gen_response_model(
        "DeleteResponse",
        data_type=dict[str, int],
        data_desc="删除影响的行数",
    ),
    description="批量删除指定ID的记忆内容。"
)
async def delete_memory(memory_ids: list[str] = Body(..., description="记忆 ID 列表")):
    """
    删除指定 ID 的记忆内容
    :param memory_ids: 记忆 ID 列表
    :return: 删除结果
    """
    affected_rows = 0
    for mid in memory_ids:
        affected_rows += await mem.delete(mid)
    return R.success(data={"affected_rows": affected_rows})


@router.post(
    "/clear",
    summary="清空用户的所有记忆内容",
    response_model=gen_response_model(
        "ClearResponse",
        data_type=dict[str, int],
        data_desc="删除影响的行数",
    ),
    description="删除指定用户的全部记忆内容。不可恢复。"
)
async def clear_memory(user_identity: IMemoryUserIdentity = Body(..., description="用户身份")):
    """
    清空用户的所有记忆内容
    :param user_identity: 用户身份
    :return: 删除结果
    """
    affected_rows = await mem.clear(user_identity=user_identity)
    return R.success(data={"affected_rows": affected_rows})


@router.post(
    "/user_profile",
    summary="获取用户画像",
    response_model=gen_response_model(
        "UserProfileResponse",
        data_type=UserProfile | None,
        data_desc="用户行为画像"
    )
)
async def get_user_profile(user_identity: IMemoryUserIdentity = Body(..., description="用户身份")):
    user = await get_user_for_access(user_identity=user_identity)
    # 查询用户画像
    user_profile: UserProfile = await user_profile_ops.get_user_profile(user, query_cache=True)
    return R.success(data=user_profile)


@router.post(
    "/canonical_relations",
    summary="查询 canonical 实体关系边",
    response_model=gen_response_model(
        "CanonicalRelationsResponse",
        data_type=dict[str, Any],
        data_desc="canonical 实体关系边列表",
    ),
    description="根据用户身份与规范化实体 ID 查询该实体参与的双向关系边。"
)
async def canonical_relations(req: CanonicalRelationsRequest):
    user = await get_user_for_access(user_identity=req.user_identity)

    rows = graph_ops.find_canonical_relations(user.id, req.canonical_id, req.limit or 100)
    return R.success(data={
        "canonical_id": req.canonical_id,
        "count": len(rows),
        "relations": rows,
    })


def _handle_memories(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理记忆内容列表，去除敏感信息并格式化输出
    :param memories: 原始记忆内容列表
    :return: 处理后的记忆内容列表
    """
    processed = []
    for memory in memories:
        memory.pop("mean_vec", None)
        memory.pop("compressed_vec", None)
        sectors = coerce_json_field(memory.get("sectors"), [])
        tags = coerce_json_field(memory.get("tags"), [])
        meta = coerce_json_field(memory.get("meta"), {})

        memory["sectors"] = sectors if isinstance(sectors, list) else []
        memory["tags"] = tags if isinstance(tags, list) else []
        memory["meta"] = meta if isinstance(meta, dict) else {}
        processed.append(memory)
    return processed
