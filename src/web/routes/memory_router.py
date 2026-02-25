import json
from typing import Dict, List, Any

from fastapi import APIRouter, Query
from fastapi.params import Path, Body

from src.imemory import IMemory
from src.utils.paging import PagingResponse
from src.web.models.web_models import AddMemoryRequest, SearchMemoryRequest, HistoryMemoryRequest
from src.utils.common_result import R

router = APIRouter(prefix="/memory", tags=["memory"])

mem = IMemory()


@router.post("/add", summary="添加记忆内容")
async def add(req: AddMemoryRequest):
    """
    添加记忆内容
    :param req: 添加记忆请求模型
    :return: 添加记忆结果
    """
    meta = req.metadata or {}
    if req.tags:
        meta["tags"] = req.tags
    result = await mem.add(req.content, user_id=req.user_id, meta=meta)
    return R.success(data=result)


@router.post("/search", summary="搜索相关记忆内容")
async def search(req: SearchMemoryRequest):
    """
    搜索相关记忆内容
    :param req: 搜索记忆请求模型
    :return: 搜索结果列表
    """
    results = await mem.search(query=req.query, user_id=req.user_id, limit=req.limit, filters=req.filters)
    return R.success(data=results)


@router.post("/history", summary="获取用户的历史记忆内容")
async def history(req: HistoryMemoryRequest):
    """
    获取用户的历史记忆内容
    :param req: 历史记忆请求模型
    :return: 历史记忆列表
    """
    results: PagingResponse = await mem.history(user_id=req.user_id, current=req.current, size=req.size)
    results.records = _handle_memories(results.records if results.records else [])
    return R.success(data=results)


@router.get("/get/{memory_id}", summary="获取指定 ID 的记忆内容")
async def get(memory_id: str = Path(..., description="记忆 ID")):
    """
    获取指定 ID 的记忆内容
    :param memory_id: 记忆 ID
    :return: 记忆内容
    """
    row = await mem.get(memory_id)
    rows = _handle_memories([row] if row else [])
    return R.success(data=rows[0] if rows else None)


@router.post("/delete", summary="删除指定 ID 的记忆内容")
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


@router.get("/clear", summary="清空用户的所有记忆内容")
async def clear_memory(user_id: str = Query(..., description="用户 ID")):
    """
    清空用户的所有记忆内容
    :param user_id: 用户 ID
    :return: 删除结果
    """
    affected_rows = await mem.clear(user_id=user_id)
    return R.success(data={"affected_rows": affected_rows})


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
        memory["sectors"] = json.loads(memory.get("sectors", []))
        memory["tags"] = json.loads(memory.get("tags", []))
        memory["meta"] = json.loads(memory.get("meta", {}))
        processed.append(memory)
    return processed
