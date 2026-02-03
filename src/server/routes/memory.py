from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.memory.memory import IMemory
from src.module import app_injector
from src.utils.common_result import R

router = APIRouter(prefix="/memory", tags=["memory"])

mem: IMemory = app_injector.get(IMemory)


class AddMemoryRequest(BaseModel):
    content: str
    user_id: Optional[str] = None
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    limit: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = {}


@router.post("/add")
async def add_memory(req: AddMemoryRequest):
    try:
        meta = req.metadata or {}
        if req.tags:
            meta["tags"] = req.tags
        result = await mem.add(req.content, user_id=req.user_id, meta=meta)
        return R.success(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
