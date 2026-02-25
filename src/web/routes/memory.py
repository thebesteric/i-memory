from fastapi import APIRouter, HTTPException

from src.imemory import IMemory
from src.web.models.web_models import AddMemoryRequest
from src.utils.common_result import R

router = APIRouter(prefix="/memory", tags=["memory"])

mem = IMemory()


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
