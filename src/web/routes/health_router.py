from agile_commons.web import R
from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", summary="健康检查")
async def health_check() -> R:
    return R.success()
