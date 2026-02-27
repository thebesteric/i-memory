from fastapi import APIRouter
from web.common_result import R

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", summary="健康检查")
async def health_check() -> R:
    return R.success()
