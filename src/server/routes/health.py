from fastapi import APIRouter

from src.utils.common_result import R

router = APIRouter()


@router.get("/health")
async def health_check() -> R:
    return R.success()
