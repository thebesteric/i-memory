from typing import Any

from agile.web.common_result import gen_response_model, R
from fastapi import APIRouter
from fastapi.params import Body
from starlette import status

from src.core import user_ops
from src.memory.memory_models import IMemoryUserIdentity, IMemoryUser
from src.web.models.web_models import AuthRegisterRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="注册用户",
    response_model=gen_response_model(
        "RegisterResponse",
        data_type=None,
        data_desc="用户注册结果",
    ),
)
async def register(req: AuthRegisterRequest = Body(..., description="用户身份")):
    user_identity: IMemoryUserIdentity = req.to_identity_model()
    user: IMemoryUser | None = await user_ops.get_user(user_identity=user_identity, using_cache=True)
    if user:
        return R.error(code=400, message="User already exists")
    # 注册用户
    await user_ops.add_user(user_identity=user_identity)
    return R.success()
