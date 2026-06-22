from agile.web.common_result import gen_response_model, R
from fastapi import APIRouter
from fastapi.params import Body
from starlette import status

from infra.db.repositories import user_repo
from domain.memory.models import IMemoryUserIdentity, IMemoryUser
from interfaces.api.schemas.web_models import AuthRegisterRequest

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
    user: IMemoryUser | None = await user_repo.get_user(user_identity=user_identity, using_cache=True)
    if user:
        return R.error(code=400, message="User already exists")
    # 注册用户
    await user_repo.add_user(user_identity=user_identity)
    return R.success()
