from agile.web.common_result import R
from fastapi import APIRouter
from fastapi.params import Body
from starlette import status

from src.core import user_ops
from src.memory.graph.graph_builder import user_queue
from src.memory.memory_models import IMemoryUserIdentity, IMemoryUser

router = APIRouter(prefix="/backend", tags=["backend"])


@router.post(
    "/build-graph",
    status_code=status.HTTP_201_CREATED,
    summary="构建用户图",
)
async def build_graph(user_identity: IMemoryUserIdentity = Body(..., description="用户身份")):
    user: IMemoryUser = await user_ops.get_user(user_identity=user_identity, using_cache=True)
    if user:
        user_queue.put_nowait(user)
        return R.success(message="已触发图构建任务，请稍后查看结果")

    return R.error(code=404, message="用户不存在")
