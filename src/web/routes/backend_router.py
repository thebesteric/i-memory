from agile.web.common_result import R
from fastapi import APIRouter
from fastapi.params import Body
from starlette import status

from src.core import user_ops
from src.core.config import env
from src.core.mem_ops import mem_ops
from src.memory.graph import graph_builder
from src.memory.memory_models import IMemoryUserIdentity, IMemoryUser
from src.memory.profile.user_profile_builder import describe_user_profile

router = APIRouter(prefix="/backend", tags=["backend"])


@router.post(
    "/build-graph",
    status_code=status.HTTP_201_CREATED,
    summary="构建记忆图",
)
async def build_graph(user_identity: IMemoryUserIdentity = Body(..., description="用户身份")):
    user: IMemoryUser | None = await user_ops.get_user(user_identity=user_identity, using_cache=True)
    if user:
        # 获取用户未参与事实构建的记忆
        un_fact_join_memories = mem_ops.find_mem_by_conditions(
            conditions=["fact_joined = 0", "user_id = %s"],
            order_by=["created_at ASC"],
            params=[user.id],
            limit=env.GRAPH_MEM_COUNT_AT_MOST or 100
        )
        # 检查记忆数量要求
        least_tolerance = graph_builder.GRAPH_MEM_COUNT_AT_LEAST_TOLERANCE
        if not un_fact_join_memories or len(un_fact_join_memories) < least_tolerance:
            return R.error(
                code=400,
                data={"memories": un_fact_join_memories, "threshold": least_tolerance},
                message=f"User {user.id} has {len(un_fact_join_memories)} un-fact-joined memories, which is less than {least_tolerance}. Skipping processing."
            )
        # 加入构建队列
        graph_builder.user_queue.put_nowait(user)
        return R.success(message="已触发图构建任务，请稍后查看结果")

    return R.error(code=404, message="用户不存在")


@router.post(
    "/build-user-profile",
    status_code=status.HTTP_201_CREATED,
    summary="构建用户画像",
)
async def build_user_profile(user_identity: IMemoryUserIdentity = Body(..., description="用户身份")):
    user: IMemoryUser | None = await user_ops.get_user(user_identity=user_identity, using_cache=True)
    if user:
        # 指定构建用户画像
        await describe_user_profile(user_ids=[user.id])
        return R.success(message="已触发用户画像构建任务，请稍后查看结果")

    return R.error(code=404, message="用户不存在")
