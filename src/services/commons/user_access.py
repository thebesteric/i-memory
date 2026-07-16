from agile.utils import LogHelper

from domain.common.exceptions import UserNotFoundError
from domain.memory.models import IMemoryUser, IMemoryUserIdentity
from infra.db.repos import user_repo
from shared.config.settings import env

logger = LogHelper.get_logger(title="[USER_ACCESS]")


async def get_user_for_access(
    user_identity: IMemoryUserIdentity,
    *,
    using_cache: bool = False,
) -> IMemoryUser:
    """Resolve user for request access, optionally auto-registering when allowed."""
    user = await user_repo.get_user(user_identity=user_identity, using_cache=using_cache)
    if user:
        return user

    if env.USER_REGISTER_FORCE:
        raise UserNotFoundError(user_identity)

    user = await user_repo.add_user(user_identity)
    logger.info(f"Auto registered user for identity: {user_identity}")
    return user

