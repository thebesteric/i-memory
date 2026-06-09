from agile.commons.biz_error import BizError, ErrorCode

from src.memory.memory_models import IMemoryUserIdentity


class UserNotFoundError(BizError):
    """
    用户未找到异常
    """

    def __init__(self, user_identity: IMemoryUserIdentity):
        super().__init__(error_code=ErrorCode.IDENTITY_INVALID, message=f"User not found for identity: {user_identity}")
