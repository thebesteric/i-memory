from typing import Generic, Optional, TypeVar, Any, ClassVar, Protocol, runtime_checkable
from http import HTTPStatus

from httpx import Response
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar('T')


@runtime_checkable
class StatusWritableResponse(Protocol):
    status_code: int


class R(BaseModel, Generic[T]):
    """
    通用响应返回类
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    SUCCESS_CODE: ClassVar[HTTPStatus] = HTTPStatus.OK
    ERROR_CODE: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR

    code: Optional[int] = Field(default=None, description="响应码")
    message: Optional[str] = Field(default=None, description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    http_status: Optional[HTTPStatus] = Field(default=None, description="HTTP 状态码")
    succeed: bool = Field(default=False, description="是否成功")
    track_id: Optional[str] = Field(default=None, description="追踪 ID")

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        self.succeed: bool = False
        self.code: Optional[int] = None
        self.message: Optional[str] = None
        self.data: Optional[T] = None
        self.http_status: Optional[HTTPStatus] = None
        self.track_id: Optional[str] = None

    # 基础构建方法
    @classmethod
    def build(cls, *, is_success: bool) -> 'R[T]':
        """创建基础响应对象并指定成功状态"""
        result = cls()
        result.succeed = is_success
        if is_success:
            result.code = cls.SUCCESS_CODE.value
            result.http_status = HTTPStatus.OK
        else:
            result.code = cls.ERROR_CODE.value
            result.http_status = HTTPStatus.INTERNAL_SERVER_ERROR
        return result

    # 成功响应入口
    @classmethod
    def success(cls, *,
                code: int = SUCCESS_CODE.value,
                message: str = SUCCESS_CODE.phrase,
                data: T = None,
                http_status: Optional[HTTPStatus] = None) -> 'R[T]':
        """
        创建成功响应对象（链式调用入口）
        :return: 成功响应对象
        """
        return cls.build(is_success=True).set_code(code).set_message(message).set_data(data).set_http_status(http_status)

    # 错误响应入口
    @classmethod
    def error(cls, *, code: int = ERROR_CODE.value, message: str = ERROR_CODE.phrase, data: T = None,
              http_status: Optional[HTTPStatus] = None) -> 'R[T]':
        """
        创建错误响应对象（链式调用入口）
        :return: 错误响应对象
        """
        return cls.build(is_success=False).set_code(code).set_message(message).set_data(data).set_http_status(http_status)

    # 链式调用方法
    def set_code(self, code: int) -> 'R[T]':
        """
        设置响应码
        :param code: 响应码
        :return: 当前实例
        """
        self.code = code
        return self

    def set_message(self, msg: str) -> 'R[T]':
        """
        设置响应消息
        :param msg: 响应消息
        :return: 当前实例
        """
        self.message = msg
        return self

    def set_data(self, data: T) -> 'R[T]':
        """
        设置响应数据
        :param data: 响应数据
        :return: 当前实例
        """
        self.data = data
        return self

    def set_track_id(self, track_id: str) -> 'R[T]':
        """
        设置追踪ID
        :param track_id: 追踪ID
        :return: 当前实例
        """
        self.track_id = track_id
        return self

    def set_http_status(self, http_status: Optional[HTTPStatus], response: Optional[StatusWritableResponse] = None) -> 'R[T]':
        """
        设置HTTP状态码；如提供 response，将同时写入 response.status_code
        :param http_status: HTTP状态码，可选
        :param response: 服务端框架的 Response 对象（需可写 status_code）
        :return: 当前实例
        """
        self.http_status = http_status if http_status else self.http_status
        if response is not None and hasattr(response, "status_code") and self.http_status is not None:
            response.status_code = self.http_status.value
        return self

    @classmethod
    def extract_data(cls, result: Optional['R[T]'], default_value: T) -> T:
        """
        从响应中提取数据，失败时返回默认值
        :param result: 响应对象
        :param default_value: 默认值
        :return: 提取到的数据或默认值
        """
        if result and result.succeed:
            return result.data if result.data is not None else default_value
        return default_value
