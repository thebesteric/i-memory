import datetime
from typing import Generic, Optional, TypeVar, Any, ClassVar, Protocol, runtime_checkable
from http import HTTPStatus

from httpx import Response
from pydantic import BaseModel, ConfigDict, Field, field_serializer

T = TypeVar('T')

# 自定义 datetime 序列化逻辑：全局默认格式
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _serialize_datetime_fields(data: Any, datetime_format: str = DEFAULT_DATETIME_FORMAT) -> Any:
    # 处理 datetime 对象
    if isinstance(data, datetime.datetime):
        # 处理带时区的 datetime
        if data.tzinfo:
            return data.astimezone().strftime(datetime_format)
        return data.strftime(datetime_format)
    # 递归处理字典
    if isinstance(data, dict):
        return {k: _serialize_datetime_fields(v, datetime_format) for k, v in data.items()}
    # 递归处理列表
    if isinstance(data, (list, tuple, set)):
        serialized = [_serialize_datetime_fields(item, datetime_format) for item in data]
        return serialized if isinstance(data, list) else type(data)(serialized)
    # 处理 Pydantic 模型（转为字典后递归处理）
    if isinstance(data, BaseModel):
        return _serialize_datetime_fields(data.model_dump(), datetime_format)
    # 其他类型原样返回
    return data


@runtime_checkable
class StatusWritableResponse(Protocol):
    status_code: int


class R(BaseModel, Generic[T]):
    """
    通用响应返回类
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    SUCCESS_CODE: ClassVar[HTTPStatus] = HTTPStatus.OK
    ERROR_CODE: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR

    code: Optional[int] = Field(default=None, description="响应码")
    message: Optional[str] = Field(default=None, description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    http_status: Optional[HTTPStatus] = Field(default=None, description="HTTP 状态码")
    succeed: bool = Field(default=False, description="是否成功")
    track_id: Optional[str] = Field(default=None, description="追踪 ID")
    datetime_format: str = Field(default=DEFAULT_DATETIME_FORMAT, description="datetime 序列化格式")

    @field_serializer("data")
    def _serialize_data(self, data: Optional[T]) -> Optional[T]:
        return _serialize_datetime_fields(data, datetime_format=self.datetime_format) if data else None

    def __init__(self, /, **kwargs: Any):
        super().__init__(**kwargs)
        self.succeed: bool = False
        self.code: Optional[int] = None
        self.message: Optional[str] = None
        self.data: Optional[T] = None
        self.http_status: Optional[HTTPStatus] = None
        self.track_id: Optional[str] = None

    # 基础构建方法
    @classmethod
    def build(cls, *, is_success: bool, datetime_format: Optional[str] = None) -> 'R[T]':
        """
        创建基础响应对象并指定成功状态
        :param is_success: 是否成功
        :param datetime_format: datetime 序列化格式
        :return: 基础响应对象
        """
        result = cls()
        if datetime_format:
            result.datetime_format = datetime_format
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
                http_status: Optional[HTTPStatus] = None,
                datetime_format: Optional[str] = None) -> 'R[T]':
        """
        创建成功响应对象（链式调用入口）
        :param code: 响应码
        :param message: 响应消息
        :param data: 响应数据
        :param http_status: HTTP状态码
        :param datetime_format: datetime 序列化格式
        :return: 成功响应对象
        """
        return cls.build(
            is_success=True,
            datetime_format=datetime_format
        ).set_code(code).set_message(message).set_data(data).set_http_status(http_status)

    # 错误响应入口
    @classmethod
    def error(cls, *,
              code: int = ERROR_CODE.value,
              message: str = ERROR_CODE.phrase,
              data: T = None,
              http_status: Optional[HTTPStatus] = None,
              datetime_format: Optional[str] = None) -> 'R[T]':
        """
        创建错误响应对象（链式调用入口）
        :param code: 响应码
        :param message: 响应消息
        :param data: 响应数据
        :param http_status: HTTP状态码
        :param datetime_format: datetime 序列化格式
        :return: 错误响应对象
        """
        return cls.build(
            is_success=False,
            datetime_format=datetime_format
        ).set_code(code).set_message(message).set_data(data).set_http_status(http_status)

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

    def set_http_status(self,
                        http_status: Optional[HTTPStatus],
                        response: Optional[StatusWritableResponse] = None) -> 'R[T]':
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
    def extract_data(cls,
                     result: Optional['R[T]'],
                     default_value: T) -> T:
        """
        从响应中提取数据，失败时返回默认值
        :param result: 响应对象
        :param default_value: 默认值
        :return: 提取到的数据或默认值
        """
        if result and result.succeed:
            return result.data if result.data is not None else default_value
        return default_value
