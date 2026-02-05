from enum import Enum
from typing import Union, Dict


class TimeUnit(Enum):
    """
    时间单位枚举
    每个成员属性：(单位名称字符串, 转换为秒的乘法因子)
    支持特性：
    1. 单位与秒的互转
    2. 不同单位之间的转换
    3. 字符串解析（支持缩写/单数/复数，如：ns/nanosecond/nanoseconds）
    """
    NANO_SECONDS = "nanoseconds", 1e-9
    MICRO_SECONDS = "microseconds", 1e-6
    MILLI_SECONDS = "milliseconds", 1e-3
    SECONDS = "seconds", 1.0
    MINUTES = "minutes", 60.0
    HOURS = "hours", 3600.0
    DAYS = "days", 86400.0

    def __init__(self, unit_name: str, factor_to_second: float):
        """
        枚举成员初始化方法，绑定自定义属性
        :param unit_name: 单位的标准字符串名称（如：milliseconds）
        :param factor_to_second: 转换为秒的乘法因子
        """
        # 标准单位名称
        self.unit_name = unit_name
        # 私有转换因子，禁止外部直接修改
        self._factor_to_second = factor_to_second

    def _factor_to_seconds(self) -> float:
        """
        【私有方法】返回将此单位的值转换为秒的乘法因子，禁止外部调用
        e.g. 1 millisecond -> 0.001 seconds

        :return: 转秒乘法因子
        """
        return self._factor_to_second

    def to_seconds(self, value: float) -> float:
        """
        转换给定的数值为秒

        :param value: 当前单位的数值（浮点型）
        :return: 转换后的秒数
        """
        return value * self._factor_to_seconds()

    def convert_to(self, value: float, target: Union['TimeUnit', str]) -> float:
        """
        将给定的数值从当前单位转换为目标单位

        :param value: 当前单位的数值
        :param target: 目标单位，可以是 TimeUnit 枚举成员或字符串（如 'ms', 'minutes', 'hour'）
        :return: 转换后的目标单位数值
        :raises TypeError: 当target为非字符串/非TimeUnit类型时
        :raises ValueError: 当字符串无法解析为有效TimeUnit时
        """
        if isinstance(target, str):
            target = self.from_string(target)
        source_seconds = self.to_seconds(value)
        return source_seconds / target._factor_to_seconds()

    @classmethod
    def from_string(cls, name: str) -> 'TimeUnit':
        """
        将常见的字符串表示解析为 TimeUnit，动态生成映射，无需硬编码
        支持规则：
        1. 缩写：ns, us, ms, s, m, h, d
        2. 单数单位名：nanosecond, microsecond, ..., day
        3. 复数单位名：nanoseconds, microseconds, ..., days
        4. 枚举标准unit_name：与成员的unit_name属性完全匹配

        :param name: 时间单位的字符串表示
        :return: 对应的 TimeUnit 枚举成员
        :raises TypeError: 当name非字符串类型时
        :raises ValueError: 当字符串无法识别为有效单位时
        """
        # 严格校验输入类型
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__} instead")

        # 预处理：去除首尾空格并转小写
        s = name.strip().lower()

        # 1. 预设缩写映射规则（使用字符串键，避免不可哈希类型）
        abbr_mapping: Dict[str, TimeUnit] = {}
        abbr_groups = {
            ("ns",): TimeUnit.NANO_SECONDS,
            ("us",): TimeUnit.MICRO_SECONDS,
            ("ms",): TimeUnit.MILLI_SECONDS,
            ("s", "sec"): TimeUnit.SECONDS,
            ("m", "min"): TimeUnit.MINUTES,
            ("h", "hr"): TimeUnit.HOURS,
            ("d", "day"): TimeUnit.DAYS,
        }
        for aliases, unit in abbr_groups.items():
            for alias in aliases:
                abbr_mapping[alias] = unit

        # 2. 先尝试直接匹配标准名或单数名
        for member in cls:
            if s == member.unit_name or s == member.unit_name[:-1]:
                return member
            # 也支持枚举成员名（如 'hours' 或 'HOURS'）
            if s == member.name.lower():
                return member

        # 3. 如果是缩写，使用映射找到对应的成员并返回
        if s in abbr_mapping:
            return abbr_mapping[s]

        # 解析失败，抛出明确的异常提示
        valid_units = ', '.join(sorted(abbr_mapping.keys())) + ' or full names (singular/plural)'
        raise ValueError(f"Unknown time unit: '{name}'. Valid units are: {valid_units}")
