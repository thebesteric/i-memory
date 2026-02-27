from enum import Enum

from utils.time_unit import TimeUnit

# 记忆缓存时间
CACHE_TTL = 60
# 记忆缓存时间单位
CACHE_TTL_TIME_UNIT = TimeUnit.SECONDS
# 记忆缓存大小
CACHE_SIZE = 2000

# 记忆扇区关联度定义
SECTOR_RELATIONSHIPS = {
    "semantic": {"procedural": 0.8, "episodic": 0.6, "reflective": 0.7, "emotional": 0.4},
    "procedural": {"semantic": 0.8, "episodic": 0.6, "reflective": 0.6, "emotional": 0.3},
    "episodic": {"reflective": 0.8, "semantic": 0.6, "procedural": 0.6, "emotional": 0.7},
    "reflective": {"episodic": 0.8, "semantic": 0.7, "procedural": 0.6, "emotional": 0.6},
    "emotional": {"episodic": 0.7, "reflective": 0.6, "semantic": 0.4, "procedural": 0.3},
}

# 记忆衰减混合模型参数
HYBRID_PARAMS = {
    "tau": 3.0,  # 时间常数，控制衰减的陡峭程度。值越大，衰减越缓慢
    "beta": 2.0,  # 非线性因子，调整衰减曲线的形状，增强衰减效果
    "eta": 0.1,  # 衰减基础系数，控制整体衰减速率的幅度
    "gamma": 0.2,  # 平台系数，防止记忆完全消失，保留最小记忆强度
    "alpha_reinforce": 0.08,  # 记忆强化系数，每次访问/使用时增加的记忆强度
    "t_days": 7.0,  # 衰减时间单位（天）
    "t_max_days": 60.0,  # 最大衰减时间（天）
    "tau_hours": 1.0,  # 衰减时间单位（小时）
    "epsilon": 1e-8,  # 防止除零的极小值
}


class ModelProvider(Enum):
    """
    模型提供商枚举
    """
    OPENAI = "openai"
    GEMINI = "gemini"
    DASHSCOPE = "dashscope"


class VectorStoreProvider(Enum):
    """
    向量存储后端枚举
    """
    POSTGRES = "postgres"
    REDIS = "redis"
    VALKEY = "valkey"
