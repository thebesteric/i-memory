import json
import math
from typing import Any, Set

from utils.log_helper import LogHelper

from src.core.constants import HYBRID_PARAMS

logger = LogHelper.get_logger()

# 各评分维度的权重配置（总和 = 1.0）
SCORING_WEIGHTS = {
    "similarity": 0.35,  # 向量相似度维度权重（最高）
    "overlap": 0.20,  # 关键词重叠维度权重
    "waypoint": 0.15,  # 位置/路径点维度权重
    "recency": 0.10,  # 近期度维度权重
    "tag_match": 0.20,  # 标签匹配维度权重
}


async def compute_tag_match_score(mem: dict[str, Any], query_tokens: Set[str]) -> float:
    if not mem or not mem["tags"]:
        return 0.0
    try:
        # 获取记忆的标签字段：["美食", "北京", "旅游"]
        tags = json.loads(mem["tags"])
        if not isinstance(tags, list):
            return 0.0

        # 标准化查询 token 集合为小写
        query_tokens = set(t.lower() for t in query_tokens)
        # 计算标签与查询 token 的匹配分数
        matches = 0
        for tag in tags:
            # 将标签转为字符串
            tl = str(tag).lower()
            # 第一层：标签与查询关键词完全匹配，如：{"北京", "美食", "上海"}
            if tl in query_tokens:
                # 完全匹配权重更高，加 2 分
                matches += 2
            else:
                # 第二层：标签与查询关键词部分包含匹配（模糊匹配）如：{"北京朝阳", "美食探店", "上海旅游"}
                for query_token in query_tokens:
                    if tl in query_token or query_token in tl:
                        # 部分包含匹配权重较低，加 1 分
                        matches += 1
        # 归一化得分（0.0~1.0）
        return min(1.0, matches / max(1, len(tags) * 2))
    except Exception as e:
        logger.warning(f"Error computing tag match score: {e}, Returning 0.0")
        return 0.0


def sigmoid(x: float) -> float:
    """
    Sigmoid 激活函数（非线性变换函数），将输入映射到 (0, 1) 范围内，常用于评分归一化
    :param x: 输入值，x 越大，结果越接近 1；x 越小，结果越接近 0
    :return: 映射后的值，范围在 (0, 1) 之间
    """
    return 1.0 / (1.0 + math.exp(-x))


def boosted_sim(s: float) -> float:
    """
    提升相似度分数的函数，使用指数衰减函数对相似度进行非线性变换
    让「高相似度的特征更突出，低相似度的特征更弱化」，提升得分的区分度（避免不同相似度的特征贡献差异不明显）
    :param s: 相似度分数
    :return: 提升后的相似度分数
    """
    # tau 控制提升的强度，值越大，提升效果越明显
    return 1 - math.exp(-HYBRID_PARAMS["tau"] * s)


def compute_hybrid_score(*, sim: float, tok_ov: float, wp_wt: float, rec_sc: float, kw_score: float = 0, tag_match: float = 0) -> float:
    """
    计算混合评分函数，将多个评分维度结合，计算最终记忆评分
    采用加权求和的方式，将各维度评分乘以对应权重后相加，再通过 Sigmoid 函数归一化
    这样可以综合考虑多个因素，得到更准确的记忆相关性评分
    :param sim: 原始向量相似度，范围 [0,1]
    :param tok_ov: 关键词重叠得分，范围 [0,1]
    :param wp_wt: 位置/路径点权重得分，范围 [0,1]
    :param rec_sc: 近期度得分，范围 [0,1]
    :param kw_score: 关键词额外得分，范围 [0,1]，默认 0
    :param tag_match: 标签匹配得分，范围 [0,1]，默认 0
    :return:
    """
    s_p = boosted_sim(sim)
    raw = (SCORING_WEIGHTS["similarity"] * s_p +
           SCORING_WEIGHTS["overlap"] * tok_ov +
           SCORING_WEIGHTS["waypoint"] * wp_wt +
           SCORING_WEIGHTS["recency"] * rec_sc +
           SCORING_WEIGHTS["tag_match"] * tag_match +
           kw_score)
    return sigmoid(raw)
