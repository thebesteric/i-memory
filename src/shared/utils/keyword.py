from typing import Set


def compute_keyword_overlap(query_keywords: Set[str], content_keywords: Set[str]) -> float:
    """
    计算查询关键词与内容关键词的重叠度，考虑关键词中下划线的权重加倍
    :param query_keywords: 查询关键词集合
    :param content_keywords: 内容关键词集合
    :return: 重叠度得分，范围在 0~1 之间
    """
    matches = 0.0
    total_weight = 0.0

    for qk in query_keywords:
        # 下划线连接的关键词权重加倍
        w = 2.0 if "_" in qk else 1.0
        if qk in content_keywords:
            matches += w
        total_weight += w

    return matches / total_weight if total_weight > 0 else 0.0


def compute_token_overlap(query_tokens: Set[str], mem_tokens: Set[str]) -> float:
    """
    计算查询 token 与记忆 token 的重叠度
    :param query_tokens: 查询 token 集合
    :param mem_tokens: 记忆 token 集合
    :return: 重叠度得分，范围在 0~1 之间
    """
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(mem_tokens))
    return overlap / len(query_tokens)
