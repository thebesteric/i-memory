import math
import regex as re
from typing import List, TypedDict

from src.ops.extract import estimate_tokens


class Chunk(TypedDict):
    text: str
    start: int
    end: int
    tokens: int


def chunk_text(txt: str, target_token: int = 128, overlap: float = 0.1) -> tuple[List[Chunk], int]:
    """
    将长文本智能分块为较小的片段，以适应目标令牌数。
    @param txt: 要分块的输入文本。
    @param target_token: 目标令牌数，默认为 128。
    @param overlap: 重叠比例，默认为 0.1（10%）。
    @return: 分块后的文本片段列表，每个片段包含文本内容、起始位置、结束位置和令牌数。
    该函数首先估计输入文本的令牌数。如果总令牌数小于等于目标令牌数，则返回整个文本作为一个块。
    否则，函数将文本按段落和句子进行分割，逐步构建块，确保每个块的长度接近目标令牌数，并根据重叠比例添加重叠内容以保持上下文连续性。
    这种分块方法有助于在处理长文本时，保持信息的完整性和上下文的连贯性。
    例子：
    txt = "这是一个很长的文本，需要被分块处理。它包含多个段落和句子。"
    chunks = chunk_text(txt, target_token=50, overlap=0.1)
    {'text': '这是一个很长的文本，需要被分块处理。', 'start': 0, 'end': 26, 'tokens': 7}
    {'text': '需要被分块处理。它包含多个段落和句子。', 'start': 20, 'end': 52, 'tokens': 8}
    """

    # 估计总令牌数
    total_token = estimate_tokens(txt)

    # 如果总令牌数小于等于目标令牌数，返回整个文本作为一个块
    if total_token <= target_token:
        return [Chunk(text=txt, start=0, end=len(txt), tokens=total_token)], total_token

    # 动态估算每 token 平均字符数
    sample = txt[:min(1000, len(txt))]
    sample_tokens = estimate_tokens(sample)
    avg_cpt = len(sample) / sample_tokens if sample_tokens else 4

    # 计算每个 chunk 的最大字符数
    max_chunk_chars = target_token * avg_cpt
    # 计算 chunk 之间的重叠字符数，保证上下文连续性
    overlap_chars = math.floor(max_chunk_chars * overlap)

    # 将输入文本按段落（两个及以上换行符）进行初步分割，得到段落列表
    paras = re.split(r"\n\n+", txt)

    # 存储所有分块的列表
    chunks: List[Chunk] = []
    # 当前正在累积的分块内容
    cur_chunk = ""
    # 当前分块的起始位置
    cur_start = 0

    # 遍历每个段落
    for p in paras:
        # 按句子分割为 sentences
        sentences = re.split(r"(?<=[.;!?。；！？])\s*", p)
        for sentence in sentences:
            # 如果累加后长度超过最大 chunk 长度 max_chunk_chars，且当前 cur_chunk 不为空，则把 cur_chunk 作为一个 chunk 存入 chunks
            potential = cur_chunk + (" " if cur_chunk else "") + sentence
            if len(potential) > max_chunk_chars and len(cur_chunk) > 0:
                # 存储当前 chunk
                chunks.append(Chunk(text=cur_chunk, start=cur_start, end=cur_start + len(cur_chunk), tokens=estimate_tokens(cur_chunk)))
                # 为了保证上下文连续性，取 current_chunk 末尾 overlap_chars 个字符（重叠部分）加上当前句子 sentence 作为新 current_chunk
                ovt = cur_chunk[-overlap_chars:] if overlap_chars < len(cur_chunk) else cur_chunk
                cur_chunk = ovt + " " + sentence
                # 整起始位置 cur_start
                cur_start = cur_start + len(cur_chunk) - len(ovt) - 1
            else:
                cur_chunk = potential

    if len(cur_chunk) > 0:
        chunks.append(Chunk(text=cur_chunk, start=cur_start, end=cur_start + len(cur_chunk), tokens=estimate_tokens(cur_chunk)))
    # 返回分块列表和总令牌数
    return chunks, total_token


def agg_vec(vecs: List[List[float]]) -> List[float]:
    n = len(vecs)
    if not n: raise ValueError("no vecs")
    if n == 1: return vecs[0].copy()

    d = len(vecs[0])
    r = [0.0] * d
    for v in vecs:
        for i in range(d):
            r[i] += v[i]

    rc = 1.0 / n
    for i in range(d):
        r[i] *= rc
    return r


def join_chunks(cks: List[Chunk]) -> str:
    return " ".join(c["text"] for c in cks) if cks else ""
