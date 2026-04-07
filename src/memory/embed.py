import time
from typing import List, Dict, Optional, Any

import numpy as np

from src.core.components import get_embed_model
from src.core.mem_ops import mem_ops
from src.tools.chunking import Chunk


async def embed_batch(txt: str, sectors: List[str]) -> list[list[float]]:
    # 去重并保持顺序，避免重复扇区造成重复嵌入请求
    unique_sectors = list(dict.fromkeys(sectors))
    sector_texts = [f"[sector:{sector}]\n{txt}" for sector in unique_sectors]
    return await get_embed_model().embed_batch(sector_texts)


async def embed(txt: str, sector: Optional[str] = None) -> List[float]:
    """
    根据配置的嵌入提供者生成文本 txt 的向量
    @param txt: 待嵌入的文本
    @return: 生成的向量列表
    """
    return (await embed_batch(txt, [sector] if sector else []))[0]


async def embed_multi_sector(uid: str, txt: str, secs: List[str], chunks: Optional[List[Chunk]] = None) -> List[Dict[str, Any]]:
    """
    用于对同一文本在多个 sector（语义分区）下分别生成向量嵌入，并记录日志
    @param uid: 任务唯一标识
    @param txt: 待嵌入的文本
    @param secs: 需要生成嵌入的 sector 列表
    @param chunks: 可选的文本块列表
    @return: 包含每个 sector 的名称、生成的向量和向量维度的列表
    """
    # 日志记录（开始）
    mem_ops.ins_log(id=uid, model="multi-sector", status="pending", ts=int(time.time() * 1000), err=None)

    res = []
    try:
        # 遍历传入的 sector 列表
        for s in secs:
            # 调用 embed 对文本 txt 生成向量
            v = await embed(txt, sector=s)
            # 将 sector 名称、生成的向量和向量维度加入结果列表
            res.append({"sector": s, "vector": v, "dim": len(v)})

        # 日志记录（完成）
        mem_ops.upd_log(id=uid, status="completed", err=None)
        return res
    except Exception as e:
        # 日志记录（失败）
        mem_ops.upd_log(id=uid, status="failed", err=str(e))
        raise e


def calc_mean_vec(emb_res: List[Dict[str, Any]], all_sectors: List[str]) -> List[float]:
    """
    计算多个 sector 嵌入向量（emb_res）的均值向量（mean vector），作为整体语义特征
    @param emb_res: 包含各 sector 向量的列表
    @param all_sectors: 所有 sector 的名称列表
    @return: 计算得到的均值向量列表
    """
    if not emb_res: return []
    # 取第一个向量的维度作为均值向量的维度
    d = emb_res[0]["dim"]
    # 初始化全零向量 mean
    mean = np.zeros(d, dtype=np.float32)
    for r in emb_res:
        # 遍历所有 sector 的向量，累加到 mean
        mean += np.array(r["vector"], dtype=np.float32)
    # 累加后除以向量数，得到均值向量
    mean /= len(emb_res)
    # 返回均值向量的列表形式
    return mean.tolist()
