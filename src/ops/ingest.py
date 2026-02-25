import datetime
import json
import re
import time
import uuid
from typing import Dict, Any

from src.core.db import get_db
from src.core.dml_ops import dml_ops
from src.core.waypoints import Waypoints
from src.memory.hsg import add_hsg_memory
from src.memory.models.memory_cfg import IMemoryConfig
from src.ops.extract import extract_text
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()
db = get_db()


def split_text(t: str, sz: int) -> list[str]:
    """
    两级分割策略：
    1. 首先按段落 (\n\n) 分割，尊重文本结构
    2. 对超过 sz 的段落，继续按句子分割，最后按固定长度分割
    """
    if len(t) <= sz:
        return [t]

    def split_long_para(para: str, size: int) -> list[str]:
        """对超长段落进行进一步分割"""
        if len(para) <= size:
            return [para]

        result = []
        # 先尝试按句号、问号、感叹号、冒号分割
        sentences = re.split(r'([.!?;。！？；\n])', para)

        cur = ""
        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue

            # 检查是否为单个标点符号
            is_punctuation = sent.strip() in '.!?;。！？；'

            # 当前累积长度 + 新句子长度 > size，且已有内容
            if len(cur) + len(sent) > size and cur.strip():
                # 如果当前片段是标点符号，强制附加到前面的内容
                if is_punctuation:
                    result.append(cur + sent)
                    cur = ""
                else:
                    result.append(cur.strip())
                    cur = sent
            else:
                cur += sent

        # 处理剩余的累积内容
        if cur.strip():
            # 如果剩余部分仍然超过 size，进行固定长度切分
            if len(cur) > size:
                # 按固定长度分割
                for j in range(0, len(cur), size):
                    chunk = cur[j:j + size].strip()
                    if chunk:
                        result.append(chunk)
            else:
                result.append(cur.strip())

        # 过滤掉仅包含标点符号的段落
        result = [r for r in result if r.strip() and r.strip() not in '.!?;。！？；']

        return result if result else [para.strip()]

    secs = []
    # 第一级：按双换行拆分为段落
    paras = t.split("\n\n")
    cur = ""

    for p in paras:
        # 跳过空段落
        if not p.strip():
            continue

        # 如果当前累积 + 新段落 > sz 且已有内容，先保存当前累积
        if len(cur) + len(p) > sz and cur.strip():
            secs.append(cur.strip())
            cur = ""

        # 如果新段落本身就超过 sz，进行进一步分割
        if len(p) > sz:
            # 先处理已累积的内容
            if cur.strip():
                secs.append(cur.strip())
                cur = ""
            # 对超长段落进行二级分割
            para_secs = split_long_para(p.strip(), sz)
            secs.extend(para_secs)
        else:
            # 正常段落，累积
            cur += ("\n\n" if cur else "") + p

    # 处理最后剩余的累积内容
    if cur.strip():
        cur_stripped = cur.strip()
        # 如果最后累积的内容仍超过 sz，需要进一步分割
        if len(cur_stripped) > sz:
            para_secs = split_long_para(cur_stripped, sz)
            secs.extend(para_secs)
        else:
            secs.append(cur_stripped)

    # 过滤掉仅包含标点符号的段落
    secs = [s for s in secs if s.strip() and s.strip() not in '.!?;。！？；']

    return secs


async def mk_root(txt: str, cfg: IMemoryConfig, ex_dict: Dict, sec_count: int, meta: Dict[str, Any] = None, user_id: str = None) -> str:
    """
    创建根记忆
    说明：根记忆包含文档摘要和元数据，作为子记忆的父节点
    逻辑：
    1. 生成摘要（截取前 N 字符）
    2. 格式化根记忆内容，包含文档类型和摘要
    :param txt: 文本完整内容
    :param cfg: 配置项
    :param ex_dict: 扩展字典
    :param sec_count: 文本切分后的数量
    :param meta: 元数据
    :param user_id: 用户 ID
    :return:
    """
    metadata = ex_dict["metadata"]
    # 生成摘要作为根记忆的内容
    summary = txt[:cfg.summary_length] + "..." if len(txt) > cfg.summary_length else txt
    # 文档类型
    ctype = metadata["content_type"].upper()
    # 根记忆内容格式化
    content = f"[Document: {ctype}]\n\n{summary}\n\n[Full content split across {sec_count} sections]"

    mid = str(uuid.uuid4())
    now = datetime.datetime.now()

    try:
        full_meta = meta or {}
        full_meta.update(metadata)
        full_meta.update({
            "is_root": True,
            "ingestion_strategy": "root-child",
            "ingested_at": now.strftime("%Y-%m-%d %H:%M:%S")
        })

        dml_ops.ins_mem(
            id=mid,
            content=content,
            primary_sector="reflective",
            sectors=json.dumps([]),
            tags=json.dumps([]),
            meta=json.dumps(full_meta, default=str),
            created_at=now,
            updated_at=now,
            last_seen_at=now,
            salience=1.0,
            decay_lambda=0.1,
            segment=1,
            user_id=user_id,
            feedback_score=0
        )
        return mid
    except Exception as e:
        raise e


async def mk_child(sec_txt: str, idx: int, sec_size: int, root_id: str, meta: Dict = None, user_id: str = None) -> str:
    """
    创建子记忆
    :param sec_txt: 部分内容
    :param idx: 索引
    :param sec_size: 部门内容长度
    :param root_id: 根 ID
    :param meta: 元数据
    :param user_id: 用户 ID
    :return:
    """
    m = meta or {}
    m.update({
        "is_root": False,
        "is_child": True,
        "section_index": idx,
        "total_sections": sec_size,
        "root_id": root_id
    })
    r = await add_hsg_memory(sec_txt, [], m, user_id)
    return r["id"]


async def ingest_document(*,
                          content_type: str,
                          user_id: str = None,
                          data: Any,
                          meta: Dict[str, Any] = None,
                          cfg: IMemoryConfig = None,
                          tags: list[str] = None) -> Dict[str, Any]:
    # 使用默认配置（如果未提供）
    if not cfg:
        cfg = IMemoryConfig.create_default()

    # 解析文本和元数据（使用 jieba 分词）
    ex_dict = await extract_text(content_type, data)
    text = ex_dict["text"]
    ex_meta = ex_dict["metadata"]
    est_tok = ex_meta["estimated_tokens"]

    # 决定是否使用 root-child 结构，force_root 表示强制使用
    # 否则就根据估计的 token 数量决定
    use_rc = cfg.force_root or (est_tok > cfg.large_token_thresh)

    # Single 存储模式
    if not use_rc:
        now = datetime.datetime.now()
        m = meta or {}
        # 合并元数据
        m.update(ex_meta)
        m.update({"ingestion_strategy": "single", "ingested_at": now.strftime("%Y-%m-%d %H:%M:%S")})
        # 将记忆写入数据库
        r = await add_hsg_memory(text, tags, m, user_id)
        return {
            "root_memory_id": r["id"],
            "child_count": 0,
            "total_tokens": est_tok,
            "strategy": "single",
            "extraction": ex_meta
        }

    # Parent-Child 存储模式
    secs = split_text(text, cfg.section_length)
    logger.info(f"[INGEST] Splitting into {len(secs)} sections")

    child_ids = []
    try:
        # 创建 Waypoints 实例
        waypoints = Waypoints()
        # 创建根记忆 (传入实际分段数)
        root_id = await mk_root(text, cfg, ex_dict, len(secs), meta, user_id)
        # 创建子记忆并建立链接
        for i, s in enumerate(secs):
            child_id = await mk_child(s, i, len(secs), root_id, meta, user_id)
            child_ids.append(child_id)
            await waypoints.link(root_id, child_id, i, user_id)

        return {
            "root_memory_id": root_id,
            "child_count": len(secs),
            "total_tokens": est_tok,
            "strategy": "root-child",
            "extraction": ex_meta
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"[INGEST] Failed: {e}")
        raise e
