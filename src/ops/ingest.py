import json
import time
import uuid
from typing import Dict, Any

from src.core.constants import MemoryConstants
from src.core.db import db
from src.core.dml_ops import dml_ops
from src.memory.hsg import add_hsg_memory
from src.ops.extract import extract_text
from src.utils.log_helper import LogHelper

LARGE_TOKEN_THRESH = MemoryConstants.LARGE_TOKEN_THRESH
SECTION_SIZE = MemoryConstants.SECTION_SIZE

logger = LogHelper.get_logger()


def split_text(t: str, sz: int) -> list[str]:
    if len(t) <= sz:
        return [t]
    secs = []
    paras = t.split("\n\n")
    cur = ""
    for p in paras:
        if len(cur) + len(p) > sz and len(cur) > 0:
            secs.append(cur.strip())
            cur = p
        else:
            cur += ("\n\n" if cur else "") + p

    if cur.strip():
        secs.append(cur.strip())

    return secs


async def mk_root(txt: str, ex: Dict, meta: Dict = None, user_id: str = None) -> str:
    summ = txt[:500] + "..." if len(txt) > 500 else txt
    ctype = ex["metadata"]["content_type"].upper()
    sec_count = int(len(txt) / SECTION_SIZE) + 1
    content = f"[Document: {ctype}]\n\n{summ}\n\n[Full content split across {sec_count} sections]"

    mid = str(uuid.uuid4())
    ts = int(time.time() * 1000)

    try:
        full_meta = meta or {}
        full_meta.update(ex["metadata"])
        full_meta.update({
            "is_root": True,
            "ingestion_strategy": "root-child",
            "ingested_at": ts
        })

        dml_ops.ins_mem(
            id=mid,
            content=content,
            primary_sector="reflective",
            tags=json.dumps([]),
            meta=json.dumps(full_meta, default=str),
            created_at=ts,
            updated_at=ts,
            last_seen_at=ts,
            salience=1.0,
            decay_lambda=0.1,
            segment=1,
            user_id=user_id or "anonymous",
            feedback_score=0
        )
        return mid
    except Exception as e:
        raise e


async def mk_child(txt: str, idx: int, tot: int, rid: str, meta: Dict = None, user_id: str = None) -> str:
    m = meta or {}
    m.update({
        "is_child": True,
        "section_index": idx,
        "total_sections": tot,
        "parent_id": rid
    })
    r = await add_hsg_memory(txt, json.dumps([]), m, user_id)
    return r["id"]


async def link(rid: str, cid: str, idx: int, user_id: str = None):
    ts = int(time.time() * 1000)
    db.execute("INSERT INTO waypoints(src_id,dst_id,user_id,weight,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s)",
               (rid, cid, user_id or "anonymous", 1.0, ts, ts))
    db.commit()


async def ingest_document(*, content_type: str, data: Any, meta: Dict = None, cfg: Dict = None, user_id: str = None, tags: list = None) -> Dict[str, Any]:
    # 长语句阈值
    large_token_thresh = cfg.get("lg_thresh", LARGE_TOKEN_THRESH) if cfg else LARGE_TOKEN_THRESH
    # 文本分段大小阈值
    section_size = cfg.get("sec_sz", SECTION_SIZE) if cfg else SECTION_SIZE

    # 解析文本和元数据（使用 jieba 分词）
    ex_dict = await extract_text(content_type, data)
    text = ex_dict["text"]
    ex_meta = ex_dict["metadata"]
    est_tok = ex_meta["estimated_tokens"]

    # 决定是否使用 root-child 结构，force_root 表示强制使用
    # 否则就根据估计的 token 数量决定
    use_rc = (cfg and cfg.get("force_root")) or est_tok > large_token_thresh
    tags_json = json.dumps(tags or [])

    # single 存储模式
    if not use_rc:
        m = meta or {}
        # 合并元数据
        m.update(ex_meta)
        m.update({"ingestion_strategy": "single", "ingested_at": int(time.time() * 1000)})
        # 将记忆写入数据库
        r = await add_hsg_memory(text, tags_json, m, user_id)
        return {
            "root_memory_id": r["id"],
            "child_count": 0,
            "total_tokens": est_tok,
            "strategy": "single",
            "extraction": ex_meta
        }

    # root-child 存储模式
    secs = split_text(text, section_size)
    logger.info(f"[INGEST] Splitting into {len(secs)} sections")

    cids = []
    try:
        rid_val = await mk_root(text, ex_dict, meta, user_id)
        for i, s in enumerate(secs):
            cid = await mk_child(s, i, len(secs), rid_val, meta, user_id)
            cids.append(cid)
            await link(rid_val, cid, i, user_id)

        return {
            "root_memory_id": rid_val,
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
