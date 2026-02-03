import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="jieba")

import re
from typing import List, Set, Dict
import jieba

# 定义同义词组
SYN_GRPS = [
    ["prefer", "like", "love", "enjoy", "favor"],
    ["theme", "mode", "style", "layout"],
    ["meeting", "meet", "session", "call", "sync"],
    ["dark", "night", "black"],
    ["light", "bright", "day"],
    ["user", "person", "people", "customer"],
    ["task", "todo", "job"],
    ["note", "memo", "reminder"],
    ["time", "schedule", "when", "date"],
    ["project", "initiative", "plan"],
    ["issue", "problem", "bug"],
    ["document", "doc", "file"],
    ["question", "query", "ask"],
    # 可扩展中文同义词组，例如：
    ["用户", "人", "客户"],
    ["任务", "工作", "事项"],
    ["问题", "故障", "bug"],
]

CMAP: Dict[str, str] = {}
SLOOK: Dict[str, Set[str]] = {}

for grp in SYN_GRPS:
    can = grp[0]
    sset = set(grp)
    for w in grp:
        CMAP[w] = can
        SLOOK[can] = sset

STEM_RULES = [
    (r"ies$", "y"),
    (r"ing$", ""),
    (r"ers?$", "er"),
    (r"ed$", ""),
    (r"s$", ""),
]

EN_PAT = re.compile(r"[a-zA-Z0-9]+")
ZH_PAT = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    tokens = []
    segs = jieba.lcut(text)
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        # 英文单词
        if EN_PAT.fullmatch(seg):
            tokens.append(seg.lower())
        # 中文词
        elif ZH_PAT.fullmatch(seg):
            tokens.append(seg)
        # 其他（符号、混合等）
        # 可根据需要处理
    return tokens


def stem(tok: str) -> str:
    # 仅对英文做词干化
    if len(tok) <= 3 or not EN_PAT.fullmatch(tok):
        return tok
    for pat, rep in STEM_RULES:
        if re.search(pat, tok):
            st = re.sub(pat, rep, tok)
            if len(st) >= 3:
                return st
    return tok


def canonicalize_token(tok: str) -> str:
    if not tok:
        return ""
    low = tok.lower() if EN_PAT.fullmatch(tok) else tok
    if low in CMAP:
        return CMAP[low]
    st = stem(low) if EN_PAT.fullmatch(tok) else low
    return CMAP.get(st, st)


def canonical_tokens_from_text(text: str) -> List[str]:
    res = []
    for tok in tokenize(text):
        can = canonicalize_token(tok)
        if can and len(can) > 1:
            res.append(can)
    return res


def synonyms_for(tok: str) -> Set[str]:
    can = canonicalize_token(tok)
    return SLOOK.get(can, {can})


def build_search_doc(text: str) -> str:
    can = canonical_tokens_from_text(text)
    exp = set()
    for tok in can:
        exp.add(tok)
        syns = SLOOK.get(tok)
        if syns:
            exp.update(syns)
    return " ".join(exp)


def build_fts_query(text: str) -> str:
    can = canonical_tokens_from_text(text)
    if not can:
        return ""
    uniq = sorted(list(set(t for t in can if len(t) > 1)))
    return " OR ".join(f'"{t}"' for t in uniq)


def canonical_token_set(text: str) -> Set[str]:
    """
    获取文本的标准化 token 集合。
    :param text:
    :return:
    """
    return set(canonical_tokens_from_text(text))
