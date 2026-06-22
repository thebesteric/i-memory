import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="jieba")

import re
from typing import List, Set, Dict
import jieba
import jieba.posseg as pseg

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

STOP_FLAGS = {
    # 核心虚词
    "p", "c", "u", "uj", "um", "uv", "uz",
    # 副词/语气词/状态词/符号
    "d", "dg", "df", "y", "z", "x", "xx",
    # 无实义动词
    "v", "vd", "vshi", "vyou", "vf", "vx", "vg", "vi", "vl",
    # 无实义形容词
    "ag", "al",
    # 代词
    "rr", "rz", "rt"
}


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


def _use_pseg(text: str) -> bool:
    return ZH_PAT.search(text) is not None


def _tokens_with_pseg(text: str) -> List[str]:
    tokens = []
    for word, flag in pseg.cut(text):
        word = word.strip()
        if not word:
            continue
        if EN_PAT.fullmatch(word):
            tokens.append(word.lower())
            continue
        if (ZH_PAT.fullmatch(word) and flag not in STOP_FLAGS) or len(word) > 1:
            tokens.append(word)
    return tokens


def stem(tok: str) -> str:
    # 仅对英文做词干化
    if len(tok) <= 3 or not EN_PAT.fullmatch(tok):
        return tok
    # 处理复数、时态等变化
    for pat, rep in STEM_RULES:
        if re.search(pat, tok):
            st = re.sub(pat, rep, tok)
            if len(st) >= 3:
                return st
    return tok


def canonicalize_token(tok: str) -> str:
    if not tok:
        return ""
    # 英文单词转小写，非英文保持原样
    low = tok.lower() if EN_PAT.fullmatch(tok) else tok
    # 检查规范化后的词汇是否存在于同义词映射表 CMAP 中，仅英文生效
    if low in CMAP:
        return CMAP[low]
    # 词干提取后再映射
    st = stem(low) if EN_PAT.fullmatch(tok) else low
    return CMAP.get(st, st)


def canonical_tokens_from_text(text: str) -> List[str]:
    """
    获取文本的标准化 token 列表（含重复的 token）
    :param text: 输入文本
    :return:
    """
    res = []
    tokens = _tokens_with_pseg(text) if _use_pseg(text) else tokenize(text)
    for tok in tokens:
        can = canonicalize_token(tok)
        if not can:
            continue
        # 英文单字仍过滤，中文保留单字
        if EN_PAT.fullmatch(can) and len(can) <= 1:
            continue
        res.append(can)
    return res


def synonyms_for(tok: str) -> Set[str]:
    can = canonicalize_token(tok)
    return SLOOK.get(can, {can})


def build_search_doc(text: str) -> str:
    cans = canonical_tokens_from_text(text)
    exp = set()
    for tok in cans:
        exp.add(tok)
        syns = SLOOK.get(tok)
        if syns:
            exp.update(syns)
    return " ".join(exp)


def build_fts_query(text: str) -> str:
    can = canonical_tokens_from_text(text)
    if not can:
        return ""
    uniq = sorted(list(set(t for t in can if not (EN_PAT.fullmatch(t) and len(t) <= 1))))
    return " OR ".join(f'"{t}"' for t in uniq)


def canonical_token_set(text: str) -> Set[str]:
    """
    获取文本的标准化 token 集合（不含重复的 token）
    :param text: 输入文本
    :return:
    """
    return set(canonical_tokens_from_text(text))
