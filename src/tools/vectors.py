import json
import struct
import time
import uuid
from typing import List, Union, Any

import numpy as np


def now() -> int:
    return int(time.time() * 1000)

def rid() -> str:
    return str(uuid.uuid4())

def cos_sim(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> float:
    if isinstance(a, list):
        a = np.array(a, dtype=np.float32)
    if isinstance(b, list):
        b = np.array(b, dtype=np.float32)

    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    d = na * nb
    return dot / d if d else 0.0

def j(x: Any) -> str:
    return json.dumps(x)

def p(x: str) -> Any:
    return json.loads(x)

def vec_to_buf(v: List[float]) -> bytes:
    """
    将一个浮点数列表（向量）序列化为二进制字节流（bytes），便于高效存储或网络传输
    """
    return struct.pack(f"{len(v)}f", *v)

def buf_to_vec(buf: bytes) -> List[float]:
    cnt = len(buf) // 4
    return list(struct.unpack(f"{cnt}f", buf))
