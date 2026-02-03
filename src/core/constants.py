import re
from typing import TypedDict, List, Dict


class MemoryConstants:
    """
    记忆相关的常量配置
    """
    # 长语句阈值
    LARGE_TOKEN_THRESH = 8000
    # 文本分段大小阈值
    SECTION_SIZE = 3000