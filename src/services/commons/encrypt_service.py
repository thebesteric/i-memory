from typing import Any, Mapping

from agile.utils import LogHelper

from shared.config.settings import env
from shared.utils.encrypt_utils import EncryptionKeyTool

logger = LogHelper.get_logger(title="[ENCRYPT]")


def encryption_enabled() -> bool:
    """
    统一的加密开关判断
    """
    return bool(env.ENCRYPTION_ENABLE)


def encrypt_if_necessary(
        value: Any,
        *,
        key_b64: str | None,
        aad: str | Mapping[str, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | None = None,
) -> Any:
    """
    通用加密入口（基于原始 key）。
    - 仅在开启加密、且 value 为非空字符串时执行。
    - 若 value 已经是可解密密文，直接返回，避免重复加密。
    - 异常时降级返回原文，保证调用链可用。
    """
    if not encryption_enabled() or not isinstance(value, str) or not value:
        return value
    if not key_b64:
        return value

    # 调用方可能已传入密文：先尝试解密，成功则视为“已加密”。
    try:
        EncryptionKeyTool.decrypt(encrypted_b64=value, key_b64=key_b64, aad=aad)
        return value
    except Exception as exc:
        logger.debug("Value is not decryptable, will encrypt: %s", exc)
        pass

    try:
        return EncryptionKeyTool.encrypt(plaintext=value, key_b64=key_b64, aad=aad)
    except Exception as exc:
        logger.warning("Encrypt text fallback to raw text: %s", exc)
        return value


def decrypt_if_necessary(
        value: Any,
        *,
        key_b64: str | None,
        aad: str | Mapping[str, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | None = None,
) -> Any:
    """
    通用解密入口（基于原始 key）。
    - 仅在开启加密、且 value 为非空字符串时执行。
    - 对历史明文或无效密文保持原样返回，避免读取流程中断。
    """
    if not encryption_enabled() or not isinstance(value, str) or not value:
        return value
    if not key_b64:
        return value

    try:
        return EncryptionKeyTool.decrypt(encrypted_b64=value, key_b64=key_b64, aad=aad)
    except Exception as exc:
        logger.warning("Decrypt text fallback to raw text: %s", exc)
        return value
