from typing import Any, Mapping

from agile.utils import LogHelper

from domain.memory.models import IMemoryUser
from shared.config.settings import env
from shared.utils.encrypt_utils import EncryptionKeyTool

logger = LogHelper.get_logger(title="[ENCRYPT]")


def encrypt_if_necessary(
        user: IMemoryUser,
        plaintext: str,
        *,
        aad: str | Mapping[str, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | None = None
) -> str:
    """
    内容加密
    :param user: 用户信息
    :param plaintext: 原始数据
    :param aad: 附加数据
    :return: 加密后的数据或原始数据
    """
    encryption_enable = env.ENCRYPTION_ENABLE or False
    if encryption_enable:
        if not user.encryption_key:
            raise ValueError("Missing user encryption key while encryption is enabled")
        return EncryptionKeyTool.encrypt(plaintext=plaintext, key_b64=user.encryption_key, aad=aad)
    return plaintext


def decrypt_if_necessary(
        user: IMemoryUser,
        ciphertext: str,
        *,
        aad: str | Mapping[str, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | None = None
) -> str:
    """
    内容解密
    :param user: 用户信息
    :param ciphertext: 密文或原始数据
    :param aad: 附加数据
    :return: 解密后的数据，若非密文则原样返回
    """
    encryption_enable = env.ENCRYPTION_ENABLE or False
    if not encryption_enable:
        return ciphertext

    if not user.encryption_key:
        raise ValueError("Missing user encryption key while encryption is enabled")

    try:
        return EncryptionKeyTool.decrypt(encrypted_b64=ciphertext, key_b64=user.encryption_key, aad=aad)
    except Exception as exc:
        # 兼容历史明文数据，避免读取链路因解密失败中断。
        logger.warning(f"decrypt_if_necessary fallback to raw text for user={user.id}: {exc}")
        return ciphertext

