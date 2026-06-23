import base64
import json
import secrets
from typing import Any, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionKeyTool:
    """
    AES-256-GCM 加密工具类。
    """

    AES_256_KEY_BYTES = 32
    IV_LEN = 12
    TAG_LEN = 16
    _JSON_PRIMITIVE_TYPES = (str, int, float, bool, type(None))

    @classmethod
    def generate_aes_256_gcm_key(cls) -> str:
        """
        生成 AES-256-GCM 所需的 256 位随机密钥。

        :return: Base64 编码的 32 字节密钥字符串。
        """
        key = secrets.token_bytes(cls.AES_256_KEY_BYTES)
        return base64.b64encode(key).decode("ascii")

    @classmethod
    def encrypt(
            cls,
            plaintext: str,
            key_b64: str,
            aad: str | Mapping[str, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | None = None
    ) -> str:
        """
        使用 AES-256-GCM 加密文本。

        :param plaintext: 待加密的明文字符串（UTF-8 编码）。
        :param key_b64: Base64 编码的 AES-256 密钥（解码后必须是 32 字节）。
        :param aad: 附加认证数据（Additional Authenticated Data），不加密但参与完整性校验；解密时必须完全一致。
        :return: Base64 编码的密文载荷，格式为 IV(12B) + Ciphertext + Tag(16B)。
        """
        key = cls._decode_key(key_b64)
        iv = secrets.token_bytes(cls.IV_LEN)
        aad_bytes = cls._normalize_aad(aad)
        cipher_with_tag = AESGCM(key).encrypt(iv, plaintext.encode("utf-8"), aad_bytes)
        payload = iv + cipher_with_tag
        return base64.b64encode(payload).decode("ascii")

    @classmethod
    def decrypt(
            cls,
            encrypted_b64: str,
            key_b64: str,
            aad: str | Mapping[str, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | None = None
    ) -> str:
        """
        使用 AES-256-GCM 解密文本。

        :param encrypted_b64: Base64 编码的密文载荷，格式为 IV(12B) + Ciphertext + Tag(16B)。
        :param key_b64: Base64 编码的 AES-256 密钥（解码后必须是 32 字节）。
        :param aad: 附加认证数据（AAD），必须与加密时一致，否则认证失败。
        :return: 解密后的明文字符串（UTF-8）。
        """
        key = cls._decode_key(key_b64)
        payload = base64.b64decode(encrypted_b64, validate=True)
        min_payload_len = cls.IV_LEN + cls.TAG_LEN
        if len(payload) < min_payload_len:
            raise ValueError("Invalid encrypted payload length for AES-GCM")
        iv = payload[:cls.IV_LEN]
        cipher_with_tag = payload[cls.IV_LEN:]
        aad_bytes = cls._normalize_aad(aad)
        plaintext = AESGCM(key).decrypt(iv, cipher_with_tag, aad_bytes)
        return plaintext.decode("utf-8")

    @classmethod
    def _normalize_aad(
            cls,
            aad: str | Mapping[str, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | None
    ) -> bytes | None:
        """
        将 AAD 标准化为字节串，保证常见容器入参的序列化稳定

        :param aad: 附加认证数据
        :return: 标准化后的字节串
        """
        if aad is None:
            return None
        if isinstance(aad, str):
            return aad.encode("utf-8")
        try:
            normalized_aad = cls._canonicalize_aad(aad)
            # sort_keys + compact separators 保证同一结构的编码结果稳定。
            return json.dumps(normalized_aad, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "AAD must be JSON-compatible (str/dict/list/tuple/set and nested primitives)"
            ) from exc

    @classmethod
    def _canonicalize_aad(cls, value: Any) -> Any:
        """
        递归规范化 AAD -> dict 按 key 排序，set/frozenset 转有序列表

        :param value: 待规范化的 AAD 值
        :return: 规范化后的 AAD 值
        """
        if isinstance(value, cls._JSON_PRIMITIVE_TYPES):
            return value

        if isinstance(value, Mapping):
            return {str(k): cls._canonicalize_aad(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [cls._canonicalize_aad(item) for item in value]

        if isinstance(value, (set, frozenset)):
            normalized_items = [cls._canonicalize_aad(item) for item in value]
            # set 无序，按稳定 JSON 文本排序避免序列化抖动。
            return sorted(
                normalized_items,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            )

        raise TypeError(f"Unsupported AAD value type: {type(value)!r}")

    @classmethod
    def _decode_key(cls, key_b64: str) -> bytes:
        """
        解码并校验 AES-256 密钥长度。

        :param key_b64: Base64 编码密钥字符串。
        :return: 原始二进制密钥。
        """
        key = base64.b64decode(key_b64, validate=True)
        if len(key) != cls.AES_256_KEY_BYTES:
            raise ValueError("Invalid AES-256 key length")
        return key
