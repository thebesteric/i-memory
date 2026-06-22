import base64
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionKeyTool:
    """
    AES-256-GCM 加密工具类。
    """

    AES_256_KEY_BYTES = 32
    IV_LEN = 12
    TAG_LEN = 16

    @classmethod
    def generate_aes_256_gcm_key(cls) -> str:
        """
        生成 AES-256-GCM 所需的 256 位随机密钥。

        :return: Base64 编码的 32 字节密钥字符串。
        """
        key = secrets.token_bytes(cls.AES_256_KEY_BYTES)
        return base64.b64encode(key).decode("ascii")

    @classmethod
    def encrypt(cls, plaintext: str, key_b64: str, aad: str | None = None) -> str:
        """
        使用 AES-256-GCM 加密文本。

        :param plaintext: 待加密的明文字符串（UTF-8 编码）。
        :param key_b64: Base64 编码的 AES-256 密钥（解码后必须是 32 字节）。
        :param aad: 附加认证数据（Additional Authenticated Data），不加密但参与完整性校验；解密时必须完全一致。
        :return: Base64 编码的密文载荷，格式为 IV(12B) + Ciphertext + Tag(16B)。
        """
        key = cls._decode_key(key_b64)
        iv = secrets.token_bytes(cls.IV_LEN)
        aad_bytes = aad.encode("utf-8") if aad is not None else None
        cipher_with_tag = AESGCM(key).encrypt(iv, plaintext.encode("utf-8"), aad_bytes)
        payload = iv + cipher_with_tag
        return base64.b64encode(payload).decode("ascii")

    @classmethod
    def decrypt(cls, encrypted_b64: str, key_b64: str, aad: str | None = None) -> str:
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
        aad_bytes = aad.encode("utf-8") if aad is not None else None
        plaintext = AESGCM(key).decrypt(iv, cipher_with_tag, aad_bytes)
        return plaintext.decode("utf-8")

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
