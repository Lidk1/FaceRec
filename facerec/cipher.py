"""人脸特征向量加密（AES-256-GCM）。

设计约束：数据库只允许保存密文，任何接口、日志、导出文件都不得出现明文
特征向量。密钥支持版本号，便于平滑换钥：

    token = "v1:<key_id>:<base64(nonce(12) || ciphertext || tag)>"

AAD 绑定版本、维度与数据类型，防止密文被跨版本、跨维度重放。
"""
from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .utils import FEATURE_DIM, ensure_feature_dim, l2_normalize

TOKEN_VERSION = "v1"
_NONCE_BYTES = 12
_KEY_BYTES = 32


class CipherError(ValueError):
    """密文损坏、密钥不匹配或格式非法。"""


class FeatureCipher(ABC):
    """特征向量加解密接口。"""

    @abstractmethod
    def encrypt(self, vector) -> str: ...

    @abstractmethod
    def decrypt(self, token: str) -> np.ndarray: ...

    @abstractmethod
    def fingerprint(self) -> str:
        """密钥指纹（不可逆），用于识别密文所属密钥版本。"""


def _aad(key_id: str, dim: int) -> bytes:
    return f"{TOKEN_VERSION}|{key_id}|{dim}|float32".encode("utf-8")


class AesFeatureCipher(FeatureCipher):
    """AES-256-GCM 实现：一次一随机 nonce，密文自带完整性校验。"""

    def __init__(self, key_b64: str, key_id: str = "k1", dim: int = FEATURE_DIM):
        try:
            key = base64.b64decode(key_b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise CipherError("AES 密钥不是合法的 Base64 字符串") from exc
        if len(key) != _KEY_BYTES:
            raise CipherError(f"AES-256 密钥长度应为 {_KEY_BYTES} 字节，实际 {len(key)} 字节")
        self._aead = AESGCM(key)
        self._key_id = key_id or "k1"
        self._dim = dim

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def dim(self) -> int:
        return self._dim

    def encrypt(self, vector) -> str:
        arr = ensure_feature_dim(l2_normalize(vector), self._dim)
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = self._aead.encrypt(nonce, arr.tobytes(order="C"), _aad(self._key_id, self._dim))
        payload = base64.b64encode(nonce + ciphertext).decode("ascii")
        return f"{TOKEN_VERSION}:{self._key_id}:{payload}"

    def decrypt(self, token: str) -> np.ndarray:
        if not isinstance(token, str) or token.count(":") != 2:
            raise CipherError("特征密文格式非法")
        version, key_id, payload = token.split(":")
        if version != TOKEN_VERSION:
            raise CipherError(f"不支持的特征密文版本: {version}")
        if key_id != self._key_id:
            raise CipherError(f"密钥版本不匹配: 密文={key_id} 当前={self._key_id}")
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise CipherError("特征密文 Base64 解码失败") from exc
        if len(raw) <= _NONCE_BYTES:
            raise CipherError("特征密文长度非法")
        nonce, ciphertext = raw[:_NONCE_BYTES], raw[_NONCE_BYTES:]
        try:
            plaintext = self._aead.decrypt(nonce, ciphertext, _aad(key_id, self._dim))
        except InvalidTag as exc:
            raise CipherError("特征密文校验失败（密钥错误或被篡改）") from exc
        expected = self._dim * 4
        if len(plaintext) != expected:
            raise CipherError(f"特征明文长度异常: {len(plaintext)} != {expected}")
        arr = np.frombuffer(plaintext, dtype=np.float32).copy()
        return arr

    def fingerprint(self) -> str:
        # 仅用于展示/排查，不泄露密钥本身
        return f"{TOKEN_VERSION}:{self._key_id}"


def mask_cipher(token: str, keep: int = 12) -> str:
    """日志脱敏：只保留前缀。"""
    if not token:
        return ""
    return token[:keep] + "***"
