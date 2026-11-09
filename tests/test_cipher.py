"""人脸特征加密（AES-256-GCM）单元测试。"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from facerec.cipher import AesFeatureCipher, CipherError, mask_cipher
from facerec.config import default_aes_key
from facerec.utils import l2_normalize
from tests.fakes import make_vector


@pytest.fixture()
def cipher() -> AesFeatureCipher:
    return AesFeatureCipher(default_aes_key())


class TestRoundTrip:
    def test_encrypt_decrypt(self, cipher):
        vec = make_vector(42)
        token = cipher.encrypt(vec)
        restored = cipher.decrypt(token)
        assert restored.shape == (512,)
        assert np.allclose(restored, vec, atol=1e-6)

    def test_ciphertext_is_text_and_hides_vector(self, cipher):
        vec = make_vector(7)
        token = cipher.encrypt(vec)
        assert token.startswith("v1:k1:")
        assert isinstance(token, str)
        # 密文里不得出现明文浮点数值
        assert str(float(vec[0])) not in token
        assert len(token) > 600

    def test_same_vector_encrypts_differently(self, cipher):
        vec = make_vector(9)
        assert cipher.encrypt(vec) != cipher.encrypt(vec)

    def test_accepts_unnormalized_input(self, cipher):
        vec = make_vector(11) * 7.0
        restored = cipher.decrypt(cipher.encrypt(vec))
        assert np.isclose(np.linalg.norm(restored), 1.0, atol=1e-5)

    def test_accepts_2d_input(self, cipher):
        vec = make_vector(12).reshape(1, 512)
        assert cipher.decrypt(cipher.encrypt(vec)).shape == (512,)

    def test_wrong_dimension_rejected(self, cipher):
        with pytest.raises(ValueError):
            cipher.encrypt(np.zeros(128, dtype=np.float32))


class TestTamperAndKey:
    def test_wrong_key_fails(self):
        vec = make_vector(3)
        token = AesFeatureCipher(default_aes_key()).encrypt(vec)
        other = AesFeatureCipher(base64.b64encode(b"\x01" * 32).decode())
        with pytest.raises(CipherError):
            other.decrypt(token)

    def test_tampered_ciphertext_fails(self, cipher):
        token = cipher.encrypt(make_vector(4))
        head, payload = token.rsplit(":", 1)
        raw = bytearray(base64.b64decode(payload))
        raw[-1] ^= 0x01
        tampered = f"{head}:{base64.b64encode(bytes(raw)).decode()}"
        with pytest.raises(CipherError):
            cipher.decrypt(tampered)

    def test_truncated_token_fails(self, cipher):
        token = cipher.encrypt(make_vector(5))
        with pytest.raises(CipherError):
            cipher.decrypt(token[:40])

    def test_unknown_version_fails(self, cipher):
        token = cipher.encrypt(make_vector(6))
        with pytest.raises(CipherError):
            cipher.decrypt(token.replace("v1:", "v9:", 1))

    def test_key_id_mismatch_fails(self):
        vec = make_vector(2)
        token = AesFeatureCipher(default_aes_key(), key_id="k9").encrypt(vec)
        with pytest.raises(CipherError):
            AesFeatureCipher(default_aes_key(), key_id="k1").decrypt(token)

    def test_invalid_key_length(self):
        with pytest.raises(CipherError):
            AesFeatureCipher(base64.b64encode(b"short").decode())

    def test_invalid_base64_key(self):
        with pytest.raises(CipherError):
            AesFeatureCipher("not-base64!!")

    def test_fingerprint_has_no_secret(self, cipher):
        fp = cipher.fingerprint()
        assert fp == "v1:k1"
        assert default_aes_key() not in fp


class TestMask:
    def test_mask(self):
        assert mask_cipher("v1:k1:abcdefghijklmn", 12).endswith("***")

    def test_mask_empty(self):
        assert mask_cipher("") == ""
