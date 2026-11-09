"""配置与图裁剪工具的单元测试。"""
from __future__ import annotations

import base64

import pytest

from facerec.config import Settings, default_aes_key
from scripts.prune_graph import read_varint, tensor_name, write_varint, make_placeholder_node, Node
from facerec.utils import FEATURE_DIM


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert settings.threshold == 0.70
        assert settings.timeout_ms == 2000
        assert settings.max_samples == 5
        assert settings.cipher_key_id == "k1"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FACE_THRESHOLD", "0.55")
        monkeypatch.setenv("FACE_TIMEOUT_MS", "1500")
        monkeypatch.setenv("FACE_DB_PATH", str(tmp_path / "x.db"))
        settings = Settings.from_env()
        assert settings.threshold == 0.55
        assert settings.timeout_ms == 1500
        assert settings.db_path == tmp_path / "x.db"

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("FACE_THRESHOLD", "not-a-number")
        assert Settings.from_env().threshold == 0.70

    def test_public_dict_has_no_secret(self):
        data = Settings().as_public_dict()
        assert "aes_key" not in data
        assert "secret_key" not in data
        assert data["threshold"] == 0.70

    def test_model_version_from_filename(self, tmp_path):
        settings = Settings(model_path=tmp_path / "facenet2rec-inference.pb")
        assert settings.model_version == "facenet2rec-inference"
        assert settings.model_available is False

    def test_default_key_is_32_bytes(self):
        assert len(base64.b64decode(default_aes_key())) == 32


class TestGraphHelpers:
    def test_varint_roundtrip(self):
        for value in (0, 1, 127, 128, 300, 2**31, 2**40):
            encoded = write_varint(value)
            decoded, pos = read_varint(encoded, 0)
            assert decoded == value
            assert pos == len(encoded)

    @pytest.mark.parametrize(
        "ref,expected",
        [
            ("input", ("input", "")),
            ("node:1", ("node", "1")),
            ("^node", ("node", "^")),
            ("a/b/c:0", ("a/b/c", "0")),
        ],
    )
    def test_tensor_name(self, ref, expected):
        assert tensor_name(ref) == expected

    def test_make_placeholder_node(self):
        node = Node(make_placeholder_node("input", [-1, 160, 160, 3]).to_bytes())
        assert node.name == "input"
        assert node.op == "Placeholder"
        assert node.inputs == []
        assert FEATURE_DIM == 512
