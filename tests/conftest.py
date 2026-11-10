"""pytest 公共 fixture。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facerec.auth import AuthService  # noqa: E402
from facerec.cipher import AesFeatureCipher  # noqa: E402
from facerec.config import Settings, default_aes_key  # noqa: E402
from facerec.engine import FaceEngine  # noqa: E402
from facerec.service import FaceRecognitionService  # noqa: E402
from facerec.store import FaceStore  # noqa: E402
from tests.fakes import FakeDetector, FakeEncoder  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        model_path=tmp_path / "model-not-used.pb",
        db_path=tmp_path / "facerec_test.db",
        threshold=0.60,
        timeout_ms=2000,
        max_samples=5,
        aes_key=default_aes_key(),
    )


@pytest.fixture()
def store(settings: Settings) -> FaceStore:
    store = FaceStore(settings.db_path)
    store.init_schema()
    AuthService(store).ensure_seed_accounts()
    return store


@pytest.fixture()
def cipher(settings: Settings) -> AesFeatureCipher:
    return AesFeatureCipher(settings.aes_key, settings.cipher_key_id)


@pytest.fixture()
def fake_engine() -> FaceEngine:
    return FaceEngine(
        detector=FakeDetector(),
        encoder=FakeEncoder(),
        threshold=0.60,
        timeout_ms=2000,
        max_samples=5,
    )


@pytest.fixture()
def service(store: FaceStore, fake_engine: FaceEngine, cipher, settings) -> FaceRecognitionService:
    return FaceRecognitionService(store, fake_engine, cipher, settings)
