"""人脸识别服务层单元测试（真实 SQLite + 真实加密，模型用假编码器）。"""
from __future__ import annotations

import numpy as np
import pytest

from facerec.detector import FaceBox
from facerec.engine import RESULT_LOW_SIMILARITY, RESULT_SUCCESS, FaceEngine
from facerec.errors import (
    DuplicateUserError,
    LowQualityError,
    UserNotFoundError,
    ValidateError,
)
from facerec.service import FaceRecognitionService
from tests.fakes import FakeDetector, FakeEncoder, blank_image, make_vector


def enroll(service, images=None, user_no="2023010101", name="韩萧", **kwargs):
    if images is None:
        images = [blank_image(240, 240)]
    return service.enroll(user_no, name, images, dept="计算机学院", **kwargs)


class TestEnroll:
    def test_enroll_success(self, service):
        summary = enroll(service, [blank_image(240, 240)] * 2)
        assert summary.user_id > 0
        assert summary.samples == 2
        assert summary.user_no == "2023010101"
        assert summary.avg_quality > 0
        user = service.store.get_user(summary.user_id)
        assert user.feature_count == 2

    def test_features_are_encrypted(self, service, cipher):
        summary = enroll(service)
        rows = service.store.list_features(summary.user_id)
        assert len(rows) == 1
        token = rows[0].feature_cipher
        assert token.startswith("v1:k1:")
        # 明文向量不得出现在密文中
        vector = cipher.decrypt(token)
        assert vector.shape == (512,)
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)

    def test_duplicate_user_rejected(self, service):
        enroll(service)
        with pytest.raises(DuplicateUserError):
            enroll(service)

    def test_replace_samples_keeps_user(self, service):
        first = enroll(service)
        second = enroll(service, [blank_image(240, 240)] * 3, replace=True)
        assert second.user_id == first.user_id
        assert service.store.get_user(first.user_id).feature_count == 3

    @pytest.mark.parametrize(
        "user_no,name,message",
        [
            ("x", "韩萧", "学号"),
            ("2023010101", "", "姓名"),
        ],
    )
    def test_invalid_profile(self, service, user_no, name, message):
        with pytest.raises(ValidateError) as exc:
            enroll(service, user_no=user_no, name=name)
        assert message in exc.value.msg

    def test_no_images(self, service):
        with pytest.raises(ValidateError):
            enroll(service, images=[])

    def test_too_many_images(self, service, settings):
        images = [blank_image()] * (settings.max_samples * 2 + 1)
        with pytest.raises(ValidateError):
            enroll(service, images=images)

    def test_all_frames_invalid(self, service, fake_engine):
        fake_engine.detector = FakeDetector([])
        with pytest.raises(LowQualityError) as exc:
            enroll(service)
        assert "人脸样本全部无效" in exc.value.msg

    def test_low_quality_rejected(self, store, cipher, settings):
        engine = FaceEngine(
            FakeDetector([FaceBox(0, 0, 20, 20)]),
            FakeEncoder(make_vector(1)),
            threshold=0.6,
        )
        service = FaceRecognitionService(store, engine, cipher, settings)
        with pytest.raises(LowQualityError) as exc:
            enroll(service, [blank_image(400, 400)])
        assert "质量过低" in exc.value.msg

    def test_partial_frames_rejected_but_enrolled(self, service):
        # 第一帧多人（被拒），第二帧单人（成功入库）
        service.engine.detector = FakeDetector(
            frames=[
                [FaceBox(0, 0, 50, 50), FaceBox(100, 100, 60, 60)],
                [FaceBox(0, 0, 120, 120)],
            ]
        )
        summary = service.enroll("2023010109", "混合样本", [blank_image(240, 240)] * 2)
        assert summary.samples == 1
        assert len(summary.rejected) == 1
        assert summary.rejected[0]["code"] == 1002


class TestRecognize:
    def test_self_recognition_success(self, service):
        vector = make_vector(77)
        service.engine.encoder.vector = vector
        enroll(service)
        result = service.recognize(blank_image(240, 240), device_id="T001")
        assert result.result_type == RESULT_SUCCESS
        assert result.user_no == "2023010101"
        assert result.similarity == pytest.approx(1.0, abs=1e-5)
        records = service.store.list_recognize_logs()
        assert len(records) == 1
        assert records[0]["device_id"] == "T001"

    def test_unknown_face_low_similarity(self, service):
        service.engine.encoder.vector = make_vector(1)
        enroll(service)
        service.engine.encoder.vector = make_vector(500)
        result = service.recognize(blank_image(240, 240))
        assert result.result_type == RESULT_LOW_SIMILARITY
        assert result.user_id is None
        assert len(service.store.list_recognize_logs()) == 1

    def test_recognition_after_feature_delete(self, service):
        enroll(service)
        user = service.store.get_user_by_no("2023010101")
        service.delete_features(user.user_id)
        result = service.recognize(blank_image(240, 240))
        assert result.result_type == RESULT_LOW_SIMILARITY
        assert "特征库为空" in result.message

    def test_verify_same_person(self, service):
        data = service.verify(blank_image(240, 240), blank_image(240, 240))
        assert data["passed"] is True
        assert data["similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_recognize_without_persist(self, service):
        enroll(service)
        service.recognize(blank_image(240, 240), persist=False)
        assert service.store.list_recognize_logs() == []


class TestManagement:
    def test_list_users(self, service):
        enroll(service)
        data = service.list_users()
        assert data["total"] == 1
        assert data["items"][0]["user_no"] == "2023010101"
        assert data["items"][0]["feature_count"] == 1

    def test_get_user_detail(self, service):
        summary = enroll(service, [blank_image(240, 240)] * 2)
        data = service.get_user(summary.user_id)
        assert len(data["features"]) == 2
        assert "feature_cipher" not in data["features"][0]

    def test_delete_user(self, service):
        summary = enroll(service)
        result = service.delete_user(summary.user_id)
        assert result["deleted"] is True
        with pytest.raises(UserNotFoundError):
            service.get_user(summary.user_id)

    def test_stats(self, service):
        enroll(service)
        service.recognize(blank_image(240, 240))
        service.recognize(blank_image(240, 240), persist=True)
        stats = service.stats()
        assert stats["total"] == 2
        assert stats["users"] == 1
        assert stats["features"] == 1
        assert "识别成功" in stats["by_result_text"]

    def test_records_include_result_text(self, service):
        enroll(service)
        service.recognize(blank_image(240, 240))
        records = service.list_records()
        assert records[0]["result_text"] == "识别成功"

    def test_status(self, service):
        status = service.status()
        assert status["ready"] is True
        assert status["users"] == 0
        assert status["features"] == 0
        assert status["cipher"] == "v1:k1"

    def test_require_gallery(self, service):
        summary = enroll(service)
        service.require_gallery(summary.user_id)
        service.delete_features(summary.user_id)
        from facerec.errors import FeatureEmptyError

        with pytest.raises(FeatureEmptyError):
            service.require_gallery(summary.user_id)
