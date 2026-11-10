"""人脸识别引擎单元测试（注入假检测器/假编码器，不需要真实模型）。"""
from __future__ import annotations

import numpy as np
import pytest

from facerec.detector import FaceBox
from facerec.engine import (
    RESULT_ERROR,
    RESULT_LOW_SIMILARITY,
    RESULT_MULTI_FACE,
    RESULT_NO_FACE,
    RESULT_SUCCESS,
    RESULT_TIMEOUT,
    FaceEngine,
)
from facerec.errors import ModelUnavailableError, MultiFaceError, NoFaceError, RecognitionTimeoutError
from facerec.utils import cosine_similarity
from tests.fakes import FakeDetector, FakeEncoder, blank_image, make_vector, noisy_vector


@pytest.fixture()
def vector():
    return make_vector(100)


@pytest.fixture()
def engine(vector):
    return FaceEngine(FakeDetector(), FakeEncoder(vector), threshold=0.6, timeout_ms=2000)


class TestExtract:
    def test_returns_vector_and_quality(self, engine, vector):
        sample = engine.extract(blank_image(240, 240))
        assert sample.vector.shape == (512,)
        assert np.allclose(sample.vector, vector)
        assert sample.face_count == 1
        assert 0.0 <= sample.quality <= 1.0

    def test_no_face_raises(self, vector):
        engine = FaceEngine(FakeDetector([]), FakeEncoder(vector))
        with pytest.raises(NoFaceError):
            engine.extract(blank_image())

    def test_multi_face_raises_when_single_required(self, vector):
        boxes = [FaceBox(0, 0, 80, 80), FaceBox(120, 120, 90, 90)]
        engine = FaceEngine(FakeDetector(boxes), FakeEncoder(vector))
        with pytest.raises(MultiFaceError):
            engine.extract(blank_image(240, 240))

    def test_multi_face_allowed_picks_main(self, vector):
        boxes = [FaceBox(0, 0, 60, 60), FaceBox(40, 40, 160, 160)]
        engine = FaceEngine(FakeDetector(boxes), FakeEncoder(vector))
        sample = engine.extract(blank_image(240, 240), require_single=False)
        assert sample.face_count == 2
        assert sample.box.area == 160 * 160

    def test_model_unavailable(self):
        engine = FaceEngine(FakeDetector(), FakeEncoder(available=False))
        with pytest.raises(ModelUnavailableError):
            engine.extract(blank_image())

    def test_timeout(self, vector):
        engine = FaceEngine(FakeDetector(), FakeEncoder(vector), timeout_ms=-1)
        with pytest.raises(RecognitionTimeoutError):
            engine.extract(blank_image())

    def test_extract_many_collects_errors(self, vector):
        boxes = [FaceBox(0, 0, 50, 50), FaceBox(100, 100, 60, 60)]
        detector = FakeDetector(boxes)
        engine = FaceEngine(detector, FakeEncoder(vector), max_samples=5)
        samples, errors = engine.extract_many([blank_image()] * 3)
        assert samples == []
        assert len(errors) == 3
        assert errors[0]["code"] == 1002

    def test_extract_many_stops_at_max(self, vector):
        engine = FaceEngine(FakeDetector(), FakeEncoder(vector), max_samples=2)
        samples, errors = engine.extract_many([blank_image()] * 5)
        assert len(samples) == 2
        assert errors == []


class TestMatch:
    def test_match_best_user(self, engine, vector):
        gallery = [
            (1, "U1", "张三", make_vector(1)),
            (2, "U2", "李四", vector),
            (3, "U3", "王五", make_vector(3)),
        ]
        user, score = engine.match(vector, gallery)
        assert user == (2, "U2", "李四")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_match_empty_gallery(self, engine, vector):
        assert engine.match(vector, []) == (None, 0.0)

    def test_compare(self, engine, vector):
        assert engine.compare(vector, vector) == pytest.approx(1.0, abs=1e-6)
        assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)


class TestRecognize:
    def test_success(self, engine, vector):
        gallery = [(7, "2023010101", "韩萧", vector)]
        result = engine.recognize(blank_image(240, 240), gallery)
        assert result.result_type == RESULT_SUCCESS
        assert result.passed
        assert result.user_id == 7
        assert result.user_no == "2023010101"
        assert result.similarity == pytest.approx(1.0, abs=1e-5)
        assert result.threshold == 0.6
        assert result.as_dict()["result_text"] == "识别成功"

    def test_low_similarity(self, engine):
        gallery = [(7, "2023010101", "韩萧", make_vector(999))]
        result = engine.recognize(blank_image(240, 240), gallery)
        assert result.result_type == RESULT_LOW_SIMILARITY
        assert not result.passed
        assert result.similarity < 0.6

    def test_no_face(self, vector):
        engine = FaceEngine(FakeDetector([]), FakeEncoder(vector))
        result = engine.recognize(blank_image(), [])
        assert result.result_type == RESULT_NO_FACE
        assert result.face_count == 0
        assert result.as_dict()["result_text"] == "未检测到人脸"

    def test_multi_face(self, vector):
        boxes = [FaceBox(0, 0, 80, 80), FaceBox(120, 120, 90, 90)]
        engine = FaceEngine(FakeDetector(boxes), FakeEncoder(vector))
        result = engine.recognize(blank_image(240, 240), [])
        assert result.result_type == RESULT_MULTI_FACE
        assert result.face_count == 2

    def test_timeout_result(self, vector):
        engine = FaceEngine(FakeDetector(), FakeEncoder(vector), timeout_ms=-1)
        result = engine.recognize(blank_image(), [(1, "U1", "张三", vector)])
        assert result.result_type == RESULT_TIMEOUT

    def test_empty_gallery(self, engine):
        result = engine.recognize(blank_image(), [])
        assert result.result_type == RESULT_LOW_SIMILARITY
        assert "特征库为空" in result.message

    def test_model_not_ready(self):
        engine = FaceEngine(FakeDetector(), FakeEncoder(available=False))
        result = engine.recognize(blank_image(), [])
        assert result.result_type == RESULT_ERROR
        assert not result.passed

    def test_status_payload(self, engine):
        status = engine.status()
        assert status["ready"] is True
        assert status["feature_dim"] == 512
        assert status["threshold"] == 0.6

    def test_same_person_noisy_still_passes(self, engine, vector):
        probe = noisy_vector(vector, 0.008, seed=3)
        engine.encoder.vector = probe
        result = engine.recognize(blank_image(), [(1, "U1", "张三", vector)])
        assert result.passed
        assert result.similarity > 0.9
