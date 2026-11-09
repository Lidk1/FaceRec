"""向量数学与相似度判定单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from facerec.utils import (
    cosine_similarity,
    ensure_feature_dim,
    face_quality,
    l2_normalize,
    valid_user_no,
)


class TestNormalize:
    def test_unit_norm(self):
        vec = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
        assert np.isclose(np.linalg.norm(vec), 1.0)

    def test_zero_vector(self):
        vec = l2_normalize(np.zeros(512, dtype=np.float32))
        assert np.allclose(vec, 0.0)

    def test_dtype_preserved_as_float32(self):
        vec = l2_normalize(np.arange(4, dtype=np.float64))
        assert vec.dtype == np.float32

    def test_negative_values(self):
        vec = l2_normalize(np.array([-3.0, 0.0, 4.0]))
        assert vec[0] < 0 and vec[2] > 0


class TestCosine:
    def test_identical_vectors(self):
        vec = l2_normalize(np.arange(512, dtype=np.float32))
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        vec = l2_normalize(np.arange(512, dtype=np.float32))
        assert cosine_similarity(vec, -vec) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.ones(512, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity(np.zeros(512, dtype=np.float32), np.zeros(128, dtype=np.float32))

    def test_result_is_clipped(self):
        a = np.array([1.0, 1.0], dtype=np.float32)
        assert -1.0 <= cosine_similarity(a, a) <= 1.0

    def test_similar_vectors_score_high(self):
        rng = np.random.RandomState(7)
        base = rng.randn(512).astype(np.float32)
        noisy = base + rng.randn(512).astype(np.float32) * 0.01
        assert cosine_similarity(l2_normalize(base), l2_normalize(noisy)) > 0.99

    def test_unrelated_vectors_score_low(self):
        rng = np.random.RandomState(8)
        a = l2_normalize(rng.randn(512).astype(np.float32))
        b = l2_normalize(rng.randn(512).astype(np.float32))
        assert abs(cosine_similarity(a, b)) < 0.2


class TestFeatureDim:
    def test_accepts_512(self):
        vec = ensure_feature_dim(np.zeros(512, dtype=np.float32))
        assert vec.size == 512

    def test_rejects_other_dims(self):
        with pytest.raises(ValueError):
            ensure_feature_dim(np.zeros(128, dtype=np.float32))

    def test_flattens_2d_input(self):
        vec = ensure_feature_dim(np.zeros((1, 512), dtype=np.float32))
        assert vec.size == 512


class TestQualityAndValidation:
    def test_quality_in_range(self):
        score = face_quality(None, face_area_ratio=0.2, face_size=160)
        assert 0.0 <= score <= 1.0

    def test_quality_penalises_tiny_face(self):
        big = face_quality(None, face_area_ratio=0.2, face_size=200)
        small = face_quality(None, face_area_ratio=0.01, face_size=60)
        assert small < big

    @pytest.mark.parametrize("value", ["2023010101", "T-001", "ab_cd"])
    def test_valid_user_no(self, value):
        assert valid_user_no(value)

    @pytest.mark.parametrize("value", ["", "a", "中文号码", "x" * 21, "with space"])
    def test_invalid_user_no(self, value):
        assert not valid_user_no(value)
