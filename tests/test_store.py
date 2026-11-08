"""数据访问层单元测试（临时 SQLite 库）。"""
from __future__ import annotations

import pytest

from facerec.errors import DuplicateUserError, StoreError, UserNotFoundError
from facerec.store import FaceStore


class TestUser:
    def test_create_and_get(self, store):
        user_id = store.create_user("2023010101", "韩萧", "计算机学院", "13800000000")
        user = store.get_user(user_id)
        assert user.user_no == "2023010101"
        assert user.user_name == "韩萧"
        assert user.dept == "计算机学院"
        assert user.feature_count == 0

    def test_duplicate_user_no_rejected(self, store):
        store.create_user("2023010101", "韩萧")
        with pytest.raises(DuplicateUserError):
            store.create_user("2023010101", "另一个人")

    def test_get_unknown_user(self, store):
        with pytest.raises(UserNotFoundError):
            store.get_user(999)

    def test_get_by_no(self, store):
        store.create_user("T001", "王琳博")
        assert store.get_user_by_no("T001").user_name == "王琳博"
        assert store.get_user_by_no("NOPE") is None

    def test_list_search_and_paging(self, store):
        for i in range(5):
            store.create_user(f"20230101{i:02d}", f"学生{i}", "计算机学院")
        assert store.count_users() == 5
        assert len(store.list_users(limit=2)) == 2
        assert len(store.list_users(limit=2, offset=4)) == 1
        assert len(store.list_users(keyword="学生3")) == 1
        assert len(store.list_users(keyword="2023010101")) == 1

    def test_delete_user_cascades_features(self, store):
        user_id = store.create_user("2023010199", "临时用户")
        store.add_feature(user_id, "v1:k1:abc", "fake-1.0")
        assert store.delete_user(user_id) is True
        assert store.get_user_by_no("2023010199") is None
        assert store.count_features() == 0


class TestFeature:
    def test_auto_feature_no(self, store):
        user_id = store.create_user("2023010102", "井泉清")
        f1 = store.add_feature(user_id, "v1:k1:a", "fake-1.0", 0.9)
        f2 = store.add_feature(user_id, "v1:k1:b", "fake-1.0", 0.8)
        features = store.list_features(user_id)
        assert len(features) == 2
        assert [f.feature_no for f in features] == [1, 2]
        assert all(f.user_id == user_id for f in features)
        assert features[0].feature_id == f1
        assert features[1].feature_id == f2

    def test_explicit_feature_no(self, store):
        user_id = store.create_user("2023010103", "王琳博")
        store.add_feature(user_id, "v1:k1:c", "fake-1.0", feature_no=3)
        assert store.list_features(user_id)[0].feature_no == 3

    def test_feature_dict_hides_cipher(self, store):
        user_id = store.create_user("2023010104", "李定坤")
        store.add_feature(user_id, "v1:k1:secret-cipher", "fake-1.0")
        data = store.list_features(user_id)[0].as_dict()
        assert "feature_cipher" not in data
        assert data["feature_dim"] == 512

    def test_gallery_only_valid(self, store):
        user_id = store.create_user("2023010105", "梁爽")
        store.add_feature(user_id, "v1:k1:g", "fake-1.0")
        assert len(store.gallery()) == 1
        assert store.gallery()[0][1] == "2023010105"

    def test_delete_features(self, store):
        user_id = store.create_user("2023010106", "朱佳豪")
        store.add_feature(user_id, "v1:k1:d", "fake-1.0")
        store.add_feature(user_id, "v1:k1:e", "fake-1.0")
        assert store.delete_features(user_id) == 2
        assert store.list_features(user_id) == []
        assert store.count_features() == 0


class TestRecognizeLog:
    def test_add_and_list(self, store):
        user_id = store.create_user("2023010107", "测试用户")
        store.add_recognize_log(1, 0.91, 0.7, 1, 45, user_id=user_id, device_id="T001")
        store.add_recognize_log(2, None, 0.7, 0, 12, device_id="T001")
        records = store.list_recognize_logs(limit=10)
        assert len(records) == 2
        assert records[0]["result_type"] == 2  # 新的在前
        assert records[1]["user_no"] == "2023010107"
        assert records[1]["similarity"] == pytest.approx(0.91)

    def test_stats(self, store):
        user_id = store.create_user("2023010108", "统计用户")
        store.add_recognize_log(1, 0.9, 0.7, 1, 40, user_id=user_id)
        store.add_recognize_log(1, 0.8, 0.7, 1, 50, user_id=user_id)
        store.add_recognize_log(4, 0.4, 0.7, 1, 60)
        stats = store.stats()
        assert stats["total"] == 3
        assert stats["success"] == 2
        assert stats["success_rate"] == pytest.approx(0.6667, abs=1e-3)
        assert stats["avg_similarity"] == pytest.approx(0.85)
        assert stats["by_result"][4] == 1


class TestConfigAndHealth:
    def test_ping(self, store):
        assert store.ping() is True

    def test_config_upsert(self, store):
        store.set_config("face.similarity.threshold", "0.70", "DECIMAL", "阈值")
        assert store.get_config("face.similarity.threshold") == "0.70"
        store.set_config("face.similarity.threshold", "0.65")
        assert store.get_config("face.similarity.threshold") == "0.65"
        assert store.get_config("missing.key", "default") == "default"

    def test_store_error_on_unwritable_path(self, tmp_path):
        store = FaceStore(tmp_path / "no-such-dir" / "a" / "b.db")
        with pytest.raises(StoreError):
            store.connect()
