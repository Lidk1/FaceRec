"""HTTP 接口与页面单元测试（Flask test client，不依赖真实模型）。"""
from __future__ import annotations

import base64
import io

import pytest

from facerec.app import create_app
from tests.fakes import blank_image

# 1×1 PNG，用于验证 Base64 / multipart 图像解析链路
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_B64 = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()


@pytest.fixture()
def app(settings, service):
    return create_app(settings, service)


@pytest.fixture()
def client(app):
    return app.test_client()


def enroll_user(client, user_no="2023010101", name="韩萧"):
    return client.post(
        "/api/face/enroll",
        json={
            "user_no": user_no,
            "user_name": name,
            "dept": "计算机学院",
            "images": [PNG_B64],
        },
    )


class TestSystem:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["code"] == 200
        assert body["data"]["service"] == "facerec"

    def test_status(self, client):
        body = client.get("/api/face/status").get_json()
        assert body["data"]["ready"] is True
        assert body["data"]["engine"]["feature_dim"] == 512
        assert body["data"]["threshold"] == 0.60

    def test_pages(self, client):
        for path, marker in (
            ("/", "识别终端"),
            ("/enroll", "人脸注册"),
            ("/library", "人脸特征库"),
        ):
            resp = client.get(path)
            assert resp.status_code == 200
            assert marker in resp.get_data(as_text=True)


class TestEnrollApi:
    def test_enroll_base64(self, client):
        resp = enroll_user(client)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["code"] == 200
        assert body["data"]["user_no"] == "2023010101"
        assert body["data"]["samples"] == 1
        assert "注册成功" in body["msg"]

    def test_enroll_multipart(self, client):
        resp = client.post(
            "/api/face/enroll",
            data={
                "user_no": "2023010202",
                "user_name": "井泉清",
                "images": (io.BytesIO(PNG_1PX), "face.png"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["data"]["samples"] == 1

    def test_duplicate_returns_409(self, client):
        enroll_user(client)
        resp = enroll_user(client)
        assert resp.status_code == 409
        body = resp.get_json()
        assert body["code"] == 1006
        assert "已注册" in body["msg"]

    def test_missing_image_returns_400(self, client):
        resp = client.post("/api/face/enroll", json={"user_no": "2023010101", "user_name": "韩萧"})
        assert resp.status_code == 400
        assert resp.get_json()["code"] == 400

    def test_invalid_user_no_returns_400(self, client):
        resp = client.post(
            "/api/face/enroll",
            json={"user_no": "x", "user_name": "韩萧", "images": [PNG_B64]},
        )
        assert resp.status_code == 400
        assert "学号" in resp.get_json()["msg"]

    def test_bad_base64_returns_400(self, client):
        resp = client.post(
            "/api/face/enroll",
            json={"user_no": "2023010101", "user_name": "韩萧", "images": ["data:image/png;base64,broken"]},
        )
        assert resp.status_code == 400


class TestRecognizeApi:
    def test_recognize_success(self, client):
        enroll_user(client)
        resp = client.post("/api/face/recognize", json={"image": PNG_B64, "device_id": "T001"})
        body = resp.get_json()
        assert body["code"] == 200
        assert body["data"]["passed"] is True
        assert body["data"]["user_no"] == "2023010101"
        assert body["data"]["result_type"] == 1

    def test_recognize_requires_one_image(self, client):
        resp = client.post("/api/face/recognize", json={"images": [PNG_B64, PNG_B64]})
        assert resp.status_code == 400

    def test_recognize_no_face(self, client, service):
        from facerec.engine import RESULT_NO_FACE
        from tests.fakes import FakeDetector

        service.engine.detector = FakeDetector([])
        resp = client.post("/api/face/recognize", json={"image": PNG_B64})
        body = resp.get_json()
        assert body["data"]["result_type"] == RESULT_NO_FACE
        assert body["data"]["passed"] is False

    def test_verify(self, client):
        resp = client.post("/api/face/verify", json={"images": [PNG_B64, PNG_B64]})
        body = resp.get_json()
        assert body["data"]["passed"] is True
        assert body["data"]["similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_verify_needs_two_images(self, client):
        resp = client.post("/api/face/verify", json={"images": [PNG_B64]})
        assert resp.status_code == 400


class TestLibraryApi:
    def test_list_and_detail(self, client):
        enroll_user(client)
        users = client.get("/api/face/users").get_json()["data"]
        assert users["total"] == 1
        user_id = users["items"][0]["user_id"]

        detail = client.get(f"/api/face/users/{user_id}").get_json()["data"]
        assert len(detail["features"]) == 1
        assert "feature_cipher" not in detail["features"][0]

    def test_unknown_user_404(self, client):
        resp = client.get("/api/face/users/9999")
        assert resp.status_code == 404
        assert resp.get_json()["code"] == 1007

    def test_delete_features_then_recognize(self, client):
        enroll_user(client)
        user_id = client.get("/api/face/users").get_json()["data"]["items"][0]["user_id"]
        resp = client.delete(f"/api/face/users/{user_id}/features")
        assert resp.get_json()["data"]["deleted"] == 1
        result = client.post("/api/face/recognize", json={"image": PNG_B64}).get_json()["data"]
        assert result["passed"] is False

    def test_delete_user(self, client):
        enroll_user(client)
        user_id = client.get("/api/face/users").get_json()["data"]["items"][0]["user_id"]
        assert client.delete(f"/api/face/users/{user_id}").get_json()["code"] == 200
        assert client.get("/api/face/users").get_json()["data"]["total"] == 0

    def test_records_and_stats(self, client):
        enroll_user(client)
        client.post("/api/face/recognize", json={"image": PNG_B64})
        records = client.get("/api/face/records").get_json()["data"]["items"]
        assert len(records) == 1
        assert records[0]["result_text"] == "识别成功"

        stats = client.get("/api/face/stats").get_json()["data"]
        assert stats["total"] == 1
        assert stats["success"] == 1
        assert stats["users"] == 1


class TestErrorHandling:
    def test_unknown_api_404_json(self, client):
        resp = client.get("/api/not-exist")
        assert resp.status_code == 404
        assert resp.get_json()["code"] == 404

    def test_method_not_allowed(self, client):
        resp = client.put("/api/face/status")
        assert resp.status_code == 405
        assert resp.get_json()["code"] == 405
