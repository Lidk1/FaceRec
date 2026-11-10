"""人脸识别业务服务层。

编排"检测 → 特征提取 → 加密 → 入库"与"检测 → 特征比对 → 判定 → 留痕"
两条链路，是控制器（HTTP 接口）与底层引擎 / DAO 之间的唯一入口。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cipher import AesFeatureCipher, CipherError, FeatureCipher
from .config import Settings
from .detector import FaceBox, HaarFaceDetector
from .encoder import FaceEncoder, FaceNet2RecEncoder
from .engine import (
    RESULT_TEXT,
    FaceEngine,
    RecognitionResult,
)
from .errors import (
    DuplicateUserError,
    FeatureEmptyError,
    LowQualityError,
    ModelUnavailableError,
    UserNotFoundError,
    ValidateError,
)
from .store import FaceStore, UserRecord
from .utils import cv2_status, elapsed_ms, now_text, valid_user_no

# 注册样本的最低质量分（清晰度 + 尺寸 + 占比加权）
MIN_ENROLL_QUALITY = 0.10


@dataclass
class EnrollSummary:
    user_id: int
    user_no: str
    user_name: str
    samples: int
    rejected: List[dict]
    avg_quality: float
    cost_ms: int
    model_version: str

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "user_no": self.user_no,
            "user_name": self.user_name,
            "samples": self.samples,
            "avg_quality": self.avg_quality,
            "rejected": self.rejected,
            "cost_ms": self.cost_ms,
            "model_version": self.model_version,
        }


class FaceRecognitionService:
    """人脸识别核心服务。"""

    def __init__(
        self,
        store: FaceStore,
        engine: FaceEngine,
        cipher: FeatureCipher,
        settings: Settings,
    ) -> None:
        self.store = store
        self.engine = engine
        self.cipher = cipher
        self.settings = settings

    # ------------------------------------------------------------------ 构造
    @classmethod
    def build(cls, settings: Settings, *, store: Optional[FaceStore] = None) -> "FaceRecognitionService":
        """按配置装配默认实现（检测器 / 编码器 / 加密器）。"""
        store = store or FaceStore(settings.db_path)
        store.init_schema()
        from .auth import AuthService

        AuthService(store).ensure_seed_accounts()
        detector = HaarFaceDetector(
            min_size=settings.min_face_size,
            scale_factor=settings.detector_scale,
            min_neighbors=settings.detector_neighbors,
        )
        encoder = FaceNet2RecEncoder(settings.model_path, model_version=settings.model_version)
        engine = FaceEngine(
            detector=detector,
            encoder=encoder,
            threshold=settings.threshold,
            timeout_ms=settings.timeout_ms,
            max_samples=settings.max_samples,
        )
        cipher = AesFeatureCipher(settings.aes_key, settings.cipher_key_id)
        return cls(store, engine, cipher, settings)

    # ------------------------------------------------------------------ 状态
    def status(self) -> dict:
        return {
            "ready": self.engine.ready,
            "engine": self.engine.status(),
            "threshold": self.engine.threshold,
            "users": self.store.count_users(),
            "features": self.store.count_features(),
            "cipher": self.cipher.fingerprint(),
            "model_version": self.engine.encoder.model_version,
            "opencv": cv2_status(),
        }

    def warmup(self) -> dict:
        encoder = self.engine.encoder
        ok = False
        if isinstance(encoder, FaceNet2RecEncoder):
            ok = encoder.warmup()
        return {"warmed": ok, "model_version": encoder.model_version}

    # ------------------------------------------------------------------ 注册
    def enroll(
        self,
        user_no: str,
        user_name: str,
        images: Sequence[bytes | str],
        *,
        dept: Optional[str] = None,
        phone: Optional[str] = None,
        replace: bool = False,
    ) -> EnrollSummary:
        """注册用户并写入加密人脸特征。

        Args:
            user_no: 学号/工号，全局唯一。
            user_name: 姓名。
            images: 采集到的多帧人脸图像（原始字节或 Base64/dataURL）。
            dept/phone: 可选资料。
            replace: 已存在用户时是否覆盖其人脸样本（保留用户资料）。
        """
        started = time.perf_counter()
        user_no = (user_no or "").strip()
        user_name = (user_name or "").strip()
        if not valid_user_no(user_no):
            raise ValidateError("学号/工号需为 2~20 位字母、数字、下划线或短横线")
        if not user_name or len(user_name) > 20:
            raise ValidateError("姓名不能为空且不超过 20 个字符")
        if not images:
            raise ValidateError("请至少采集 1 帧人脸图像")
        if len(images) > self.settings.max_samples * 2:
            raise ValidateError(f"单次最多提交 {self.settings.max_samples * 2} 帧图像")

        samples, rejected = self.engine.extract_many(list(images)[: self.settings.max_samples * 2])
        if not samples:
            reason = rejected[0]["message"] if rejected else "未提取到有效人脸"
            raise LowQualityError(f"人脸样本全部无效：{reason}")

        avg_quality = round(float(np.mean([s.quality for s in samples])), 4)
        if max(s.quality for s in samples) < MIN_ENROLL_QUALITY:
            raise LowQualityError("人脸图像质量过低，请正对摄像头、保持光线充足后重试")

        existing = self.store.get_user_by_no(user_no)
        if existing is not None and not replace:
            raise DuplicateUserError(f"学号/工号 {user_no} 已注册")
        if existing is not None:
            user_id = existing.user_id
            self.store.delete_features(user_id)
        else:
            user_id = self.store.create_user(user_no, user_name, dept, phone)

        try:
            for sample in samples:
                token = self.cipher.encrypt(sample.vector)
                self.store.add_feature(
                    user_id,
                    token,
                    model_version=self.engine.encoder.model_version,
                    quality_score=sample.quality,
                    feature_dim=int(sample.vector.size),
                )
        except Exception:
            # 写入失败时回滚用户，避免"有用户无特征"的脏数据
            if existing is None:
                self.store.delete_user(user_id)
            raise

        return EnrollSummary(
            user_id=user_id,
            user_no=user_no,
            user_name=user_name,
            samples=len(samples),
            rejected=rejected,
            avg_quality=avg_quality,
            cost_ms=elapsed_ms(started),
            model_version=self.engine.encoder.model_version,
        )

    # ------------------------------------------------------------------ 识别
    def gallery(self) -> List[Tuple[int, str, str, np.ndarray]]:
        """解密特征库，返回 [(user_id, user_no, user_name, vector), ...]。"""
        if not self.engine.ready:
            raise ModelUnavailableError(self.engine._not_ready_reason())
        result = []
        for user_id, user_no, user_name, token in self.store.gallery():
            try:
                vector = self.cipher.decrypt(token)
            except CipherError:
                # 单条密文损坏不影响其它用户识别，跳过并继续
                continue
            result.append((user_id, user_no, user_name, vector))
        return result

    def recognize(self, image, *, device_id: Optional[str] = None, persist: bool = True) -> RecognitionResult:
        """刷脸核验并记录识别日志。"""
        gallery = self.gallery()
        result = self.engine.recognize(image, gallery, device_id=device_id)
        if persist:
            self.store.add_recognize_log(
                result_type=result.result_type,
                similarity=result.similarity,
                threshold=result.threshold,
                face_count=result.face_count,
                cost_ms=result.cost_ms,
                user_id=result.user_id,
                device_id=device_id,
            )
        return result

    def verify(self, image_a, image_b) -> dict:
        """1:1 比对：判断两张图是否为同一人（不查特征库）。"""
        sample_a = self.engine.extract(image_a, require_single=True)
        sample_b = self.engine.extract(image_b, require_single=True)
        score = self.engine.compare(sample_a.vector, sample_b.vector)
        return {
            "similarity": round(float(score), 5),
            "threshold": self.engine.threshold,
            "passed": bool(score >= self.engine.threshold),
            "quality_a": sample_a.quality,
            "quality_b": sample_b.quality,
        }

    # ------------------------------------------------------------------ 管理
    def list_users(self, keyword: Optional[str] = None, limit: int = 50, offset: int = 0) -> dict:
        users = self.store.list_users(keyword, limit, offset)
        return {
            "total": self.store.count_users(),
            "items": [user.as_dict() for user in users],
        }

    def get_user(self, user_id: int) -> dict:
        user = self.store.get_user(user_id)
        data = user.as_dict()
        data["features"] = [f.as_dict() for f in self.store.list_features(user_id)]
        return data

    def delete_features(self, user_id: int) -> dict:
        user = self.store.get_user(user_id)
        removed = self.store.delete_features(user_id)
        return {"user_id": user.user_id, "deleted": removed}

    def delete_user(self, user_id: int) -> dict:
        user = self.store.get_user(user_id)
        ok = self.store.delete_user(user_id)
        if not ok:
            raise UserNotFoundError(f"用户 {user_id} 不存在")
        return {"user_id": user_id, "user_no": user.user_no, "deleted": True}

    def list_records(self, limit: int = 20, offset: int = 0) -> List[dict]:
        """识别记录，附结果中文描述。"""
        rows = self.store.list_recognize_logs(limit, offset)
        for row in rows:
            row["result_text"] = RESULT_TEXT.get(int(row["result_type"]), "未知")
        return rows

    def stats(self) -> dict:
        data = self.store.stats()
        data["by_result_text"] = {
            RESULT_TEXT.get(code, str(code)): count for code, count in data["by_result"].items()
        }
        data["users"] = self.store.count_users()
        data["features"] = self.store.count_features()
        return data

    def require_gallery(self, user_id: int) -> None:
        if not self.store.list_features(user_id):
            raise FeatureEmptyError()
