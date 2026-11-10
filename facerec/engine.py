"""人脸识别引擎：检测 → 特征提取 → 相似度比对 → 判定。

引擎只负责"单张人脸"的数学部分，不访问数据库；特征库遍历与识别记录
落库由 :class:`facerec.service.FaceRecognitionService` 编排。这样既便于
单元测试（注入假检测器/假编码器），也便于后续替换模型。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .detector import FaceBox, FaceDetector, DetectionResult
from .encoder import FaceEncoder
from .errors import (
    LowQualityError,
    ModelUnavailableError,
    MultiFaceError,
    NoFaceError,
    RecognitionTimeoutError,
)
from .utils import cosine_similarity, decode_image, elapsed_ms, face_quality

# 识别结果类型，与 rec_recognize_log.result_type 保持一致
RESULT_SUCCESS = 1
RESULT_NO_FACE = 2
RESULT_MULTI_FACE = 3
RESULT_LOW_SIMILARITY = 4
RESULT_TIMEOUT = 5
RESULT_ERROR = 6

RESULT_TEXT = {
    RESULT_SUCCESS: "识别成功",
    RESULT_NO_FACE: "未检测到人脸",
    RESULT_MULTI_FACE: "检测到多张人脸",
    RESULT_LOW_SIMILARITY: "相似度不足",
    RESULT_TIMEOUT: "识别超时",
    RESULT_ERROR: "识别异常",
}


@dataclass
class FaceSample:
    """单帧人脸的提取结果。"""

    box: FaceBox
    vector: np.ndarray
    quality: float
    face_count: int = 1

    def as_dict(self) -> dict:
        return {
            "box": self.box.as_dict(),
            "quality": self.quality,
            "face_count": self.face_count,
            "dim": int(self.vector.size),
        }


@dataclass
class RecognitionResult:
    """一次刷脸核验的最终结果。"""

    result_type: int
    similarity: Optional[float] = None
    user_id: Optional[int] = None
    user_no: Optional[str] = None
    user_name: Optional[str] = None
    face_count: int = 0
    cost_ms: int = 0
    threshold: float = 0.0
    boxes: List[FaceBox] = field(default_factory=list)
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.result_type == RESULT_SUCCESS

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "result_type": self.result_type,
            "result_text": RESULT_TEXT.get(self.result_type, "未知结果"),
            "similarity": round(self.similarity, 5) if self.similarity is not None else None,
            "threshold": self.threshold,
            "user_id": self.user_id,
            "user_no": self.user_no,
            "user_name": self.user_name,
            "face_count": self.face_count,
            "cost_ms": self.cost_ms,
            "boxes": [b.as_dict() for b in self.boxes],
            "message": self.message or RESULT_TEXT.get(self.result_type, ""),
        }


class FaceEngine:
    """人脸检测 + 特征提取 + 比对。"""

    def __init__(
        self,
        detector: FaceDetector,
        encoder: FaceEncoder,
        threshold: float = 0.75,
        timeout_ms: int = 2000,
        max_samples: int = 5,
    ) -> None:
        self.detector = detector
        self.encoder = encoder
        self.threshold = float(threshold)
        self.timeout_ms = int(timeout_ms)
        self.max_samples = int(max_samples)

    # ------------------------------------------------------------------ 状态
    @property
    def ready(self) -> bool:
        return bool(self.detector.available and self.encoder.available)

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "detector": self.detector.name,
            "encoder": self.encoder.name,
            "model_version": self.encoder.model_version,
            "feature_dim": self.encoder.dim,
            "threshold": self.threshold,
            "timeout_ms": self.timeout_ms,
        }

    # ------------------------------------------------------------------ 提取
    def extract(self, image, *, require_single: bool = True) -> FaceSample:
        """从图像中提取主人脸特征。

        Args:
            image: 图像字节流或 BGR ndarray。
            require_single: 为 True 时，检出多张人脸直接抛 :class:`MultiFaceError`。

        Raises:
            NoFaceError: 未检测到人脸。
            MultiFaceError: 存在多张人脸且要求单人。
            LowQualityError: 人脸过小或过于模糊。
            ModelUnavailableError: 模型未就绪。
        """
        if not self.ready:
            raise ModelUnavailableError(self._not_ready_reason())

        bgr = decode_image(image) if not hasattr(image, "shape") else image
        started = time.perf_counter()
        detection: DetectionResult = self.detector.detect(bgr)
        if detection.count == 0:
            raise NoFaceError()
        if detection.count > 1:
            if require_single:
                raise MultiFaceError(f"检测到 {detection.count} 张人脸，请仅保留一人")
        box = detection.main or detection.boxes[0]
        vector = self.encoder.encode(bgr, box)

        height, width = bgr.shape[:2]
        ratio = box.area / float(max(1, width * height))
        gray = None
        try:
            import cv2

            gray = cv2.cvtColor(bgr[box.y : box.y + box.h, box.x : box.x + box.w], cv2.COLOR_BGR2GRAY)
        except Exception:  # pragma: no cover - OpenCV 不可用时跳过清晰度评分
            gray = None
        quality = face_quality(gray, ratio, min(box.w, box.h))

        sample = FaceSample(box=box, vector=vector, quality=quality, face_count=detection.count)
        if elapsed_ms(started) > self.timeout_ms:
            raise RecognitionTimeoutError()
        return sample

    def extract_many(self, images: Sequence) -> Tuple[List[FaceSample], List[dict]]:
        """批量提取（注册场景），返回成功样本与逐帧错误说明。"""
        samples: List[FaceSample] = []
        errors: List[dict] = []
        for index, image in enumerate(images):
            try:
                samples.append(self.extract(image, require_single=True))
            except (NoFaceError, MultiFaceError, LowQualityError, RecognitionTimeoutError) as exc:
                errors.append({"index": index, "code": exc.code, "message": exc.msg})
            if len(samples) >= self.max_samples:
                break
        return samples, errors

    # ------------------------------------------------------------------ 比对
    def compare(self, vector_a, vector_b) -> float:
        return cosine_similarity(vector_a, vector_b)

    def match(
        self,
        vector,
        gallery: Sequence[Tuple[int, str, str, np.ndarray]],
    ) -> Tuple[Optional[Tuple[int, str, str]], float]:
        """在特征库中做 1:N 比对。

        Args:
            vector: 待比对特征。
            gallery: [(user_id, user_no, user_name, vector), ...]

        Returns:
            (命中用户三元组或 None, 最高相似度)。
        """
        best_user = None
        best_score = -1.0
        for user_id, user_no, user_name, feature in gallery:
            score = self.compare(vector, feature)
            if score > best_score:
                best_score = score
                best_user = (user_id, user_no, user_name)
        if not gallery:
            return None, 0.0
        return best_user, max(0.0, best_score)

    def recognize(
        self,
        image,
        gallery: Sequence[Tuple[int, str, str, np.ndarray]],
        *,
        device_id: Optional[str] = None,
    ) -> RecognitionResult:
        """完整刷脸核验：检测 → 提取 → 1:N 比对 → 阈值判定。

        该方法是纯函数式流程（不落库），失败场景全部转换为带结果类型的
        :class:`RecognitionResult`，由服务层记录识别日志。
        """
        started = time.perf_counter()
        threshold = self.threshold
        if not self.ready:
            return RecognitionResult(
                result_type=RESULT_ERROR,
                threshold=threshold,
                cost_ms=elapsed_ms(started),
                message=self._not_ready_reason(),
            )
        try:
            bgr = decode_image(image) if not hasattr(image, "shape") else image
            detection = self.detector.detect(bgr)
        except Exception as exc:  # noqa: BLE001
            return RecognitionResult(
                result_type=RESULT_ERROR,
                threshold=threshold,
                cost_ms=elapsed_ms(started),
                message=f"图像解析失败: {exc}",
            )

        face_count = detection.count
        boxes = detection.boxes
        base = RecognitionResult(
            result_type=RESULT_ERROR,
            face_count=face_count,
            threshold=threshold,
            boxes=boxes,
        )
        if face_count == 0:
            base.result_type = RESULT_NO_FACE
            base.message = RESULT_TEXT[RESULT_NO_FACE]
            base.cost_ms = elapsed_ms(started)
            return base
        if face_count > 1:
            base.result_type = RESULT_MULTI_FACE
            base.message = f"检测到 {face_count} 张人脸，禁止多人同时识别"
            base.cost_ms = elapsed_ms(started)
            return base

        box = detection.main or boxes[0]
        try:
            vector = self.encoder.encode(bgr, box)
        except Exception as exc:  # noqa: BLE001
            base.result_type = RESULT_ERROR
            base.message = f"特征提取失败: {exc}"
            base.cost_ms = elapsed_ms(started)
            return base

        user, score = self.match(vector, gallery)
        base.similarity = round(float(score), 5)
        base.cost_ms = elapsed_ms(started)
        if base.cost_ms > self.timeout_ms:
            base.result_type = RESULT_TIMEOUT
            base.message = f"识别耗时 {base.cost_ms}ms 超过 {self.timeout_ms}ms 阈值"
            return base
        if user is None:
            base.result_type = RESULT_LOW_SIMILARITY
            base.message = "特征库为空，请先完成人脸注册"
            return base
        if score >= threshold:
            base.result_type = RESULT_SUCCESS
            base.user_id, base.user_no, base.user_name = user
            base.message = RESULT_TEXT[RESULT_SUCCESS]
        else:
            base.result_type = RESULT_LOW_SIMILARITY
            base.message = f"最高相似度 {score:.3f} 低于阈值 {threshold:.2f}"
        return base

    # ------------------------------------------------------------------ 内部
    def _not_ready_reason(self) -> str:
        if not self.detector.available:
            return "人脸检测器未就绪（缺少 OpenCV 或级联模型）"
        if not self.encoder.available:
            return f"人脸特征模型未就绪：{getattr(self.encoder, 'load_error', None) or '模型文件缺失'}"
        return "人脸引擎未就绪"
