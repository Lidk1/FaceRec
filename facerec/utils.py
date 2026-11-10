"""通用工具：图像编解码、向量归一化、余弦相似度、时间与校验。"""
from __future__ import annotations

import base64
import re
import time
from datetime import datetime

import numpy as np

try:  # OpenCV 为运行期依赖；纯逻辑单元测试可在无 OpenCV 环境下运行
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - 环境缺失时走降级分支
    cv2 = None
    CV2_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

FEATURE_DIM = 512
USER_NO_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,20}$")


class ImageDecodeError(ValueError):
    """图像字节流无法解析。"""


def cv2_available() -> bool:
    return cv2 is not None


def cv2_status() -> dict:
    """OpenCV 运行状态，便于打包产物排查导入失败原因。"""
    return {
        "available": cv2 is not None,
        "version": getattr(cv2, "__version__", None),
        "error": CV2_IMPORT_ERROR,
    }


def now_text() -> str:
    """毫秒精度时间字符串，统一落库格式。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def elapsed_ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


def decode_image(data: bytes | np.ndarray):
    """把上传字节流解码为 BGR ndarray（OpenCV 约定）。

    Args:
        data: 原始图像字节（JPG/PNG）或已是 ndarray。

    Raises:
        ImageDecodeError: 无法解码或图像为空。
    """
    if isinstance(data, np.ndarray):
        return data
    if cv2 is None:
        raise ImageDecodeError("未安装 OpenCV，无法解码图像")
    if not data:
        raise ImageDecodeError("图像内容为空")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageDecodeError("图像格式不受支持或文件已损坏")
    return image


def decode_base64_image(text: str):
    """解析 dataURL / 纯 Base64 字符串。"""
    if not text or not isinstance(text, str):
        raise ImageDecodeError("图像数据为空")
    payload = text.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:  # noqa: BLE001 - 统一转换为业务可读异常
        raise ImageDecodeError("Base64 图像数据解析失败") from exc
    if len(raw) < 64:
        raise ImageDecodeError("图像数据过短")
    return decode_image(raw)


def l2_normalize(vector) -> np.ndarray:
    """L2 归一化，零向量返回全零向量（避免除零）。"""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return np.zeros_like(arr)
    return (arr / norm).astype(np.float32)


def cosine_similarity(a, b) -> float:
    """余弦相似度，取值范围 [-1, 1]；零向量返回 0。"""
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError(f"特征维度不一致: {va.shape} vs {vb.shape}")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))


def face_quality(gray_crop, face_area_ratio: float, face_size: int) -> float:
    """人脸质量评分（0~1）。

    综合清晰度（Laplacian 方差）、人脸像素尺寸与画面占比，用于注册时
    过滤模糊、过小的人脸样本。
    """
    clarity = 0.0
    if cv2 is not None and gray_crop is not None and getattr(gray_crop, "size", 0) > 0:
        variance = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
        clarity = min(1.0, variance / 120.0)
    size_score = min(1.0, max(0.0, (face_size - 60) / 140.0))
    ratio_score = min(1.0, face_area_ratio / 0.18)
    return round(0.55 * clarity + 0.25 * size_score + 0.20 * ratio_score, 4)


def valid_user_no(value: str) -> bool:
    return bool(value) and bool(USER_NO_PATTERN.match(value))


def ensure_feature_dim(vector, dim: int = FEATURE_DIM) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"特征维度应为 {dim}，实际 {arr.size}")
    return arr
