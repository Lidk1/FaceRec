"""FaceNet2Rec / FaceNet 特征提取器。

模型说明
--------
采用 Inception-ResNet-v1 结构、VGGFace2 数据集训练的人脸表征模型
（权重文件 ``20180402-114759.pb``，对应 FaceNet2Rec 训练基线），输入
160×160 RGB 人脸、输出 512 维浮点特征向量。

推理后端
--------
通过 OpenCV DNN 加载冻结计算图，纯 CPU 推理，不依赖 TensorFlow / PyTorch
运行时，便于打包为单个可执行产物；接口与实现分离，替换推理后端不影响业务层。
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .detector import FaceBox, expand_box
from .errors import ModelUnavailableError
from .utils import FEATURE_DIM, l2_normalize

try:  # pragma: no cover - 依赖环境
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class FaceEncoder(ABC):
    """人脸特征提取接口。"""

    name: str = "encoder"
    dim: int = FEATURE_DIM
    model_version: str = "unknown"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def encode(self, bgr, box: Optional[FaceBox] = None) -> np.ndarray:
        """提取单张人脸的 L2 归一化特征向量。"""


class FaceNet2RecEncoder(FaceEncoder):
    """FaceNet2Rec 特征提取实现（OpenCV DNN 后端）。"""

    name = "facenet2rec"
    input_size = (160, 160)

    def __init__(self, model_path: str | Path, model_version: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        self.model_version = model_version or f"facenet2rec-{self.model_path.stem}"
        self._net = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ 状态
    @property
    def available(self) -> bool:
        return self.model_path.is_file() and cv2 is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def warmup(self) -> bool:
        """预加载并跑一次空推理，减少首个请求的冷启动开销。"""
        if not self.available:
            self._load_error = f"模型文件不存在或 OpenCV 不可用: {self.model_path}"
            return False
        try:
            self._ensure_net()
            dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
            self._forward(dummy)
            return True
        except Exception as exc:  # noqa: BLE001 - 加载失败统一降级为"模型未就绪"
            self._load_error = f"模型加载失败: {exc}"
            self._net = None
            return False

    def _ensure_net(self):
        if self._net is not None:
            return self._net
        if not self.available:
            raise ModelUnavailableError(f"模型文件不存在或 OpenCV 不可用: {self.model_path}")
        with self._lock:
            if self._net is None:
                try:
                    # 通过文件字节加载，规避 OpenCV 在中文路径下无法打开文件的问题
                    model_bytes = np.fromfile(str(self.model_path), dtype=np.uint8)
                    net = cv2.dnn.readNetFromTensorflow(model_bytes)
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self._net = net
                except Exception as exc:  # noqa: BLE001
                    self._load_error = f"模型加载失败: {exc}"
                    raise ModelUnavailableError(self._load_error) from exc
        return self._net

    # ------------------------------------------------------------------ 推理
    def encode(self, bgr, box: Optional[FaceBox] = None) -> np.ndarray:
        net = self._ensure_net()
        height, width = bgr.shape[:2]
        if box is not None:
            x1, y1, x2, y2 = expand_box(box, width, height, margin=0.18)
        else:  # 已裁剪的人脸图，整幅使用
            x1, y1, x2, y2 = 0, 0, width, height
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("人脸裁剪区域为空")
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        vector = self._forward(resized)
        return vector

    def _forward(self, face_bgr) -> np.ndarray:
        net = self._ensure_net()
        blob = cv2.dnn.blobFromImage(
            face_bgr,
            scalefactor=1.0 / 128.0,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        with self._lock:  # cv2.dnn.Net 非线程安全
            net.setInput(blob)
            output = net.forward()
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.size != self.dim:
            raise ModelUnavailableError(f"模型输出维度异常: {vector.size}，期望 {self.dim}")
        return l2_normalize(vector)

    def benchmark(self, rounds: int = 3) -> dict:
        """对空图做若干轮前向推理，返回平均耗时（毫秒）。"""
        if not self.available:
            return {"available": False, "cost_ms": None}
        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
        self._ensure_net()
        costs = []
        for _ in range(max(1, rounds)):
            started = time.perf_counter()
            self._forward(dummy)
            costs.append((time.perf_counter() - started) * 1000)
        return {"available": True, "cost_ms": round(sum(costs) / len(costs), 1), "rounds": len(costs)}
