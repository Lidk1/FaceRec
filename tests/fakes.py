"""测试替身：假检测器、假特征提取器、确定性向量。"""
from __future__ import annotations

import numpy as np

from facerec.detector import DetectionResult, FaceBox, FaceDetector, select_main_face
from facerec.encoder import FaceEncoder

DIM = 512


def make_vector(seed: int, dim: int = DIM) -> np.ndarray:
    """可复现的 512 维单位向量。"""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def noisy_vector(base: np.ndarray, noise: float, seed: int = 0) -> np.ndarray:
    """在基准向量上叠加噪声，用于构造"同一人"样本。"""
    rng = np.random.RandomState(seed)
    vec = base + rng.randn(base.size).astype(np.float32) * noise
    return vec / np.linalg.norm(vec)


class FakeDetector(FaceDetector):
    name = "fake-detector"

    def __init__(self, boxes_per_call=None, available: bool = True, frames=None):
        """boxes_per_call: 每次 detect 返回的人脸框列表（None 表示一张居中脸）。

        frames: 逐帧脚本，按调用顺序消费，最后一帧重复使用（用于"首帧多人、
        次帧单人"这类多帧场景）。
        """
        self._boxes = boxes_per_call
        self._available = available
        self._frames = [list(frame) for frame in frames] if frames is not None else None
        self._index = 0
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, bgr) -> DetectionResult:
        self.calls += 1
        height, width = getattr(bgr, "shape", (100, 100))[:2]
        if self._frames is not None:
            boxes = self._frames[min(self._index, len(self._frames) - 1)]
            self._index += 1
        elif self._boxes is None:
            boxes = [FaceBox(0, 0, max(1, width // 2), max(1, height // 2))]
        else:
            boxes = list(self._boxes)
        return DetectionResult(
            boxes=boxes,
            main=select_main_face(boxes, width, height),
            image_width=width,
            image_height=height,
            cost_ms=1,
        )


class FakeEncoder(FaceEncoder):
    name = "fake-encoder"
    dim = DIM
    model_version = "fake-1.0"

    def __init__(self, vector=None, available: bool = True):
        self.vector = vector if vector is not None else make_vector(1)
        self._available = available
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def encode(self, bgr, box=None) -> np.ndarray:
        self.calls += 1
        return self.vector.copy()


def blank_image(width: int = 200, height: int = 200) -> np.ndarray:
    """测试用灰度三通道图（不依赖 OpenCV）。"""
    return np.zeros((height, width, 3), dtype=np.uint8)
