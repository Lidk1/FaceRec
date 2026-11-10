"""人脸检测：OpenCV Haar 级联检测 + 主人脸选取。

检测层独立成类，仅暴露 detect(bgr) -> DetectionResult，便于替换为 MTCNN
等更强检测器而不影响上层业务。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .assets import opencv_readable_path
from .utils import cv2_available

try:  # pragma: no cover - 依赖环境
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class FaceBox:
    """人脸框（左上角坐标 + 宽高）。"""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "area": self.area}


@dataclass
class DetectionResult:
    """一次检测的完整结果。"""

    boxes: List[FaceBox] = field(default_factory=list)
    main: Optional[FaceBox] = None
    image_width: int = 0
    image_height: int = 0
    cost_ms: int = 0

    @property
    def count(self) -> int:
        return len(self.boxes)


def select_main_face(boxes: Sequence[FaceBox], width: int, height: int) -> Optional[FaceBox]:
    """选取主人脸。

    规则（对应 SRS FR-04）：面积明显最大的脸优先；当存在多张面积接近
    （≥ 最大面积 70%）的人脸时，取离画面中心最近的一张，避免边缘路人脸
    抢占判定。

    Args:
        boxes: 候选人脸框（不应为空）。
        width: 画面宽度。
        height: 画面高度。

    Returns:
        主人脸框；候选为空时返回 None。
    """
    if not boxes:
        return None
    max_area = max(b.area for b in boxes)
    candidates = [b for b in boxes if b.area >= max_area * 0.7]
    if len(candidates) == 1:
        return candidates[0]

    cx0, cy0 = width / 2.0, height / 2.0
    # 归一化中心距离：0 表示正中，1 表示画面角点
    diag = max(1.0, (width**2 + height**2) ** 0.5)

    def score(box: FaceBox) -> float:
        dist = ((box.cx - cx0) ** 2 + (box.cy - cy0) ** 2) ** 0.5 / diag
        return (box.area / max_area) - 0.6 * dist

    return max(candidates, key=score)


class FaceDetector(ABC):
    """人脸检测接口。"""

    name: str = "detector"

    @abstractmethod
    def detect(self, bgr) -> DetectionResult: ...

    @property
    @abstractmethod
    def available(self) -> bool: ...


class HaarFaceDetector(FaceDetector):
    """基于 OpenCV Haar 级联的轻量检测器（CPU，单帧约 20~60ms）。"""

    name = "haar-cascade-frontalface"

    def __init__(
        self,
        min_size: int = 80,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        max_width: int = 800,
    ) -> None:
        self.min_size = max(24, int(min_size))
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.max_width = int(max_width)
        self._cascades = []
        if cv2 is not None:
            # alt2 对人脸姿态/光照更稳健，作为主检测器；default 作为兜底
            for xml_name in (
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_frontalface_default.xml",
            ):
                try:
                    xml = opencv_readable_path(Path(cv2.data.haarcascades) / xml_name)
                    cascade = cv2.CascadeClassifier(str(xml))
                    if not cascade.empty():
                        self._cascades.append(cascade)
                except Exception:  # noqa: BLE001 - 缺少级联文件时跳过
                    continue
        self._cascade = self._cascades[0] if self._cascades else None

    @property
    def available(self) -> bool:
        return bool(self._cascades)

    def detect(self, bgr) -> DetectionResult:
        import time

        started = time.perf_counter()
        if not self._cascades or cv2 is None:
            return DetectionResult(cost_ms=0)
        height, width = bgr.shape[:2]
        # 高分辨率画面先缩放再检测：Haar 检测耗时随面积增长，缩到 800px 宽
        # 可把单帧检测压到 30ms 内，检测框再按比例映射回原图裁剪。
        scale = 1.0
        work = bgr
        if width > self.max_width:
            scale = self.max_width / float(width)
            work = cv2.resize(bgr, (self.max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = []
        for cascade in self._cascades:
            rects = cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(max(24, int(self.min_size * scale)),) * 2,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(rects):
                break
        boxes = []
        for (x, y, w, h) in rects:
            boxes.append(
                FaceBox(
                    int(round(x / scale)),
                    int(round(y / scale)),
                    int(round(w / scale)),
                    int(round(h / scale)),
                )
            )
        # 过滤嵌套框（同一张脸被重复检测）
        boxes = _suppress_nested(boxes)
        boxes.sort(key=lambda b: b.area, reverse=True)
        cost = int(round((time.perf_counter() - started) * 1000))
        return DetectionResult(
            boxes=boxes,
            main=select_main_face(boxes, width, height),
            image_width=width,
            image_height=height,
            cost_ms=cost,
        )


def _suppress_nested(boxes: List[FaceBox], iou_threshold: float = 0.4) -> List[FaceBox]:
    """简单的同类框去重：保留面积更大的框。"""
    kept: List[FaceBox] = []
    for box in sorted(boxes, key=lambda b: b.area, reverse=True):
        if all(_iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    return kept


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def expand_box(box: FaceBox, width: int, height: int, margin: float = 0.18) -> tuple:
    """向外扩边并裁剪到图像范围内，返回 (x1, y1, x2, y2)。"""
    mx = int(box.w * margin)
    my = int(box.h * margin)
    x1 = max(0, box.x - mx)
    y1 = max(0, box.y - my)
    x2 = min(width, box.x + box.w + mx)
    y2 = min(height, box.y + box.h + my)
    return x1, y1, x2, y2
