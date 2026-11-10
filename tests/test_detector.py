"""人脸检测与主人脸选取单元测试。"""
from __future__ import annotations

from facerec.detector import (
    FaceBox,
    _iou,
    _suppress_nested,
    expand_box,
    select_main_face,
)


class TestSelectMainFace:
    def test_empty_returns_none(self):
        assert select_main_face([], 640, 480) is None

    def test_single_face(self):
        box = FaceBox(10, 10, 100, 100)
        assert select_main_face([box], 640, 480) is box

    def test_largest_face_wins(self):
        small = FaceBox(0, 0, 80, 80)
        large = FaceBox(500, 300, 200, 200)
        assert select_main_face([small, large], 800, 600) is large

    def test_near_center_preferred_among_similar_sizes(self):
        left = FaceBox(20, 200, 150, 150)     # 面积 22500，靠近左边缘
        center = FaceBox(320, 200, 145, 145)  # 面积 21025（≥ 70%），居中
        assert select_main_face([left, center], 800, 600) is center

    def test_dominant_face_beats_center_small_face(self):
        big_edge = FaceBox(0, 0, 400, 400)
        tiny_center = FaceBox(300, 200, 90, 90)
        assert select_main_face([big_edge, tiny_center], 800, 600) is big_edge


class TestNestedSuppression:
    def test_iou_identical(self):
        box = FaceBox(10, 10, 100, 100)
        assert _iou(box, box) == 1.0

    def test_iou_disjoint(self):
        assert _iou(FaceBox(0, 0, 50, 50), FaceBox(100, 100, 50, 50)) == 0.0

    def test_drop_nested_keep_larger(self):
        big = FaceBox(0, 0, 200, 200)
        inner = FaceBox(50, 50, 100, 100)  # IoU ≈ 0.14，不抑制
        same = FaceBox(5, 5, 195, 195)     # IoU ≈ 0.95，抑制
        kept = _suppress_nested([big, inner, same])
        assert big in kept
        assert len(kept) == 2


class TestExpandBox:
    def test_expand_clamped_to_image(self):
        x1, y1, x2, y2 = expand_box(FaceBox(0, 0, 100, 100), 200, 200, margin=0.2)
        assert (x1, y1) == (0, 0)
        assert x2 == 120 and y2 == 120

    def test_expand_interior(self):
        x1, y1, x2, y2 = expand_box(FaceBox(50, 50, 100, 100), 400, 400, margin=0.1)
        assert (x1, y1, x2, y2) == (40, 40, 160, 160)
