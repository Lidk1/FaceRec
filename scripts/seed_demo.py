"""演示数据初始化：用真实人脸照片注册演示用户，并跑一轮识别验证。

用法::

    python scripts/seed_demo.py                 # 首次初始化
    python scripts/seed_demo.py --force         # 清空重建（覆盖已有样本）
    python scripts/seed_demo.py --dry-run       # 只打印将要注册的清单

数据来源：仓库自带 ``facenet_facerecognition/test_img``、``emb_img`` 中
课程小组采集的人脸照片；照片仅用于实时提取特征，不落库。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from facerec.config import Settings  # noqa: E402
from facerec.service import FaceRecognitionService  # noqa: E402

ASSET_ROOT = PROJECT_ROOT.parent / "facenet_facerecognition"
TEST_IMG = ASSET_ROOT / "test_img"
EMB_IMG = ASSET_ROOT / "emb_img"

# (学号, 姓名, 学院/部门, 图片列表)
DEMO_USERS = [
    ("2023010101", "韩萧", "计算机学院 / 软件工程 2023", [
        TEST_IMG / "hanxiao_02.jpg", TEST_IMG / "hanxiao_03.jpg", TEST_IMG / "hanxiao_04.jpg",
        EMB_IMG / "hanxiao_01.jpg", EMB_IMG / "hanxiao_02.jpg",
    ]),
    ("2023010102", "井泉清", "计算机学院 / 软件工程 2023", [
        TEST_IMG / "jingquanqing_01.jpg", EMB_IMG / "jingquanqing_01.jpg",
        EMB_IMG / "jingquanqing_02.jpg", EMB_IMG / "jingquanqing_04.jpg",
    ]),
    ("2023010103", "王琳博", "计算机学院 / 软件工程 2023", [
        TEST_IMG / "wanglinbo_01.jpg", EMB_IMG / "wanglinbo_01.jpg",
        EMB_IMG / "wanglinbo_02.jpg", EMB_IMG / "wanglinbo_03.jpg",
    ]),
    ("2023010104", "李定坤", "计算机学院 / 软件工程 2023", [
        TEST_IMG / "lidingkun_01.jpg", EMB_IMG / "lidingkun_01.jpg",
        EMB_IMG / "lidingkun_02.jpg", EMB_IMG / "lidingkun_03.jpg",
    ]),
    ("2022010201", "梁爽", "计算机学院 / 教学实验中心", [
        EMB_IMG / "liangshuang_01.jpg", EMB_IMG / "liangshuang_02.jpg",
        EMB_IMG / "liangshuang_03.jpg", EMB_IMG / "liangshuang_04.jpg",
        EMB_IMG / "liangshuang_05.jpg",
    ]),
    ("2022010202", "朱佳豪", "计算机学院 / 网络工程 2022", [
        EMB_IMG / "zhujiahao_01.jpg", EMB_IMG / "zhujiahao_02.jpg",
        EMB_IMG / "zhujiahao_03.jpg", EMB_IMG / "zhujiahao_04.jpg",
        EMB_IMG / "zhujiahao_05.jpg",
    ]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化人脸识别演示数据")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的用户样本")
    parser.add_argument("--dry-run", action="store_true", help="只打印注册清单")
    parser.add_argument("--no-recognize", action="store_true", help="不执行识别验证")
    args = parser.parse_args()

    settings = Settings.from_env()
    print(f"数据库: {settings.db_path}")
    print(f"模型  : {settings.model_path} (存在={settings.model_available})")
    print(f"阈值  : {settings.threshold}\n")

    for user_no, name, dept, images in DEMO_USERS:
        available = [p for p in images if p.is_file()]
        print(f"[{user_no}] {name} · {dept} · {len(available)} 张照片")
        for path in available:
            print(f"    - {path.relative_to(ASSET_ROOT)}")
    if args.dry_run:
        return 0

    service = FaceRecognitionService.build(settings)
    print("\n模型预热:", service.warmup())

    enrolled = []
    for user_no, name, dept, images in DEMO_USERS:
        files = [p.read_bytes() for p in images if p.is_file()]
        if not files:
            print(f"跳过 {name}: 未找到照片")
            continue
        try:
            summary = service.enroll(user_no, name, files, dept=dept, replace=args.force)
            enrolled.append((user_no, name, summary))
            print(
                f"注册成功: {name}({user_no}) 用户ID={summary.user_id} "
                f"样本={summary.samples} 质量={summary.avg_quality} 耗时={summary.cost_ms}ms"
            )
            for item in summary.rejected:
                print(f"    拒绝第 {item['index'] + 1} 帧: {item['message']}")
        except Exception as exc:  # noqa: BLE001 - 演示脚本需继续处理其它用户
            print(f"注册失败: {name} -> {exc}")

    if args.no_recognize or not enrolled:
        return 0

    print("\n=== 识别验证（用注册照片回测） ===")
    image_map = {user_no: images for user_no, _name, _dept, images in DEMO_USERS}
    for user_no, name, _summary in enrolled:
        image = next((p for p in image_map[user_no] if p.is_file()), None)
        if image is None:
            continue
        result = service.recognize(image.read_bytes(), device_id="T001")
        mark = "PASS" if result.passed else "FAIL"
        print(
            f"{mark} {name}: 命中={result.user_name or '无'} 相似度={result.similarity} "
            f"阈值={result.threshold} 耗时={result.cost_ms}ms 结果={result.message}"
        )

    print("\n=== 陌生人识别（应判为相似度不足） ===")
    stranger = EMB_IMG / "other_01.jpg"
    if stranger.is_file():
        result = service.recognize(stranger.read_bytes(), device_id="T001")
        print(f"陌生人: 结果={result.message} 相似度={result.similarity} 通过={result.passed}")

    print("\n=== 演示用户间最高相似度（用于阈值校准） ===")
    gallery = service.gallery()
    by_user: dict[int, list] = {}
    for user_id, user_no, user_name, vector in gallery:
        by_user.setdefault(user_id, [vector])
    users = {uid: (user_no, user_name) for uid, user_no, user_name, _ in gallery}
    import numpy as np

    uids = list(by_user)
    for i, uid in enumerate(uids):
        for other in uids[i + 1:]:
            sim = max(float(np.dot(a, b)) for a in by_user[uid] for b in by_user[other])
            print(f"{users[uid][1]} ↔ {users[other][1]}: {sim:.3f}")

    print("\n完成。启动 Web 服务: python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
