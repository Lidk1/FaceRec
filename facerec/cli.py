"""命令行工具：无浏览器环境下的注册 / 识别 / 统计。

示例::

    python -m facerec.cli status
    python -m facerec.cli enroll --user-no 2023010101 --name 韩萧 --images a.jpg b.jpg
    python -m facerec.cli recognize --image test.jpg
    python -m facerec.cli bench --image test.jpg -n 5
    python -m facerec.cli users --keyword 韩
    python -m facerec.cli stats
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Settings
from .service import FaceRecognitionService


def _service() -> FaceRecognitionService:
    return FaceRecognitionService.build(Settings.from_env())


def _print(data) -> None:
    if isinstance(data, (dict, list)):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(data)


def cmd_status(args) -> int:
    service = _service()
    _print(service.status())
    return 0


def cmd_enroll(args) -> int:
    service = _service()
    files = []
    for item in args.images:
        path = Path(item)
        if not path.is_file():
            print(f"跳过不存在的文件: {path}", file=sys.stderr)
            continue
        files.append(path.read_bytes())
    if not files:
        print("未提供有效图片", file=sys.stderr)
        return 2
    summary = service.enroll(
        args.user_no, args.name, files, dept=args.dept, phone=args.phone, replace=args.replace
    )
    _print(summary.as_dict())
    return 0


def cmd_recognize(args) -> int:
    service = _service()
    image = Path(args.image).read_bytes()
    result = service.recognize(image, device_id=args.device)
    _print(result.as_dict())
    return 0 if result.passed else 1


def cmd_bench(args) -> int:
    service = _service()
    image = Path(args.image).read_bytes()
    costs = []
    result = None
    for _ in range(max(1, args.n)):
        started = time.perf_counter()
        result = service.recognize(image, persist=False)
        costs.append((time.perf_counter() - started) * 1000)
    costs.sort()
    _print(
        {
            "rounds": len(costs),
            "avg_ms": round(sum(costs) / len(costs), 1),
            "p50_ms": round(costs[len(costs) // 2], 1),
            "min_ms": round(costs[0], 1),
            "max_ms": round(costs[-1], 1),
            "threshold_ms": service.engine.timeout_ms,
            "last_result": result.as_dict() if result else None,
        }
    )
    return 0


def cmd_users(args) -> int:
    _print(_service().list_users(args.keyword, limit=args.limit))
    return 0


def cmd_records(args) -> int:
    _print(_service().list_records(limit=args.limit))
    return 0


def cmd_stats(args) -> int:
    _print(_service().stats())
    return 0


def cmd_delete_features(args) -> int:
    _print(_service().delete_features(args.user_id))
    return 0


def cmd_delete_user(args) -> int:
    _print(_service().delete_user(args.user_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facerec", description="人脸识别系统命令行工具")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="查看服务状态").set_defaults(func=cmd_status)

    p = sub.add_parser("enroll", help="注册人脸")
    p.add_argument("--user-no", required=True, help="学号/工号")
    p.add_argument("--name", required=True, help="姓名")
    p.add_argument("--dept", default=None, help="学院/部门")
    p.add_argument("--phone", default=None, help="联系方式")
    p.add_argument("--images", nargs="+", required=True, help="人脸照片路径（1~5 张）")
    p.add_argument("--replace", action="store_true", help="覆盖已存在用户的人脸样本")
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("recognize", help="识别单张图片")
    p.add_argument("--image", required=True, help="图片路径")
    p.add_argument("--device", default=None, help="终端编号")
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("bench", help="识别耗时压测")
    p.add_argument("--image", required=True, help="图片路径")
    p.add_argument("-n", type=int, default=5, help="循环次数")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("users", help="用户列表")
    p.add_argument("--keyword", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_users)

    p = sub.add_parser("records", help="识别记录")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_records)

    sub.add_parser("stats", help="识别统计").set_defaults(func=cmd_stats)

    p = sub.add_parser("delete-features", help="删除某人脸特征")
    p.add_argument("--user-id", type=int, required=True)
    p.set_defaults(func=cmd_delete_features)

    p = sub.add_parser("delete-user", help="注销用户")
    p.add_argument("--user-id", type=int, required=True)
    p.set_defaults(func=cmd_delete_user)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - CLI 统一错误输出
        print(f"执行失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
