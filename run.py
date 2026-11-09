"""启动入口。

用法::

    python run.py                     # 默认 127.0.0.1:8080，并预加载模型
    python run.py --host 0.0.0.0 --port 8080
    python run.py --no-warmup         # 跳过模型预热（首个请求会较慢）
    python run.py --check             # 只做自检后退出（用于部署验证）
"""
from __future__ import annotations

import argparse
import sys
import time

from facerec.app import create_app
from facerec.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FaceNet2Rec 校园人脸识别系统")
    parser.add_argument("--host", default=None, help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认 8080")
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动时的模型预热")
    parser.add_argument("--check", action="store_true", help="自检后退出（校验配置、模型、数据库）")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    host = args.host or settings.host
    port = args.port or settings.port

    app = create_app(settings)
    service = app.extensions["facerec"]["service"]

    status = service.status()
    print("=" * 66)
    print(f" {app.config['APP_NAME']}  v{app.config['APP_VERSION']}")
    print(f" 模型: {status['model_version']}   就绪: {status['ready']}")
    print(f" 模型文件: {settings.model_path}")
    opencv = status.get("opencv", {})
    print(f" OpenCV: {opencv.get('version') or '不可用'}")
    if opencv.get("error"):
        print(f" OpenCV 导入失败: {opencv['error']}")
    print(f" 阈值: {status['threshold']}   特征库: {status['users']} 人 / {status['features']} 条特征")
    print(f" 数据: {settings.db_path}")
    print(f" 访问: http://{host}:{port}/")
    print("=" * 66)

    if args.check:
        if status["ready"]:
            print("自检通过：模型、数据库、加密组件均已就绪。")
            return 0
        print("自检失败：模型未就绪，请检查 model/ 目录与依赖安装。")
        return 1

    if not args.no_warmup and not args.debug:
        started = time.perf_counter()
        result = service.warmup()
        cost = int((time.perf_counter() - started) * 1000)
        print(f"模型预热: {result}  耗时 {cost}ms")
        if not result.get("warmed"):
            print("警告：模型预热失败，识别接口将返回“模型未就绪”，请检查日志。")

    if not args.no_browser and not args.debug:
        try:
            import threading
            import webbrowser

            threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
        except Exception:  # noqa: BLE001 - 无桌面环境时静默
            pass

    app.run(host=host, port=port, debug=args.debug, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
