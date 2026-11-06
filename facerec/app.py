"""Flask 应用工厂与页面路由。"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from flask import Blueprint, Flask, render_template

from . import __version__
from .api import api_bp, register_error_handlers
from .config import APP_ROOT, DATA_ROOT, Settings
from .service import FaceRecognitionService

WEB_DIR = APP_ROOT / "web"
LOG_DIR = DATA_ROOT / "logs"

logger = logging.getLogger("facerec")

pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/")
def terminal_page():
    return render_template("terminal.html", active="terminal")


@pages_bp.get("/enroll")
def enroll_page():
    return render_template("enroll.html", active="enroll")


@pages_bp.get("/library")
def library_page():
    return render_template("library.html", active="library")


def setup_logging(log_dir: Path = LOG_DIR, level: int = logging.INFO) -> None:
    """控制台 + 轮转文件日志；重复调用不会叠加 handler。"""
    root = logging.getLogger("facerec")
    if getattr(root, "_configured", False):
        return
    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "facerec.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # 日志目录不可写时仅保留控制台输出
        root.warning("日志目录不可写，已降级为仅控制台输出: %s", log_dir)

    root._configured = True  # type: ignore[attr-defined]


def create_app(
    settings: Optional[Settings] = None,
    service: Optional[FaceRecognitionService] = None,
) -> Flask:
    """创建 Flask 应用。

    Args:
        settings: 运行配置，默认从环境变量加载。
        service: 业务服务，单元测试可注入替身（跳过真实模型与数据库）。
    """
    settings = settings or Settings.from_env()
    app = Flask(
        __name__,
        static_folder=str(WEB_DIR / "static"),
        template_folder=str(WEB_DIR / "templates"),
        static_url_path="/static",
    )
    app.config["APP_NAME"] = "FaceNet2Rec 校园人脸识别系统"
    app.config["APP_VERSION"] = __version__
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    app.json.ensure_ascii = False  # 中文直接输出，避免 \uXXXX

    setup_logging()

    if service is None:
        service = FaceRecognitionService.build(settings)
    app.extensions["facerec"] = {"settings": settings, "service": service}

    app.register_blueprint(api_bp)
    app.register_blueprint(pages_bp)
    register_error_handlers(app)

    @app.context_processor
    def _inject_globals():
        return {
            "app_name": app.config["APP_NAME"],
            "app_version": app.config["APP_VERSION"],
            "active": "",
            "threshold": settings.threshold,
            "max_samples": settings.max_samples,
            "model_name": settings.model_version,
        }

    logger.info("应用初始化完成: model=%s ready=%s db=%s", settings.model_version, service.status()["ready"], settings.db_path)
    return app
