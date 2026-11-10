"""运行配置。

所有配置项都可以通过环境变量覆盖，默认值保证"克隆即可运行"：

    FACE_MODEL_PATH      FaceNet2Rec 冻结图（.pb）路径
    FACE_DB_PATH         SQLite 演示库路径
    FACE_THRESHOLD       余弦相似度判定阈值，默认 0.70（实测校准，见开发文档）
    FACE_TIMEOUT_MS      单帧识别超时阈值，默认 2000ms
    FACE_AES_KEY         Base64 编码的 32 字节特征加密密钥
    FACE_WEB_HOST/PORT   Web 服务监听地址与端口
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _is_frozen() -> bool:
    """是否运行在 PyInstaller 打包产物中。"""
    return bool(getattr(sys, "frozen", False))


def _resource_root() -> Path:
    """只读资源根目录（web、model）。

    PyInstaller 6.x 的 onedir 布局把数据放进 ``_internal``，不同版本/模式下
    ``sys._MEIPASS`` 的指向略有差异，这里按“是否包含 model 或 web 目录”探测，
    兼容源码运行、onedir 与 onefile 三种形态。
    """
    if _is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        candidates = [Path(getattr(sys, "_MEIPASS", exe_dir)), exe_dir, exe_dir / "_internal"]
        for candidate in candidates:
            if (candidate / "model").is_dir() or (candidate / "web").is_dir():
                return candidate
        return exe_dir
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    """可写数据根目录（data、logs），始终位于可执行文件旁边，便于携带数据。"""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return _resource_root()


# 项目根目录：<root>/facerec/config.py -> <root>
PROJECT_ROOT = _resource_root()
APP_ROOT = PROJECT_ROOT
DATA_ROOT = _data_root()

# FaceNet（Inception-ResNet-v1，训练集 VGGFace2）冻结图，输出 512 维特征。
# 发布版 .pb 含训练期队列与 cond 分支，需先用 scripts/prune_graph.py 裁剪为
# 纯推理图（facenet2rec-inference.pb），OpenCV DNN 才能加载。
DEFAULT_MODEL_FILE = "facenet2rec-inference.pb"
SOURCE_MODEL_FILE = "20180402-114759.pb"

# 默认密钥仅用于课程演示；生产环境必须通过 FACE_AES_KEY 覆盖。
_DEMO_KEY_SEED = b"FaceRec-CampusFacePay-Demo-Key-2026"


def default_aes_key() -> str:
    """返回演示用 AES-256 密钥（Base64）。"""
    return base64.b64encode(hashlib.sha256(_DEMO_KEY_SEED).digest()).decode("ascii")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """不可变配置对象，便于单元测试构造独立实例。"""

    project_root: Path = APP_ROOT
    model_path: Path = field(default_factory=lambda: APP_ROOT / "model" / DEFAULT_MODEL_FILE)
    db_path: Path = field(default_factory=lambda: DATA_ROOT / "data" / "facerec.db")

    # 人脸识别参数
    threshold: float = 0.70
    timeout_ms: int = 2000
    max_samples: int = 5
    min_face_size: int = 80
    detector_scale: float = 1.1
    detector_neighbors: int = 5

    # 特征加密
    aes_key: str = field(default_factory=default_aes_key)
    cipher_key_id: str = "k1"

    # Web
    host: str = "127.0.0.1"
    port: int = 8080
    secret_key: str = "facerec-demo-secret"
    max_upload_mb: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = Path(_env_str("FACE_MODEL_PATH", str(APP_ROOT / "model" / DEFAULT_MODEL_FILE)))
        db_path = Path(_env_str("FACE_DB_PATH", str(DATA_ROOT / "data" / "facerec.db")))
        return cls(
            model_path=model_path,
            db_path=db_path,
            threshold=_env_float("FACE_THRESHOLD", 0.70),
            timeout_ms=_env_int("FACE_TIMEOUT_MS", 2000),
            max_samples=_env_int("FACE_MAX_SAMPLES", 5),
            min_face_size=_env_int("FACE_MIN_SIZE", 80),
            aes_key=_env_str("FACE_AES_KEY", default_aes_key()),
            cipher_key_id=_env_str("FACE_CIPHER_KEY_ID", "k1"),
            host=_env_str("FACE_WEB_HOST", "127.0.0.1"),
            port=_env_int("FACE_WEB_PORT", 8080),
            secret_key=_env_str("FACE_SECRET_KEY", "facerec-demo-secret"),
            max_upload_mb=_env_int("FACE_MAX_UPLOAD_MB", 8),
        )

    @property
    def model_available(self) -> bool:
        return self.model_path.is_file()

    @property
    def model_version(self) -> str:
        return self.model_path.stem or DEFAULT_MODEL_FILE.split(".")[0]

    def as_public_dict(self) -> dict:
        """对外暴露的只读配置（不含密钥）。"""
        return {
            "threshold": self.threshold,
            "timeout_ms": self.timeout_ms,
            "max_samples": self.max_samples,
            "min_face_size": self.min_face_size,
            "model_path": str(self.model_path),
            "model_version": self.model_version,
            "model_available": self.model_available,
            "db_path": str(self.db_path),
        }
