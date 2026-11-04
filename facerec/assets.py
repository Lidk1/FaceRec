"""资源路径适配。

OpenCV 的 ``imread`` / ``CascadeClassifier`` 使用窄字符文件接口，在中文路径
（Windows 上非常常见）下会打开失败。项目里所有图片读取都走 ``imdecode`` +
``np.fromfile``，但级联模型只能按文件名加载，因此这里在检测到非 ASCII 路径时
把资源复制到一个可写的 ASCII 临时目录，再交给 OpenCV。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_CACHE_DIR_NAME = "facerec_assets"


def is_ascii_path(path: str | Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _candidate_dirs() -> list[Path]:
    """按优先级给出可用的 ASCII 临时目录（去重）。"""
    candidates: list[Path] = []
    for raw in (
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
        tempfile.gettempdir(),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp") if os.environ.get("LOCALAPPDATA") else None,
        os.path.join(os.environ.get("SystemRoot", ""), "Temp") if os.environ.get("SystemRoot") else None,
    ):
        if not raw:
            continue
        path = Path(raw)
        if is_ascii_path(path) and path not in candidates:
            candidates.append(path)
    return candidates


def opencv_readable_path(path: str | Path) -> Path:
    """返回 OpenCV 可以打开的资源路径。

    Args:
        path: 原始资源路径（可能包含中文）。

    Returns:
        原路径（纯 ASCII 时）或 ASCII 临时目录中的副本路径。

    Raises:
        FileNotFoundError: 源文件不存在。
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    if is_ascii_path(str(src)):
        return src

    for base_dir in _candidate_dirs():
        target_dir = base_dir / _CACHE_DIR_NAME
        target = target_dir / src.name
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != src.stat().st_size:
                shutil.copyfile(src, target)
            return target
        except OSError:
            continue
    # 找不到可写 ASCII 目录时退回原路径，由调用方给出"模型未就绪"提示
    return src
