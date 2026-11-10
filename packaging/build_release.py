"""构建 Windows 可执行产物（PyInstaller onedir）并打 zip。

用法::

    python packaging/build_release.py            # 构建 + 打 zip
    python packaging/build_release.py --no-zip   # 只构建目录

产物::

    dist/FaceRec/FaceRec.exe        # 可执行文件
    dist/FaceRec/_internal/         # 依赖、模型、前端资源
    dist/FaceRec/data/facerec.db    # 预置演示库
    dist/FaceRec-1.0.0-win64.zip    # 分发包
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_NAME = "FaceRec"
VERSION = "1.0.0"


def run_pyinstaller() -> None:
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        APP_NAME,
        "--onedir",
        "--console",
        "--add-data",
        f"web{';' if sys.platform == 'win32' else ':'}web",
        "--add-data",
        f"model/facenet2rec-inference.pb{';' if sys.platform == 'win32' else ':'}model",
        "--add-data",
        f"facerec/schema_sqlite.sql{';' if sys.platform == 'win32' else ':'}facerec",
        "--collect-submodules",
        "facerec",
        "--collect-all",
        "cv2",
        "--collect-all",
        "numpy",
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "PyQt5",
        "--exclude-module",
        "pytest",
        "run.py",
    ]
    print("执行:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def assemble() -> Path:
    app_dir = DIST / APP_NAME
    data_dir = app_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    src_db = ROOT / "data" / "facerec.db"
    if src_db.is_file():
        shutil.copyfile(src_db, data_dir / "facerec.db")
        print(f"已复制演示库: {src_db} -> {data_dir / 'facerec.db'}")
    (app_dir / "start.bat").write_text(
        '@echo off\r\nchcp 65001 >nul\r\ncd /d "%~dp0"\r\nFaceRec.exe --no-browser %*\r\npause\r\n',
        encoding="utf-8",
    )
    (app_dir / "启动说明.txt").write_text(
        "FaceNet2Rec 校园人脸识别系统 V1.0.0\n"
        "=====================================\n\n"
        "1. 双击 FaceRec.exe（或 start.bat）启动；\n"
        "2. 浏览器打开 http://127.0.0.1:8080/login 登录（首次自动跳转登录页）；\n"
        "3. 演示账号：普通用户 demo/demo123 · 管理员 admin/admin123；可注册新用户；\n"
        "4. 登录后：识别终端 / 人脸注册 / 特征库（仅管理员）；\n"
        "5. 首次使用请允许浏览器访问摄像头；停止服务请关闭控制台或 Ctrl+C。\n\n"
        "换端口：FaceRec.exe --port 8090\n"
        "自检：  FaceRec.exe --check\n\n"
        "数据在 data/，日志在 logs/；删除 data/facerec.db 可重置（账号表 fr_account 同库）。\n",
        encoding="utf-8",
    )
    return app_dir


def make_zip(app_dir: Path) -> Path:
    out = DIST / f"{APP_NAME}-{VERSION}-win64.zip"
    if out.exists():
        out.unlink()
    root_name = f"{APP_NAME}-{VERSION}-win64"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in app_dir.rglob("*"):
            if path.is_file():
                arc = Path(root_name) / path.relative_to(app_dir)
                zf.write(path, arc.as_posix())
    print(f"分发包: {out}  ({out.stat().st_size / 1048576:.1f} MB)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-zip", action="store_true", help="只构建目录，不打包 zip")
    parser.add_argument("--skip-build", action="store_true", help="复用已有 dist 目录，只重新组装")
    args = parser.parse_args()

    if not args.skip_build:
        run_pyinstaller()
    app_dir = assemble()
    if not args.no_zip:
        make_zip(app_dir)
    print(f"完成: {app_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
