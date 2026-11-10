"""HTTP 接口层（Flask Blueprint）。

统一返回结构：``{"code": 200, "msg": "success", "data": {...}}``。
业务异常由 :class:`facerec.errors.FaceRecError` 统一转换为带错误码的响应，
不向前端暴露堆栈信息。
"""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from .auth import AuthService, admin_required, login_required
from .errors import FaceRecError, ValidateError
from .utils import ImageDecodeError, decode_base64_image

logger = logging.getLogger("facerec.api")

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def ok(data=None, msg: str = "success"):
    return jsonify({"code": 200, "msg": msg, "data": data})


def fail(msg: str, code: int = 500, http_status: int = 200, data=None):
    return jsonify({"code": code, "msg": msg, "data": data}), http_status


def _service():
    return current_app.extensions["facerec"]["service"]


def _auth() -> AuthService:
    return current_app.extensions["facerec"]["auth"]


def _payload() -> dict:
    """读取请求参数：JSON 优先，multipart/form 其次。"""
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        return body
    if request.form:
        return request.form.to_dict()
    return {}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _collect_images() -> list:
    """从 JSON（image/images）或 multipart（image/images 文件）中收集图像。"""
    images: list = []
    if request.files:
        files = request.files.getlist("images") + request.files.getlist("image")
        for storage in files:
            if storage and storage.filename:
                images.append(storage.read())
        if images:
            return images
    body = _payload()
    raw = body.get("images")
    if raw is None and body.get("image") is not None:
        raw = [body["image"]]
    if not raw:
        raise ValidateError("请求中缺少图像数据（image / images）")
    if not isinstance(raw, list):
        raise ValidateError("images 字段必须为数组")
    for item in raw:
        if isinstance(item, str):
            try:
                images.append(decode_base64_image(item))
            except ImageDecodeError as exc:
                raise ValidateError(str(exc)) from exc
        else:
            raise ValidateError("图像需为 Base64 字符串")
    return images


def _int_arg(name: str, default: int, minimum: int = 1, maximum: int = 200) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# 系统
# ---------------------------------------------------------------------------


@api_bp.post("/auth/login")
def auth_login():
    body = _payload()
    account = _auth().login(body.get("username", ""), body.get("password", ""), body.get("role", "user"))
    AuthService.write_session(account)
    return ok(account.as_public_dict(), msg="登录成功")


@api_bp.post("/auth/register")
def auth_register():
    body = _payload()
    account = _auth().register(body.get("username", ""), body.get("password", ""), body.get("display_name", ""))
    return ok(account.as_public_dict(), msg="注册成功，请返回登录")


@api_bp.post("/auth/logout")
def auth_logout():
    AuthService.clear_session()
    return ok(None, msg="已退出登录")


@api_bp.get("/auth/me")
def auth_me():
    account = AuthService.current()
    if not account:
        return fail("未登录", 1010, 401)
    return ok(account.as_dict())


@api_bp.get("/health")
def health():
    svc = _service()
    status = svc.status()
    return ok(
        {
            "service": "facerec",
            "ready": status["ready"],
            "model_version": status["model_version"],
            "threshold": status["threshold"],
            "users": status["users"],
            "features": status["features"],
        }
    )


@api_bp.get("/face/status")
@login_required
def face_status():
    return ok(_service().status())


# ---------------------------------------------------------------------------
# 注册 / 识别
# ---------------------------------------------------------------------------


@api_bp.post("/face/enroll")
@login_required
def enroll():
    body = _payload()
    images = _collect_images()
    summary = _service().enroll(
        user_no=body.get("user_no", ""),
        user_name=body.get("user_name", ""),
        images=images,
        dept=body.get("dept"),
        phone=body.get("phone"),
        replace=_as_bool(body.get("replace", False)),
    )
    return ok(summary.as_dict(), msg=f"注册成功，已采集 {summary.samples} 帧人脸样本")


@api_bp.post("/face/recognize")
@login_required
def recognize():
    body = _payload()
    images = _collect_images()
    if len(images) != 1:
        raise ValidateError("识别接口一次只接受 1 帧图像")
    result = _service().recognize(images[0], device_id=body.get("device_id"))
    return ok(result.as_dict(), msg=result.message)


@api_bp.post("/face/verify")
@login_required
def verify():
    body = _payload()
    raw = body.get("images") or [body.get("image_a"), body.get("image_b")]
    if not raw or len(raw) != 2 or not all(raw):
        raise ValidateError("1:1 比对需要提供两张图像")
    try:
        first, second = (decode_base64_image(item) if isinstance(item, str) else item for item in raw)
    except ImageDecodeError as exc:
        raise ValidateError(str(exc)) from exc
    return ok(_service().verify(first, second))


# ---------------------------------------------------------------------------
# 用户 / 特征库
# ---------------------------------------------------------------------------


@api_bp.get("/face/users")
@login_required
def list_users():
    return ok(
        _service().list_users(
            keyword=request.args.get("keyword") or None,
            limit=_int_arg("limit", 50),
            offset=max(0, int(request.args.get("offset", 0) or 0)),
        )
    )


@api_bp.get("/face/users/<int:user_id>")
@login_required
def get_user(user_id: int):
    return ok(_service().get_user(user_id))


@api_bp.delete("/face/users/<int:user_id>/features")
@admin_required
def delete_features(user_id: int):
    return ok(_service().delete_features(user_id), msg="人脸特征已删除")


@api_bp.delete("/face/users/<int:user_id>")
@admin_required
def delete_user(user_id: int):
    return ok(_service().delete_user(user_id), msg="用户已注销")


# ---------------------------------------------------------------------------
# 识别记录 / 统计
# ---------------------------------------------------------------------------


@api_bp.get("/face/records")
@login_required
def list_records():
    return ok(
        {
            "items": _service().list_records(
                limit=_int_arg("limit", 20),
                offset=max(0, int(request.args.get("offset", 0) or 0)),
            )
        }
    )


@api_bp.get("/face/stats")
@admin_required
def stats():
    return ok(_service().stats())


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------


def register_error_handlers(app) -> None:
    @app.errorhandler(FaceRecError)
    def _handle_business_error(exc: FaceRecError):
        logger.warning("业务异常 code=%s msg=%s", exc.code, exc.msg)
        return fail(exc.msg, exc.code, exc.http_status, exc.data)

    @app.errorhandler(413)
    def _handle_too_large(_exc):
        limit = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
        return fail(f"上传内容过大，单次不超过 {limit}MB", 413, 413)

    @app.errorhandler(404)
    def _handle_not_found(_exc):
        if request.path.startswith("/api/"):
            return fail("接口不存在", 404, 404)
        return fail("页面不存在", 404, 404)

    @app.errorhandler(405)
    def _handle_method(_exc):
        return fail("请求方法不支持", 405, 405)

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception):
        logger.exception("未捕获异常: %s", exc)
        return fail("系统内部异常，请稍后重试", 500, 500)
