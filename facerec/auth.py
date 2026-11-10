"""Web 登录会话与账号认证（独立于人脸档案 fr_user）。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional

from flask import jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .errors import AuthError, DuplicateAccountError, ForbiddenError, ValidateError
from .store import AccountRecord, FaceStore
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")
ROLE_USER = "user"
ROLE_ADMIN = "admin"


@dataclass
class SessionAccount:
    account_id: int
    username: str
    display_name: str
    role: str

    def as_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
        }


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise ValidateError("用户名须为 3~32 位字母、数字或下划线")
    return username


def validate_password(password: str, *, registering: bool = False) -> str:
    password = password or ""
    if len(password) < 6:
        raise ValidateError("密码至少 6 位")
    if registering and len(password) > 64:
        raise ValidateError("密码不超过 64 位")
    return password


class AuthService:
    """账号注册、登录与默认演示账号初始化。"""

    def __init__(self, store: FaceStore) -> None:
        self.store = store

    def ensure_seed_accounts(self) -> None:
        if self.store.count_accounts() > 0:
            return
        self.store.create_account(
            "admin",
            generate_password_hash("admin123"),
            "系统管理员",
            ROLE_ADMIN,
        )
        self.store.create_account(
            "demo",
            generate_password_hash("demo123"),
            "演示用户",
            ROLE_USER,
        )

    def register(self, username: str, password: str, display_name: str) -> AccountRecord:
        username = validate_username(username)
        password = validate_password(password, registering=True)
        display_name = (display_name or "").strip() or username
        if len(display_name) > 30:
            raise ValidateError("显示名称不超过 30 字")
        return self.store.create_account(username, generate_password_hash(password), display_name, ROLE_USER)

    def login(self, username: str, password: str, role: str) -> AccountRecord:
        username = validate_username(username)
        password = validate_password(password)
        role = (role or ROLE_USER).strip().lower()
        if role not in {ROLE_USER, ROLE_ADMIN}:
            raise ValidateError("无效登录角色")
        account = self.store.get_account_by_username(username)
        if account is None or account.status != 1:
            raise AuthError("用户名或密码错误")
        if not check_password_hash(account.password_hash, password):
            raise AuthError("用户名或密码错误")
        if account.role != role:
            hint = "管理员" if role == ROLE_ADMIN else "普通用户"
            raise AuthError(f"该账号不是{hint}账号，请切换登录入口")
        return account

    @staticmethod
    def write_session(account: AccountRecord) -> None:
        session.clear()
        session.permanent = True
        session["account_id"] = account.account_id
        session["username"] = account.username
        session["display_name"] = account.display_name
        session["role"] = account.role

    @staticmethod
    def clear_session() -> None:
        session.clear()

    @staticmethod
    def current() -> Optional[SessionAccount]:
        aid = session.get("account_id")
        if not aid:
            return None
        return SessionAccount(
            account_id=int(aid),
            username=str(session.get("username", "")),
            display_name=str(session.get("display_name", "")),
            role=str(session.get("role", ROLE_USER)),
        )


def login_required(view: Callable):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("account_id"):
            if request.path.startswith("/api/"):
                return jsonify({"code": 1010, "msg": "请先登录", "data": None}), 401
            return redirect(url_for("pages.login_page", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view: Callable):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("account_id"):
            if request.path.startswith("/api/"):
                return jsonify({"code": 1010, "msg": "请先登录", "data": None}), 401
            return redirect(url_for("pages.login_page", next=request.path))
        if session.get("role") != ROLE_ADMIN:
            if request.path.startswith("/api/"):
                return jsonify({"code": 1012, "msg": "需要管理员权限", "data": None}), 403
            return redirect(url_for("pages.terminal_page"))
        return view(*args, **kwargs)

    return wrapped
