"""数据访问层（SQLite 演示库）。

类职责与《类图与数据库设计说明书》中的 BaseDao / UserDao / FaceFeatureDao /
RecognitionDao 对应，只是把 JDBC 换成标准库 sqlite3，SQL 保持可读。
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .errors import DuplicateUserError, StoreError, UserNotFoundError
from .utils import now_text

_SCHEMA_FILE = Path(__file__).with_name("schema_sqlite.sql")


@dataclass
class UserRecord:
    user_id: int
    user_no: str
    user_name: str
    dept: Optional[str]
    phone: Optional[str]
    status: int
    create_time: str
    feature_count: int = 0

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "user_no": self.user_no,
            "user_name": self.user_name,
            "dept": self.dept,
            "phone": self.phone,
            "status": self.status,
            "create_time": self.create_time,
            "feature_count": self.feature_count,
        }


@dataclass
class FeatureRecord:
    feature_id: int
    user_id: int
    feature_no: int
    feature_cipher: str
    feature_dim: int
    model_version: str
    quality_score: Optional[float]
    create_time: str

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "user_id": self.user_id,
            "feature_no": self.feature_no,
            "feature_dim": self.feature_dim,
            "model_version": self.model_version,
            "quality_score": self.quality_score,
            "create_time": self.create_time,
        }


class FaceStore:
    """人脸库存储。连接按操作创建，天然适配多线程 Web 服务。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._write_lock = threading.RLock()
        self._schema = _SCHEMA_FILE.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ 基础
    def connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            return conn
        except sqlite3.Error as exc:  # pragma: no cover - 磁盘/权限异常
            raise StoreError(f"数据库连接失败: {exc}") from exc

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self.connect() as conn:
            conn.executescript(self._schema)

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ 用户
    def create_user(self, user_no: str, user_name: str, dept: str | None = None, phone: str | None = None) -> int:
        now = now_text()
        try:
            with self._write_lock, self.connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO fr_user (user_no, user_name, dept, phone, status, create_time) VALUES (?, ?, ?, ?, 1, ?)",
                    (user_no, user_name, dept, phone, now),
                )
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper():
                raise DuplicateUserError(f"学号/工号 {user_no} 已注册") from exc
            raise StoreError(str(exc)) from exc

    def get_user(self, user_id: int) -> UserRecord:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT u.*, (SELECT COUNT(*) FROM fr_feature f
                                 WHERE f.user_id = u.user_id AND f.feature_status = 1) AS feature_count
                   FROM fr_user u WHERE u.user_id = ?""",
                (user_id,),
            ).fetchone()
        if row is None:
            raise UserNotFoundError(f"用户 {user_id} 不存在")
        return _row_to_user(row)

    def get_user_by_no(self, user_no: str) -> Optional[UserRecord]:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT u.*, (SELECT COUNT(*) FROM fr_feature f
                                 WHERE f.user_id = u.user_id AND f.feature_status = 1) AS feature_count
                   FROM fr_user u WHERE u.user_no = ?""",
                (user_no,),
            ).fetchone()
        return _row_to_user(row) if row else None

    def list_users(self, keyword: str | None = None, limit: int = 50, offset: int = 0) -> List[UserRecord]:
        sql = """SELECT u.*, (SELECT COUNT(*) FROM fr_feature f
                               WHERE f.user_id = u.user_id AND f.feature_status = 1) AS feature_count
                 FROM fr_user u"""
        params: list = []
        if keyword:
            sql += " WHERE u.user_no LIKE ? OR u.user_name LIKE ?"
            like = f"%{keyword}%"
            params += [like, like]
        sql += " ORDER BY u.user_id DESC LIMIT ? OFFSET ?"
        params += [max(1, limit), max(0, offset)]
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_user(row) for row in rows]

    def count_users(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM fr_user").fetchone()[0])

    def delete_user(self, user_id: int) -> bool:
        with self._write_lock, self.connect() as conn:
            cursor = conn.execute("DELETE FROM fr_user WHERE user_id = ?", (user_id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ 特征
    def add_feature(
        self,
        user_id: int,
        feature_cipher: str,
        model_version: str,
        quality_score: float | None = None,
        feature_dim: int = 512,
        feature_no: int | None = None,
    ) -> int:
        now = now_text()
        with self._write_lock, self.connect() as conn:
            if feature_no is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(feature_no), 0) FROM fr_feature WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                feature_no = int(row[0]) + 1
            cursor = conn.execute(
                """INSERT INTO fr_feature
                   (user_id, feature_no, feature_cipher, feature_dim, model_version,
                    quality_score, feature_status, create_time)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                (user_id, feature_no, feature_cipher, feature_dim, model_version, quality_score, now),
            )
            return int(cursor.lastrowid)

    def list_features(self, user_id: int, only_valid: bool = True) -> List[FeatureRecord]:
        sql = "SELECT * FROM fr_feature WHERE user_id = ?"
        if only_valid:
            sql += " AND feature_status = 1"
        sql += " ORDER BY feature_no"
        with self.connect() as conn:
            rows = conn.execute(sql, (user_id,)).fetchall()
        return [_row_to_feature(row) for row in rows]

    def gallery(self) -> List[Tuple[int, str, str, str]]:
        """全部有效特征：[(user_id, user_no, user_name, cipher), ...]。"""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT f.user_id, u.user_no, u.user_name, f.feature_cipher
                   FROM fr_feature f JOIN fr_user u ON u.user_id = f.user_id
                   WHERE f.feature_status = 1 AND u.status = 1
                   ORDER BY f.user_id, f.feature_no"""
            ).fetchall()
        return [(int(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in rows]

    def count_features(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM fr_feature WHERE feature_status = 1").fetchone()[0])

    def delete_features(self, user_id: int) -> int:
        with self._write_lock, self.connect() as conn:
            cursor = conn.execute("DELETE FROM fr_feature WHERE user_id = ?", (user_id,))
            return int(cursor.rowcount)

    # ------------------------------------------------------------------ 记录
    def add_recognize_log(
        self,
        result_type: int,
        similarity: float | None,
        threshold: float,
        face_count: int,
        cost_ms: int,
        user_id: int | None = None,
        device_id: str | None = None,
    ) -> int:
        with self._write_lock, self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO fr_recognize_log
                   (user_id, device_id, result_type, similarity, threshold, face_count, cost_ms, recognize_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, device_id, result_type, similarity, threshold, face_count, cost_ms, now_text()),
            )
            return int(cursor.lastrowid)

    def list_recognize_logs(self, limit: int = 20, offset: int = 0) -> List[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT r.record_id, r.user_id, u.user_no, u.user_name, r.device_id,
                          r.result_type, r.similarity, r.threshold, r.face_count, r.cost_ms, r.recognize_time
                   FROM fr_recognize_log r LEFT JOIN fr_user u ON u.user_id = r.user_id
                   ORDER BY r.record_id DESC LIMIT ? OFFSET ?""",
                (max(1, limit), max(0, offset)),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict:
        """识别统计：总量、成功率、平均相似度/耗时、失败原因分布、近 7 日趋势。"""
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM fr_recognize_log").fetchone()[0]
            success = conn.execute("SELECT COUNT(*) FROM fr_recognize_log WHERE result_type = 1").fetchone()[0]
            avg_sim = conn.execute("SELECT AVG(similarity) FROM fr_recognize_log WHERE result_type = 1").fetchone()[0]
            avg_cost = conn.execute("SELECT AVG(cost_ms) FROM fr_recognize_log").fetchone()[0]
            rows = conn.execute(
                """SELECT result_type, COUNT(*) AS cnt FROM fr_recognize_log
                   GROUP BY result_type ORDER BY result_type"""
            ).fetchall()
            daily = conn.execute(
                """SELECT substr(recognize_time, 1, 10) AS day, COUNT(*) AS cnt,
                          SUM(CASE WHEN result_type = 1 THEN 1 ELSE 0 END) AS success
                   FROM fr_recognize_log
                   WHERE recognize_time >= date('now', '-6 days')
                   GROUP BY day ORDER BY day"""
            ).fetchall()
        result = {
            "total": int(total),
            "success": int(success),
            "success_rate": round(success / total, 4) if total else 0.0,
            "avg_similarity": round(float(avg_sim), 4) if avg_sim is not None else None,
            "avg_cost_ms": round(float(avg_cost), 1) if avg_cost is not None else None,
            "by_result": {int(r["result_type"]): int(r["cnt"]) for r in rows},
            "daily": [dict(row) for row in daily],
        }
        return result

    # ------------------------------------------------------------------ 配置
    def get_config(self, key: str, default: str | None = None) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute("SELECT config_value FROM fr_config WHERE config_key = ?", (key,)).fetchone()
        return str(row[0]) if row else default

    def set_config(self, key: str, value: str, value_type: str = "STRING", desc: str | None = None) -> None:
        with self._write_lock, self.connect() as conn:
            conn.execute(
                """INSERT INTO fr_config (config_key, config_value, value_type, config_desc)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(config_key) DO UPDATE SET config_value = excluded.config_value""",
                (key, str(value), value_type, desc),
            )


def _row_to_user(row: sqlite3.Row) -> UserRecord:
    keys = row.keys()
    return UserRecord(
        user_id=int(row["user_id"]),
        user_no=str(row["user_no"]),
        user_name=str(row["user_name"]),
        dept=row["dept"],
        phone=row["phone"],
        status=int(row["status"]),
        create_time=str(row["create_time"]),
        feature_count=int(row["feature_count"]) if "feature_count" in keys else 0,
    )


def _row_to_feature(row: sqlite3.Row) -> FeatureRecord:
    return FeatureRecord(
        feature_id=int(row["feature_id"]),
        user_id=int(row["user_id"]),
        feature_no=int(row["feature_no"]),
        feature_cipher=str(row["feature_cipher"]),
        feature_dim=int(row["feature_dim"]),
        model_version=str(row["model_version"]),
        quality_score=row["quality_score"],
        create_time=str(row["create_time"]),
    )
