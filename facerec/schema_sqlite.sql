-- ---------------------------------------------------------------------------
-- 人脸识别系统 —— 演示库（SQLite）
--
-- 说明：课程交付的标准数据库为 MySQL 8.0（见 sql/01_schema.sql，来自
-- 《类图与数据库设计说明书》）。为了让演示包做到"零配置启动"、并让单元
-- 测试可以脱离数据库服务运行，本系统内置同一逻辑模型的 SQLite 版本。
-- 关系模型与 MySQL 版保持一致：用户 1:N 人脸特征、用户 1:N 识别记录。
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fr_user (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_no     TEXT    NOT NULL UNIQUE,          -- 学号 / 工号，全局唯一
    user_name   TEXT    NOT NULL,                 -- 姓名
    dept        TEXT,                             -- 学院 / 部门
    phone       TEXT,                             -- 联系方式
    status      INTEGER NOT NULL DEFAULT 1,       -- 1 正常 2 停用
    create_time TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS fr_feature (
    feature_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES fr_user(user_id) ON DELETE CASCADE,
    feature_no    INTEGER NOT NULL,               -- 同一用户样本序号 1~5
    feature_cipher TEXT   NOT NULL,               -- AES-256-GCM 密文
    feature_dim   INTEGER NOT NULL DEFAULT 512,
    model_version TEXT    NOT NULL,
    quality_score REAL,
    feature_status INTEGER NOT NULL DEFAULT 1,    -- 1 有效 0 失效
    create_time   TEXT    NOT NULL,
    UNIQUE (user_id, feature_no)
);

CREATE TABLE IF NOT EXISTS fr_recognize_log (
    record_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER REFERENCES fr_user(user_id) ON DELETE SET NULL,
    device_id     TEXT,
    result_type   INTEGER NOT NULL,               -- 1 成功 2 无人脸 3 多人 4 相似度不足 5 超时 6 异常
    similarity    REAL,
    threshold     REAL    NOT NULL,
    face_count    INTEGER NOT NULL DEFAULT 0,
    cost_ms       INTEGER NOT NULL DEFAULT 0,
    recognize_time TEXT   NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feature_user ON fr_feature(user_id, feature_status);
CREATE INDEX IF NOT EXISTS idx_user_no ON fr_user(user_no);
CREATE INDEX IF NOT EXISTS idx_recog_time ON fr_recognize_log(recognize_time);
CREATE INDEX IF NOT EXISTS idx_recog_result ON fr_recognize_log(result_type, recognize_time);

CREATE TABLE IF NOT EXISTS fr_config (
    config_key   TEXT PRIMARY KEY,
    config_value TEXT NOT NULL,
    value_type   TEXT NOT NULL DEFAULT 'STRING',
    config_desc  TEXT
);

-- 系统登录账号（与 fr_user 人脸档案分离：账号用于 Web 登录，fr_user 用于刷脸身份）
CREATE TABLE IF NOT EXISTS fr_account (
    account_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    display_name  TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK (role IN ('user', 'admin')),
    status        INTEGER NOT NULL DEFAULT 1,   -- 1 正常 2 停用
    create_time   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_account_role ON fr_account(role, status);
