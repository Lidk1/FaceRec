-- ============================================================================
--  FaceNet2Rec 校园人脸识别系统 —— 正式库建表脚本（MySQL 8.0）
--
--  依据《类图与数据库设计说明书》第 3 章：命名规范“模块前缀_业务名”，
--  字符集 utf8mb4 / 排序规则 utf8mb4_0900_ai_ci，引擎 InnoDB，
--  金额 DECIMAL(12,2)、时间 DATETIME(3)、状态 TINYINT。
--
--  说明：本脚本覆盖人脸识别子系统实际使用的表（用户、角色权限、人脸特征、
--  商家/终端、识别记录、系统参数）。账户与交易相关表见设计文档完整脚本。
--
--  版本：V1.0.0    适用：MySQL 8.0.36
-- ============================================================================

DROP DATABASE IF EXISTS `face_rec`;
CREATE DATABASE `face_rec` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE `face_rec`;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------------------------------------------------------
-- 角色表：学生、教职工、管理员
-- ----------------------------------------------------------------------------
CREATE TABLE `sys_role` (
  `role_id`     INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '角色主键',
  `role_code`   VARCHAR(20)  NOT NULL COMMENT '角色编码：STUDENT / TEACHER / ADMIN',
  `role_name`   VARCHAR(30)  NOT NULL COMMENT '角色名称',
  `role_desc`   VARCHAR(200) NULL DEFAULT NULL COMMENT '角色职责说明',
  `create_time` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`role_id`),
  UNIQUE KEY `uk_role_code` (`role_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='角色表';

-- ----------------------------------------------------------------------------
-- 权限表：功能权限点（删除特征、导出流水等）
-- ----------------------------------------------------------------------------
CREATE TABLE `sys_permission` (
  `perm_id`     INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '权限主键',
  `perm_code`   VARCHAR(50)  NOT NULL COMMENT '权限编码，如 feature:delete',
  `perm_name`   VARCHAR(50)  NOT NULL COMMENT '权限名称',
  `module`      VARCHAR(30)  NOT NULL COMMENT '所属功能模块',
  `create_time` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`perm_id`),
  UNIQUE KEY `uk_perm_code` (`perm_code`),
  KEY `idx_perm_module` (`module`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='权限表';

-- ----------------------------------------------------------------------------
-- 角色权限关联表
-- ----------------------------------------------------------------------------
CREATE TABLE `sys_role_permission` (
  `role_id`    INT UNSIGNED NOT NULL COMMENT '角色 ID，外键 sys_role.role_id',
  `perm_id`    INT UNSIGNED NOT NULL COMMENT '权限 ID，外键 sys_permission.perm_id',
  `grant_time` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '授权时间',
  PRIMARY KEY (`role_id`, `perm_id`),
  KEY `idx_rp_perm` (`perm_id`),
  CONSTRAINT `fk_rp_role` FOREIGN KEY (`role_id`) REFERENCES `sys_role` (`role_id`)
    ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_rp_perm` FOREIGN KEY (`perm_id`) REFERENCES `sys_permission` (`perm_id`)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='角色权限关联表';

-- ----------------------------------------------------------------------------
-- 用户表：学生、教职工、管理员
-- ----------------------------------------------------------------------------
CREATE TABLE `sys_user` (
  `user_id`         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '用户主键',
  `user_no`         VARCHAR(20)  NOT NULL COMMENT '学号/工号，全局唯一',
  `user_name`       VARCHAR(50)  NOT NULL COMMENT '真实姓名',
  `gender`          CHAR(1)      NOT NULL DEFAULT '0' COMMENT '性别：0 未知，1 男，2 女',
  `phone`           VARCHAR(20)  NULL DEFAULT NULL COMMENT '联系方式',
  `password_hash`   VARCHAR(100) NULL DEFAULT NULL COMMENT '口令摘要（人脸子系统可为空）',
  `role_id`         INT UNSIGNED NOT NULL COMMENT '角色 ID，外键 sys_role.role_id',
  `college`         VARCHAR(100) NULL DEFAULT NULL COMMENT '所属学院/部门',
  `class_name`      VARCHAR(50)  NULL DEFAULT NULL COMMENT '班级（仅学生）',
  `job_title`       VARCHAR(50)  NULL DEFAULT NULL COMMENT '职称（仅教职工）',
  `user_status`     TINYINT      NOT NULL DEFAULT 1 COMMENT '状态：1 正常，2 停用，3 注销',
  `last_login_time` DATETIME     NULL DEFAULT NULL COMMENT '最近登录时间',
  `create_time`     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
  `update_time`     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后修改时间',
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uk_user_no` (`user_no`),
  KEY `idx_user_role` (`role_id`),
  KEY `idx_user_status` (`user_status`),
  KEY `idx_user_name` (`user_name`),
  CONSTRAINT `ck_user_gender` CHECK (`gender` IN ('0','1','2')),
  CONSTRAINT `fk_user_role` FOREIGN KEY (`role_id`) REFERENCES `sys_role` (`role_id`)
    ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户表';

-- ----------------------------------------------------------------------------
-- 人脸特征表：只保存 512 维特征的 AES-256-GCM 密文，绝无原始图像
-- ----------------------------------------------------------------------------
CREATE TABLE `face_feature` (
  `feature_id`     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '特征主键',
  `user_id`        BIGINT UNSIGNED NOT NULL COMMENT '所属用户，外键 sys_user.user_id',
  `feature_no`     TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '样本序号，同一用户 1~5',
  `feature_cipher` TEXT NOT NULL COMMENT 'AES-256-GCM 密文（v1:key_id:Base64）',
  `feature_dim`    SMALLINT UNSIGNED NOT NULL DEFAULT 512 COMMENT '特征维度',
  `model_version`  VARCHAR(50)  NOT NULL DEFAULT 'facenet2rec-inference' COMMENT '生成该特征的模型版本',
  `quality_score`  DECIMAL(5,4) NULL DEFAULT NULL COMMENT '样本质量评分 0~1',
  `feature_status` TINYINT      NOT NULL DEFAULT 1 COMMENT '状态：1 有效，0 失效',
  `create_time`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '采集入库时间',
  PRIMARY KEY (`feature_id`),
  UNIQUE KEY `uk_feature_user_no` (`user_id`, `feature_no`),
  KEY `idx_feature_status` (`feature_status`),
  CONSTRAINT `ck_feature_dim` CHECK (`feature_dim` = 512),
  CONSTRAINT `fk_feature_user` FOREIGN KEY (`user_id`) REFERENCES `sys_user` (`user_id`)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='人脸特征表（仅密文）';

-- ----------------------------------------------------------------------------
-- 商户表
-- ----------------------------------------------------------------------------
CREATE TABLE `mch_merchant` (
  `merchant_id`     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '商户主键',
  `merchant_no`     VARCHAR(20)  NOT NULL COMMENT '商户编号',
  `merchant_name`   VARCHAR(100) NOT NULL COMMENT '商户名称',
  `merchant_type`   TINYINT      NOT NULL DEFAULT 1 COMMENT '类型：1 食堂，2 超市，3 其他',
  `location`        VARCHAR(200) NULL DEFAULT NULL COMMENT '经营位置',
  `contact_phone`   VARCHAR(20)  NULL DEFAULT NULL COMMENT '联系电话',
  `merchant_status` TINYINT      NOT NULL DEFAULT 1 COMMENT '状态：1 营业，0 停用',
  `create_time`     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '建档时间',
  PRIMARY KEY (`merchant_id`),
  UNIQUE KEY `uk_merchant_no` (`merchant_no`),
  KEY `idx_merchant_name` (`merchant_name`),
  CONSTRAINT `ck_merchant_type` CHECK (`merchant_type` IN (1,2,3))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='商户表';

-- ----------------------------------------------------------------------------
-- 消费终端表
-- ----------------------------------------------------------------------------
CREATE TABLE `dev_terminal` (
  `device_id`      VARCHAR(32) NOT NULL COMMENT '终端编号',
  `merchant_id`    BIGINT UNSIGNED NOT NULL COMMENT '所属商户，外键 mch_merchant.merchant_id',
  `device_name`    VARCHAR(50) NOT NULL COMMENT '终端名称',
  `ip_addr`        VARCHAR(45) NULL DEFAULT NULL COMMENT '内网 IP',
  `device_status`  TINYINT     NOT NULL DEFAULT 1 COMMENT '状态：1 在线，0 离线',
  `last_heartbeat` DATETIME    NULL DEFAULT NULL COMMENT '最近心跳时间',
  `create_time`    DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '登记时间',
  PRIMARY KEY (`device_id`),
  KEY `idx_device_merchant` (`merchant_id`),
  KEY `idx_device_status` (`device_status`),
  CONSTRAINT `fk_device_merchant` FOREIGN KEY (`merchant_id`) REFERENCES `mch_merchant` (`merchant_id`)
    ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='消费终端表';

-- ----------------------------------------------------------------------------
-- 识别记录表：每次刷脸核验的结果、相似度、耗时与失败原因
-- ----------------------------------------------------------------------------
CREATE TABLE `rec_recognize_log` (
  `record_id`      BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '记录主键',
  `user_id`        BIGINT UNSIGNED NULL DEFAULT NULL COMMENT '命中用户，识别失败为空',
  `device_id`      VARCHAR(32) NULL DEFAULT NULL COMMENT '终端编号，外键 dev_terminal.device_id',
  `result_type`    TINYINT     NOT NULL COMMENT '1 成功，2 无人脸，3 多人，4 相似度不足，5 超时，6 异常',
  `similarity`     DECIMAL(6,5) NULL DEFAULT NULL COMMENT '最高余弦相似度 [-1,1]',
  `threshold`      DECIMAL(6,5) NOT NULL DEFAULT 0.70000 COMMENT '本次判定阈值',
  `face_count`     TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '画面检出人脸数',
  `cost_ms`        INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '识别耗时（毫秒）',
  `recognize_time` DATETIME(3) NOT NULL COMMENT '识别时间（毫秒精度）',
  PRIMARY KEY (`record_id`),
  KEY `idx_recog_time` (`recognize_time`),
  KEY `idx_recog_user` (`user_id`, `recognize_time`),
  KEY `idx_recog_result` (`result_type`, `recognize_time`),
  KEY `idx_recog_device` (`device_id`, `recognize_time`),
  CONSTRAINT `fk_recog_user` FOREIGN KEY (`user_id`) REFERENCES `sys_user` (`user_id`)
    ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_recog_device` FOREIGN KEY (`device_id`) REFERENCES `dev_terminal` (`device_id`)
    ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='人脸识别记录表';

-- ----------------------------------------------------------------------------
-- 系统参数表
-- ----------------------------------------------------------------------------
CREATE TABLE `sys_config` (
  `config_id`    INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '参数主键',
  `config_key`   VARCHAR(50)  NOT NULL COMMENT '参数键，如 face.similarity.threshold',
  `config_value` VARCHAR(200) NOT NULL COMMENT '参数值',
  `value_type`   VARCHAR(20)  NOT NULL DEFAULT 'STRING' COMMENT 'STRING / DECIMAL / INT / BOOL',
  `config_desc`  VARCHAR(200) NULL DEFAULT NULL COMMENT '参数说明',
  `update_by`    BIGINT UNSIGNED NULL DEFAULT NULL COMMENT '最后修改人',
  `update_time`  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后修改时间',
  PRIMARY KEY (`config_id`),
  UNIQUE KEY `uk_config_key` (`config_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='系统参数表';

-- ----------------------------------------------------------------------------
-- 视图：用户-特征汇总（不暴露任何密文）
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `v_user_face` AS
SELECT u.user_id, u.user_no, u.user_name, r.role_code, r.role_name,
       u.college, u.user_status,
       COUNT(f.feature_id) AS feature_count,
       MAX(f.create_time)  AS last_enroll_time
FROM sys_user u
JOIN sys_role r ON r.role_id = u.role_id
LEFT JOIN face_feature f ON f.user_id = u.user_id AND f.feature_status = 1
GROUP BY u.user_id, u.user_no, u.user_name, r.role_code, r.role_name, u.college, u.user_status;

-- ----------------------------------------------------------------------------
-- 视图：按日、按终端统计识别效果（成功率、平均相似度、失败分布）
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `v_recog_daily_stat` AS
SELECT DATE(recognize_time) AS stat_date,
       device_id,
       COUNT(*)                                        AS total_cnt,
       SUM(result_type = 1)                            AS success_cnt,
       ROUND(SUM(result_type = 1) / COUNT(*), 4)       AS success_rate,
       ROUND(AVG(similarity), 4)                       AS avg_similarity,
       ROUND(AVG(cost_ms), 1)                          AS avg_cost_ms,
       SUM(result_type = 2)                            AS no_face_cnt,
       SUM(result_type = 3)                            AS multi_face_cnt,
       SUM(result_type = 4)                            AS low_similarity_cnt,
       SUM(result_type = 5)                            AS timeout_cnt
FROM rec_recognize_log
GROUP BY DATE(recognize_time), device_id;

SET FOREIGN_KEY_CHECKS = 1;
