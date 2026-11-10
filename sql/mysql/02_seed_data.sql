-- ============================================================================
--  FaceNet2Rec 校园人脸识别系统 —— 测试数据（MySQL 8.0）
--
--  执行顺序：01_schema.sql -> 02_seed_data.sql
--  说明：人脸特征密文由运行期密钥生成，不能预置到 SQL 中；执行完本脚本后，
--        请通过 `python scripts/seed_demo.py` 或 POST /api/face/enroll 录入
--        演示用户的人脸特征（密文与 FACE_AES_KEY 保持一致）。
-- ============================================================================

USE `face_rec`;
SET NAMES utf8mb4;

-- ----------------------------------------------------------------------------
-- 角色
-- ----------------------------------------------------------------------------
INSERT INTO `sys_role` (`role_code`, `role_name`, `role_desc`) VALUES
 ('STUDENT', '学生',       '校园消费主体，可注册人脸、刷脸核验、查询本人记录'),
 ('TEACHER', '教职工',     '校园消费主体，具备记录导出权限'),
 ('ADMIN',   '系统管理员', '负责特征库维护、识别统计与参数配置');

-- ----------------------------------------------------------------------------
-- 权限点
-- ----------------------------------------------------------------------------
INSERT INTO `sys_permission` (`perm_code`, `perm_name`, `module`) VALUES
 ('face:enroll',    '人脸注册',     '人脸识别'),
 ('face:recognize', '人脸核验',     '人脸识别'),
 ('feature:query',  '特征库查询',   '特征库管理'),
 ('feature:delete', '删除人脸特征', '特征库管理'),
 ('stat:view',      '识别统计查看', '模型监测'),
 ('config:update',  '参数配置',     '系统配置');

INSERT INTO `sys_role_permission` (`role_id`, `perm_id`)
SELECT r.role_id, p.perm_id
FROM `sys_role` r JOIN `sys_permission` p
WHERE (r.role_code = 'STUDENT' AND p.perm_code IN ('face:enroll', 'face:recognize'))
   OR (r.role_code = 'TEACHER' AND p.perm_code IN ('face:enroll', 'face:recognize'))
   OR (r.role_code = 'ADMIN'   AND p.perm_code IN ('feature:query', 'feature:delete', 'stat:view', 'config:update'));

-- ----------------------------------------------------------------------------
-- 管理员与演示用户（口令由统一认证模块维护，此处留空）
-- ----------------------------------------------------------------------------
INSERT INTO `sys_user`
 (`user_no`, `user_name`, `gender`, `phone`, `password_hash`, `role_id`, `college`, `class_name`, `job_title`)
SELECT 'admin', '系统管理员', '0', '13800000000', NULL, role_id, NULL, NULL, NULL
FROM `sys_role` WHERE `role_code` = 'ADMIN';

INSERT INTO `sys_user`
 (`user_no`, `user_name`, `gender`, `phone`, `password_hash`, `role_id`, `college`, `class_name`)
SELECT '2023010101', '韩萧', '1', '13800000101', NULL, role_id, '计算机学院', '软件工程 2023'
FROM `sys_role` WHERE `role_code` = 'STUDENT'
UNION ALL
SELECT '2023010102', '井泉清', '1', '13800000102', NULL, role_id, '计算机学院', '软件工程 2023'
FROM `sys_role` WHERE `role_code` = 'STUDENT'
UNION ALL
SELECT '2023010103', '王琳博', '1', '13800000103', NULL, role_id, '计算机学院', '软件工程 2023'
FROM `sys_role` WHERE `role_code` = 'STUDENT'
UNION ALL
SELECT '2023010104', '李定坤', '1', '13800000104', NULL, role_id, '计算机学院', '软件工程 2023'
FROM `sys_role` WHERE `role_code` = 'STUDENT'
UNION ALL
SELECT '2022010201', '梁爽', '2', '13800000201', NULL, role_id, '计算机学院', '软件工程 2022'
FROM `sys_role` WHERE `role_code` = 'STUDENT'
UNION ALL
SELECT '2022010202', '朱佳豪', '1', '13800000202', NULL, role_id, '计算机学院', '软件工程 2022'
FROM `sys_role` WHERE `role_code` = 'STUDENT';

-- ----------------------------------------------------------------------------
-- 商户与消费终端
-- ----------------------------------------------------------------------------
INSERT INTO `mch_merchant` (`merchant_no`, `merchant_name`, `merchant_type`, `location`, `contact_phone`) VALUES
 ('M001', '第一食堂 · 一楼档口', 1, '第一食堂一层', '0571-88000001'),
 ('M002', '校园超市 · 东区店',  2, '东区生活广场', '0571-88000002');

INSERT INTO `dev_terminal` (`device_id`, `merchant_id`, `device_name`, `ip_addr`, `device_status`)
SELECT 'T001', `merchant_id`, '一食堂 1 号终端', '192.168.10.11', 1 FROM `mch_merchant` WHERE `merchant_no` = 'M001'
UNION ALL
SELECT 'T002', `merchant_id`, '超市东区收银台', '192.168.10.21', 1 FROM `mch_merchant` WHERE `merchant_no` = 'M002';

-- ----------------------------------------------------------------------------
-- 系统参数（阈值经实测校准为 0.70，见开发技术文档 4.6）
-- ----------------------------------------------------------------------------
INSERT INTO `sys_config` (`config_key`, `config_value`, `value_type`, `config_desc`) VALUES
 ('face.similarity.threshold', '0.70', 'DECIMAL', '人脸比对余弦相似度判定阈值'),
 ('face.recognize.timeout_ms', '2000', 'INT',     '单帧识别超时阈值（毫秒）'),
 ('face.sample.count',         '5',    'INT',     '注册时采集的有效人脸样本上限'),
 ('face.min.size',             '80',   'INT',     '可接受的最小人脸边长（像素）'),
 ('face.model.version',        'facenet2rec-inference', 'STRING', '当前推理模型版本'),
 ('session.user.timeout_min',  '60',   'INT',     '普通用户会话超时（分钟）'),
 ('session.admin.timeout_min', '30',   'INT',     '管理员会话超时（分钟）');

-- ----------------------------------------------------------------------------
-- 校验
-- ----------------------------------------------------------------------------
SELECT '用户' AS 指标, COUNT(*) AS 数量 FROM `sys_user`
UNION ALL SELECT '角色', COUNT(*) FROM `sys_role`
UNION ALL SELECT '权限点', COUNT(*) FROM `sys_permission`
UNION ALL SELECT '商户', COUNT(*) FROM `mch_merchant`
UNION ALL SELECT '终端', COUNT(*) FROM `dev_terminal`
UNION ALL SELECT '系统参数', COUNT(*) FROM `sys_config`;
