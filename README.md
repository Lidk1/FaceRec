# FaceNet2Rec 校园人脸识别系统（FaceRec）

基于自研训练 FaceNet2Rec（Inception-ResNet-v1 / VGGFace2，512 维特征）的校园人脸识别子系统，
提供 **人脸注册 → 加密入库 → 实时刷脸核验 → 识别记录与统计** 的完整链路，是校园人脸支付系统
（《基于 FaceNet2Rec 的校园人脸支付系统 V1.0》）的人脸识别能力实现。

- **检测**：OpenCV Haar 级联（alt2 主检测器 + default 兜底），单帧 20~60ms
- **特征**：FaceNet2Rec 冻结图经 OpenCV DNN 做 CPU 推理，输出 512 维 L2 归一化向量，约 40ms/帧
- **比对**：余弦相似度 1:N 比对，阈值默认 0.70（实测校准，见开发文档）
- **安全**：特征向量 AES-256-GCM 加密存储，密钥带版本号；原始人脸图像不落盘
- **界面**：中文 Web 终端（摄像头实时识别 / 注册 / 特征库管理），无需前端构建工具
- **工程**：Flask 分层架构、统一异常与返回结构、143 个 pytest 单元测试、PyInstaller 可执行产物

## 快速开始

```bash
# 1) 安装依赖（Python 3.9+）
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

# 2) 准备模型：把 FaceNet 冻结图放到 model/ 并裁成纯推理图
python scripts/prune_graph.py model/20180402-114759.pb model/facenet2rec-inference.pb

# 3) 初始化演示数据（使用课程小组真实照片注册 6 位演示用户）
python scripts/seed_demo.py --force

# 4) 启动
python run.py            # 浏览器自动打开 http://127.0.0.1:8080/
```

详细步骤（含模型获取、环境变量、常见问题）见 [docs/开发技术文档.md](docs/开发技术文档.md) 与
[docs/启动说明.md](docs/启动说明.md)。

## 目录结构

```
FaceRec/
├─ facerec/                 # 业务包
│  ├─ app.py                # Flask 应用工厂
│  ├─ api.py                # REST 接口（/api/face/*）
│  ├─ service.py            # 人脸识别服务（注册/识别/管理）
│  ├─ engine.py             # 检测→提取→比对→判定
│  ├─ detector.py           # Haar 检测 + 主人脸选取
│  ├─ encoder.py            # FaceNet2Rec（OpenCV DNN）推理
│  ├─ cipher.py             # AES-256-GCM 特征加解密
│  ├─ store.py              # SQLite 数据访问（对应 BaseDao/UserDao/...）
│  ├─ cli.py                # 命令行工具
│  └─ schema_sqlite.sql     # 演示库建表脚本
├─ web/                     # 前端页面（模板 + 原生 JS/CSS）
├─ scripts/                 # prune_graph.py 模型图裁剪、seed_demo.py 演示数据
├─ tests/                   # pytest 单元测试（143 例）
├─ sql/mysql/               # MySQL 建表与测试数据脚本（正式库）
├─ docs/                    # 开发技术文档、启动说明
├─ model/                   # 模型权重（不入库，见文档）
├─ run.py                   # 启动入口
└─ requirements*.txt
```

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/face/status` | 模型/特征库状态、阈值、引擎信息 |
| POST | `/api/face/enroll` | 人脸注册（JSON Base64 或多文件 multipart） |
| POST | `/api/face/recognize` | 刷脸核验（1 帧），返回用户、相似度、耗时 |
| POST | `/api/face/verify` | 1:1 人脸比对 |
| GET | `/api/face/users` | 用户列表（关键字/分页） |
| GET | `/api/face/users/{id}` | 用户与特征明细（不返回密文） |
| DELETE | `/api/face/users/{id}/features` | 删除人脸特征 |
| DELETE | `/api/face/users/{id}` | 注销用户 |
| GET | `/api/face/records` | 识别记录 |
| GET | `/api/face/stats` | 识别成功率、失败分布、趋势 |

统一响应：`{"code": 200, "msg": "success", "data": {...}}`。完整参数与错误码见开发文档第 6 章。

## 测试

```bash
pytest                 # 143 个用例，约 5 秒
```

覆盖：向量数学与阈值、AES-GCM 加解密与防篡改、主人脸选取、引擎判定分支、
服务层注册/识别/幂等/统计、HTTP 接口与错误码、图裁剪工具、配置加载。

## 分支模型（多人协作）

| 分支 | 用途 | 合并方向 |
| --- | --- | --- |
| `main` | 可发布版本，仅接受 release 合并 | — |
| `develop` | 集成分支，日常合入 feature | → main |
| `feature/face-core` | 检测、特征提取、模型图裁剪 | → develop |
| `feature/api-web` | 接口层与前端页面 | → develop |
| `feature/tests-docs` | 单元测试与文档 | → develop |
| `release/v1.0.0` | 发布冻结与打包 | → main |

提交信息：`feat(face): 新增主人脸选取`、`fix(api): 修正 404 返回码` 等。

## 模型与许可

推理权重为 FaceNet 公开预训练模型（Inception-ResNet-v1，VGGFace2），仅用于课程原型验证；
模型文件不随仓库分发，获取与裁剪步骤见文档。项目中引用的 Haar 级联与 OpenCV 遵循其各自许可。

## 文档

- [开发技术文档](docs/开发技术文档.md)：环境搭建、项目结构、核心逻辑、接口说明、数据库、测试、性能与安全
- [启动说明](docs/启动说明.md)：源码启动与可执行产物启动
- [提交清单](docs/提交清单.md)：本次交付物对照
