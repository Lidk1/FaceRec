"""统一业务异常与错误码。

接口层只返回 code / msg / data 三段式结构，底层异常不得直接暴露给前端。
"""
from __future__ import annotations


class FaceRecError(Exception):
    """业务异常基类。code 为对外错误码，http_status 为 HTTP 状态码。"""

    code = 500
    http_status = 500
    default_msg = "系统内部异常，请稍后重试"

    def __init__(self, msg: str | None = None, *, code: int | None = None, data=None):
        self.msg = msg or self.default_msg
        if code is not None:
            self.code = code
        self.data = data
        super().__init__(self.msg)


class ValidateError(FaceRecError):
    code = 400
    http_status = 400
    default_msg = "请求参数不合法"


class NoFaceError(FaceRecError):
    code = 1001
    http_status = 200
    default_msg = "未检测到人脸，请正对摄像头并保持光线充足"


class MultiFaceError(FaceRecError):
    code = 1002
    http_status = 200
    default_msg = "画面中存在多张人脸，请仅保留一人进行识别"


class LowQualityError(FaceRecError):
    code = 1003
    http_status = 200
    default_msg = "人脸图像质量过低，请靠近摄像头重新采集"


class ModelUnavailableError(FaceRecError):
    code = 1004
    http_status = 503
    default_msg = "人脸模型未就绪，请检查模型文件与依赖环境"


class RecognitionTimeoutError(FaceRecError):
    code = 1005
    http_status = 200
    default_msg = "识别超时，已降级为刷卡通道"


class DuplicateUserError(FaceRecError):
    code = 1006
    http_status = 409
    default_msg = "该学号/工号已注册"


class UserNotFoundError(FaceRecError):
    code = 1007
    http_status = 404
    default_msg = "用户不存在"


class FeatureEmptyError(FaceRecError):
    code = 1008
    http_status = 404
    default_msg = "该用户尚未采集有效人脸特征"


class StoreError(FaceRecError):
    code = 1009
    http_status = 503
    default_msg = "数据存储不可用，请稍后重试"


class AuthError(FaceRecError):
    code = 1010
    http_status = 401
    default_msg = "未登录或会话已过期"


class DuplicateAccountError(FaceRecError):
    code = 1011
    http_status = 409
    default_msg = "该用户名已被注册"


class ForbiddenError(FaceRecError):
    code = 1012
    http_status = 403
    default_msg = "无权限执行此操作"
