# coding:utf-8
"""
文件名: error_codes.py
描述: 定义通用的API异常和错误码
"""
from enum import Enum

class ErrorCode(Enum):
    """通用错误码"""
    SUCCESS = 0             # 成功
    UNKNOWN_ERROR = -1      # 未知错误
    NETWORK_ERROR = 1001    # 网络错误
    TIMEOUT_ERROR = 1002    # 超时错误
    SERVER_ERROR = 1003     # 服务端错误 (如 5xx)
    RATE_LIMIT_ERROR = 1004 # 请求限流 (如 429)
    AUTHENTICATION_ERROR = 2001 # 认证失败
    PERMISSION_DENIED = 2002    # 权限不足
    INVALID_PARAMETER = 3001  # 无效参数
    RESOURCE_NOT_FOUND = 4004 # 资源未找到

class ApiException(Exception):
    """基础API异常类"""
    def __init__(self, code: ErrorCode, message: str, error_id: str = None):
        self.code = code
        self.message = message
        self.error_id = error_id
        super().__init__(f"Error {code.value} ({code.name}): {message}" + (f" [ErrorID: {error_id}]" if error_id else ""))

class NetworkError(ApiException):
    """网络相关错误"""
    def __init__(self, message: str, error_id: str = None):
        super().__init__(ErrorCode.NETWORK_ERROR, message, error_id)

class TimeoutError(ApiException):
    """请求超时错误"""
    def __init__(self, message: str, error_id: str = None):
        super().__init__(ErrorCode.TIMEOUT_ERROR, message, error_id)

class ServerError(ApiException):
    """服务器内部错误"""
    def __init__(self, message: str, error_id: str = None):
        super().__init__(ErrorCode.SERVER_ERROR, message, error_id)

class RateLimitError(ApiException):
    """请求限流错误"""
    def __init__(self, message: str, error_id: str = None):
        super().__init__(ErrorCode.RATE_LIMIT_ERROR, message, error_id)

class AuthenticationError(ApiException):
    """认证错误"""
    def __init__(self, message: str, error_id: str = None):
        super().__init__(ErrorCode.AUTHENTICATION_ERROR, message, error_id) 