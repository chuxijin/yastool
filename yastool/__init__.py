# yastool/__init__.py
"""YASTool: Yet Another Simple Toolkit for Python Projects"""

__version__ = "0.1.0" #初始版本

# 导出最常用的接口
from .common import (
    get_logger,
    setup_logging,
    LogCategory,
    LogContext,
    log_exception
)
from .decorators import (
    api_method,
    log_operation,
    cache_response,
    retry_on_failure,
    with_log_context,
    BackoffStrategy
)
from .models import (
    ErrorCode,
    ApiException,
    NetworkError,
    TimeoutError,
    ServerError,
    RateLimitError
)

__all__ = [
    # Logging
    'get_logger',
    'setup_logging',
    'LogCategory',
    'LogContext',
    'log_exception',
    # Decorators
    'api_method',
    'log_operation',
    'cache_response',
    'retry_on_failure',
    'with_log_context',
    'BackoffStrategy',
    # Models
    'ErrorCode',
    'ApiException',
    'NetworkError',
    'TimeoutError',
    'ServerError',
    'RateLimitError'
] 