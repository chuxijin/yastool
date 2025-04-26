# yastool/models/__init__.py
from .error_codes import (
    ErrorCode,
    ApiException,
    NetworkError,
    TimeoutError,
    ServerError,
    RateLimitError,
    AuthenticationError
)

__all__ = [
    'ErrorCode',
    'ApiException',
    'NetworkError',
    'TimeoutError',
    'ServerError',
    'RateLimitError',
    'AuthenticationError'
] 