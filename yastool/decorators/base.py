# coding:utf-8
"""
文件名: base.py
描述: 提供基础和通用的装饰器
创建日期: 2024-07-28
版本: 1.0.0
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Dict, Optional, List, Union


# 本地模块
from ..common.logger import get_logger, LogContext, generate_context_id, clear_context_value, generate_error_id, set_context_value
from ..models.error_codes import ApiException

# 获取模块日志记录器
logger = get_logger(__name__)

def with_log_context(context_name: Optional[str] = None,
                     log_level: int = logging.DEBUG,
                     error_log_level: int = logging.WARNING,
                     log_args: bool = True,
                     sensitive_keys: Optional[List[str]] = None,
                     context_attributes: Optional[List[str]] = None):
    """
    装饰器：为函数调用添加一个独立的日志上下文作用域。

    在函数开始时记录进入上下文，结束或异常时记录离开上下文。
    自动生成并管理 ctx_id。

    参数:
        context_name (str, 可选): 上下文的名称，默认使用函数名。
        log_level (int): 记录进入/成功退出消息的日志级别，默认DEBUG。
        error_log_level (int): 记录异常退出消息的日志级别，默认WARNING。
        log_args (bool): 是否在进入日志中记录函数参数。默认True。
        sensitive_keys (List[str], 可选): 需要屏蔽的参数/属性键名。
        context_attributes (List[str], 可选): 要从 'self' 实例提取并记录的属性名。
    """
    default_sensitive = ['password', 'token', 'secret', 'key', 'credential', 'auth', 'cookies']
    sensitive_keys_set = set(default_sensitive + [k.lower() for k in sensitive_keys]) if sensitive_keys else set(default_sensitive)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 确定logger实例
            instance = args[0] if args and hasattr(args[0], func.__name__) else None
            current_logger = getattr(instance, 'logger', logger) if instance else logger

            ctx_name = context_name or func.__name__
            # 生成上下文ID并设置
            ctx_id = generate_context_id(ctx_name)
            set_context_value("ctx_id", ctx_id)

            # 记录上下文开始
            if current_logger.isEnabledFor(log_level):
                log_message = f"进入上下文: {ctx_name}"
                details = []
                extra_context_start = {"ctx_id": ctx_id} # 确保ctx_id在日志中

                # 记录实例属性
                if instance and context_attributes:
                    attr_details = {}
                    for attr in context_attributes:
                        if hasattr(instance, attr):
                            value = getattr(instance, attr)
                            if attr.lower() in sensitive_keys_set:
                                value = '******' if value else None
                            attr_details[attr] = repr(value)
                        else:
                             attr_details[attr] = "<Not Found>"
                    if attr_details:
                        details.append(f"ContextAttrs: {attr_details}")
                        # 不建议将这些属性直接添加到线程上下文，只记录
                        # extra_context_start.update({f"ctx_{k}": v for k, v in attr_details.items()})

                # 记录参数
                if log_args:
                    arg_details = {}
                    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                    arg_idx_offset = 1 if instance else 0
                    for i, arg_val in enumerate(args[arg_idx_offset:]):
                         arg_name = arg_names[i + arg_idx_offset] if (i + arg_idx_offset) < len(arg_names) else f"arg_{i}"
                         if arg_name.lower() in sensitive_keys_set:
                              arg_details[arg_name] = '******'
                         else:
                              arg_details[arg_name] = repr(arg_val)

                    for k, v in kwargs.items():
                        if k.lower() in sensitive_keys_set:
                             arg_details[k] = '******'
                        else:
                             arg_details[k] = repr(v)

                    if arg_details:
                        details.append(f"Args: {arg_details}")

                if details:
                    log_message += " (" + ", ".join(details) + ")"

                current_logger.log(log_level, log_message, extra_context=extra_context_start)

            try:
                result = func(*args, **kwargs)
                # 记录成功退出
                if current_logger.isEnabledFor(log_level):
                    current_logger.log(log_level, f"离开上下文: {ctx_name} (成功)", extra_context=LogContext.all())
                return result
            except Exception as e:
                # 记录异常退出
                if current_logger.isEnabledFor(error_log_level):
                    # 如果没有 error_id，生成一个
                    error_id = LogContext.get("error_id")
                    if not error_id:
                        error_id = generate_error_id(f"ctx_error_{ctx_name}", func.__name__)
                        # 注意: 这里设置的 error_id 会在 finally 中被清理，除非异常被上层捕获

                    current_logger.log(
                        error_log_level,
                        f"离开上下文: {ctx_name} (异常: {type(e).__name__}: {str(e)})",
                        exc_info=False, # 通常只记录错误信息，避免重复traceback
                        extra_context=LogContext.all()
                    )
                raise
            finally:
                # 清理此上下文设置的 ctx_id
                clear_context_value("ctx_id")
                # 注意：不要在这里清理 error_id，它可能由外层操作设置并需要保留
        return wrapper
    return decorator 