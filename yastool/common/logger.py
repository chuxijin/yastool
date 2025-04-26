# coding:utf-8
"""
文件名: logger.py
描述: 日志系统实现，包括日志上下文管理、分类、格式化等
作者: PanMaster团队 (Adapted for yastool)
创建日期: 2023-04-03
最后修改: 2024-07-28
版本: 1.1.0
"""

# 标准库
import functools
import os
import logging
import datetime
import threading
import json
import traceback
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

# ==================== 日志级别和分类 ====================

class LogCategory(Enum):
    """
    日志分类枚举

    定义了不同类型的日志分类，用于区分不同模块和功能的日志
    """
    UI = "ui"               # UI操作相关日志
    API = "api"             # API调用相关日志
    BUSINESS = "business"   # 业务逻辑相关日志
    SYSTEM = "system"       # 系统操作相关日志
    ERROR = "error"         # 错误处理相关日志
    PERFORMANCE = "perf"    # 性能监控相关日志
    DEBUG = "debug"         # 通用调试日志

class LogLevelConfig:
    """
    日志级别配置

    管理不同分类和模块的日志级别，提供默认配置和模块映射
    """

    # 默认日志级别配置
    DEFAULT_LEVELS = {
        LogCategory.UI: logging.DEBUG,
        LogCategory.API: logging.DEBUG,
        LogCategory.BUSINESS: logging.INFO,
        LogCategory.SYSTEM: logging.INFO,
        LogCategory.ERROR: logging.ERROR,
        LogCategory.PERFORMANCE: logging.DEBUG,
        LogCategory.DEBUG: logging.DEBUG
    }

    # 模块与日志分类的映射 (示例，用户可以覆盖)
    # 用户可以通过 LogManager.update_module_categories() 更新此映射
    MODULE_CATEGORIES = {
        # 'myapp.view': LogCategory.UI,
        # 'myapp.api': LogCategory.API,
        # 'myapp.services': LogCategory.BUSINESS,
        # 'another_app.utils': LogCategory.SYSTEM
    }

    @classmethod
    def get_level_for_module(cls, module_name: str) -> int:
        """
        根据模块名获取对应的日志级别

        参数:
            module_name (str): 模块名称

        返回:
            int: 相应的日志级别
        """
        # 模块级别优先 - 精确匹配
        if module_name in cls.MODULE_CATEGORIES:
            category = cls.MODULE_CATEGORIES[module_name]
            return cls.DEFAULT_LEVELS.get(category, logging.INFO) # 使用get以防category不在DEFAULT_LEVELS中

        # 查找最匹配的模块前缀
        matched_prefix = None
        for prefix in cls.MODULE_CATEGORIES:
            if module_name.startswith(prefix):
                if matched_prefix is None or len(prefix) > len(matched_prefix):
                    matched_prefix = prefix

        if matched_prefix:
            category = cls.MODULE_CATEGORIES[matched_prefix]
            return cls.DEFAULT_LEVELS.get(category, logging.INFO)

        # 尝试从分类名称推断
        parts = module_name.split('.')
        for part in reversed(parts):
            try:
                # 尝试将模块名的一部分视为分类名
                category = LogCategory(part.lower()) # 假设分类名可能是模块名的一部分
                return cls.DEFAULT_LEVELS.get(category, logging.INFO)
            except ValueError:
                continue

        # 默认级别
        return logging.INFO

    @classmethod
    def update_module_categories(cls, mapping: Dict[str, LogCategory]):
        """更新模块到分类的映射"""
        cls.MODULE_CATEGORIES.update(mapping)

    @classmethod
    def set_category_level(cls, category: LogCategory, level: int):
        """设置特定分类的日志级别"""
        if isinstance(level, str):
            level_int = logging.getLevelName(level.upper())
            if not isinstance(level_int, int):
                print(f"警告: 无效的日志级别名称 '{level}' 用于分类 {category}. 使用默认 INFO.")
                level_int = logging.INFO
        elif isinstance(level, int):
            level_int = level
        else:
            print(f"警告: 无效的日志级别类型 '{type(level)}' 用于分类 {category}. 使用默认 INFO.")
            level_int = logging.INFO

        cls.DEFAULT_LEVELS[category] = level_int
        # 更新已存在的相关logger级别
        for module, cat in cls.MODULE_CATEGORIES.items():
            if cat == category:
                logger = logging.getLogger(module)
                # 确保logger已存在且不是占位符
                if not isinstance(logger, logging.PlaceHolder):
                    logger.setLevel(level_int)

# ==================== 日志上下文管理 ====================

class LogContext:
    """
    线程安全的日志上下文管理器

    提供一种在线程内共享上下文数据的机制，主要用于日志记录。
    存储操作ID、错误ID等标识符，以便在日志中自动添加相关上下文信息。

    标准上下文键:
        - operation_id: 操作ID，格式为 "{function_name}_{timestamp}"
        - error_id: 错误ID，格式为 "{error_type}_{function_name}_{timestamp}"
        - retry_id: 重试ID，格式为 "retry_{timestamp}"
        - auth_id: 认证ID，格式为 "auth_{timestamp}"
        - ctx_id: 上下文ID，格式为 "ctx_{timestamp}"
        - trace_id: 分布式跟踪ID (可选)
        - user_id: 用户标识 (可选)
    """

    _local = threading.local()

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        设置上下文值

        参数:
            key (str): 上下文键名，建议使用标准键名
            value (Any): 上下文值，如果为None则移除该键
        """
        if not hasattr(cls._local, "context"):
            cls._local.context = {}

        if value is None:
            cls._local.context.pop(key, None)
        else:
            cls._local.context[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        获取上下文值

        参数:
            key (str): 上下文键名
            default (Any, 可选): 键不存在时返回的默认值

        返回:
            Any: 键对应的值，如果键不存在则返回默认值
        """
        return getattr(cls._local, "context", {}).get(key, default)

    @classmethod
    def clear(cls) -> None:
        """
        清除当前线程的所有上下文数据

        注意: 通常应该在请求/操作结束时调用此方法以避免内存泄漏
        """
        cls._local.context = {}

    @classmethod
    def all(cls) -> Dict[str, Any]:
        """
        获取所有上下文数据

        返回:
            Dict[str, Any]: 包含所有上下文键值对的字典
        """
        return getattr(cls._local, "context", {}).copy()

    def __enter__(self):
        """进入上下文管理器，保存当前上下文"""
        self._original_context = self.all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，恢复原始上下文"""
        # 在退出时清除或恢复之前的状态可能更安全
        # 简单起见，这里只清除，避免状态污染
        # 如果需要恢复，可以使用 self._local.context = self._original_context
        self.clear()

    @classmethod
    def context_wrapper(cls, **context_vars):
        """
        装饰器：为函数调用设置临时的上下文变量

        示例:
            @LogContext.context_wrapper(user_id="user123", trace_id="trace-abc")
            def process_request():
                logger.info("处理请求") # 日志会自动包含 user_id 和 trace_id
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                original_context = cls.all()
                try:
                    for key, value in context_vars.items():
                        cls.set(key, value)
                    return func(*args, **kwargs)
                finally:
                    # 恢复原始上下文
                    current_keys = set(cls.all().keys())
                    original_keys = set(original_context.keys())
                    # 移除新增的key
                    for key in current_keys - original_keys:
                        cls.set(key, None)
                    # 恢复被修改/删除的key
                    for key in original_keys:
                        cls.set(key, original_context.get(key))
                    # 处理被删除后又恢复的key
                    for key in original_keys - current_keys:
                         cls.set(key, original_context.get(key))
            return wrapper
        return decorator


# ==================== 上下文辅助函数 ====================

def generate_operation_id(function_name: str) -> str:
    """
    生成操作ID

    参数:
        function_name (str): 函数名称

    返回:
        str: 格式为"{function_name}_{timestamp_ms}"的操作ID
    """
    return f"{function_name}_{int(time.time() * 1000)}" # 使用毫秒增加唯一性

def generate_error_id(error_type: str = "error", function_name: str = "unknown") -> str:
    """
    生成错误ID并设置到日志上下文

    参数:
        error_type (str): 错误类型，默认为"error"
        function_name (str): 函数名称，默认为"unknown"

    返回:
        str: 格式为"{error_type}_{function_name}_{timestamp_ms}"的错误ID
    """
    func_name_safe = function_name if function_name else "unknown"
    error_id = f"{error_type}_{func_name_safe}_{int(time.time() * 1000)}"
    LogContext.set("error_id", error_id)
    return error_id

def generate_context_id(context_type: str = "ctx") -> str:
    """
    生成上下文ID

    参数:
        context_type (str): 上下文类型，默认为"ctx"

    返回:
        str: 格式为"{context_type}_{timestamp_ms}"的上下文ID
    """
    return f"{context_type}_{int(time.time() * 1000)}"

def set_operation_context(operation_id: str) -> None:
    """设置操作上下文"""
    LogContext.set("operation_id", operation_id)

def clear_operation_context() -> None:
    """清理操作上下文"""
    LogContext.set("operation_id", None)

def set_context_value(key: str, value: Any) -> None:
    """设置上下文值的便捷方法"""
    LogContext.set(key, value)

def clear_context_value(key: str) -> None:
    """清理指定上下文值的便捷方法"""
    LogContext.set(key, None)

# ==================== 上下文日志记录器 ====================

class ContextLogger:
    """
    带上下文的日志记录器

    自动从LogContext获取上下文信息，并通过 logging 的 extra 传递。
    """

    def __init__(self, logger: logging.Logger, module_name: str):
        """
        初始化上下文日志记录器

        参数:
            logger (logging.Logger): 标准的Logger实例
            module_name (str): 模块名称，用于确定日志级别
        """
        self._logger = logger
        self._module_name = module_name
        # 设置模块特定的日志级别 (确保在logger创建后设置)
        self.update_level()

    def update_level(self):
        """根据配置更新日志级别"""
        level = LogLevelConfig.get_level_for_module(self._module_name)
        self._logger.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:
        """检查是否启用了指定的日志级别"""
        return self._logger.isEnabledFor(level)

    def _get_merged_context(self, extra_context: Optional[Dict] = None) -> Dict:
        """获取线程上下文并合并额外上下文"""
        context_data = LogContext.all()
        if extra_context:
            # 确保 extra_context 是字典
            if isinstance(extra_context, dict):
                context_data.update(extra_context)
            else:
                # 如果不是字典，记录一个警告或错误，或者忽略
                self._logger.warning("传递给 extra_context 的值不是字典，已被忽略。")
        return context_data

    def _log(self, level, msg, args, exc_info=None, stack_info=None, extra_context=None, **kwargs):
        """Internal log method to handle common logic."""
        if self.isEnabledFor(level):
            log_ctx = self._get_merged_context(extra_context)

            # --- Auto Error ID Generation for ERROR/CRITICAL ---
            if level >= logging.ERROR and "error_id" not in log_ctx:
                try:
                    # 尝试回溯调用栈找到调用者函数名
                    # limit=3: _log -> public method (e.g., error) -> original caller
                    frame = traceback.extract_stack(limit=3)[-3]
                    func_name = frame.name
                except IndexError:
                    # 如果栈深度不够，使用更简单的方式
                    try:
                        frame = traceback.extract_stack(limit=2)[-2]
                        func_name = frame.name
                    except Exception:
                        func_name = "unknown_frame"
                except Exception:
                     func_name = "unknown_frame" # 其他潜在错误

                err_type = "auto_critical" if level == logging.CRITICAL else "auto_error"
                generate_error_id(err_type, func_name)
                log_ctx = self._get_merged_context(extra_context) # Re-fetch context with new error_id

            # Prepare the final 'extra' dict for the underlying logger
            final_extra = {'log_context': log_ctx}

            # Pass other relevant kwargs like exc_info, stack_info directly
            # Use *args instead of args tuple directly in the call
            self._logger.log(level, msg, *args,
                             exc_info=exc_info,
                             stack_info=stack_info,
                             extra=final_extra,
                             **kwargs) # Pass remaining kwargs (e.g., stacklevel if needed)

    def debug(self, msg: str, *args, **kwargs) -> None:
        extra_context = kwargs.pop('extra_context', None)
        # Pass args directly to _log
        self._log(logging.DEBUG, msg, args, extra_context=extra_context, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        extra_context = kwargs.pop('extra_context', None)
        self._log(logging.INFO, msg, args, extra_context=extra_context, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        extra_context = kwargs.pop('extra_context', None)
        self._log(logging.WARNING, msg, args, extra_context=extra_context, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        extra_context = kwargs.pop('extra_context', None)
        exc_info = kwargs.pop('exc_info', None) # Allow passing exc_info
        stack_info = kwargs.pop('stack_info', None)
        self._log(logging.ERROR, msg, args, exc_info=exc_info, stack_info=stack_info, extra_context=extra_context, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        extra_context = kwargs.pop('extra_context', None)
        exc_info = kwargs.pop('exc_info', None)
        stack_info = kwargs.pop('stack_info', None)
        self._log(logging.CRITICAL, msg, args, exc_info=exc_info, stack_info=stack_info, extra_context=extra_context, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        extra_context = kwargs.pop('extra_context', None)
        exc_info = kwargs.pop('exc_info', None)
        stack_info = kwargs.pop('stack_info', None)
        self._log(level, msg, args, exc_info=exc_info, stack_info=stack_info, extra_context=extra_context, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        # Ensure exc_info is True by default for exception()
        # Pass other kwargs along to the error method
        self.error(msg, *args, exc_info=exc_info, **kwargs)

# ==================== 自定义 Formatter ====================

class ContextFormatter(logging.Formatter):
    """
    自定义日志格式化器，将上下文信息插入日志级别之后。
    """
    def format(self, record: logging.LogRecord) -> str:
        context_str = ""
        if hasattr(record, 'log_context') and record.log_context:
            context_parts = []
            # 定义标准上下文键及其显示前缀 (与ContextLogger中类似但独立)
            standard_keys = {
                "operation_id": "OpID", "error_id": "ErrID", "retry_id": "RetryID",
                "auth_id": "AuthID", "ctx_id": "CtxID", "trace_id": "TraceID",
                "user_id": "User"
            }
            context_data = record.log_context # 从 record 中获取

            # 处理标准键
            for key, prefix in standard_keys.items():
                value = context_data.get(key)
                if value:
                    context_parts.append(f"[{prefix}:{value}]")

            # 添加不在标准列表中的其他上下文键值对
            other_keys = set(context_data.keys()) - set(standard_keys.keys())
            if other_keys:
                other_parts = [f"{k}={v}" for k, v in context_data.items() if k in other_keys]
                # 为了简洁，可以将其他键合并
                context_parts.append(f"[Other:{','.join(other_parts)}]")

            context_str = " ".join(context_parts)

        # 将格式化后的上下文添加到 record 中，以便格式字符串使用
        record.context_str = context_str
        # 添加一个空格字段，仅当 context_str 非空时才有值
        record.context_space = " " if context_str else ""

        # 调用父类的 format 方法，使用包含 %(context_str)s 的格式字符串
        return super().format(record)

# ==================== 全局错误处理 (已移除UI依赖) ====================

class GlobalErrorHandler:
    """
    全局异常处理器 (无UI依赖版本 + 可选UI显示方法)

    提供统一的异常捕获和处理机制，避免程序崩溃，并记录详细的错误信息。
    提供可选的 show_error 方法用于手动显示GUI错误。
    """
    _original_excepthook = None
    main_window = None # 添加 main_window 属性以存储父窗口引用

    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        全局异常处理函数

        用于设置为sys.excepthook，捕获所有未处理的异常。
        """
        # 忽略 KeyboardInterrupt (Ctrl+C)
        if issubclass(exc_type, KeyboardInterrupt):
            if GlobalErrorHandler._original_excepthook:
                 GlobalErrorHandler._original_excepthook(exc_type, exc_value, exc_traceback)
            else:
                 # 默认行为
                 import sys
                 sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # 获取根记录器或特定错误记录器
        logger = logging.getLogger("global_errors")
        if not logger.hasHandlers(): # 如果没有配置handler，则添加到根logger
             logger = logging.getLogger()

        # 生成唯一的错误ID
        error_id = f"global_{exc_type.__name__}_{int(time.time() * 1000)}"

        # 格式化异常信息
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)

        # 记录严重错误日志
        log_message = (
            f"[全局异常 ErrID:{error_id}] 捕获到未处理的异常: {exc_type.__name__}: {str(exc_value)}\n"
            f"详细堆栈跟踪:\n{tb_text}"
        )
        logger.critical(log_message)

        # --- 尝试显示 GUI 错误提示 --- 
        # 检查是否有可用的 QApplication 实例和 main_window (或 qfluentwidgets 可用)
        gui_possible = False
        try:
            from PySide6.QtWidgets import QApplication
            if QApplication.instance():
                gui_possible = True
        except ImportError:
            pass # PySide6 不可用
        except Exception:
            pass # QApplication.instance() 可能出错
            
        # 只有在 GUI 环境可能可用时才尝试显示
        if gui_possible:
            # 格式化用户友好的错误消息
            dialog_title = f"程序发生未处理的错误 (ID: {error_id})"
            dialog_message = (
                f"错误类型: {exc_type.__name__}\n"
                f"错误信息: {str(exc_value)}\n\n"
                f"建议您保存工作并重启程序。\n"
                f"如果问题持续出现，请联系技术支持并提供错误ID。"
            )
            # 调用 show_error 显示对话框，使用 self 调用以访问 main_window
            # 注意：show_error 内部有自己的 try-except 和 GUI 库导入
            GlobalErrorHandler.show_error(dialog_title, dialog_message)
        else:
            # 如果 GUI 不可用，可以在控制台打印简化提示
             print(f"\n!!! 系统发生严重错误 !!!\n错误ID: {error_id}\n请检查日志文件获取详细信息。\n", file=sys.stderr)

        # 如果有原始的 hook，调用它 (可选，取决于是否要完全覆盖)
        # if GlobalErrorHandler._original_excepthook:
        #     GlobalErrorHandler._original_excepthook(exc_type, exc_value, exc_traceback)

    @staticmethod
    def show_error(title: str, message: str, parent: Optional[Any] = None):
        """
        显示错误消息框 (需要 qfluentwidgets 和 PySide6)

        用于在捕获到异常时手动显示错误消息。如果未安装必要的GUI库，
        或者在非GUI线程调用且无法切换，此方法将记录错误并返回False。

        参数:
            title (str): 消息框标题
            message (str): 错误消息
            parent (Any, 可选): 父窗口对象 (例如 QWidget)。如果为None，会尝试使用 GlobalErrorHandler.main_window 或活动窗口。

        返回:
            bool: 如果消息框成功显示并被用户关闭，则返回 True 或消息框的结果，否则返回 False。
        """
        logger = logging.getLogger(__name__) # 使用当前模块的 logger
        try:
            # 延迟导入，避免硬依赖
            from qfluentwidgets import MessageBox
            from PySide6.QtWidgets import QApplication

            # 检查 QApplication 实例是否存在
            app = QApplication.instance()
            if not app:
                logger.error("无法显示错误消息框：QApplication 未运行。")
                return False

            # 确定父窗口
            target_parent = parent
            if not target_parent:
                target_parent = GlobalErrorHandler.main_window
                if not target_parent:
                    target_parent = QApplication.activeWindow()
                    # 如果仍然没有父窗口，可能需要创建一个临时的，但这通常不推荐
                    # if not target_parent:
                    #     logger.warning("未找到合适的父窗口用于显示错误消息框。")

            # 显示错误消息
            # 需要在主线程执行UI操作
            def _show():
                msg_box = MessageBox(title, message, target_parent)
                msg_box.yesButton.setText("确定")
                msg_box.cancelButton.hide()
                return msg_box.exec() # exec() 返回值通常是 QDialog.Accepted 或 QDialog.Rejected

            # 检查是否在主线程
            if QApplication.instance().thread() == threading.current_thread():
                return _show()
            else:
                # 尝试在主线程中调用 (如果可能)
                # 注意：这依赖于 Qt 的事件循环，如果调用时事件循环未运行或不同，可能无效
                try:
                    from PySide6.QtCore import QTimer, QMetaObject, Qt
                    # 使用 QMetaObject.invokeMethod 可能更健壮，但实现稍复杂
                    # QTimer.singleShot 是一个简单的方式，但不保证立即执行
                    result = [None] # 使用列表在闭包中传递结果
                    QTimer.singleShot(0, lambda: result.__setitem__(0, _show()))
                    # 这里无法同步等待结果，所以返回 True 表示尝试了显示
                    logger.info("尝试在主线程中异步显示错误消息框。")
                    # 或者返回一个Future/Promise如果需要等待结果
                    return True # 表示尝试启动显示
                except ImportError:
                     logger.error("无法显示错误消息框：不在主线程且无法导入 PySide6.QtCore。")
                     return False
                except Exception as e:
                     logger.error(f"尝试在主线程显示错误消息框时出错: {e}", exc_info=True)
                     return False


        except ImportError as e:
            logger.error(f"无法显示错误消息框，缺少必要的库：{e} (请确保安装了 qfluentwidgets 和 PySide6)")
            return False
        except Exception as e:
            logger.error(f"显示错误消息框失败: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def install(main_window: Optional[Any] = None):
        """
        安装全局异常处理器，并可选择设置主窗口引用。

        参数:
            main_window (Any, 可选): 应用程序主窗口引用，用于 show_error 的默认父窗口。
        """
        import sys
        GlobalErrorHandler.main_window = main_window # 存储主窗口引用
        if sys.excepthook != GlobalErrorHandler.handle_exception:
            GlobalErrorHandler._original_excepthook = sys.excepthook
            sys.excepthook = GlobalErrorHandler.handle_exception
            logging.getLogger(__name__).info("全局异常处理器已安装。")

    @staticmethod
    def uninstall():
        """卸载全局异常处理器，恢复原始状态"""
        import sys
        if sys.excepthook == GlobalErrorHandler.handle_exception:
            sys.excepthook = GlobalErrorHandler._original_excepthook
            GlobalErrorHandler._original_excepthook = None
            logging.getLogger(__name__).info("全局异常处理器已卸载。")

# ==================== 日志管理器 ====================

class LogManager:
    """
    全局日志管理器 (单例模式)

    提供日志系统的初始化、配置和管理功能。
    """

    # 日志级别名称到值的映射
    LEVELS = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir: Union[str, Path] = 'logs',
                 default_console_level: Union[str, int] = logging.INFO,
                 default_file_level: Union[str, int] = logging.DEBUG,
                 log_filename_format: str = '{date}.log',
                 # 在格式字符串中添加 %(context_space)s%(context_str)s
                 log_format: str = '%(asctime)s - %(name)s - %(levelname)s%(context_space)s%(context_str)s - [%(threadName)s:%(thread)d] - %(message)s',
                 archive_days: Optional[int] = 30,
                 install_global_handler: bool = True,
                 third_party_levels: Optional[Dict[str, Union[str, int]]] = None):
        """
        初始化日志管理器 (支持配置)

        参数:
            log_dir: 日志文件存放目录
            default_console_level: 控制台默认日志级别
            default_file_level: 文件默认日志级别
            log_filename_format: 日志文件名格式 ({date}会被替换)
            log_format: 日志记录格式
            archive_days: 保留日志天数 (None表示不归档)
            install_global_handler: 是否安装全局异常处理器
            third_party_levels: 第三方库日志级别配置 {logger_name: level}
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            self.log_filename_format = log_filename_format
            # self.log_format = log_format # 存储格式字符串以备用 (不需要了，直接传给Formatter)
            self.archive_days = archive_days

            # 解析级别
            console_level = self._parse_level(default_console_level, logging.INFO)
            file_level = self._parse_level(default_file_level, logging.DEBUG)

            # 获取根记录器
            self.root_logger = logging.getLogger()
            self.root_logger.setLevel(logging.DEBUG) # 根记录器捕捉所有，由handler过滤

            # 清除现有处理器 (避免重复添加)
            for handler in self.root_logger.handlers[:]:
                self.root_logger.removeHandler(handler)
                handler.close() # 关闭处理器

            # 创建格式器 - 使用新的 ContextFormatter
            formatter = ContextFormatter(log_format) # 使用修改后的格式

            # --- 控制台处理器 ---
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(console_level)
            self.console_handler.setFormatter(formatter) # 使用新 Formatter
            self.root_logger.addHandler(self.console_handler)

            # --- 文件处理器 ---
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            log_filename = self.log_filename_format.format(date=today)
            self.log_file = self.log_dir / log_filename

            # 使用 RotatingFileHandler 或 TimedRotatingFileHandler 可能更好
            # 这里为了简单，保持 FileHandler，但归档逻辑需要外部触发或定时任务
            try:
                self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
                self.file_handler.setLevel(file_level)
                self.file_handler.setFormatter(formatter) # 使用新 Formatter
                self.root_logger.addHandler(self.file_handler)
            except Exception as e:
                self.root_logger.error(f"无法创建日志文件处理器 {self.log_file}: {e}")
                self.file_handler = None # 标记文件处理器创建失败

            # 配置第三方库日志级别
            self._configure_third_party_loggers(third_party_levels)

            # 安装全局错误处理器
            if install_global_handler:
                GlobalErrorHandler.install()

            # 初始化完成
            self.root_logger.info("日志系统初始化完成.")
            self._initialized = True

    def _parse_level(self, level: Union[str, int], default: int) -> int:
        """解析日志级别"""
        if isinstance(level, int):
            return level
        if isinstance(level, str):
            level_upper = level.upper()
            # 注意：这里 LEVELS 应该是类属性或实例属性
            levels_map = LogManager.LEVELS # 访问类属性
            if level_upper in levels_map:
                return levels_map[level_upper]
            else:
                # 尝试直接获取 logging 模块的级别
                level_int = logging.getLevelName(level_upper)
                if isinstance(level_int, int):
                    return level_int
                else:
                    self.root_logger.warning(f"无效的日志级别字符串 '{level}', 使用默认级别 {logging.getLevelName(default)}.")
                    return default
        self.root_logger.warning(f"无效的日志级别类型 '{type(level)}', 使用默认级别 {logging.getLevelName(default)}.")
        return default

    def _configure_third_party_loggers(self, third_party_levels: Optional[Dict[str, Union[str, int]]] = None):
        """
        配置第三方库的日志级别
        """
        default_levels = {
            'urllib3': logging.WARNING,
            'requests': logging.WARNING,
            'asyncio': logging.INFO, # asyncio 调试信息可能很多
            # 'werkzeug': logging.WARNING, # 如果使用Flask/Werkzeug
            # 'sqlalchemy': logging.WARNING, # 如果使用SQLAlchemy
        }
        config_levels = third_party_levels or {}
        final_levels = {**default_levels, **config_levels} # 用户配置覆盖默认配置

        for logger_name, level in final_levels.items():
            parsed_level = self._parse_level(level, logging.WARNING) # 默认给警告
            logging.getLogger(logger_name).setLevel(parsed_level)

    def get_logger(self, name: str) -> ContextLogger:
        """
        获取带上下文的日志记录器

        参数:
            name (str): 记录器名称，通常使用 __name__

        返回:
            ContextLogger: 带上下文的日志记录器实例
        """
        if not self._initialized:
            print("警告: 日志系统尚未初始化，请先调用 LogManager() 或 setup_logging()。将使用默认配置初始化。")
            # 可以选择返回一个基本的logger或者抛出异常
            # return logging.getLogger(name) # 返回标准logger
            self.__init__() # 尝试默认初始化

        logger = logging.getLogger(name)
        # 如果logger是新创建的，需要确保它继承了根logger的级别设置
        # 或者通过ContextLogger内部的update_level来设置
        return ContextLogger(logger, name)

    def update_log_level(self, console_level: Optional[Union[str, int]] = None,
                           file_level: Optional[Union[str, int]] = None):
        """动态更新控制台和文件的日志级别"""
        if console_level is not None:
            level = self._parse_level(console_level, self.console_handler.level)
            self.console_handler.setLevel(level)
            self.root_logger.info(f"控制台日志级别更新为: {logging.getLevelName(level)}")
        if file_level is not None and self.file_handler:
            level = self._parse_level(file_level, self.file_handler.level)
            self.file_handler.setLevel(level)
            self.root_logger.info(f"文件日志级别更新为: {logging.getLevelName(level)}")

    def update_module_categories(self, mapping: Dict[str, LogCategory]):
        """更新模块到分类的映射，并更新现有logger级别"""
        LogLevelConfig.update_module_categories(mapping)
        # 更新所有已知logger的级别
        for name, logger_instance in logging.root.manager.loggerDict.items():
            if isinstance(logger_instance, logging.Logger): # 跳过 PlaceHolder
                 level = LogLevelConfig.get_level_for_module(name)
                 logger_instance.setLevel(level)
        self.root_logger.info("模块日志分类映射已更新。")

    def set_category_level(self, category: LogCategory, level: Union[str, int]):
        """设置特定分类的日志级别，并更新相关logger"""
        level_int = self._parse_level(level, logging.INFO)
        LogLevelConfig.set_category_level(category, level_int)
        self.root_logger.info(f"日志分类 '{category.name}' 级别已更新为: {logging.getLevelName(level_int)}")

    def set_module_level(self, module_prefix: str, level: Union[str, int]):
        """
        设置特定模块前缀的日志级别
        """
        level_int = self._parse_level(level, logging.INFO)
        updated_loggers = []
        for logger_name, logger_instance in logging.root.manager.loggerDict.items():
            if logger_name.startswith(module_prefix) and isinstance(logger_instance, logging.Logger):
                logger_instance.setLevel(level_int)
                updated_loggers.append(logger_name)
        # 也可能需要更新根logger或其他父logger？取决于具体场景
        # logger = logging.getLogger(module_prefix) # 获取或创建logger
        # logger.setLevel(level_int)
        if updated_loggers:
             self.root_logger.info(f"模块前缀 '{module_prefix}' 的日志级别已更新为: {logging.getLevelName(level_int)} (影响: {updated_loggers})" + (' - 可能需要重启应用使所有logger生效' if '.' in module_prefix else '')) # 提示
        else:
            # 如果没有匹配的现有logger，可以考虑创建一个并设置级别
            logging.getLogger(module_prefix).setLevel(level_int)
            self.root_logger.info(f"模块前缀 '{module_prefix}' 的日志级别已设置为: {logging.getLevelName(level_int)} (新logger或无子logger)")

    def archive_logs(self):
        """
        归档旧日志文件 (基于初始化时设置的 archive_days)
        """
        if self.archive_days is None or self.archive_days <= 0:
            return

        try:
            # 计算截止日期
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.archive_days)

            # 归档目录
            archive_dir = self.log_dir / 'archive'
            archive_dir.mkdir(exist_ok=True)

            archived_count = 0
            error_count = 0

            # 遍历日志目录中的文件
            for item in self.log_dir.iterdir():
                if item.is_file() and item.suffix == '.log': # 假设日志文件以.log结尾
                    try:
                        # 尝试从文件名解析日期 (需要更健壮的解析逻辑)
                        # 假设格式是 YYYY-MM-DD.log
                        file_date_str = item.stem
                        # 兼容 YYYY-MM-DD 或 {prefix}-YYYY-MM-DD 等格式
                        date_part = file_date_str.split('-')[-3:] # 取最后三部分
                        if len(date_part) == 3:
                            file_date = datetime.datetime.strptime('-'.join(date_part), '%Y-%m-%d')
                        else:
                            continue # 无法解析日期

                        # 如果日志文件早于截止日期，则归档
                        if file_date.date() < cutoff_date.date(): # 比较日期部分即可
                            archive_path = archive_dir / item.name
                            # 如果目标文件已存在，可以选择覆盖或添加时间戳
                            if archive_path.exists():
                                archive_path = archive_dir / f"{item.stem}_{int(time.time())}{item.suffix}"
                            item.rename(archive_path)
                            archived_count += 1
                            # self.root_logger.info(f"已归档日志文件: {item.name}") # 可能产生过多日志
                    except ValueError:
                         # 文件名不是期望的日期格式，跳过
                         self.root_logger.debug(f"跳过无法解析日期的日志文件: {item.name}")
                         continue
                    except Exception as e:
                        self.root_logger.error(f"归档日志文件 {item.name} 时出错: {str(e)}")
                        error_count += 1

            if archived_count > 0:
                self.root_logger.info(f"日志归档完成: {archived_count} 个文件已移动到 {archive_dir}")
            if error_count > 0:
                 self.root_logger.warning(f"日志归档期间发生 {error_count} 个错误。")

        except Exception as e:
            self.root_logger.error(f"执行日志归档任务时出错: {str(e)}")

# ==================== 便捷函数 (推荐用法) ====================

# 全局日志管理器实例的引用
_log_manager_instance: Optional[LogManager] = None

def setup_logging(**kwargs) -> LogManager:
    """
    初始化并获取日志管理器实例 (推荐的初始化方法)

    参数:
        **kwargs: 传递给 LogManager 构造函数的参数
                  (log_dir, default_console_level, etc.)

    返回:
        LogManager: 单例日志管理器实例
    """
    global _log_manager_instance
    if _log_manager_instance is None or not _log_manager_instance._initialized:
        _log_manager_instance = LogManager(**kwargs)
    elif kwargs: # 如果已初始化但提供了参数，则警告或更新配置
        # 简单的做法是警告用户重复初始化
        # 获取一个logger实例来记录警告，避免打印到stdout/stderr
        logger = _log_manager_instance.get_logger(__name__)
        logger.warning("日志系统已初始化，忽略本次 setup_logging 的参数。如需更改配置，请使用 LogManager 的实例方法。")
        # 或者可以实现更新逻辑：
        # _log_manager_instance.update_config(**kwargs) # 需要实现 update_config 方法
    return _log_manager_instance

def get_logger(name: str) -> ContextLogger:
    """
    获取指定名称的带上下文的日志记录器

    如果日志系统未初始化，会尝试使用默认配置进行初始化。

    参数:
        name (str): 记录器名称，通常使用 __name__

    返回:
        ContextLogger: 带上下文的日志记录器实例
    """
    manager = setup_logging() # 获取或初始化管理器
    return manager.get_logger(name)

def log_exception(logger_name: str = "exception", message: str = "捕获到异常",
                  exc_info=True, stack_info=False, **kwargs) -> None:
    """
    记录当前异常信息 (修复版)

    在 except 块中调用，自动记录当前异常的详细信息，并包含上下文。

    参数:
        logger_name (str): 用于记录异常的logger名称
        message (str): 记录异常时的消息前缀
        exc_info (bool): 是否包含异常信息 (type, value, traceback)
        stack_info (bool): 是否包含堆栈信息 (当前线程的调用栈)
        **kwargs: 传递给 logger.error 的其他参数 (例如 extra_context)
                  注意: exc_info 和 stack_info 会被优先处理。
    """
    logger = get_logger(logger_name)
    # Error ID generation is now handled inside ContextLogger._log

    # Pass relevant arguments and remaining kwargs to the logger's error method.
    # ContextLogger.error will handle the context merging and passing 'extra'.
    logger.error(
        message,
        # args can be passed if needed, but typically not for log_exception
        exc_info=exc_info,
        stack_info=stack_info,
        **kwargs # Pass extra_context etc. via kwargs
    )

# --- 配置快捷方式 ---

def set_console_level(level: Union[str, int]):
    """设置控制台日志级别"""
    manager = setup_logging()
    manager.update_log_level(console_level=level)

def set_file_level(level: Union[str, int]):
    """设置文件日志级别"""
    manager = setup_logging()
    manager.update_log_level(file_level=level)

def add_module_category_mapping(mapping: Dict[str, LogCategory]):
    """添加模块到分类的映射"""
    manager = setup_logging()
    manager.update_module_categories(mapping)

def configure_category_level(category: LogCategory, level: Union[str, int]):
    """配置特定分类的日志级别"""
    manager = setup_logging()
    manager.set_category_level(category, level)

def configure_module_level(module_prefix: str, level: Union[str, int]):
    """配置特定模块前缀的日志级别"""
    manager = setup_logging()
    manager.set_module_level(module_prefix, level) 