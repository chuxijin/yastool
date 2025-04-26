# yastool/common/__init__.py
from .logger import (
    # 日志配置与管理
    LogManager, 
    setup_logging,
    LogCategory,
    LogLevelConfig,
    # 上下文管理
    LogContext,
    generate_operation_id,
    generate_error_id,
    generate_context_id,
    set_context_value,
    clear_context_value,
    # 获取Logger实例
    get_logger,
    ContextLogger,
    # 异常记录
    log_exception,
    # 配置快捷方式
    set_console_level,
    set_file_level,
    add_module_category_mapping,
    configure_category_level,
    configure_module_level,
    # 全局错误处理 (可选)
    GlobalErrorHandler
)

__all__ = [
    'LogManager',
    'setup_logging',
    'LogCategory',
    'LogLevelConfig',
    'LogContext',
    'generate_operation_id',
    'generate_error_id',
    'generate_context_id',
    'set_context_value',
    'clear_context_value',
    'get_logger',
    'ContextLogger',
    'log_exception',
    'set_console_level',
    'set_file_level',
    'add_module_category_mapping',
    'configure_category_level',
    'configure_module_level',
    'GlobalErrorHandler'
] 