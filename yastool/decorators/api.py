# coding:utf-8
"""
文件名: api.py
描述: 提供API请求相关的装饰器，包括重试、缓存、日志、User-Agent等
作者: PanMaster团队 (Adapted for yastool)
创建日期: 2023-04-03
最后修改: 2024-07-28
版本: 1.1.0
"""

# 标准库
import datetime
from pathlib import Path
import sys
import time
import random
import json
import platform
import functools
import threading
import logging
from enum import Enum
from functools import wraps
from typing import Callable, Any, Dict, Optional, List, Union
from collections import defaultdict



# 本地模块 (使用相对导入)
from ..models.error_codes import ApiException, ErrorCode, NetworkError, RateLimitError, ServerError, TimeoutError, AuthenticationError
from ..common.logger import (
    ContextLogger,
    get_logger,
    LogContext,
    generate_operation_id,
    generate_error_id,
    set_context_value,
    clear_context_value
)

# ==================== 常量定义 ====================

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1
_DEFAULT_CACHE_MAX_SIZE = 1000
_MAX_PERF_SAMPLES = 100  # 性能监控样本数量限制

# 获取模块日志记录器
# 使用 __name__ 获取当前模块的logger ("yastool.decorators.api")
logger = get_logger(__name__)

# ==================== 性能监控相关 ====================

_metrics_lock = threading.RLock()
_call_counts = defaultdict(int)
_error_counts = defaultdict(int)
_total_durations = defaultdict(float)
_perf_samples = defaultdict(list)

def _update_metrics(decorator_name: str, func_name: str, duration: float, error: bool = False):
    """
    内部函数：更新性能指标

    参数:
        decorator_name (str): 装饰器名称
        func_name (str): 函数名称
        duration (float): 执行时间(毫秒)
        error (bool): 是否发生错误
    """
    # 使用更结构化的键
    metric_key = f"{decorator_name}::{func_name}"
    with _metrics_lock:
        _call_counts[metric_key] += 1
        _total_durations[metric_key] += duration

        # 保留最近的性能样本，用于计算百分位数
        samples = _perf_samples[metric_key]
        samples.append(duration)
        # 使用固定大小的列表，移除旧样本
        if len(samples) > _MAX_PERF_SAMPLES:
             _perf_samples[metric_key] = samples[-_MAX_PERF_SAMPLES:]

        if error:
            _error_counts[metric_key] += 1

def get_performance_metrics() -> Dict[str, Dict[str, Any]]:
    """
    获取所有通过装饰器收集的性能指标。

    返回:
        Dict: 格式为 { "decorator_name::func_name": {metrics} }
              metrics 包含: calls, errors, avg_time_ms, error_rate_percent, p95_time_ms
    """
    with _metrics_lock:
        # 创建副本以避免在迭代时修改
        call_counts = _call_counts.copy()
        error_counts = _error_counts.copy()
        total_durations = _total_durations.copy()
        perf_samples = {k: sorted(v) for k, v in _perf_samples.items()} # 排序用于计算百分位

    metrics = {}
    all_keys = set(call_counts.keys()) | set(error_counts.keys()) | set(total_durations.keys())

    for key in all_keys:
        calls = call_counts.get(key, 0)
        errors = error_counts.get(key, 0)
        total_time = total_durations.get(key, 0.0)
        samples = perf_samples.get(key, [])

        if calls > 0:
            avg_time = total_time / calls
            error_rate = (errors / calls * 100)
            # 计算 P95 (需要足够样本)
            p95_time = samples[int(len(samples) * 0.95)] if len(samples) >= 20 else (samples[-1] if samples else avg_time)

            metrics[key] = {
                "calls": calls,
                "errors": errors,
                "avg_time_ms": round(avg_time, 2),
                "p95_time_ms": round(p95_time, 2),
                "error_rate_percent": round(error_rate, 2)
            }
        elif key in perf_samples: # 处理只有样本但调用计数为0的罕见情况
             metrics[key] = {"calls": 0, "errors": 0, "avg_time_ms": 0, "p95_time_ms": 0, "error_rate_percent": 0, "samples": len(samples)}

    return metrics

def reset_performance_metrics():
    """重置所有性能指标计数器"""
    with _metrics_lock:
        _call_counts.clear()
        _error_counts.clear()
        _total_durations.clear()
        _perf_samples.clear()
    logger.info("API装饰器性能指标已重置。")

def export_performance_metrics_to_json(filepath: Optional[Union[str, Path]] = None) -> Union[str, None]:
    """
    将性能指标导出为JSON格式

    参数:
        filepath (str | Path, 可选): 导出文件路径。如不提供，则返回JSON字符串。

    返回:
        str | None: 如果未提供文件路径，则返回JSON字符串，否则返回None。
    """
    metrics_data = get_performance_metrics()
    now = datetime.datetime.now().isoformat()

    # 格式化为更易读的结构
    formatted_metrics = {
        "export_timestamp": now,
        "metrics": metrics_data,
        "summary": {
            "total_tracked_methods": len(metrics_data),
            "total_calls": sum(m.get("calls", 0) for m in metrics_data.values()),
            "total_errors": sum(m.get("errors", 0) for m in metrics_data.values()),
            "overall_avg_error_rate": round(sum(m.get("error_rate_percent", 0) for m in metrics_data.values()) / len(metrics_data), 2) if metrics_data else 0,
        }
    }

    if filepath:
        try:
            fp = Path(filepath)
            fp.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
            with fp.open('w', encoding='utf-8') as f:
                json.dump(formatted_metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"性能指标已导出到 {fp.resolve()}")
            return None
        except Exception as e:
            logger.error(f"导出性能指标到 {filepath} 失败: {e}", exc_info=True)
            # 失败时可以选择返回JSON字符串作为后备
            # return json.dumps(formatted_metrics, indent=2, ensure_ascii=False)
            return None
    else:
        return json.dumps(formatted_metrics, indent=2, ensure_ascii=False)

# TODO: register_metrics_endpoint 需要具体的Web框架集成，暂时注释掉
# def register_metrics_endpoint(app=None, endpoint="/api/metrics/decorators"):
#     """
#     注册装饰器指标监控端点（示例，需要Web框架集成）
#     """
#     if app is None:
#         logger.warning("未提供Web应用实例，无法注册指标端点。")
#         return
#     try:
#         # 示例: Flask
#         if hasattr(app, 'route'):
#             @app.route(endpoint)
#             def decorator_metrics_endpoint():
#                 from flask import jsonify # 需要安装 Flask
#                 return jsonify(get_performance_metrics())
#             logger.info(f"装饰器性能指标端点已注册: {endpoint}")
#         else:
#              logger.error("不支持的应用类型，无法注册指标端点。请自行集成。")
#     except Exception as e:
#         logger.error(f"注册指标监控端点失败: {str(e)}", exc_info=True)

# ==================== User-Agent 相关 ====================

def get_system_info() -> str:
    """获取格式化的系统信息用于User-Agent。"""
    system = platform.system()
    version = platform.release()
    machine = platform.machine()

    if system == "Windows":
        # 尝试获取更准确的Windows版本
        try:
            ver = sys.getwindowsversion()
            win_version = f"Windows NT {ver.major}.{ver.minor}"
        except AttributeError:
             win_version = f"Windows NT {version}" # 回退
        # 规范化常见版本
        if win_version == "Windows NT 10.0": win_version = "Windows NT 10.0; Win64; x64"
        elif win_version == "Windows NT 6.3": win_version = "Windows NT 6.3; Win64; x64" # Win 8.1
        elif win_version == "Windows NT 6.2": win_version = "Windows NT 6.2; Win64; x64" # Win 8
        elif win_version == "Windows NT 6.1": win_version = "Windows NT 6.1; Win64; x64" # Win 7
        return win_version
    elif system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        arch = "ARM64" if machine == 'arm64' else "Intel"
        return f"Macintosh; {arch} Mac OS X {mac_ver.replace('.', '_')}"
    elif system == "Linux":
        # 可以尝试获取发行版信息，但这比较复杂且不可靠
        return f"X11; Linux {machine}"
    return f"Unknown; {system} {version}"

def generate_chrome_user_agent() -> str:
    """生成随机的Chrome User-Agent。"""
    # 使用近期版本范围
    major_versions = list(range(100, 125))
    major = random.choice(major_versions)
    minor = random.randint(0, 9)
    build = random.randint(4000, 6000)
    patch = random.randint(100, 200)
    chrome_version = f"{major}.{minor}.{build}.{patch}"
    system_info = get_system_info()
    # 包含 Sec-CH-UA 信息可能更真实，但这里保持简单
    return f"Mozilla/5.0 ({system_info}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36"

def generate_edge_user_agent() -> str:
    """生成随机的Edge User-Agent。"""
    major_versions = list(range(100, 125))
    major = random.choice(major_versions)
    minor = random.randint(0, 9)
    build = random.randint(1000, 2000)
    patch = random.randint(10, 100)
    edge_version = f"{major}.{minor}.{build}.{patch}"
    # Edge通常基于对应的Chrome版本
    chrome_version = edge_version # 简化处理
    system_info = get_system_info()
    return f"Mozilla/5.0 ({system_info}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36 Edg/{edge_version}"

def generate_firefox_user_agent() -> str:
    """生成随机的Firefox User-Agent。"""
    major_versions = list(range(95, 126))
    major = random.choice(major_versions)
    minor = random.randint(0, 9)
    firefox_version = f"{major}.{minor}"
    system_info = get_system_info()
    # Firefox 版本号也出现在 rv: 部分
    return f"Mozilla/5.0 ({system_info}; rv:{firefox_version}) Gecko/20100101 Firefox/{firefox_version}"

def get_browser_user_agent(browser_type: Optional[str] = None) -> str:
    """
    根据浏览器类型获取User-Agent。支持 'chrome', 'edge', 'firefox'。
    如果 browser_type 为 None 或不支持，则随机选择一个。
    """
    valid_browsers = {
        "chrome": generate_chrome_user_agent,
        "edge": generate_edge_user_agent,
        "firefox": generate_firefox_user_agent
    }

    selected_type = browser_type.lower() if browser_type else random.choice(list(valid_browsers.keys()))

    generator = valid_browsers.get(selected_type, random.choice(list(valid_browsers.values())))
    return generator()


# ==================== 装饰器定义 ====================

def use_dynamic_user_agent(browser_type: Optional[str] = None):
    """
    装饰器：为被装饰的类方法调用动态生成并临时设置User-Agent。

    要求被装饰方法的'self'对象具有一个用于获取请求头的方法，
    通常命名为 `_get_headers()` 或 `get_headers()`，并且返回一个字典。
    它会尝试修改这个字典中的 'User-Agent' 键。

    参数:
        browser_type (str, 可选): 'chrome', 'edge', 'firefox' 或 None (随机)。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            header_method_name = None
            if hasattr(self, '_get_headers') and callable(self._get_headers):
                header_method_name = '_get_headers'
            elif hasattr(self, 'get_headers') and callable(self.get_headers):
                 header_method_name = 'get_headers'

            if not header_method_name:
                logger.warning(f"对象 {type(self).__name__} 没有找到 '_get_headers' 或 'get_headers' 方法，@use_dynamic_user_agent 将无效。")
                return func(self, *args, **kwargs)

            original_get_headers = getattr(self, header_method_name)
            original_headers_dict = original_get_headers() # 调用一次获取原始结构

            # 动态生成User-Agent
            user_agent = get_browser_user_agent(browser_type)
            logger.debug(f"使用动态User-Agent ({browser_type or 'random'} -> {user_agent.split(' ')[0]}...) for {func.__name__}")

            # 临时替换方法
            def patched_get_headers(*h_args, **h_kwargs):
                # 确保调用原始方法以获取基础headers
                headers = original_get_headers(*h_args, **h_kwargs)
                if not isinstance(headers, dict):
                     logger.error(f"{header_method_name} 返回的不是字典，无法设置User-Agent。类型: {type(headers)}")
                     return headers # 返回原始结果
                headers['User-Agent'] = user_agent
                return headers

            setattr(self, header_method_name, patched_get_headers)

            try:
                return func(self, *args, **kwargs)
            finally:
                # 恢复原始方法
                setattr(self, header_method_name, original_get_headers)
        return wrapper
    return decorator

class BackoffStrategy(Enum):
    """重试退避策略"""
    CONSTANT = "constant"      # 固定延迟
    LINEAR = "linear"          # 线性增长 (n * delay)
    EXPONENTIAL = "exponential"  # 指数增长 (factor^n * delay)
    FIBONACCI = "fibonacci"    # 斐波那契增长 (fib(n) * delay)
    JITTER = "jitter"          # 在基础延迟上加随机抖动

def retry_on_failure(max_retries: int = DEFAULT_MAX_RETRIES,
                     delay: float = DEFAULT_RETRY_DELAY,
                     retry_on_exceptions: Optional[List[type]] = None,
                     retry_on_api_codes: Optional[List[Union[int, ErrorCode]]] = None,
                     backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
                     max_delay: float = 60.0,
                     factor: float = 2.0, # 用于指数退避
                     jitter: bool = True): # 是否在所有策略上都添加抖动
    """
    增强版重试装饰器。

    在函数调用失败时自动重试，支持多种退避策略、特定异常/错误码重试和抖动。

    参数:
        max_retries (int): 最大重试次数 (不包括首次尝试)。
        delay (float): 基础重试延迟(秒)。
        retry_on_exceptions (List[type], 可选): 只在捕获到这些类型的异常时重试。
                                              默认为 [TimeoutError, ConnectionError, ServerError, RateLimitError]。
                                              注意：这里的TimeoutError和ServerError是ApiException的子类。
                                              如果要捕获内置的TimeoutError，需要显式添加。
        retry_on_api_codes (List[Union[int, ErrorCode]], 可选): 只在ApiException的错误码匹配时重试。
                                                      例如 [429, 500, 502, 503, 504, ErrorCode.RATE_LIMIT_ERROR]。
        backoff (BackoffStrategy): 退避策略。
        max_delay (float): 最大延迟时间(秒)。
        factor (float): 指数退避的底数 (仅用于EXPONENTIAL策略)。
        jitter (bool): 是否在计算出的延迟上添加随机抖动 (推荐开启以避免惊群效应)。
    """
    if retry_on_exceptions is None:
        # 默认重试网络相关、服务器错误和限流错误
        retry_on_exceptions = [TimeoutError, ConnectionError, ServerError, RateLimitError]

    # 转换错误码为整数，方便比较
    normalized_retry_codes = set()
    if retry_on_api_codes:
        for code in retry_on_api_codes:
            normalized_retry_codes.add(code.value if isinstance(code, ErrorCode) else int(code))

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果第一个参数是类实例且有logger，则使用它
            instance_logger = None
            if args and hasattr(args[0], 'logger') and isinstance(getattr(args[0], 'logger'), ContextLogger):
                instance_logger = getattr(args[0], 'logger')
            current_logger = instance_logger or logger # 使用实例logger或模块logger

            # 生成重试ID并设置上下文
            retry_context_id = generate_operation_id(f"retry_{func.__name__}")
            set_context_value("retry_id", retry_context_id)

            last_error = None
            fib_prev, fib_curr = 0, 1 # 用于斐波那契

            for attempt in range(max_retries + 1):
                start_time = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                    # 成功时，如果进行过重试，记录成功信息
                    if attempt > 0:
                        current_logger.info(f"重试成功: {func.__name__} 在第 {attempt+1} 次尝试成功。", extra_context=LogContext.all())

                    # 收集指标 (只有最后一次尝试或首次成功才记录总时间)
                    duration = (time.monotonic() - start_time) * 1000
                    _update_metrics("retry_on_failure", func.__name__, duration, error=False)
                    return result

                except Exception as e:
                    duration = (time.monotonic() - start_time) * 1000
                    last_error = e
                    error_name = type(e).__name__
                    error_msg = str(e)

                    should_retry = False
                    # 检查是否是可重试的异常类型
                    if any(isinstance(e, exc_type) for exc_type in retry_on_exceptions):
                        should_retry = True
                    # 如果是ApiException，检查错误码是否可重试
                    elif isinstance(e, ApiException) and e.code.value in normalized_retry_codes:
                         should_retry = True

                    # 如果不满足重试条件或者已达到最大尝试次数
                    if not should_retry or attempt == max_retries:
                        current_logger.warning(
                            f"重试终止: {func.__name__} 失败，尝试 {attempt+1}/{max_retries+1} 次。" + 
                            f" 错误: {error_name}('{error_msg}')" +
                            (f" (Code: {e.code.name})" if isinstance(e, ApiException) else "") +
                            (f" [ErrorID: {getattr(e, 'error_id', 'N/A')}]") +
                            (" (不可重试)" if not should_retry else " (达到最大重试次数)"),
                            extra_context=LogContext.all()
                        )
                        # 收集指标
                        _update_metrics("retry_on_failure", func.__name__, duration, error=True)
                        raise e # 抛出最后一次的异常

                    # --- 计算退避延迟 ---
                    current_delay = 0
                    if backoff == BackoffStrategy.CONSTANT:
                        current_delay = delay
                    elif backoff == BackoffStrategy.LINEAR:
                        current_delay = delay * (attempt + 1)
                    elif backoff == BackoffStrategy.EXPONENTIAL:
                        current_delay = delay * (factor ** attempt)
                    elif backoff == BackoffStrategy.FIBONACCI:
                        if attempt > 1:
                            fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
                        elif attempt == 1:
                            fib_prev, fib_curr = 0, 1 # 确保第二次尝试delay=1*delay
                        else: # attempt == 0
                            fib_prev, fib_curr = 0, 0 # 首次延迟为0，由基础delay决定
                        current_delay = delay * fib_curr if fib_curr > 0 else delay
                    elif backoff == BackoffStrategy.JITTER:
                        # Jitter策略通常是在其他策略基础上加随机性，这里简单实现为线性+随机
                        base_delay = delay * (attempt + 1)
                        current_delay = random.uniform(delay, max(delay, base_delay)) # 随机范围
                    else:
                        current_delay = delay # 默认为固定

                    # 应用最大延迟限制
                    current_delay = min(current_delay, max_delay)

                    # 添加抖动 (如果启用)
                    if jitter and backoff != BackoffStrategy.JITTER:
                        jitter_amount = current_delay * random.uniform(0.1, 0.5) # 增加延迟的10%-50%
                        wait_time = current_delay + jitter_amount
                    else:
                        wait_time = current_delay

                    # 确保等待时间不为负或过小
                    wait_time = max(0.1, wait_time) # 至少等待0.1秒

                    # 生成错误ID (每次重试前生成一个新的，关联到重试ID)
                    # error_id = generate_error_id(f"retry_attempt_{attempt+1}", func.__name__)
                    # 或者使用原始异常的error_id (如果存在且需要保留)
                    error_id = getattr(e, 'error_id', None)
                    if not error_id:
                        error_id = generate_error_id(f"retry_attempt_{attempt+1}", func.__name__)
                    else:
                        set_context_value("error_id", error_id) # 确保上下文中的ID是最新的

                    current_logger.warning(
                        f"操作失败将重试: {func.__name__} 尝试 {attempt+1}/{max_retries+1} 失败。" + 
                        f" 错误: {error_name}('{error_msg}')" +
                        (f" (Code: {e.code.name})" if isinstance(e, ApiException) else "") +
                        f" [ErrorID: {error_id}]. 等待 {wait_time:.2f} 秒后重试...",
                        extra_context=LogContext.all()
                    )

                    time.sleep(wait_time)
                    clear_context_value("error_id") # 清理当前尝试的错误ID，避免带到下一次

            # 如果循环结束还没成功 (理论上应该在循环内抛出异常)
            if last_error:
                current_logger.error(f"重试逻辑异常: {func.__name__} 循环结束但仍有错误。最后错误: {last_error}", extra_context=LogContext.all())
                _update_metrics("retry_on_failure", func.__name__, 0, error=True)
                raise last_error
            else: # 理论上不会执行到这里
                 current_logger.error(f"重试逻辑异常: {func.__name__} 循环结束但没有错误记录。", extra_context=LogContext.all())
                 # 可能表示成功了但没在循环中返回？
                 _update_metrics("retry_on_failure", func.__name__, 0, error=True) # 标记为错误
                 raise RuntimeError(f"{func.__name__} retry logic finished unexpectedly.")

            # finally 块不再需要，上下文清理委托给调用者或上层装饰器
            # finally:
            #     clear_context_value("retry_id")
        return wrapper
    return decorator

def cache_response(ttl: int = 300, max_size: int = _DEFAULT_CACHE_MAX_SIZE):
    """
    响应缓存装饰器 (线程安全，LRU特性)。

    缓存函数返回值一定时间，避免重复请求，并限制缓存大小。
    使用简单的字典实现近似LRU（当缓存满时移除随机一项，非严格LRU）。

    参数:
        ttl (int): 缓存有效期（秒）。默认300秒 (5分钟)。ttl<=0表示永久缓存。
        max_size (int): 最大缓存条目数。默认1000。

    注意:
        - 缓存键基于函数名、args和kwargs生成。确保参数是可哈希的。
        - 对于类方法，'self'实例不会包含在缓存键中（通常期望的行为）。
        - 提供了 cache_info() 和 cache_clear() 方法。
    """
    def decorator(func):
        cache: Dict[tuple, Any] = {}
        timestamps: Dict[tuple, float] = {}
        stats = {"hits": 0, "misses": 0}
        lock = threading.RLock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果第一个参数是类实例，从缓存键中排除它
            key_args = args
            instance = None
            if args and hasattr(args[0], func.__name__):
                 instance = args[0]
                 key_args = args[1:]

            # 生成缓存键 (确保kwargs有序)
            frozen_kwargs = frozenset(sorted(kwargs.items()))
            # 包含函数名以区分不同函数的相同参数
            cache_key = (func.__name__, key_args, frozen_kwargs)

            try:
                 hash(cache_key)
            except TypeError:
                 logger.warning(f"缓存键不可哈希: {func.__name__} 参数 {key_args}, {frozen_kwargs}. 无法使用缓存。")
                 # 记录指标 (算作miss，但时间为0)
                 _update_metrics("cache_response", func.__name__, 0, error=False)
                 return func(*args, **kwargs)

            now = time.monotonic() # 使用monotonic时钟更适合测量时间间隔

            with lock:
                # 检查缓存是否有效
                if cache_key in cache:
                    is_expired = False
                    if ttl > 0:
                        is_expired = (now - timestamps[cache_key]) > ttl

                    if not is_expired:
                        stats["hits"] += 1
                        # 获取实例logger或模块logger
                        current_logger = getattr(instance, 'logger', logger) if instance else logger
                        # DEBUG级别日志，避免过多输出
                        if current_logger.isEnabledFor(logging.DEBUG):
                            remaining = "永久" if ttl <= 0 else f"{ttl - (now - timestamps[cache_key]):.1f}秒"
                            current_logger.debug(f"缓存命中: {func.__name__}, Key: {str(cache_key)[:100]}..., 剩余有效期: {remaining}")
                        # 记录指标 (命中时间近似为0)
                        _update_metrics("cache_response", func.__name__, 0, error=False)
                        return cache[cache_key]
                    else:
                         # 缓存过期，移除旧条目
                         cache.pop(cache_key, None)
                         timestamps.pop(cache_key, None)

            # --- 缓存未命中或过期 --- 
            start_time = time.monotonic()
            result = func(*args, **kwargs)
            duration = (time.monotonic() - start_time) * 1000 # 毫秒

            with lock:
                stats["misses"] += 1
                # 限制缓存大小 (近似LRU：满时随机移除一个)
                # 注意：更精确的LRU需要 OrderedDict 或类似结构
                if len(cache) >= max_size:
                    try:
                        # 移除一个随机键 (比查找最旧键更快，但在高并发下可能非最优)
                        removed_key = random.choice(list(cache.keys()))
                        cache.pop(removed_key, None)
                        timestamps.pop(removed_key, None)
                    except IndexError: # 缓存为空，忽略
                        pass
                    except Exception as e:
                        logger.error(f"缓存清理时出错: {e}") # 捕获移除过程中的异常

                # 添加新缓存 (只有在缓存大小允许时)
                if len(cache) < max_size:
                    cache[cache_key] = result
                    timestamps[cache_key] = time.monotonic()

                    # 获取实例logger或模块logger
                    current_logger = getattr(instance, 'logger', logger) if instance else logger
                    if current_logger.isEnabledFor(logging.DEBUG):
                        hit_rate = stats["hits"] / (stats["hits"] + stats["misses"]) * 100 if (stats["hits"] + stats["misses"]) > 0 else 0
                        current_logger.debug(f"缓存更新: {func.__name__}, Key: {str(cache_key)[:100]}..., 耗时: {duration:.2f}ms, 命中率: {hit_rate:.1f}%, 缓存大小: {len(cache)}/{max_size}")
                else:
                    # 缓存已满，无法添加新条目
                    current_logger = getattr(instance, 'logger', logger) if instance else logger
                    current_logger.warning(f"缓存已满 ({max_size} 条目)，无法缓存 {func.__name__} 的结果。考虑增加 max_size。")

            # 记录指标
            _update_metrics("cache_response", func.__name__, duration, error=False)
            return result

        def cache_info() -> Dict[str, Any]:
            """返回缓存统计信息。"""
            with lock:
                num_items = len(cache)
                total_accesses = stats["hits"] + stats["misses"]
                hit_rate = stats["hits"] / total_accesses * 100 if total_accesses > 0 else 0
                return {
                    "hits": stats["hits"],
                    "misses": stats["misses"],
                    "size": num_items,
                    "max_size": max_size,
                    "ttl_seconds": ttl,
                    "hit_rate_percent": round(hit_rate, 2)
                }

        def cache_clear() -> None:
            """清除此函数的缓存。"""
            with lock:
                cache.clear()
                timestamps.clear()
                stats["hits"] = 0
                stats["misses"] = 0
            # 获取logger有点困难，因为没有实例
            logger.info(f"函数 {func.__name__} 的缓存已清除。")

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper
    return decorator

def _infer_operation_type(func_name: str) -> str:
    """根据函数名推断操作类型（简单规则）。"""
    name_lower = func_name.lower()
    if any(verb in name_lower for verb in ['get', 'fetch', 'query', 'list', 'search', 'check', 'describe', 'retrieve', 'head', 'options']):
        return "查询/获取"
    elif any(verb in name_lower for verb in ['upload', 'download', 'transfer', 'sync', 'backup']):
        return "文件传输"
    elif any(verb in name_lower for verb in ['create', 'add', 'new', 'put', 'post', 'insert', 'register']):
        return "创建/添加"
    elif any(verb in name_lower for verb in ['update', 'modify', 'set', 'patch', 'edit']):
        return "更新/修改"
    elif any(verb in name_lower for verb in ['delete', 'remove', 'cancel', 'purge']):
        return "删除/取消"
    elif any(verb in name_lower for verb in ['login', 'logout', 'auth', 'verify', 'authenticate']):
        return "认证/用户操作"
    elif any(verb in name_lower for verb in ['move', 'copy', 'rename', 'link']):
        return "文件/资源操作"
    else:
        return '通用操作'

def log_operation(operation_name: Optional[str] = None,
                  operation_type: Optional[str] = None,
                  log_level: int = logging.INFO, # 控制开始/结束日志的级别
                  error_log_level: int = logging.ERROR, # 控制错误日志的级别
                  log_args: bool = True, # 是否记录函数参数
                  log_return: bool = False, # 是否记录函数返回值 (可能包含敏感信息)
                  sensitive_keys: Optional[List[str]] = None, # 需要屏蔽的参数/属性键名
                  context_attributes: Optional[List[str]] = None, # 要从self实例记录的属性
                  handle_exceptions: bool = True, # 是否将未知异常包装成ApiException
                  collect_metrics: bool = True):
    """
    增强版操作日志装饰器。

    记录函数调用的开始、结束和异常，自动管理操作ID和上下文。

    参数:
        operation_name (str, 可选): 操作的描述性名称，默认使用函数名。
        operation_type (str, 可选): 操作类型，默认根据函数名推断。
        log_level (int): 记录开始/成功结束消息的日志级别，默认INFO。
        error_log_level (int): 记录失败/异常消息的日志级别，默认ERROR。
        log_args (bool): 是否记录函数调用的参数 (args 和 kwargs)。默认True。
        log_return (bool): 是否记录函数的返回值。默认False (返回值可能很大或敏感)。
        sensitive_keys (List[str], 可选): 在记录参数或属性时需要屏蔽的键名列表。
                                          默认包含 ['password', 'token', 'secret', 'key', 'credential', 'auth', 'cookies']。
        context_attributes (List[str], 可选): 要从 'self' 实例提取并记录到日志上下文的属性名列表。
        handle_exceptions (bool): 是否捕获所有未知异常并包装为 ApiException(ErrorCode.UNKNOWN_ERROR)。默认True。
        collect_metrics (bool): 是否收集性能指标。默认True。
    """
    default_sensitive = ['password', 'token', 'secret', 'key', 'credential', 'auth', 'cookies', 'session']
    sensitive_keys_set = set(default_sensitive + [k.lower() for k in sensitive_keys]) if sensitive_keys else set(default_sensitive)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 确定logger实例
            instance = args[0] if args and hasattr(args[0], func.__name__) else None
            current_logger = getattr(instance, 'logger', logger) if instance else logger

            # 确定操作名称和类型
            op_name = operation_name or func.__name__
            op_type = operation_type or _infer_operation_type(op_name)

            # 生成操作ID并设置上下文
            op_id = generate_operation_id(op_name)
            set_context_value("operation_id", op_id)

            # 记录开始日志 (仅当级别允许)
            start_log_enabled = current_logger.isEnabledFor(log_level)
            extra_context_start = {} # 用于日志记录的额外上下文

            if start_log_enabled:
                log_message = f"{op_type}: 开始 - {op_name}"
                details = []

                # 记录实例属性
                if instance and context_attributes:
                    attr_details = {}
                    for attr in context_attributes:
                        if hasattr(instance, attr):
                            value = getattr(instance, attr)
                            if attr.lower() in sensitive_keys_set:
                                value = '******' if value else None
                            attr_details[attr] = repr(value) # 使用repr更安全
                        else:
                             attr_details[attr] = "<Not Found>"
                    if attr_details:
                        details.append(f"ContextAttrs: {attr_details}")
                        extra_context_start.update({f"ctx_{k}": v for k, v in attr_details.items()}) # 添加到上下文

                # 记录参数
                if log_args:
                    arg_details = {}
                    # 处理位置参数 (忽略 self)
                    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                    arg_idx_offset = 1 if instance else 0
                    for i, arg_val in enumerate(args[arg_idx_offset:]):
                         arg_name = arg_names[i + arg_idx_offset] if (i + arg_idx_offset) < len(arg_names) else f"arg_{i}"
                         if arg_name.lower() in sensitive_keys_set:
                              arg_details[arg_name] = '******'
                         else:
                              arg_details[arg_name] = repr(arg_val)

                    # 处理关键字参数
                    for k, v in kwargs.items():
                        if k.lower() in sensitive_keys_set:
                             arg_details[k] = '******'
                        else:
                             arg_details[k] = repr(v)

                    if arg_details:
                        details.append(f"Args: {arg_details}")
                        # extra_context_start.update({f"arg_{k}": v for k,v in arg_details.items()}) # 参数通常不加到线程上下文

                if details:
                    log_message += " (" + ", ".join(details) + ")"

                current_logger.log(log_level, log_message, extra_context=extra_context_start)

            start_time = time.monotonic()
            error_occurred = False
            result_value = None
            error_id = None # 初始化错误ID

            try:
                result_value = func(*args, **kwargs)
                return result_value # 显式返回值
            except ApiException as e:
                error_occurred = True
                error_id = e.error_id or generate_error_id(f"api_{e.code.name}", op_name) # 使用已有或生成新的
                set_context_value("error_id", error_id) # 确保错误ID在上下文中
                current_logger.log(
                    error_log_level,
                    f"{op_type}: 失败 - {op_name}. Code: {e.code.name}, Msg: {e.message}",
                    exc_info=False, # ApiException通常不需要完整traceback日志
                    extra_context=LogContext.all() # 包含所有上下文
                )
                raise # 重新抛出原始API异常
            except Exception as e:
                error_occurred = True
                error_id = generate_error_id("unknown_error", op_name) # 未知错误生成ID
                set_context_value("error_id", error_id)
                current_logger.log(
                    error_log_level,
                    f"{op_type}: 异常 - {op_name}. Type: {type(e).__name__}, Msg: {str(e)}",
                    exc_info=True, # 对未知异常记录traceback
                    extra_context=LogContext.all()
                )
                if handle_exceptions:
                    raise ApiException(
                        code=ErrorCode.UNKNOWN_ERROR,
                        message=f"{op_name} 时发生未处理的异常: {type(e).__name__} - {str(e)}",
                        error_id=error_id
                    ) from e # 保留原始异常链
                else:
                    raise # 重新抛出原始异常
            finally:
                elapsed_time_ms = (time.monotonic() - start_time) * 1000
                # 记录结束日志 (仅当级别允许，并且不是因异常退出)
                if not error_occurred and current_logger.isEnabledFor(log_level):
                    log_message = f"{op_type}: 完成 - {op_name}. 耗时: {elapsed_time_ms:.2f}ms"
                    if log_return:
                        try:
                            return_repr = repr(result_value)
                            # 检查是否包含敏感词？(简单检查，可能误判)
                            if any(sk in return_repr.lower() for sk in sensitive_keys_set):
                                 log_message += ", 返回值: <包含敏感内容>"
                            else:
                                 log_message += f", 返回值: {return_repr[:200]}{ '...' if len(return_repr)>200 else '' }"
                        except Exception:
                             log_message += ", 返回值: <无法表示>"
                    current_logger.log(log_level, log_message, extra_context=LogContext.all())

                # 收集指标 (如果启用)
                if collect_metrics:
                    _update_metrics("log_operation", func.__name__, elapsed_time_ms, error=error_occurred)

                # 清理当前操作在上下文中设置的值
                clear_context_value("operation_id")
                if error_id: # 只在发生错误时清理错误ID
                    clear_context_value("error_id")
                # 清理可能由这个装饰器添加的 ctx_ 开头的属性
                if start_log_enabled and instance and context_attributes:
                     for attr in context_attributes:
                         clear_context_value(f"ctx_{attr}")

        return wrapper
    return decorator

def api_method(operation_name: Optional[str] = None,
               max_retries: int = DEFAULT_MAX_RETRIES,
               retry_delay: float = DEFAULT_RETRY_DELAY,
               retry_backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
               retry_exceptions: Optional[List[type]] = None,
               retry_codes: Optional[List[Union[int, ErrorCode]]] = None,
               cache_ttl: Optional[int] = None, # None表示不缓存
               cache_max_size: int = _DEFAULT_CACHE_MAX_SIZE,
               enable_retry: bool = True,  # 是否启用重试
               enable_cache: bool = False, # 是否启用缓存，默认不启用
               log_args: bool = True,
               log_return: bool = False,
               sensitive_keys: Optional[List[str]] = None,
               context_attributes: Optional[List[str]] = None,
               handle_exceptions: bool = True):
    """
    通用API方法组合装饰器。

    按顺序应用: 日志 -> 缓存 -> 重试。
    注意装饰器执行顺序与应用顺序相反。

    参数:
        operation_name: 操作名称 (用于日志)。
        max_retries: 最大重试次数。
        retry_delay: 重试基础延迟。
        retry_backoff: 重试退避策略。
        retry_exceptions: 重试的异常类型列表。
        retry_codes: 重试的API错误码列表。
        cache_ttl: 缓存时间 (秒)，None表示不缓存。
        cache_max_size: 缓存最大条目数。
        enable_retry: 是否启用重试功能，默认为True。
        enable_cache: 是否启用缓存功能，默认为False。
        log_args: 是否记录参数。
        log_return: 是否记录返回值。
        sensitive_keys: 屏蔽的敏感键。
        context_attributes: 记录到日志的实例属性。
        handle_exceptions: 是否包装未知异常。

    返回:
        Callable: 应用了所有选定功能的装饰器。
    """
    def decorator(func):
        # 应用装饰器 (从内到外，执行顺序相反)
        decorated_func = func

        # 1. 重试 (最内层，先执行重试逻辑)
        if enable_retry:
            decorated_func = retry_on_failure(
                max_retries=max_retries,
                delay=retry_delay,
                retry_on_exceptions=retry_exceptions,
                retry_on_api_codes=retry_codes,
                backoff=retry_backoff
            )(decorated_func)

        # 2. 缓存 (如果启用了缓存)
        if enable_cache and cache_ttl is not None:
            decorated_func = cache_response(
                ttl=cache_ttl,
                max_size=cache_max_size
            )(decorated_func)

        # 3. 日志记录 (在重试和缓存外部，记录最终结果或失败)
        decorated_func = log_operation(
            operation_name=operation_name,
            log_args=log_args,
            log_return=log_return,
            sensitive_keys=sensitive_keys,
            context_attributes=context_attributes,
            handle_exceptions=handle_exceptions,
            collect_metrics=True # 指标收集在log_operation内部完成
        )(decorated_func)

        # 保留原始函数的签名
        return wraps(func)(decorated_func)

    return decorator 