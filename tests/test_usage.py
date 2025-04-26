# tests/test_usage.py

import time
import random
import logging
from typing import Optional

# --- 1. 导入 yastool 组件 ---
from yastool.common.logger import (
    setup_logging,
    get_logger,
    LogContext,
    log_exception,
    GlobalErrorHandler # 导入但不在此处安装，通常在主程序入口安装
)
from yastool.decorators.api import api_method, BackoffStrategy
from yastool.models.error_codes import ApiException, ErrorCode, NetworkError

# --- 2. 初始化日志系统 ---
# 配置日志，输出到控制台和文件 (logs/test.log)
# 使用 DEBUG 级别，以便观察装饰器的详细日志
setup_logging(
    log_dir='logs', # 日志会存放在 tests/logs 目录下
    log_filename_format='test_{date}.log',
    default_console_level='DEBUG', # 控制台显示 DEBUG 及以上
    default_file_level='DEBUG',    # 文件记录 DEBUG 及以上
    log_format='%(asctime)s - %(levelname)s%(context_space)s%(context_str)s - %(name)s:%(lineno)d - %(message)s'
)

# 获取 logger
logger = get_logger(__name__) # 使用当前模块名 'tests.test_usage'

# --- 3. 定义一个模拟 API 调用的类和方法 ---
class MockApiClient:
    def __init__(self):
        self.call_count = 0
        self.network_error_count = 0
        self.rate_limit_count = 0
        self.logger = get_logger(type(self).__name__) # 使用类名作为 logger 名

    @api_method(
        operation_name="get_mock_data", # 自定义操作名
        enable_retry=True,
        max_retries=3,          # 最多重试3次
        retry_delay=0.2,        # 基础延迟0.2秒
        retry_backoff=BackoffStrategy.EXPONENTIAL, # 指数退避
        # 重试网络错误和特定的API错误码 (429-限流, 503-服务不可用)
        retry_exceptions=[NetworkError, ConnectionError], # 添加内置 ConnectionError
        retry_codes=[ErrorCode.RATE_LIMIT_ERROR, 503],
        enable_cache=True,
        cache_ttl=2,            # 缓存2秒 (方便演示)
        log_args=True,
        log_return=True,        # 开启返回值记录以供演示
        context_attributes=['call_count'] # 记录 self.call_count 到日志上下文
    )
    def fetch_data(self, item_id: str, force_error: Optional[str] = None) -> dict:
        """模拟获取数据的 API 方法"""
        self.call_count += 1
        self.logger.info(f"内部调用 fetch_data (第 {self.call_count} 次)")

        # --- 模拟各种失败场景 ---
        if force_error == "network":
            self.network_error_count += 1
            if self.network_error_count <= 2: # 前两次抛出网络错误
                 self.logger.warning(f"模拟网络错误 (第 {self.network_error_count} 次)")
                 raise ConnectionError("模拟的网络连接中断")
            else:
                 self.logger.info("网络错误后恢复")
                 self.network_error_count = 0 # 重置计数器

        elif force_error == "ratelimit":
             self.rate_limit_count += 1
             if self.rate_limit_count <= 2: # 前两次返回 429
                 self.logger.warning(f"模拟触发速率限制 (第 {self.rate_limit_count} 次)")
                 raise ApiException(code=ErrorCode.RATE_LIMIT_ERROR, message="请求过于频繁")
             else:
                 self.logger.info("速率限制解除")
                 self.rate_limit_count = 0

        elif force_error == "server_error":
            self.logger.error("模拟发生不可恢复的服务器内部错误")
            raise ApiException(code=ErrorCode.SERVER_ERROR, message="服务器内部处理失败")

        elif force_error == "notfound":
             self.logger.warning("模拟资源未找到错误")
             raise ApiException(code=ErrorCode.RESOURCE_NOT_FOUND, message=f"项目 {item_id} 未找到")

        # 模拟耗时
        time.sleep(random.uniform(0.1, 0.3))

        # 模拟成功响应
        response = {
            "id": item_id,
            "data": f"Data for {item_id}",
            "timestamp": time.time(),
            "internal_call_count": self.call_count
        }
        self.logger.info(f"成功获取项目 {item_id} 的数据")
        return response

# --- 4. 演示调用流程 ---
def run_demo():
    logger.info("===== 开始 yastool 使用演示 =====")
    client = MockApiClient()

    # --- 场景 1: 首次成功调用 (无缓存, 无错误) ---
    logger.info("\n----- 场景 1: 首次成功调用 -----")
    LogContext.set("request_type", "initial_fetch")
    try:
        data1 = client.fetch_data("item-001")
        logger.info(f"场景 1 结果: {data1['data']}")
    except Exception as e:
        logger.error(f"场景 1 意外失败: {e}", exc_info=True)
    finally:
        LogContext.clear() # 清理本次请求的上下文

    time.sleep(0.1) # 短暂间隔

    # --- 场景 2: 缓存命中 ---
    logger.info("\n----- 场景 2: 缓存命中 (2秒内) -----")
    LogContext.set("request_type", "cached_fetch")
    try:
        data2 = client.fetch_data("item-001")
        logger.info(f"场景 2 结果: {data2['data']} (应与场景1时间戳相同或接近)")
        assert data1['timestamp'] == data2['timestamp'] # 验证缓存命中
        logger.info("缓存命中验证成功!")
    except Exception as e:
        logger.error(f"场景 2 意外失败: {e}", exc_info=True)
    finally:
        LogContext.clear()

    # --- 场景 3: 缓存失效 ---
    logger.info("\n----- 场景 3: 缓存失效 (等待超过2秒) -----")
    logger.info("等待 2.5 秒使缓存失效...")
    time.sleep(2.5)
    LogContext.set("request_type", "expired_cache_fetch")
    try:
        data3 = client.fetch_data("item-001")
        logger.info(f"场景 3 结果: {data3['data']} (时间戳应更新)")
        assert data1['timestamp'] != data3['timestamp']
        logger.info("缓存失效验证成功!")
    except Exception as e:
        logger.error(f"场景 3 意外失败: {e}", exc_info=True)
    finally:
        LogContext.clear()

    # --- 场景 4: 可重试的网络错误 ---
    logger.info("\n----- 场景 4: 可重试的网络错误 -----")
    LogContext.set("request_type", "retry_network")
    try:
        # 前两次调用会抛 ConnectionError 并被重试，第三次成功
        data4 = client.fetch_data("item-002", force_error="network")
        logger.info(f"场景 4 结果: {data4['data']} (经历重试后成功)")
    except ConnectionError:
         logger.error("场景 4 失败: 网络错误重试次数耗尽 (不应发生)")
    except Exception as e:
        logger.error(f"场景 4 意外失败: {e}", exc_info=True)
    finally:
        LogContext.clear()

    # --- 场景 5: 可重试的 API 限流错误 ---
    logger.info("\n----- 场景 5: 可重试的 API 限流错误 -----")
    LogContext.set("request_type", "retry_ratelimit")
    try:
        # 前两次调用会抛 429 ApiException 并被重试，第三次成功
        data5 = client.fetch_data("item-003", force_error="ratelimit")
        logger.info(f"场景 5 结果: {data5['data']} (速率限制后成功)")
    except ApiException as e:
         if e.code == ErrorCode.RATE_LIMIT_ERROR:
              logger.error("场景 5 失败: 速率限制重试次数耗尽 (不应发生)")
         else:
             logger.error(f"场景 5 发生意外 API 错误: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"场景 5 意外失败: {e}", exc_info=True)
    finally:
        LogContext.clear()

    # --- 场景 6: 不可重试的 API 错误 (NotFound) ---
    logger.info("\n----- 场景 6: 不可重试的 API 错误 (NotFound) -----")
    LogContext.set("request_type", "non_retryable_api_error")
    try:
        client.fetch_data("item-unknown", force_error="notfound")
        logger.error("场景 6 错误: NotFound 错误未被抛出 (逻辑错误)")
    except ApiException as e:
        if e.code == ErrorCode.RESOURCE_NOT_FOUND:
            logger.info(f"场景 6 成功捕获 NotFound 错误: {e.message} (ErrID: {e.error_id})")
        else:
            logger.error(f"场景 6 捕获到非预期的 API 错误: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"场景 6 意外失败: {e}", exc_info=True)
    finally:
        LogContext.clear()

    # --- 场景 7: 捕获未处理异常 (演示 log_exception) ---
    logger.info("\n----- 场景 7: 捕获其他异常 -----")
    LogContext.set("request_type", "unhandled_exception_test")
    try:
        result = 10 / 0
    except Exception:
        # 使用 log_exception 记录当前异常，会自动包含上下文和 error_id
        log_exception(logger_name=__name__, message="计算失败")
        logger.info("场景 7 成功捕获并记录了除零异常")
    finally:
        LogContext.clear()

    logger.info("\n===== yastool 使用演示结束 =====")

if __name__ == "__main__":
    # 提示：GlobalErrorHandler.install() 通常在应用程序的主入口点调用一次即可
    # GlobalErrorHandler.install()
    run_demo()
    # 可以在这里调用 GlobalErrorHandler.show_error("测试", "这是一个测试错误") 来测试GUI提示 (如果环境支持) 