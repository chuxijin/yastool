# tests/test_gui_error.py

import sys
import logging

# --- 导入必要的库 ---
# GUI 库 (尝试导入，如果失败则禁用GUI测试)
try:
    from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
    from qfluentwidgets import setTheme, Theme # 用于设置主题，使界面更好看
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"无法导入 GUI 库 (PySide6 或 qfluentwidgets): {e}\nGUI 测试将被跳过。请确保已安装 PySide6 和 qfluentwidgets。")
    GUI_AVAILABLE = False

# yastool 组件
from yastool.common.logger import (
    setup_logging,
    get_logger,
    GlobalErrorHandler
)

# --- 初始化日志 --- 
# (可以复用 test_usage.py 的配置或单独配置)
setup_logging(
    log_dir='logs',
    log_filename_format='test_gui_{date}.log',
    default_console_level='INFO',
    default_file_level='DEBUG',
    log_format='%(asctime)s - %(levelname)s%(context_space)s%(context_str)s - %(name)s:%(lineno)d - %(message)s'
)
logger = get_logger(__name__)

# --- 定义测试函数 ---
def trigger_manual_error_dialog():
    """手动触发错误对话框"""
    logger.info("正在手动触发 show_error...")
    # 调用 show_error，它会尝试使用安装时设置的主窗口作为父窗口
    success = GlobalErrorHandler.show_error(
        "手动触发错误",
        "这是一个通过 GlobalErrorHandler.show_error() 手动显示的消息。"
    )
    if success:
        logger.info("手动错误对话框已成功显示并关闭。")
    else:
        logger.error("手动错误对话框未能成功显示。请检查日志和GUI环境。")

def trigger_unhandled_exception():
    """触发一个未处理的异常，由全局处理器捕获"""
    logger.info("正在触发一个未处理的 ZeroDivisionError...")
    # 这个错误会被 GlobalErrorHandler.handle_exception 捕获
    # 如果安装时传递了 main_window，handle_exception 内部会调用 show_error
    result = 1 / 0
    print(result) # 这行不会执行

# --- 创建简单的 GUI 应用程序 --- 
class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Yastool GUI Error Handler Test')
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout(self)

        # 按钮 1: 手动显示错误
        self.btn_manual = QPushButton('手动触发 show_error()', self)
        self.btn_manual.clicked.connect(trigger_manual_error_dialog)
        layout.addWidget(self.btn_manual)

        # 按钮 2: 触发未处理异常
        self.btn_unhandled = QPushButton('触发未处理异常 (全局捕获)', self)
        self.btn_unhandled.clicked.connect(trigger_unhandled_exception)
        layout.addWidget(self.btn_unhandled)

# --- 主程序入口 --- 
if __name__ == "__main__":
    if GUI_AVAILABLE:
        app = QApplication(sys.argv)

        # 设置 Fluent UI 主题 (可选)
        setTheme(Theme.DARK) # 或 Theme.LIGHT

        # 创建主窗口
        window = TestWindow()

        # !!! 关键: 安装全局错误处理器，并传递主窗口引用 !!!
        GlobalErrorHandler.install(main_window=window)
        logger.info("已安装全局异常处理器，并设置了主窗口。")

        window.show()

        logger.info("测试窗口已显示。请点击按钮测试错误处理。")

        sys.exit(app.exec())
    else:
        logger.warning("GUI 环境不可用，无法运行 GUI 测试。")
        sys.exit(1) # 以错误码退出 