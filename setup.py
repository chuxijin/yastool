# coding:utf-8
import setuptools
from pathlib import Path
# import re # 不再需要 re 模块

# 读取 README.md 作为 long_description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 不再从 __init__.py 读取版本号，直接在此处定义
HARDCODED_VERSION = "0.1.0"

setuptools.setup(
    name="yastool", # 包名
    version=HARDCODED_VERSION, # 使用硬编码的版本号
    author="Chuxijin", # 修改为你的名字或团队名
    author_email="chuxijin@163.com", # 修改为你的邮箱
    description="Yet Another Simple Toolkit for Python Projects (Logging & Decorators)", # 简短描述
    long_description=long_description,
    long_description_content_type="text/markdown", # README 格式
    url="https://github.com/Chuxijin/yastool", # 修改为你的项目仓库地址
    packages=setuptools.find_packages(where=".", include=["yastool*"]), # 自动查找yastool下的包
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License", # 选择一个开源许可证
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Logging",
        "Topic :: Utilities",
    ],
    python_requires='>=3.8', # 指定最低Python版本
    install_requires=[ # 核心库目前无外部依赖
        # 'requests>=2.20.0', # 如果未来有依赖，在此添加
    ],
    extras_require={ # 可选依赖
        'gui': [
            'PySide6>=6.0.0',    # Qt GUI 绑定
            # 'qfluentwidgets>=1.0.0' # 移除，假定由系统/用户自行提供
        ]
        # 可以添加其他可选依赖组，例如 'web': ['flask>=2.0']
    }
) 