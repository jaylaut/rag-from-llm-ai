# ============================================================================
# conftest.py —— pytest 全局配置（测试的"公共底座"）
#
# 位置：tests/conftest.py
#
# 【本文件是干什么的】
#   pytest 启动测试时，会自动加载根目录下名为 conftest.py 的文件。
#   这里做两件事：
#     1. 把 src/ 目录加入 Python 的模块搜索路径（sys.path），
#        这样测试里才能 `from rag.ingestion.registry import ...`；
#     2. 提供测试共用的"临时目录"fixture（fixture = pytest 的依赖注入机制）。
# ============================================================================

# ---------- 导入区 ----------
# sys：Python 标准库，提供 sys.path（模块搜索路径列表）。
import sys

# Path：Python 标准库，处理路径。
from pathlib import Path

# pytest：测试框架。
import pytest

# ============================================================================
# 第 1 步：把项目根目录和 src/ 加入 sys.path
# ============================================================================
# __file__ 是本文件路径 tests/conftest.py；.resolve().parent 是 tests/ 目录；
# 再 .parent 是项目根目录。
ROOT_DIR = Path(__file__).resolve().parent.parent   # 项目根目录
SRC_DIR = ROOT_DIR / "src"                           # src 目录（rag 包所在地）

# 把 src/ 目录加入 sys.path（这样 `from rag...` 才能被找到，与 pip install -e
# 后的包名一致）。同时保留项目根目录（有些工具链按根目录找）。
# 注意用 insert(0, ...) 放到最前面，优先于已安装的其它同名包。
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))


# ============================================================================
# 第 2 步：提供测试共用的 fixture
# ============================================================================
@pytest.fixture
def tmp_registry_path(tmp_path):
    """提供一个"临时目录里的注册表路径"。

    【为什么用临时目录】
        测试绝不能污染真实数据。pytest 内置的 tmp_path fixture 会给每个测试
        一个独立的临时目录，测试结束后自动清理。我们把注册表指向这个临时
        目录里的文件，测试里随便读写都不会影响 data/index/registry.json。

    【参数】
        tmp_path : pytest 内置 fixture，指向本次测试专属的临时目录

    【返回】
        Path —— 临时目录里的 registry.json 路径
    """
    return tmp_path / "registry.json"


# ============================================================================
# 第 3 步：注册自定义标记 + 集成测试默认跳过
# ============================================================================
def pytest_configure(config):
    """pytest 启动钩子：注册自定义标记，消除"未注册标记"警告。

    参数：
        config : pytest 的配置对象
    """
    # 注册 integration 标记：表示"依赖外部服务（如 Ollama）的集成测试"
    config.addinivalue_line("markers", "integration: 依赖外部服务的集成测试")


def pytest_addoption(parser):
    """添加命令行选项 --run-integration。

    参数：
        parser : pytest 的命令行参数解析器
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="运行集成测试（依赖外部服务，如 Ollama）",
    )


def pytest_collection_modifyitems(config, items):
    """收集完测试后，根据 --run-integration 决定是否跳过集成测试。

    【企业级规范】
        集成测试依赖外部服务（Ollama、数据库等），默认在 CI/普通运行里跳过，
        只在显式传 --run-integration 时才执行。这样：
        · 普通单测：快、稳定、不依赖环境；
        · 集成测试：需要时单独跑，验证真实链路。

    参数：
        config : pytest 配置对象
        items  : 收集到的所有测试项列表
    """
    # 若没传 --run-integration，就把所有 integration 标记的测试跳过
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(
            reason="需要 --run-integration 选项才运行（依赖外部服务）"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

