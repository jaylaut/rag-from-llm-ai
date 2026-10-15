# ============================================================
# logger.py —— 全项目共用的日志设施（基础设施，不属于任何业务模块）
#
# 位置：src/rag/utils/logger.py
#
# 【本文件是干什么的】
#   把程序运行过程记录下来，同时输出到两个地方：
#     1. 控制台（人实时看进度，只显示 INFO 及以上，保持清爽）
#     2. 日志文件（机器事后排查，记录 DEBUG 及以上，最全最细）
#
# 【企业级增强点（相对基础版 logger 的三处升级）】
#   1. trace_id（链路追踪 ID）：一次"运行"生成一个唯一 ID，写进每行日志。
#      好处：几十万文件批量处理时，事后能凭这个 ID 把"同一次运行"的
#      所有日志串起来，快速定位是哪次运行出的问题。
#   2. 幂等共享（保留）：同名 Logger 重复调用直接返回已配置实例，
#      保证"入口配一次，全项目复用"，配置不会漂移。
#   3. 异常上下文（log.exception）：专门记录"崩溃现场"（异常类型+堆栈），
#      排查线上问题时不抓瞎。
# ============================================================

# ---------- 导入区 ----------
# logging：Python 标准库的日志模块，提供 Logger/Handler/Formatter 等核心类。
import logging

# uuid：Python 标准库，生成"全局唯一 ID"（trace_id 用它生成）。
import uuid

# datetime：Python 标准库，取当前时间，给日志文件名加时间戳。
from datetime import datetime

# Path：Python 标准库 pathlib 的类，安全处理文件路径。
from pathlib import Path

# ---------- 日志格式模板 ----------
# 每条日志最终打印成：时间 | trace_id | 级别 | 模块:行号 | 消息
#   %(asctime)s   时间（精确到毫秒）
#   %(trace_id)s  自定义字段：链路追踪 ID（下面对应到 formatter 里注册）
#   %(levelname)-7s 级别，-7s 固定占 7 格左对齐，视觉整齐
#   %(name)s      Logger 名（默认 "rag"）
#   %(lineno)d    打印日志的代码行号（排查时直接跳过去）
#   %(message)s   日志正文
LOG_FORMAT = (
    "%(asctime)s | %(trace_id)s | %(levelname)-7s | "
    "%(name)s:%(lineno)d | %(message)s"
)

# 时间显示格式：年-月-日 时:分:秒（精确到秒）
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------- 自定义 Filter：给每条日志注入 trace_id ----------
# logging.Filter 是日志系统的一个"过滤器"钩子。每产生一条日志，都会先经过
# 所有 filter 的 filter() 方法。我们借这个机制，在每条日志的"上下文字典"
# （record.__dict__）里塞进一个 trace_id，这样上面的 %(trace_id)s 占位符
# 就能被正确填充。
class TraceIdFilter(logging.Filter):
    """给每条日志记录注入一个 trace_id（链路追踪 ID）。

    【原理】
        Python logging 里，每条日志会被包装成一个 LogRecord 对象，格式模板
        %(xxx)s 里的 xxx 就是从 LogRecord 的属性里取值。默认 LogRecord 没有
        trace_id 这个属性，所以我们写一个 Filter，在 filter() 里动态给它
        加上这个属性，格式化时就能读到了。
    """

    def __init__(self, trace_id: str):
        """构造函数：保存本次运行的 trace_id。

        参数：
            trace_id : str —— 本次运行的唯一标识（由 uuid 生成）
        """
        super().__init__()          # 调用父类 Filter 的构造，保持正确初始化
        self.trace_id = trace_id    # 存下来，供 filter() 使用

    def filter(self, record: logging.LogRecord) -> bool:
        """每条日志都会经过这里，把 trace_id 写进日志记录。

        参数：
            record : logging.LogRecord —— 一条日志的记录对象

        返回：
            bool —— 恒返回 True，表示"这条日志放行，不被过滤掉"
        """
        # 给 record 动态加一个 trace_id 属性（Python 对象可随意加属性）
        record.trace_id = self.trace_id
        return True   # True = 放行；若返回 False 则这条日志被丢弃


# ---------- 核心函数：配置并返回 Logger ----------
def setup_logger(
        name: str = "rag",                        # Logger 名（同名=同一实例的共享凭证）
        log_dir: Path = Path("data/logs"),        # 日志文件目录
        run_name: str = "app",                    # 运行标识（决定日志文件名前缀）
        console_level: int = logging.INFO,        # 控制台只显示 INFO 及以上
        file_level: int = logging.DEBUG,          # 文件记录 DEBUG 及以上（最全）
        separate_error_file: bool = False,        # 是否额外生成只收 ERROR 的文件
        include_third_party: bool = False,        # 是否把第三方库告警也记下来
) -> logging.Logger:
    """配置并返回一个"双通道"Logger（控制台 + 文件）。全项目统一入口。

    标准用法（两行）：
        from rag.utils.logger import setup_logger
        log = setup_logger()

    参数说明（只列关键两个，其余见类型标注）：
        run_name   本次运行的名字，日志文件名为 <run_name>_<时间>_<trace_id>.log。
        separate_error_file  True 时额外生成一个只收 ERROR 的文件，排查效率翻倍。
        include_third_party  True 时把 faiss/requests 等库的 WARNING 也存进日志文件。
    """

    # ---- 1. 生成本次运行的 trace_id ----
    # uuid.uuid4() 生成一个随机唯一 ID，.hex 取其 32 位十六进制表示。
    # 截取前 8 位足够唯一（2^32 种组合），且不占空间、日志里好读。
    trace_id = uuid.uuid4().hex[:8]

    # ---- 2. 拿到 Logger（同名全局唯一）----
    logger = logging.getLogger(name)

    # ---- 3. 幂等判断：已配置过就直接返回，防止日志打两份 ----
    if logger.handlers:
        return logger

    # ---- 4. Logger 总级别设最低，具体粗细由各 handler 决定 ----
    logger.setLevel(logging.DEBUG)

    # ---- 5. 确保日志目录存在 ----
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- 6. 生成带时间戳和 trace_id 的文件名（每次运行独立，历史不覆盖）----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 7. 创建 Formatter，并把自定义的 trace_id 字段注册进去 ----
    # 注意：默认 Formatter 不认识 %(trace_id)s，必须通过 "defaults" 参数
    # 提供一个默认值，否则遇到没有 trace_id 的 record 会抛 KeyError。
    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
        defaults={"trace_id": trace_id},   # 兜底默认值，配合 TraceIdFilter 使用
    )

    # ---- 8. 创建 TraceIdFilter 实例，挂到各 handler 上 ----
    trace_filter = TraceIdFilter(trace_id)

    # ========== 通道一：控制台（INFO 及以上）==========
    console_handler = logging.StreamHandler()   # StreamHandler 输出到控制台
    console_handler.setLevel(console_level)     # 低于 INFO 的不显示
    console_handler.setFormatter(formatter)
    console_handler.addFilter(trace_filter)     # 注入 trace_id
    logger.addHandler(console_handler)

    # ========== 通道二：主日志文件（DEBUG 及以上，最全）==========
    main_log_file = log_dir / f"{run_name}_{timestamp}_{trace_id}.log"
    # encoding="utf-8" 必须显式指定，否则 Windows 中文乱码
    file_handler = logging.FileHandler(main_log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(trace_filter)
    logger.addHandler(file_handler)

    # ========== 可选通道三：ERROR 专属文件 ==========
    error_log_file = None
    if separate_error_file:
        error_log_file = log_dir / f"{run_name}_error_{timestamp}_{trace_id}.log"
        error_handler = logging.FileHandler(error_log_file, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)   # 只收 ERROR 及以上
        error_handler.setFormatter(formatter)
        error_handler.addFilter(trace_filter)
        logger.addHandler(error_handler)

    # ========== 可选：捕获第三方库日志 ==========
    if include_third_party:
        root_logger = logging.getLogger()       # 无参 = 根 Logger，所有库日志都汇到这
        root_logger.setLevel(logging.WARNING)   # 库日志只收 WARNING 以上
        root_logger.addHandler(file_handler)    # 复用文件 handler，同一时间线

    # ---- 9. 把关键信息挂到 logger 对象上，供运行报告引用 ----
    # logging.Logger 类本身没有这三个属性，我们是"动态附加"自定义属性
    # （Python 对象允许随意加属性），供运行结束报告引用日志路径/trace_id。
    # 用 # type: ignore[attr-defined] 告知 mypy：这是故意的动态属性，不是笔误。
    logger.log_file_path = main_log_file     # type: ignore[attr-defined]  # 主日志文件路径
    logger.error_file_path = error_log_file  # type: ignore[attr-defined]  # 错误日志文件路径
    logger.trace_id = trace_id               # type: ignore[attr-defined]  # 本次运行的 trace_id

    # ---- 10. 首条日志，确认配置生效 ----
    logger.info(
        "日志系统就绪 | 运行=%s | 控制台=%s | 文件=%s | %s",
        run_name,
        logging.getLevelName(console_level),   # 数字级别转可读字符串（20→INFO）
        logging.getLevelName(file_level),
        main_log_file.resolve(),               # resolve 转绝对路径
    )
    return logger
