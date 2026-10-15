# ============================================================================
# detector.py —— 存量变更检测（类型 A）
#
# 位置：src/rag/ingestion/detector.py
#
# 【本文件是干什么的】
#   增量更新有两类来源（对应《P1-索引构建.md》2.5.3.1）：
#     · 类型 A：raw/ 目录内的文档被修改/删除 —— 靠"目录 diff"发现差集
#     · 类型 B：用户临时上传/外部目录注入的新文档 —— 靠"主动入站"登记
#   本文件负责【类型 A】：扫描 raw/ 全集，与注册表比对，找出三类变化：
#     新增（注册表没有）、变更（哈希不同）、删除（注册表有、磁盘没了）。
#
# 【为什么必须用内容哈希而不是文件修改时间 mtime（对应文档 2.5.4）】
#   mtime 有两个方向的坑：
#     · 假变更：git checkout、解压、云盘同步会刷新 mtime，内容却没变 → 白重嵌
#     · 漏变更：某些工具改内容却不更新 mtime → 改了却检索不到
#   内容哈希是"内容本身"算出来的，骗不了它。
# ============================================================================

# ---------- 导入区 ----------
# Path：Python 标准库 pathlib 里的类，安全处理文件/目录路径。
# dataclass：Python 标准库，定义"数据容器"类（自动生成 __init__ 等样板代码）。
from dataclasses import dataclass, field
from pathlib import Path

# List：Python 的类型标注工具（List[Path] 表示"由 Path 组成的列表"）。
# 项目内：内容哈希函数。注意：这里复用 registry 里的 compute_content_hash，
# 保证"判重"和"判变"用的是同一把尺子（对应文档 2.1.2 决策4：内容 hash 是全
# 系统判断"变没变"的统一依据）。
# 项目内：文档注册表（比对的对象）。
from rag.ingestion.registry import DocumentRegistry, compute_content_hash

# 项目内：日志设施。
from rag.utils.logger import setup_logger

# 模块级日志对象：本文件所有函数共用。
log = setup_logger()


# ============================================================================
# 数据类：一次变更检测的结果（三类差集）
# ============================================================================
@dataclass
class ChangeSet:
    """一次变更检测的结果容器。

    【字段】
        added   新增的文档（磁盘有、注册表无）
        changed 变更的文档（磁盘有、注册表也有，但哈希不同）
        deleted 删除的 doc_id（注册表有、磁盘没了）
    """
    added: list[Path] = field(default_factory=list)    # 新增文件路径列表
    changed: list[Path] = field(default_factory=list)  # 变更文件路径列表
    deleted: list[str] = field(default_factory=list)   # 被删除的 doc_id 列表

    # 便捷属性：本次是否有任何变化（无变化时调用方可跳过后续处理）
    @property
    def is_empty(self) -> bool:
        """判断本次是否"零变更"。

        【返回】
            bool —— 三类差集都为空则返回 True
        """
        return not (self.added or self.changed or self.deleted)


# ============================================================================
# 函数：扫描 raw/ 下所有可处理的文件
# ============================================================================
def scan_raw_files(raw_dir: Path) -> list[Path]:
    """递归扫描 raw/ 目录下所有"受支持格式"的文件。

    【原理】
        rglob("*") 递归遍历目录下所有文件（含任意层级子目录）。
        再按扩展名过滤，只保留 settings.SUPPORTED_SUFFIXES 里声明的格式。

    【参数】
        raw_dir : Path —— raw 语料目录路径

    【返回】
        List[Path] —— 受支持格式的文件路径列表
    """
    # 从全局配置读取"允许处理的扩展名集合"（.md/.html/.ipynb/.xml 等）
    from rag.config.settings import settings

    files = []
    for f in raw_dir.rglob("*"):              # 递归遍历所有文件
        # 判断：是文件 且 扩展名（转小写）在受支持集合里
        if f.is_file() and f.suffix.lower() in settings.SUPPORTED_SUFFIXES:
            files.append(f)
    return files


# ============================================================================
# 函数：计算"清洗后正文"的哈希
# ============================================================================
def _file_content_hash(file_path: Path) -> str:
    """计算一个文件的内容哈希。

    【重要说明（对应文档 2.5.4 关键点）】
        变更检测应使用"清洗后正文"的哈希，而非原始文件的哈希。
        原因：排版变化（改个空格、重新保存）不应算内容变化。
        因此本函数会走"解析 → 清洗"得到正文，再对正文算哈希。

        但为避免与 parser/cleaner 产生强耦合（它们在第 3 轮才完整实现），
        这里采用"延迟导入 + 容错降级"策略：
        · 若能导入 parser 和 cleaner → 用"清洗后正文"哈希（最准确）；
        · 若尚未实现 → 退回"原始文件内容"哈希（保证本模块可独立运行、可测试）。

    【参数】
        file_path : Path —— 文件路径

    【返回】
        str —— 40 位 SHA1 指纹
    """
    # 尝试走"解析 + 清洗"的准确哈希（延迟导入，避免未实现时报错）
    try:
        from rag.ingestion.cleaner import clean  # 清洗：去噪去重筛选
        from rag.ingestion.parser import parse_file  # 解析：格式→文本
        text = parse_file(file_path)                          # 第 1 步：解析
        if text is not None:                                  # 解析成功（非 None）
            # 第 2 步：清洗。这里用一个新的空集合做去重（单文件场景，无跨文件去重）
            cleaned = clean(text, seen_hashes=set())
            if cleaned is not None:                           # 清洗通过
                return compute_content_hash(cleaned)          # 对清洗后正文算哈希
    except ImportError:
        # parser/cleaner 尚未实现（开发早期阶段）→ 走降级逻辑
        pass

    # 降级：直接对原始文件内容哈希（不够精确，但保证可运行）
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return compute_content_hash(text)


# ============================================================================
# 核心函数：检测变更（类型 A 入口）
# ============================================================================
def detect_changes(raw_dir: Path, registry: DocumentRegistry) -> ChangeSet:
    """检测存量文档的变化，返回三类差集。

    【完整流程（对应文档 2.5.4 第①步）】
        1. 扫描 raw/ 目录，得到磁盘上所有文件的清单；
        2. 对每个文件算 content_hash；
        3. 与注册表比对，分成三类：
           · 新增：注册表里没有这个 doc_id
           · 变更：注册表里有，但 content_hash 不同
           · 删除：注册表里有且 active，但磁盘上文件没了

    【参数】
        raw_dir  : Path             —— raw 语料目录
        registry : DocumentRegistry —— 文档注册表（比对对象）

    【返回】
        ChangeSet —— 包含 added / changed / deleted 三类差集
    """
    changes = ChangeSet()

    # ---- 第 1 步：遍历磁盘文件，判断"新增 / 变更" ----
    disk_doc_ids = set()   # 记录磁盘上出现的 doc_id（相对路径），用于后面判"删除"

    for file_path in scan_raw_files(raw_dir):
        # doc_id = 文件相对 raw/ 的路径（稳定、可回溯来源，对应文档 2.5.3）。
        # relative_to 计算相对路径；as_posix() 把反斜杠统一转成正斜杠，
        # 保证 Windows/Linux 跨平台时 doc_id 一致（否则同一文件在不同系统
        # 会得到不同 doc_id，导致误判）。
        doc_id = file_path.relative_to(raw_dir).as_posix()
        disk_doc_ids.add(doc_id)

        h = _file_content_hash(file_path)          # 算内容哈希
        record = registry.records.get(doc_id)      # 查注册表

        if record is None:
            # 注册表没有 → 新增
            changes.added.append(file_path)
        elif record.content_hash != h:
            # 哈希不同 → 变更
            changes.changed.append(file_path)
        # 哈希相同 → 没变，不处理

    # ---- 第 2 步：找"注册表有、磁盘没了"的删除项 ----
    for doc_id, record in registry.records.items():
        # 只在"active 且 磁盘已不存在"时判为删除
        if record.status == "active" and doc_id not in disk_doc_ids:
            changes.deleted.append(doc_id)

    # ---- 第 3 步：打汇总日志 ----
    log.info(
        "变更检测完成：新增 %d / 变更 %d / 删除 %d",
        len(changes.added), len(changes.changed), len(changes.deleted),
    )
    return changes
