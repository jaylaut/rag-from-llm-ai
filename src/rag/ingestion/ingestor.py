# ============================================================================
# ingestor.py —— 新文档入站（类型 B）
#
# 位置：src/rag/ingestion/ingestor.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.5.4.5 节。增量更新的第二类来源（类型 B）：
#   用户临时上传 / raw/ 之外某目录新冒出一批文档，靠"主动入站"登记。
#
#   与 detector（类型 A）的对称关系（对应文档 3.1 说明）：
#     · detector 检测"存量变化"（扫描 raw/ 全集找差集）
#     · ingestor 摄入"新文档"（主动来报到，按内容判重）
#   两者后段复用同一套"解析→清洗→分块→向量化→写入"内核。
#
# 【三个关键设计点（对应文档 2.5.4.5）】
#   1. 判重靠 content_hash 而非 doc_id（临时文件路径每次变，用路径判重会重复入库）
#   2. 不做删除判定（只增不减，避免误删刚上传的内容）
#   3. 复用同一套内核（与类型 A 完全相同的 process_document）
# ============================================================================

# ---------- 导入区 ----------
# Path：Python 标准库，处理路径。
from pathlib import Path

# Optional：Python 类型标注（Optional[X] 表示"X 或 None"）。
# 项目内：嵌入器 / 向量存储协议。
from rag.ingestion.embedder import Embedder

# 项目内：单文档内核（与类型 A 复用同一套）。
from rag.ingestion.pipeline import process_document

# 项目内：注册表 + 内容哈希（判重用）。
from rag.ingestion.registry import DocumentRegistry, compute_content_hash
from rag.ingestion.vectorstore import VectorStore

# 项目内：日志。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 单个文件入站
# ============================================================================
def ingest_file(
    path: Path,
    embedder: Embedder,
    vectorstore: VectorStore,
    registry: DocumentRegistry,
    seen_hashes: set | None = None,
) -> str:
    """摄入单个新文件（类型 B 核心入口）。

    【流程（对应文档 2.5.4.5）】
        ① 算 content_hash
        ② 查 hash_index 是否已入库过
           ├─ 已存在 → 去重，返回"已存在"
           └─ 不存在 → 走完整链路：解析→清洗→分块→向量化→入库→登记

    【参数】
        path        : Path             要入站的文件路径
        embedder    : Embedder         嵌入器
        vectorstore : VectorStore      向量存储
        registry    : DocumentRegistry 注册表
        seen_hashes : set              全局去重集合（批量入站时共享）

    【返回】
        str —— 状态："已存在" / "已入库" / "已跳过"
    """
    if seen_hashes is None:
        seen_hashes = set()

    # ---- 第 1 步：读内容、算哈希（判重的基础）----
    # 注意：这里对"原始内容"算哈希做快速判重。更严谨的做法是"清洗后正文"
    # 哈希，但入站判重只需识别"同一份文件"，原始哈希已足够快且够用。
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    h = compute_content_hash(raw_text)

    # ---- 第 2 步：查反向索引，判重 ----
    existing_doc_id = registry.find_by_hash(h)
    if existing_doc_id is not None:
        log.info("内容已存在（doc_id=%s），跳过去重", existing_doc_id)
        return "已存在"

    # ---- 第 3 步：走完整链路（复用 process_document）----
    # doc_id 用文件路径本身（临时上传的文件路径可能每次都变，但没关系，
    # 因为判重靠的是 content_hash 而非 doc_id）。
    doc_id = path.name   # 用文件名作为 doc_id（上传场景无稳定相对路径）

    result = process_document(path, doc_id, embedder, vectorstore, seen_hashes)

    if result is None:
        log.info("入站跳过（清洗淘汰）：%s", path)
        return "已跳过"

    # ---- 第 4 步：登记到注册表 ----
    registry.upsert(
        doc_id=doc_id,
        content_hash=result.content_hash,
        chunk_ids=result.chunk_ids,
        embedding_model=getattr(embedder, "model", "unknown"),
    )

    log.info("文档已入站：%s（%d 块）", path, result.chunk_count)
    return "已入库"


# ============================================================================
# 目录批量入站
# ============================================================================
def ingest_directory(
    directory: Path,
    embedder: Embedder,
    vectorstore: VectorStore,
    registry: DocumentRegistry,
) -> dict:
    """批量摄入某目录下的一批新文档。

    【原理（对应文档 2.5.4.5）】
        遍历目录下所有文件，逐个调用 ingest_file，最后统计结果。
        共享同一个 seen_hashes，实现跨文件去重。

    【参数】
        directory   : Path             目录路径
        embedder    : Embedder         嵌入器
        vectorstore : VectorStore      向量存储
        registry    : DocumentRegistry 注册表

    【返回】
        dict —— 统计结果，如 {"已入库": 3, "已存在": 2, "已跳过": 1}
    """
    stats = {"已入库": 0, "已存在": 0, "已跳过": 0}   # 计数器
    seen_hashes: set = set()   # 共享去重集合（跨文件精确去重）

    # directory.iterdir() 遍历目录下的一级子项
    for f in directory.iterdir():
        if f.is_file():                                   # 只处理文件
            result = ingest_file(f, embedder, vectorstore, registry, seen_hashes)
            stats[result] = stats.get(result, 0) + 1      # 累加对应状态计数

    log.info("目录批量入站完成：%s", stats)
    return stats
