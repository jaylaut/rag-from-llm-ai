# ============================================================================
# pipeline.py —— 单文档处理内核 + 首次全量编排
#
# 位置：src/rag/ingestion/pipeline.py
#
# 【本文件是干什么的】
#   把 P1 的"五步主线"串起来，构成"一篇文档进 → chunk + 向量出"的内核：
#     原始文件 → ① 解析 → ② 清洗 → ③ 分块 → ④ 向量化 → ⑤ 入库
#   这个内核是全量构建和增量更新【唯一复用点】（对应文档 3.2 节）。
#
# 【为什么设计成"无状态函数 + 依赖注入"】
#   - 无状态：不遍历目录、不持有全局状态，输入输出清晰，好测试；
#   - 依赖注入：embedder 和 vectorstore 通过参数传入，测试时注入 Fake，
#     生产时注入真实实现，不硬编码依赖。
# ============================================================================

# ---------- 导入区 ----------
# Path：Python 标准库，处理路径。
from pathlib import Path

# 项目内：各模块
from rag.ingestion.bm25 import BM25Index  # ⑥ BM25 稀疏索引

# Optional / List：Python 类型标注。
from rag.ingestion.cleaner import clean  # ② 清洗
from rag.ingestion.embedder import Embedder  # ④ 向量化（协议）
from rag.ingestion.parser import parse_file  # ① 解析
from rag.ingestion.registry import DocumentRegistry, compute_content_hash  # 登记
from rag.ingestion.splitter import split  # ③ 分块
from rag.ingestion.vectorstore import VectorStore  # ⑤ 入库（协议）
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 处理结果数据类
# ============================================================================
class ProcessResult:
    """单文档处理的结果。

    字段：
        doc_id       : str        文档标识（相对路径）
        content_hash : str        清洗后正文的哈希
        chunk_ids    : list[str]  切出的块 ID
        chunk_texts  : list[str]  切出的块文本（BM25 分词索引用）
        chunk_count  : int        块数量
    """

    def __init__(self, doc_id: str, content_hash: str,
                 chunk_ids: list[str], chunk_texts: list[str]):
        self.doc_id = doc_id
        self.content_hash = content_hash
        self.chunk_ids = chunk_ids
        self.chunk_texts = chunk_texts
        self.chunk_count = len(chunk_ids)


# ============================================================================
# 核心内核：处理单个文档
# ============================================================================
def process_document(
    file_path: Path,
    doc_id: str,
    embedder: Embedder,
    vectorstore: VectorStore,
    seen_hashes: set[str],
    bm25: BM25Index | None = None,
) -> ProcessResult | None:
    """处理单个文档：解析 → 清洗 → 分块 → 向量化 → 入库（+ BM25）。

    【完整流程】
        ① 解析（parser）：格式 → 文本
        ② 清洗（cleaner）：去噪/去重/筛选
        ③ 分块（splitter）：标题切 + 父子块
        ④ 向量化（embedder）：文本 → 向量
        ⑤ 入库（vectorstore）：向量写入 FAISS
        ⑥ BM25（可选）：chunk 文本加入 BM25 稀疏索引

    【参数】
        file_path    : Path         要处理的文件
        doc_id       : str          文档标识（相对路径）
        embedder     : Embedder     嵌入器（协议，可注入 Fake/Ollama）
        vectorstore  : VectorStore  向量存储（协议，可注入 FAISS 实现）
        seen_hashes  : Set[str]     全局去重集合（跨文档精确去重）
        bm25         : BM25Index|None  BM25 索引（可选；传入则同步构建稀疏索引）

    【返回】
        ProcessResult ：处理成功，含 chunk_ids/chunk_texts 等
        None          ：被清洗淘汰（去噪/去重/筛选不通过）
    """
    # ---- ① 解析 ----
    text = parse_file(file_path)
    if text is None:                     # 不支持的格式，跳过
        log.debug("跳过（格式不支持）：%s", doc_id)
        return None

    # ---- ② 清洗 ----
    text = clean(text, seen_hashes)
    if text is None:                     # 被去噪/去重/筛选淘汰
        log.debug("跳过（清洗淘汰）：%s", doc_id)
        return None

    # ---- ③ 分块 ----
    chunks = split(text, doc_id=doc_id)
    if not chunks:                       # 没切出任何块（极端情况）
        return None

    # ---- ④ 向量化 ----
    chunk_texts = [c.text for c in chunks]           # 取每块的文本
    vectors = embedder.embed(chunk_texts)            # 批量嵌入
    chunk_ids = [c.chunk_id for c in chunks]         # 取每块的 ID

    # ---- ⑤ 入库 ----
    vectorstore.add(chunk_ids, vectors)              # 向量写入 FAISS

    # ---- ⑥ BM25 稀疏索引（可选）----
    if bm25 is not None:
        # 把 chunk 文本 + chunk_id 加入 BM25（内部会 jieba 分词）
        bm25.add_documents(chunk_texts, ids=chunk_ids)

    # ---- 计算清洗后正文哈希（供注册表登记）----
    content_hash = compute_content_hash(text)

    return ProcessResult(
        doc_id=doc_id, content_hash=content_hash,
        chunk_ids=chunk_ids, chunk_texts=chunk_texts,
    )


# ============================================================================
# 首次全量构建编排（跑在空注册表上 = 增量的退化情形）
# ============================================================================
def build_index(
    raw_dir: Path,
    embedder: Embedder,
    vectorstore: VectorStore,
    registry: DocumentRegistry,
    bm25: BM25Index | None = None,
) -> dict:
    """首次全量构建：扫描 raw/ 全部文档，逐个走 process_document。

    【原理（对应文档 3.2 核心原则）】
        全量 = 增量的退化情形。注册表为空时，所有文档都是"新增"，
        所以首次全量构建不需要单独一套逻辑——就是遍历所有文件，逐个处理。

    【参数】
        raw_dir     : Path             raw 语料目录
        embedder    : Embedder         嵌入器
        vectorstore : VectorStore      向量存储
        registry    : DocumentRegistry 注册表
        bm25        : BM25Index|None   BM25 索引（可选；传入则同步构建稀疏索引）

    【返回】
        dict —— 构建报告 {处理数, 跳过数}
    """
    # 延迟导入 detector 的扫描函数（避免循环依赖）
    from rag.ingestion.detector import scan_raw_files

    seen_hashes: set[str] = set()     # 全局去重集合
    processed = 0                     # 成功入库数
    skipped = 0                       # 被淘汰/跳过数

    # 遍历 raw/ 下所有受支持文件
    for file_path in scan_raw_files(raw_dir):
        # doc_id = 相对路径（统一正斜杠）
        doc_id = file_path.relative_to(raw_dir).as_posix()

        result = process_document(file_path, doc_id, embedder, vectorstore, seen_hashes, bm25)

        if result is None:
            skipped += 1
            continue

        # 登记到注册表
        registry.upsert(
            doc_id=doc_id,
            content_hash=result.content_hash,
            chunk_ids=result.chunk_ids,
            embedding_model=getattr(embedder, "model", "unknown"),
        )
        processed += 1

        # 每处理 100 篇打一条进度日志
        if processed % 100 == 0:
            log.info("全量构建进度：已处理 %d 篇", processed)

    log.info("全量构建完成：处理 %d 篇 / 跳过 %d 篇", processed, skipped)
    return {"processed": processed, "skipped": skipped}
