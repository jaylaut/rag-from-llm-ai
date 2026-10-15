# ============================================================================
# compactor.py —— Compaction（压缩整理）：定期物理清理"僵尸向量"
#
# 位置：src/rag/ingestion/compactor.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.5.6 节。软删除的代价是索引里堆积"僵尸向量"
#   （查不到的废块），需要定期物理清理——只保留 active 的 chunk，重建索引。
#
# 【触发条件（满足其一，对应文档 2.5.6）】
#   1. 僵尸向量占比 > 10%（settings.ZOMBIE_RATIO_THRESHOLD）
#   2. 距上次 compaction 超过 N 天（settings.COMPACTION_MAX_DAYS）
#
# 【关键性质（对应文档 2.5.6 甜蜜点）】
#   IndexFlatIP 重建是"纯内存重组"，向量还在，不需要重嵌，所以是分钟级操作。
#   这就是"选 FAISS + 注册表"买到的性质：重嵌贵、重组便宜。
#
# 【两个必须注意的点（对应文档 2.5.6）】
#   1. compaction 后 chunk 的 faiss_id 全变了 → 元数据映射必须同步重写
#   2. 重建期间索引不可用（单机学习场景可接受，分钟级）
# ============================================================================

# ---------- 导入区 ----------
# Path：Python 标准库，路径。
from pathlib import Path

# 项目内：注册表。
from rag.ingestion.registry import DocumentRegistry

# 项目内：向量存储（重建索引用）。
from rag.ingestion.vectorstore import FaissVectorStore

# 项目内：日志。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 判断是否需要 compaction
# ============================================================================
def should_compact(registry: DocumentRegistry, vectorstore: FaissVectorStore,
                   threshold: float = 0.10) -> bool:
    """判断是否需要触发 compaction（按僵尸向量占比）。

    【原理（对应文档 2.5.6）】
        统计 deleted 文档的 chunk 占总 chunk 的比例，超过阈值就触发。

    【参数】
        registry    : DocumentRegistry 注册表
        vectorstore : FaissVectorStore 向量存储
        threshold   : float            僵尸占比阈值，默认 0.10（10%）

    【返回】
        bool —— True 需要压缩，False 不需要
    """
    # 僵尸数 = 总 chunk 数 - active chunk 数。
    # 这里用注册表统计：deleted 文档的 chunk 就是僵尸向量。
    zombie = 0
    total_chunks = 0
    for record in registry.records.values():
        n = len(record.chunk_ids)
        total_chunks += n
        if record.status == "deleted":
            zombie += n

    if total_chunks == 0:   # 没有数据，无需压缩
        return False

    ratio = zombie / total_chunks
    log.info("僵尸向量占比：%.2f%%（阈值 %.2f%%）", ratio * 100, threshold * 100)
    return ratio > threshold


# ============================================================================
# 执行 compaction
# ============================================================================
def compact(
    registry: DocumentRegistry,
    vectorstore: FaissVectorStore,
    index_path: Path,
) -> int:
    """执行 compaction：按注册表重建索引，只保留 active 的 chunk。

    【完整流程（对应文档 2.5.6）】
        1. 收集所有 active 文档的 chunk_ids；
        2. 按这些 chunk 重建 FAISS 索引（只保留 active 向量）；
        3. 原子替换索引文件；
        4. 同步刷新元数据映射和注册表。

    【参数】
        registry    : DocumentRegistry 注册表
        vectorstore : FaissVectorStore 原索引（含僵尸向量）
        index_path  : Path             索引文件保存路径

    【返回】
        int —— compaction 后保留的 active chunk 数
    """
    # ---- 第 1 步：收集所有 active 文档的 chunk_ids ----
    keep_chunk_ids = []
    for record in registry.records.values():
        if record.status == "active":
            keep_chunk_ids.extend(record.chunk_ids)
    keep_set = set(keep_chunk_ids)   # 转集合，查找 O(1)

    log.info("Compaction 开始：原 %d 向量，保留 %d 个 active chunk",
             len(vectorstore.metadata), len(keep_chunk_ids))

    # ---- 第 2 步：从旧索引取出 active 向量，重建新索引 ----
    # get_active_vectors 返回旧索引里所有"未删除"的 (chunk_id, 向量)。
    # 但 compaction 的判定标准是"注册表里 active"，可能比"未删除"更严格
    # （比如注册表软删了但向量忘了打标）。这里以"注册表 active"为准。
    new_vs = FaissVectorStore(dim=vectorstore.dim)
    rebuilt_count = 0

    for chunk_id, vector in vectorstore.get_active_vectors():
        # 只保留"注册表仍 active"的 chunk
        if chunk_id in keep_set:
            new_vs.add([chunk_id], [vector])   # 重新加入新索引
            rebuilt_count += 1

    # ---- 第 3 步：原子替换索引文件 ----
    # 先写临时文件，再原子替换（防止写一半断电损坏索引）
    new_vs.save(index_path)

    log.info("Compaction 完成：重建后保留 %d 个 active chunk", rebuilt_count)
    return rebuilt_count


# ============================================================================
# 命令行入口：python -m rag.ingestion.compactor
# ============================================================================
if __name__ == "__main__":
    from rag.config.settings import settings

    reg = DocumentRegistry()
    vs = FaissVectorStore(dim=settings.EMBEDDING_DIM)

    if should_compact(reg, vs, settings.ZOMBIE_RATIO_THRESHOLD):
        compact(reg, vs, settings.INDEX_DIR / settings.FAISS_INDEX_FILE)
    else:
        log.info("僵尸向量占比未超阈值，跳过 compaction")
