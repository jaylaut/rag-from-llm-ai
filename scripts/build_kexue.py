# ============================================================================
# build_kexue.py —— 首次全链路验证：只跑 kexue 语料（120 篇 XML）
#
# 【为什么要单独跑 kexue（对应文档 3.5 素材实施顺序）】
#   文档建议"先做 kexue，能最快看到一条完整解析链路跑通"。
#   kexue 只有 120 篇、2.4MB，几分钟能跑完，是验证全链路的最佳起点。
#   跑通后再扩展到 github/blogs/docs（量级大得多）。
# ============================================================================

# ---------- 导入区 ----------
from pathlib import Path

from rag.ingestion.embedder import OllamaEmbedder      # 真实嵌入器（调本地 Ollama）
from rag.ingestion.vectorstore import FaissVectorStore  # FAISS 向量存储
from rag.ingestion.registry import DocumentRegistry     # 文档注册表
from rag.ingestion.updater import run                   # 增量编排（全量=退化情形）
from rag.config.settings import settings                # 全局配置
from rag.utils.logger import setup_logger                # 日志

# 模块级日志对象（run_name 用 "kexue"，日志文件名可区分）
log = setup_logger(run_name="kexue")


def main() -> None:
    """主流程：对 kexue 目录做首次全量构建。"""
    # kexue 语料目录（只处理这一类，先验证全链路）
    kexue_dir = settings.RAW_DIR / "kexue"
    log.info("开始 kexue 全量构建：%s", kexue_dir)

    # ---- 依赖注入：真实实现 ----
    embedder = OllamaEmbedder(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        batch_size=settings.EMBED_BATCH_SIZE,
    )
    vectorstore = FaissVectorStore(dim=settings.EMBEDDING_DIM)
    registry = DocumentRegistry()   # 空注册表 → 全部判为"新增"

    # ---- 跑全量（空注册表 = 增量退化情形）----
    report = run(
        raw_dir=kexue_dir,
        embedder=embedder,
        vectorstore=vectorstore,
        registry=registry,
        full=True,
    )

    # ---- 保存向量索引（run 内部保存了注册表，这里补保存 FAISS 索引）----
    index_path = settings.INDEX_DIR / settings.FAISS_INDEX_FILE
    vectorstore.save(index_path)
    log.info("FAISS 索引已保存：%s", index_path)

    # ---- 输出最终报告 ----
    log.info("kexue 全量构建报告：%s", report)


if __name__ == "__main__":
    main()
