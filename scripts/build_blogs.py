# ============================================================================
# build_blogs.py —— 用 blogs 语料做首次真实验证（159 篇，非重复内容）
#
# 【为什么用 blogs 而非 kexue】
#   排查发现 kexue 的 120 个 feed 文件是抓取时的重复数据（内容完全一致），
#   只能验证"去重功能"，无法验证"全链路在真实不同内容上工作"。
#   blogs 有 145 篇 HTML + 6 xml + 6 jsonl，是真实的不同文章，
#   用它做首次全链路验证更有说服力。
# ============================================================================

from pathlib import Path

from rag.ingestion.embedder import OllamaEmbedder
from rag.ingestion.vectorstore import FaissVectorStore
from rag.ingestion.registry import DocumentRegistry
from rag.ingestion.updater import run
from rag.config.settings import settings
from rag.utils.logger import setup_logger

log = setup_logger(run_name="blogs")


def main() -> None:
    """对 blogs 目录做首次全量构建（真实验证）。"""
    blogs_dir = settings.RAW_DIR / "blogs"
    log.info("开始 blogs 全量构建：%s", blogs_dir)

    embedder = OllamaEmbedder(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        batch_size=settings.EMBED_BATCH_SIZE,
    )
    vectorstore = FaissVectorStore(dim=settings.EMBEDDING_DIM)
    registry = DocumentRegistry()

    report = run(
        raw_dir=blogs_dir,
        embedder=embedder,
        vectorstore=vectorstore,
        registry=registry,
        full=True,
    )

    # 保存 FAISS 索引
    index_path = settings.INDEX_DIR / settings.FAISS_INDEX_FILE
    vectorstore.save(index_path)
    log.info("FAISS 索引已保存：%s", index_path)

    log.info("blogs 全量构建报告：%s", report)


if __name__ == "__main__":
    main()
