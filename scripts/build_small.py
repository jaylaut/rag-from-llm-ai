# ============================================================================
# build_small.py —— 小规模全量构建：blogs + kexue（首次完整验证用）
#
# 【为什么单独写这个】
#   blogs 和 kexue 是"小目录"（几万块以内，几分钟~几小时），可以快速验证
#   全链路。github（13926 篇）+ docs（1571 篇）是 21 小时量级，需单独挂机。
#
#   ⚠️ 关键：不能用 run(raw_dir=blogs) + run(raw_dir=kexue) 分两次跑，
#   因为 detect_changes 是"扫描 X 目录 vs 注册表全集"，第二次跑 kexue 时
#   blogs 的文档会被误判为"删除"。必须把两个目录合并成一个 raw 集合统一跑。
# ============================================================================

import shutil
from pathlib import Path

from rag.ingestion.embedder import OllamaEmbedder
from rag.ingestion.vectorstore import FaissVectorStore
from rag.ingestion.registry import DocumentRegistry
from rag.ingestion.updater import run
from rag.config.settings import settings
from rag.utils.logger import setup_logger

log = setup_logger(run_name="small")


def main() -> None:
    """对 blogs + kexue 做统一全量构建。"""
    # 临时合并目录：把 blogs 和 kexue 都放进去，作为统一 raw 集合
    merged = Path("data/raw/_small_merged")
    if merged.exists():
        shutil.rmtree(merged)   # 清理上次残留
    merged.mkdir(parents=True)

    # 用 symlink 或 copy 把两个目录并进来（Windows 上 symlink 需权限，用 copy 稳妥）
    for sub in ["blogs", "kexue"]:
        src = settings.RAW_DIR / sub
        dst = merged / sub
        shutil.copytree(src, dst)
    log.info("已合并 blogs + kexue 到临时目录：%s", merged)

    embedder = OllamaEmbedder(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        batch_size=settings.EMBED_BATCH_SIZE,
    )
    vectorstore = FaissVectorStore(dim=settings.EMBEDDING_DIM)
    registry = DocumentRegistry()

    report = run(
        raw_dir=merged,
        embedder=embedder,
        vectorstore=vectorstore,
        registry=registry,
        full=True,
    )

    vectorstore.save(settings.INDEX_DIR / settings.FAISS_INDEX_FILE)
    log.info("blogs+kexue 全量构建报告：%s", report)
    log.info("合并后 active 向量：%d，注册表 active chunk：%d",
             vectorstore.count_active(), registry.count_active_chunks())

    # 清理临时目录
    shutil.rmtree(merged)
    log.info("已清理临时合并目录")


if __name__ == "__main__":
    main()
