# ============================================================================
# updater.py —— 增量编排：检测 → 处理 → 写入 → 对账
#
# 位置：src/rag/ingestion/updater.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.5.4 节的"四段增量流水线"，是全量和增量【共用的
#   唯一入口】。核心原则（对应文档 3.2）：全量 = 增量的退化情形——注册表为空
#   时，所有文档都是"新增"，所以首次全量构建跑的就是增量逻辑。
#
# 【四段流水线】
#   ① 变更检测：扫描 raw/ → hash 比对注册表 → 分三类差集
#   ② 变更处理：只重处理差集（新增/变更走内核，删除走软删）
#   ③ 索引写入：向量追加 / 软删除
#   ④ 对账收尾：一致性校验 + 增量报告
#
# 【本文件实现的企业级能力（对应文档 2.5.7）】
#   1. 幂等键（doc_id + version）：重复处理不产生重复数据
#   2. 断点续传（job.json）：中断后重跑，已完成的跳过
#   3. 失败重试 + dead-letter：单篇失败不阻断整批，重试后仍失败记入死信
#   4. 对账：active chunk 数 == FAISS 向量数
# ============================================================================

# ---------- 导入区 ----------
# argparse：Python 标准库，解析命令行参数。
import argparse

# json：Python 标准库，断点续传的 job.json 读写。
import json

# time：Python 标准库，计时（算耗时）。
import time

# Path：Python 标准库，路径。
from pathlib import Path

# Optional：Python 类型标注（Optional[X] 表示"X 或 None"）。
# 项目内：各模块
from rag.ingestion.detector import detect_changes  # ① 变更检测
from rag.ingestion.embedder import Embedder, OllamaEmbedder  # 嵌入器
from rag.ingestion.pipeline import process_document  # 单文档内核
from rag.ingestion.registry import DocumentRegistry  # 注册表
from rag.ingestion.vectorstore import FaissVectorStore, VectorStore  # 向量存储
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 幂等键 + 断点续传：job.json 管理
# ============================================================================
class JobState:
    """任务进度状态（断点续传的载体）。

    【原理（对应文档 2.5.7 断点续传）】
        全量构建要跑 21 小时，中途可能断电/中断。job.json 记录"本批处理的
        文档清单 + 进度"，中断后重跑，已完成的跳过，从断点继续。
    """

    def __init__(self, job_path: Path):
        """构造函数。

        参数：
            job_path : Path —— job.json 的路径
        """
        self.job_path = job_path
        self.completed: set = set()   # 已完成的 doc_id 集合

    def load(self) -> None:
        """从磁盘加载进度（若存在）。"""
        if self.job_path.exists():
            data = json.loads(self.job_path.read_text(encoding="utf-8"))
            self.completed = set(data.get("completed", []))

    def save(self) -> None:
        """把进度保存到磁盘。"""
        self.job_path.parent.mkdir(parents=True, exist_ok=True)
        self.job_path.write_text(
            json.dumps({"completed": sorted(self.completed)}, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_done(self, doc_id: str) -> bool:
        """判断某文档是否已完成（用于跳过）。"""
        return doc_id in self.completed

    def mark_done(self, doc_id: str) -> None:
        """标记某文档已完成。"""
        self.completed.add(doc_id)


# ============================================================================
# 失败重试 + dead-letter
# ============================================================================
def _process_with_retry(
    file_path: Path,
    doc_id: str,
    embedder: Embedder,
    vectorstore: VectorStore,
    seen_hashes: set,
    max_retry: int,
    backoff_factor: float,
):
    """带重试地处理单个文档。

    【原理（对应文档 2.5.7 失败重试）】
        瞬时故障（Ollama 偶发超时、网络抖动）重试几次大概率能恢复。
        指数退避：第 n 次重试前等待 base * factor^(n-1) 秒，避免打爆服务。

    【参数】
        （略，见各模块说明）
        max_retry      : int   最大重试次数
        backoff_factor : float 退避系数

    【返回】
        ProcessResult / None（成功/淘汰），或抛异常（重试耗尽）
    """

    last_exception: Exception | None = None   # 记录最后一次异常，重试耗尽后抛出
    for attempt in range(max_retry):
        try:
            return process_document(file_path, doc_id, embedder, vectorstore, seen_hashes)
        except Exception as e:   # 捕获所有异常（网络、解析、嵌入都可能失败）
            last_exception = e
            wait = backoff_factor ** attempt   # 指数退避：1, 2, 4, ...
            log.warning(
                "处理失败（第 %d/%d 次）：%s，%s 秒后重试",
                attempt + 1, max_retry, doc_id, wait,
            )
            time.sleep(wait)

    # 重试耗尽，抛出最后的异常（由上层捕获，记入 dead-letter）。
    # 理论上走到这里 last_exception 一定不为 None（循环里至少捕获过一次），
    # 用 assert 向 mypy 和读者保证这一点。
    assert last_exception is not None
    raise last_exception


# ============================================================================
# 对账：一致性校验
# ============================================================================
def reconcile(registry: DocumentRegistry, vectorstore: VectorStore) -> bool:
    """对账：校验注册表与向量索引的一致性。

    【原理（对应文档 2.5.4 第④步）】
        注册表 active 文档的 chunk 总数，必须等于 FAISS 里的 active 向量数。
        不一致说明有 bug 或数据漂移。

    【参数】
        registry    : DocumentRegistry 注册表
        vectorstore : VectorStore      向量存储

    【返回】
        bool —— True 一致，False 不一致
    """
    registry_count = registry.count_active_chunks()     # 注册表统计
    vector_count = vectorstore.count_active()           # FAISS 统计

    ok = (registry_count == vector_count)
    if ok:
        log.info("对账通过：注册表 %d 块 == FAISS %d 向量", registry_count, vector_count)
    else:
        log.error("对账失败：注册表 %d 块 != FAISS %d 向量", registry_count, vector_count)
    return ok


# ============================================================================
# 主流程：增量更新（也是全量入口）
# ============================================================================
def run(
    raw_dir: Path | None = None,
    embedder: Embedder | None = None,
    vectorstore: VectorStore | None = None,
    registry: DocumentRegistry | None = None,
    full: bool = False,
    allow_delete: bool = True,
) -> dict:
    """增量更新主流程（全量构建也走这里）。

    【完整流程（对应文档 2.5.4 四段流水线）】
        ① 变更检测 → ② 变更处理 → ③ 索引写入 → ④ 对账收尾

    ⚠️【重要约定：raw_dir 必须是"文档全集"】
        detect_changes 的删除判定假设"raw_dir 是真相来源全集"——注册表里有、
        但 raw_dir 里没有的文档，会被判定为"删除"（软删除）。
        因此：
        · 正常全量/增量：raw_dir 传 settings.RAW_DIR（整个 data/raw/）；
        · 绝不能分目录多次调用（如先 run(blogs) 再 run(kexue)）——第二次会把
          blogs 的文档误判为"删除"！
        · 若只想处理某个子目录的"新增"，应传 allow_delete=False 关闭删除判定。

    【参数】
        raw_dir      : Path               raw 语料目录（None 用配置默认；必须是全集！）
        embedder     : Embedder           嵌入器（None 用 OllamaEmbedder）
        vectorstore  : VectorStore        向量存储（None 用 FaissVectorStore）
        registry     : DocumentRegistry   注册表（None 用默认路径）
        full         : bool               是否全量（实际不影响逻辑，仅日志标识）
        allow_delete : bool               是否执行删除判定（默认 True；子目录局部处理时传 False）

    【返回】
        dict —— 增量报告（新增/变更/删除/重嵌块数/耗时/失败数）
    """
    # ---- 依赖注入：参数未传时，用默认实现（生产配置）----
    from rag.config.settings import settings

    if raw_dir is None:
        raw_dir = settings.RAW_DIR
    if embedder is None:
        embedder = OllamaEmbedder(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            batch_size=settings.EMBED_BATCH_SIZE,
        )
    if vectorstore is None:
        vectorstore = FaissVectorStore(dim=settings.EMBEDDING_DIM)
    if registry is None:
        registry = DocumentRegistry()

    start_time = time.time()   # 记录开始时间

    # ---- 断点续传：加载进度 ----
    # 关键：job 文件路径跟随 registry 的路径（同目录），而非硬编码 settings。
    # 这样测试传入临时 registry 时，job 也写到临时目录，不会污染真实数据。
    # registry.path 是 registry.json 的位置，job.json 放它旁边。
    job_path = registry.path.parent / settings.JOB_FILE
    job = JobState(job_path)
    job.load()

    # ---- ① 变更检测 ----
    changes = detect_changes(raw_dir, registry)

    # 无变化：直接返回（假变更免疫）
    if changes.is_empty:
        log.info("无变更，跳过处理")
        return {"新增": 0, "变更": 0, "删除": 0, "重嵌块数": 0, "耗时_秒": 0, "失败": 0}

    seen_hashes: set[str] = set()   # 全局去重集合（精确去重的哈希池）
    dead_letter = []             # 死信列表（重试仍失败的文档）
    reembedded_chunks = 0        # 重嵌块数统计

    # ---- ②③ 变更处理 + 写入 ----
    # 处理"新增 + 变更"（都要重新走内核）
    for file_path in changes.added + changes.changed:
        doc_id = file_path.relative_to(raw_dir).as_posix()

        # 断点续传：已完成则跳过
        if job.is_done(doc_id):
            continue

        try:
            result = _process_with_retry(
                file_path, doc_id, embedder, vectorstore, seen_hashes,
                settings.MAX_RETRY, settings.RETRY_BACKOFF_FACTOR,
            )
            if result is None:
                continue   # 被清洗淘汰，不是失败
            # 登记到注册表
            registry.upsert(
                doc_id=doc_id,
                content_hash=result.content_hash,
                chunk_ids=result.chunk_ids,
                embedding_model=settings.EMBEDDING_MODEL,
            )
            reembedded_chunks += result.chunk_count
            job.mark_done(doc_id)   # 标记完成
            job.save()              # 及时落盘进度
        except Exception as e:
            # 重试耗尽仍失败 → 记入死信
            log.error("文档处理失败，记入死信：%s（%s）", doc_id, e)
            dead_letter.append({"doc_id": doc_id, "error": str(e)})

    # 处理"删除"（软删除：改注册表状态 + 向量打标）
    # 防护：allow_delete=False 时跳过删除判定（子目录局部处理时，避免误删
    # 全集里其它目录的文档）。
    if allow_delete:
        for doc_id in changes.deleted:
            registry.mark_deleted(doc_id)                    # 注册表软删除
            vectorstore.mark_deleted(registry.get_chunk_ids(doc_id))  # 向量软删除
            log.info("软删除文档：%s", doc_id)
    else:
        # 关闭删除判定：但仍要提示用户有 N 篇"看起来该删"的文档被保留
        if changes.deleted:
            log.warning("allow_delete=False，跳过 %d 篇删除（保留在索引中）", len(changes.deleted))

    # ---- ④ 对账 + 收尾 ----
    registry.save()   # 保存注册表
    if settings.ENABLE_RECONCILE:
        reconcile(registry, vectorstore)   # 一致性校验

    # 保存死信（若有）
    if dead_letter:
        # 死信文件也跟随 registry 目录（测试隔离），而非硬编码 settings.INDEX_DIR
        dead_letter_path = registry.path.parent / settings.DEAD_LETTER_FILE
        dead_letter_path.write_text(
            json.dumps(dead_letter, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- 生成增量报告 ----
    elapsed = time.time() - start_time
    report = {
        "新增": len(changes.added),
        "变更": len(changes.changed),
        "删除": len(changes.deleted),
        "重嵌块数": reembedded_chunks,
        "失败": len(dead_letter),
        "耗时_秒": round(elapsed, 2),
    }
    log.info("增量报告：%s", report)
    return report


# ============================================================================
# 命令行入口：python -m rag.ingestion.updater [--full]
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 索引增量更新（全量=增量退化情形）")
    parser.add_argument("--full", action="store_true", help="全量构建（注册表为空时自动等价于全量）")
    args = parser.parse_args()

    run(full=args.full)
