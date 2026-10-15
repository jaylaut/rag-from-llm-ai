# ============================================================================
# test_updater.py —— 增量编排（updater）的单元测试
#
# 【测什么】
#   验证 run() 的四段流水线：变更检测→处理→写入→对账，以及幂等、
#   增量闭环（新增/变更/删除）。用 FakeEmbedder + 临时目录，不依赖 Ollama。
#   对应文档 2.5.4 和 4.2 的增量闭环验证项。
# ============================================================================

from rag.ingestion.embedder import FakeEmbedder
from rag.ingestion.registry import DocumentRegistry
from rag.ingestion.updater import JobState, reconcile, run
from rag.ingestion.vectorstore import FaissVectorStore


# ----------------------------------------------------------------------------
# 辅助：写一个够长、纯中文的文档
# ----------------------------------------------------------------------------
def _write_doc(raw_dir, rel_path, body):
    f = raw_dir / rel_path
    f.parent.mkdir(parents=True, exist_ok=True)
    # 注意：正文必须超过 cleaner 的筛选阈值（min_chars=100），否则会被过滤。
    # 所以 body 之外再拼接一段足够长的中文内容兜底。
    long_body = (f"{body}这是一段用于测试增量更新的中文内容，长度足够通过筛选，"
                 f"确保能够正常入库和检索。")
    f.write_text(
        f"# 文档\n\n## 内容\n{long_body * 3}\n",
        encoding="utf-8",
    )
    return f


# ----------------------------------------------------------------------------
# 测试 1：首次全量（空注册表 → 全是新增）
# ----------------------------------------------------------------------------
def test_run_first_time_all_added(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_doc(raw_dir, "a.md", "第一篇")
    _write_doc(raw_dir, "b.md", "第二篇")

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    report = run(raw_dir, embedder, vs, reg, full=True)

    assert report["新增"] == 2            # 两篇都是新增
    assert len(reg.records) == 2          # 注册表登记 2 篇
    # 对账：注册表 active chunk == FAISS active 向量
    assert reg.count_active_chunks() == vs.count_active()


# ----------------------------------------------------------------------------
# 测试 2：增量闭环——新增
# ----------------------------------------------------------------------------
def test_run_incremental_add(tmp_path):
    raw_dir = tmp_path / "raw"
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    _write_doc(raw_dir, "a.md", "原始文档")
    run(raw_dir, embedder, vs, reg)          # 第一次构建

    # 新增一篇
    _write_doc(raw_dir, "b.md", "新增文档")
    report = run(raw_dir, embedder, vs, reg)  # 第二次增量

    assert report["新增"] == 1              # 只识别出 1 篇新增
    assert report["变更"] == 0
    assert report["删除"] == 0


# ----------------------------------------------------------------------------
# 测试 3：增量闭环——变更
# ----------------------------------------------------------------------------
def test_run_incremental_change(tmp_path):
    raw_dir = tmp_path / "raw"
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    _write_doc(raw_dir, "a.md", "原始内容")
    run(raw_dir, embedder, vs, reg)

    # 修改内容
    _write_doc(raw_dir, "a.md", "修改后的内容")
    report = run(raw_dir, embedder, vs, reg)

    assert report["变更"] == 1              # 识别出 1 篇变更


# ----------------------------------------------------------------------------
# 测试 4：增量闭环——删除
# ----------------------------------------------------------------------------
def test_run_incremental_delete(tmp_path):
    raw_dir = tmp_path / "raw"
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    _write_doc(raw_dir, "a.md", "内容")
    _write_doc(raw_dir, "b.md", "内容二")
    run(raw_dir, embedder, vs, reg)

    # 删除 a.md
    (raw_dir / "a.md").unlink()
    report = run(raw_dir, embedder, vs, reg)

    assert report["删除"] == 1              # 识别出 1 篇删除
    # 软删除生效：a.md 的 chunk 在向量里被标记 deleted
    assert reg.records["a.md"].status == "deleted"


# ----------------------------------------------------------------------------
# 测试 5：幂等（同一批跑两次，第二次零变更）
# ----------------------------------------------------------------------------
def test_run_idempotent(tmp_path):
    raw_dir = tmp_path / "raw"
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    _write_doc(raw_dir, "a.md", "内容")
    run(raw_dir, embedder, vs, reg)

    report2 = run(raw_dir, embedder, vs, reg)   # 内容没变，再跑一次

    assert report2["新增"] == 0
    assert report2["变更"] == 0
    assert report2["删除"] == 0


# ----------------------------------------------------------------------------
# 测试 6：对账函数
# ----------------------------------------------------------------------------
def test_reconcile(tmp_path):
    from rag.ingestion.registry import DocumentRegistry

    reg = DocumentRegistry(path=tmp_path / "reg.json")
    vs = FaissVectorStore(dim=4)
    # 注册表登记 1 篇，1 个 chunk；向量库也 1 个 active
    reg.upsert("d1", "h1", ["c0"], "m1")
    vs.add(["c0"], [[1.0, 0.0, 0.0, 0.0]])
    assert reconcile(reg, vs) is True   # 一致

    # 破坏一致性：向量库多一个 active
    vs.add(["c_extra"], [[0.0, 1.0, 0.0, 0.0]])
    assert reconcile(reg, vs) is False  # 不一致


# ----------------------------------------------------------------------------
# 测试 7：JobState 断点续传
# ----------------------------------------------------------------------------
def test_job_state_persist(tmp_path):
    job = JobState(tmp_path / "job.json")
    assert job.is_done("a.md") is False
    job.mark_done("a.md")
    job.save()

    # 重新加载
    job2 = JobState(tmp_path / "job.json")
    job2.load()
    assert job2.is_done("a.md") is True


# ----------------------------------------------------------------------------
# 测试 8：allow_delete=False 时不做删除判定（防止子目录局部处理误删）
# ----------------------------------------------------------------------------
def test_run_allow_delete_false(tmp_path):
    """分目录局部处理时，allow_delete=False 应避免误删全集其它目录的文档。"""
    raw_dir = tmp_path / "raw"
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    # 先完整构建一篇文档
    _write_doc(raw_dir, "blogs/a.md", "博客文档")
    run(raw_dir, embedder, vs, reg)
    assert "blogs/a.md" in reg.records   # blogs 已入库

    # 模拟"只处理 kexue 子目录"的场景：raw_dir 指向一个只含 kexue 的子目录
    kexue_dir = tmp_path / "raw_kexue"
    _write_doc(kexue_dir, "kexue_doc.md", "kexue文档")

    # allow_delete=False：不应把 blogs/a.md 判为删除
    run(kexue_dir, embedder, vs, reg, allow_delete=False)

    # blogs 文档应仍是 active（未被误删）
    assert reg.records["blogs/a.md"].status == "active"
    # kexue 文档作为新增入库
    assert "kexue_doc.md" in reg.records
