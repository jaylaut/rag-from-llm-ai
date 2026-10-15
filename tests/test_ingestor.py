# ============================================================================
# test_ingestor.py —— 新文档入站（类型 B）的单元测试
#
# 【测什么】
#   验证 ingest_file / ingest_directory 的入站、判重（content_hash）、
#   不误删、统计等行为。对应文档 2.5.4.5 和 4.2 的类型 B 验证项。
# ============================================================================

from rag.ingestion.embedder import FakeEmbedder
from rag.ingestion.ingestor import ingest_directory, ingest_file
from rag.ingestion.registry import DocumentRegistry
from rag.ingestion.vectorstore import FaissVectorStore


# ----------------------------------------------------------------------------
# 辅助：写一个够长、纯中文的文档
# ----------------------------------------------------------------------------
def _write_doc(tmp_path, name, title):
    f = tmp_path / name
    # 注意：正文必须超过 cleaner 的筛选阈值（min_chars=100），否则会被过滤。
    # 所以这里写一段足够长的中文内容（重复 title 多次凑够长度）。
    body = (f"这是关于{title}的详细介绍内容，用于测试入站流程是否正常工作。"
            f"该内容包含足够多的中文字符，确保能够通过中文密度和正文长度的双重筛选，"
            f"从而顺利入库。") * 3
    f.write_text(f"# {title}\n\n## 内容\n{body}\n", encoding="utf-8")
    return f


# ----------------------------------------------------------------------------
# 测试 1：临时上传新增（可检索）
# ----------------------------------------------------------------------------
def test_ingest_file_new(tmp_path):
    f = _write_doc(tmp_path, "upload.md", "临时上传文档")

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    result = ingest_file(f, embedder, vs, reg)

    assert result == "已入库"               # 首次入站成功
    assert len(reg.records) == 1            # 注册表登记 1 篇
    assert vs.count_active() > 0            # 有向量入库


# ----------------------------------------------------------------------------
# 测试 2：内容去重（同一内容换路径，第二次返回"已存在"）
# ----------------------------------------------------------------------------
def test_ingest_file_dedup_by_content(tmp_path):
    f1 = _write_doc(tmp_path, "a.md", "相同内容")
    f2 = _write_doc(tmp_path, "b.md", "相同内容")   # 内容相同，文件名不同

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    assert ingest_file(f1, embedder, vs, reg) == "已入库"
    # 第二次：内容相同（哈希相同），即使路径不同也应判重
    assert ingest_file(f2, embedder, vs, reg) == "已存在"
    # 索引里没有重复向量（只有第一次入的向量）
    assert len(reg.records) == 1


# ----------------------------------------------------------------------------
# 测试 3：目录批量入站
# ----------------------------------------------------------------------------
def test_ingest_directory(tmp_path):
    dir_path = tmp_path / "uploads"
    dir_path.mkdir()
    _write_doc(dir_path, "x.md", "文档X")
    _write_doc(dir_path, "y.md", "文档Y")

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    stats = ingest_directory(dir_path, embedder, vs, reg)

    assert stats["已入库"] == 2             # 两篇都入站成功
    assert len(reg.records) == 2
