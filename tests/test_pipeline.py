# ============================================================================
# test_pipeline.py —— 单文档处理内核（pipeline）的单元测试
#
# 【测什么】
#   验证 process_document 能走通"解析→清洗→分块→向量化→入库"全链路，
#   以及 build_index 全量构建。用 FakeEmbedder 注入，不依赖 Ollama。
# ============================================================================

from rag.ingestion.embedder import FakeEmbedder
from rag.ingestion.pipeline import build_index, process_document
from rag.ingestion.registry import DocumentRegistry
from rag.ingestion.vectorstore import FaissVectorStore


# ----------------------------------------------------------------------------
# 辅助：写一个"够长、纯中文"的测试文档
# ----------------------------------------------------------------------------
def _write_chinese_md(tmp_path, rel_path, title):
    """写一个满足清洗门槛（够长、纯中文）的 Markdown 文件。"""
    f = tmp_path / rel_path
    f.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"# {title}\n\n"
        f"## 第一节\n这是关于{title}的第一段详细介绍内容，用于验证解析清洗分块的完整流程。\n\n"
        f"## 第二节\n这是关于{title}的第二段内容，包含足够的中文文字，确保能够通过中文密度筛选。\n\n"
        f"## 第三节\n第三段内容继续补充，让整篇文档的长度超过最低阈值，从而不会被清洗环节过滤掉。\n"
    )
    f.write_text(content, encoding="utf-8")
    return f


# ----------------------------------------------------------------------------
# 测试 1：process_document 走通全链路
# ----------------------------------------------------------------------------
def test_process_document_full_pipeline(tmp_path):
    f = _write_chinese_md(tmp_path, "docs/a.md", "RAG检索增强生成")

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    seen = set()

    result = process_document(f, "docs/a.md", embedder, vs, seen)

    assert result is not None               # 处理成功
    assert result.chunk_count > 0           # 切出了块
    assert result.content_hash              # 有内容哈希
    assert len(result.chunk_ids) == result.chunk_count
    # 向量已入库：active 数量 == chunk 数
    assert vs.count_active() == result.chunk_count


# ----------------------------------------------------------------------------
# 测试 2：清洗淘汰（英文文档被过滤）
# ----------------------------------------------------------------------------
def test_process_document_english_filtered(tmp_path):
    f = tmp_path / "docs" / "english.md"
    f.parent.mkdir(parents=True, exist_ok=True)   # 先创建目录再写文件
    f.write_text("# English\n\n" + "this is an english document " * 50, encoding="utf-8")

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    seen = set()

    result = process_document(f, "docs/english.md", embedder, vs, seen)
    assert result is None                   # 英文文档被过滤
    assert vs.count_active() == 0           # 没有向量入库


# ----------------------------------------------------------------------------
# 测试 3：内容去重（同一内容第二次处理被去重）
# ----------------------------------------------------------------------------
def test_process_document_deduplicates(tmp_path):
    f1 = _write_chinese_md(tmp_path, "docs/a.md", "测试主题")
    f2 = _write_chinese_md(tmp_path, "docs/b.md", "测试主题")  # 内容相同，路径不同

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    seen = set()

    r1 = process_document(f1, "docs/a.md", embedder, vs, seen)
    r2 = process_document(f2, "docs/b.md", embedder, vs, seen)

    assert r1 is not None                   # 第一次成功
    assert r2 is None                       # 第二次被去重（内容相同）


# ----------------------------------------------------------------------------
# 测试 4：build_index 全量构建（注册表为空 = 全是新增）
# ----------------------------------------------------------------------------
def test_build_index_full(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_chinese_md(raw_dir, "docs/a.md", "主题A")
    _write_chinese_md(raw_dir, "docs/b.md", "主题B")

    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    reg = DocumentRegistry(path=tmp_path / "registry.json")

    report = build_index(raw_dir, embedder, vs, reg)

    assert report["processed"] == 2         # 两篇都处理成功
    # 注册表登记了 2 篇
    assert len(reg.records) == 2
    # 对账：注册表 active chunk 数 == FAISS active 向量数
    assert reg.count_active_chunks() == vs.count_active()
