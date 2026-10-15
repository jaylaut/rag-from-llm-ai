# ============================================================================
# test_detector.py —— 存量变更检测（类型 A）的单元测试
#
# 【测什么】
#   验证 detect_changes 能正确识别三类变化：新增 / 变更 / 删除，
#   以及"无变化时差集为空"（假变更免疫，对应文档 4.2 的验证项）。
# ============================================================================

from rag.ingestion.detector import detect_changes, scan_raw_files
from rag.ingestion.registry import DocumentRegistry, compute_content_hash


# ----------------------------------------------------------------------------
# 辅助函数：在临时目录里造一个"raw/ 语料"环境
# ----------------------------------------------------------------------------
def _write(raw_dir, rel_path: str, content: str):
    """在 raw_dir 下按相对路径写入一个文件（自动创建父目录）。

    参数：
        raw_dir  : 语料根目录
        rel_path : 相对路径（如 "docs/a.md"）
        content  : 文件内容
    """
    target = raw_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)   # 确保父目录存在
    target.write_text(content, encoding="utf-8")


def _doc_id(raw_dir, rel_path: str) -> str:
    """把相对路径转成 detector 使用的 doc_id 格式（正斜杠）。"""
    return rel_path.replace("\\", "/")


# ----------------------------------------------------------------------------
# 测试 1：空注册表 + 磁盘有文件 → 全部判为"新增"
# ----------------------------------------------------------------------------
def test_all_added_when_registry_empty(tmp_path):
    raw_dir = tmp_path / "raw"
    _write(raw_dir, "docs/a.md", "第一篇文档内容")
    _write(raw_dir, "blogs/b.html", "<html>第二篇</html>")

    reg = DocumentRegistry(path=tmp_path / "registry.json")   # 空注册表

    changes = detect_changes(raw_dir, reg)

    assert len(changes.added) == 2        # 两个文件都判为新增
    assert changes.changed == []          # 无变更
    assert changes.deleted == []          # 无删除
    assert changes.is_empty is False


# ----------------------------------------------------------------------------
# 测试 2：内容变化 → 判为"变更"
# ----------------------------------------------------------------------------
def test_changed_when_content_differs(tmp_path):
    raw_dir = tmp_path / "raw"
    _write(raw_dir, "docs/a.md", "原始内容")

    reg = DocumentRegistry(path=tmp_path / "registry.json")
    # 先登记一个"旧版本"（哈希对应旧内容）
    reg.upsert(
        doc_id=_doc_id(raw_dir, "docs/a.md"),
        content_hash=compute_content_hash("原始内容"),
        chunk_ids=["c0"],
        embedding_model="m1",
    )

    # 修改磁盘文件内容
    _write(raw_dir, "docs/a.md", "修改后的内容")

    changes = detect_changes(raw_dir, reg)
    assert changes.changed is not None and len(changes.changed) == 1   # 判为变更
    assert changes.added == []
    assert changes.deleted == []


# ----------------------------------------------------------------------------
# 测试 3：磁盘文件消失 → 判为"删除"
# ----------------------------------------------------------------------------
def test_deleted_when_file_removed(tmp_path):
    raw_dir = tmp_path / "raw"
    _write(raw_dir, "docs/a.md", "内容")

    reg = DocumentRegistry(path=tmp_path / "registry.json")
    reg.upsert(
        doc_id=_doc_id(raw_dir, "docs/a.md"),
        content_hash=compute_content_hash("内容"),
        chunk_ids=["c0"],
        embedding_model="m1",
    )

    # 删除磁盘上的文件
    (raw_dir / "docs" / "a.md").unlink()

    changes = detect_changes(raw_dir, reg)
    assert changes.deleted == [_doc_id(raw_dir, "docs/a.md")]   # 判为删除
    assert changes.added == []
    assert changes.changed == []


# ----------------------------------------------------------------------------
# 测试 4：无变化 → 差集为空（假变更免疫）
# ----------------------------------------------------------------------------
def test_no_change_when_identical(tmp_path):
    raw_dir = tmp_path / "raw"
    content = "稳定的内容"
    _write(raw_dir, "docs/a.md", content)

    reg = DocumentRegistry(path=tmp_path / "registry.json")
    reg.upsert(
        doc_id=_doc_id(raw_dir, "docs/a.md"),
        content_hash=compute_content_hash(content),
        chunk_ids=["c0"],
        embedding_model="m1",
    )

    changes = detect_changes(raw_dir, reg)
    assert changes.is_empty is True        # 零变更


# ----------------------------------------------------------------------------
# 测试 5：软删除的文档，即使磁盘文件没了，也不重复判"删除"
# ----------------------------------------------------------------------------
def test_deleted_doc_not_redetected(tmp_path):
    """已软删除的文档（status=deleted）不应再被 detect_changes 判为删除。"""
    raw_dir = tmp_path / "raw"
    _write(raw_dir, "docs/a.md", "内容")

    reg = DocumentRegistry(path=tmp_path / "registry.json")
    reg.upsert(_doc_id(raw_dir, "docs/a.md"), compute_content_hash("内容"), ["c0"], "m1")
    reg.mark_deleted(_doc_id(raw_dir, "docs/a.md"))   # 先软删除

    (raw_dir / "docs" / "a.md").unlink()              # 再删文件

    changes = detect_changes(raw_dir, reg)
    # 因为已经是 deleted 状态，不会再次进入 deleted 差集
    assert _doc_id(raw_dir, "docs/a.md") not in changes.deleted


# ----------------------------------------------------------------------------
# 测试 6：scan_raw_files —— 只扫描受支持格式，忽略其它
# ----------------------------------------------------------------------------
def test_scan_raw_files_filters_suffix(tmp_path):
    raw_dir = tmp_path / "raw"
    _write(raw_dir, "a.md", "x")
    _write(raw_dir, "b.html", "x")
    _write(raw_dir, "c.ipynb", "{}")
    _write(raw_dir, "d.tex", "LaTeX")       # .tex 不在受支持集合里
    _write(raw_dir, "e.pdf", "PDF")         # .pdf 也不受支持

    files = scan_raw_files(raw_dir)
    suffixes = {f.suffix for f in files}
    assert ".md" in suffixes
    assert ".html" in suffixes
    assert ".ipynb" in suffixes
    assert ".tex" not in suffixes           # 不支持格式被过滤
    assert ".pdf" not in suffixes
