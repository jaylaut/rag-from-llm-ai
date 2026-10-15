# ============================================================================
# test_splitter.py —— 分块（splitter）的单元测试
#
# 【测什么】
#   验证按标题切、段落对齐、滑窗兜底、父子块关系（parent_id 关联）。
# ============================================================================

from rag.ingestion.splitter import (
    build_parent_map,
    split,
    split_by_heading,
    split_by_window,
)


# ----------------------------------------------------------------------------
# 测试 1：按标题切分
# ----------------------------------------------------------------------------
def test_split_by_heading():
    md = "# 大标题\n\n## 小节一\n内容一\n\n## 小节二\n内容二"
    sections = split_by_heading(md)
    # 应切出 3 段：# 大标题、## 小节一、## 小节二
    assert len(sections) == 3
    assert any("小节一" in s for s in sections)
    assert any("小节二" in s for s in sections)


# ----------------------------------------------------------------------------
# 测试 2：滑窗兜底（超长段落切分 + 重叠）
# ----------------------------------------------------------------------------
def test_split_by_window_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"   # 26 个字符
    chunks = split_by_window(text, chunk_size=10, overlap=2)
    # 步长 = 10 - 2 = 8。滑窗从 0 开始每次前进 8，一直滑到末尾：
    #   0→"abcdefghij", 8→"ijklmnopqr", 16→"qrstuvwxyz", 24→"yz"
    # 共 4 块（最后一块不满 10 字符是正常的兜底行为）。
    assert len(chunks) == 4
    # 相邻块有重叠：第 1 块末尾 2 字符 == 第 2 块开头 2 字符
    assert chunks[0][-2:] == chunks[1][:2]


# ----------------------------------------------------------------------------
# 测试 3：父子块关系（核心验证）
# ----------------------------------------------------------------------------
def test_split_parent_child_relation():
    md = "# 标题\n\n## 章节A\n这是第一段内容。\n\n这是第二段内容。\n\n## 章节B\n这是第三段内容。"
    chunks = split(md, doc_id="docs/test.md")

    # 每个子块都必须有 parent_id（指向某个父块）
    for c in chunks:
        assert c.parent_id.startswith("docs/test.md#p")
        # 子块自己的 id 唯一
        assert c.chunk_id.startswith("docs/test.md#c")

    # 章节A 的两个段落子块，应指向同一个父块
    # （章节A 是第 2 个标题小节，索引 1，父块 id 为 #p1）
    chapter_a_children = [c for c in chunks if "第一段" in c.text or "第二段" in c.text]
    assert len(chapter_a_children) == 2
    assert chapter_a_children[0].parent_id == chapter_a_children[1].parent_id


# ----------------------------------------------------------------------------
# 测试 4：超长段落滑窗兜底
# ----------------------------------------------------------------------------
def test_split_long_paragraph_windowed():
    # 一个超长段落（无换行），超过 chunk_size 会被滑窗切
    long_text = "这是一个很长的段落" * 100   # 800 字符
    md = "# 标题\n\n" + long_text
    chunks = split(md, doc_id="d", chunk_size=512, overlap=64)
    # 至少被切成 2 块（800 > 512）
    assert len(chunks) >= 2


# ----------------------------------------------------------------------------
# 测试 5：build_parent_map 能通过 parent_id 查到父块文本
# ----------------------------------------------------------------------------
def test_build_parent_map():
    md = "# 标题\n\n## 章节A\n内容A\n\n## 章节B\n内容B"
    parent_map = build_parent_map(md, doc_id="docs/test.md")
    # "# 标题" 本身也是一个标题小节（父块），所以共 3 个父块：
    #   #p0 → "# 标题"，#p1 → "## 章节A"，#p2 → "## 章节B"
    assert len(parent_map) == 3
    # 每个父块文本都包含对应内容
    assert any("章节A" in v for v in parent_map.values())
    assert any("章节B" in v for v in parent_map.values())
