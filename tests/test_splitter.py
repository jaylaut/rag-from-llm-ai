# ============================================================================
# test_splitter.py —— 分块器（splitter.py）的单元测试
#
# 【测什么】
#   1. 结构扫描：标题识别（含代码围栏内的 # 不当标题）、代码区间、未闭合围栏
#   2. 父块构建：## 为默认粒度、超限降级、无标题时硬切、**区间不重复**（回归）
#   3. 子块切分：段落对齐、短段合并、超长段滑窗（512/64）、标题不与正文拆散
#   4. 对外契约：split_document 的字段完整性、可回溯性、父子映射、is_code
#   5. 边界与异常：空文本、参数非法、\r\n 规范化、metadata 合并
#
# 【怎么跑】
#   .venv/Scripts/python.exe -m pytest tests/test_splitter.py -v
#
# 【设计原则】
#   优先断言"不变量"（块长 <= 上限、text 必须等于原文切片、chunk_id 唯一），
#   而不是死锁具体块数——块数会随策略微调而变，不变量不会。
#   只有回归 bug 的用例才断言精确的块数/字符，并在 docstring 里写明原因。
# ============================================================================
import pytest

from rag.config.settings import settings
from rag.ingestion.splitter import (
    _build_parents,
    _has_prose,
    _heading_path_at,
    _is_code_heavy,
    _paragraph_ranges,
    _scan_structure,
    _slice_by_level,
    _split_long_text,
    _split_parent_to_children,
    split_document,
)

# 一份结构完整的小文档：1 个 #、2 个 ##、1 个 ###、1 段代码
DOC = """# 安装指南

本文介绍安装步骤。

## 环境准备

需要 Python 3.12 和 Node 20。

```python
# 这是代码里的注释，不是标题
print("hello")
```

### Windows

在 Windows 上使用 PowerShell 安装。

## 快速开始

执行 `pip install -e .` 即可。
"""

# 无标点的超长段落（用来强制走滑窗兜底）
NO_PUNCT = "这是一段用来凑长度的中文没有句号" * 200
# 带句号的超长段落（用来验证"优先切在句末"）
WITH_PUNCT = "这是一个完整的句子。且带句号！还有问号？" * 120


# ============================================================================
# 一、结构扫描 _scan_structure
# ============================================================================
def test_scan_headings_and_paths():
    """标题的级别、文字、层级路径都要算对（# → ## → ### 三层链）。"""
    headings, _ = _scan_structure(DOC)
    got = [(h["level"], h["title"], h["path"]) for h in headings]

    assert got[0] == (1, "安装指南", ["安装指南"])
    assert got[1] == (2, "环境准备", ["安装指南", "环境准备"])
    assert got[2] == (3, "Windows", ["安装指南", "环境准备", "Windows"])
    # 第二个 ## 出现后，栈里的 ### 必须被弹掉，不能残留
    assert got[3] == (2, "快速开始", ["安装指南", "快速开始"])


def test_scan_offsets_are_sliceable():
    """start/end 必须能精确切出标题行——偏移量算错，回调原文就全废。"""
    headings, _ = _scan_structure(DOC)
    for h in headings:
        line = DOC[h["start"]:h["end"]]
        assert line.startswith("#")
        assert h["title"] in line


def test_scan_ignores_hash_inside_code_fence():
    """代码块里的 `# 注释` 不能被当成标题（这是代码介入 C 语言语料时的头号误判）。"""
    headings, code_ranges = _scan_structure(DOC)
    titles = [h["title"] for h in headings]

    assert "这是代码里的注释，不是标题" not in titles
    assert len(code_ranges) == 1


def test_scan_code_range_values():
    """代码区间必须是 (start, end) 元组，且能切出以 ``` 开头、``` 结尾的原文。

    【回归】旧代码写成 code_ranges.append(a, b)，list.append 只收一个参数，
    跑第一份带代码块的文档就抛 TypeError，整条索引链在第一步就断。
    """
    _, code_ranges = _scan_structure(DOC)
    start, end = code_ranges[0]
    snippet = DOC[start:end]

    assert snippet.startswith("```python")
    assert snippet.rstrip().endswith("```")


def test_scan_unclosed_fence_extends_to_eof():
    """未闭合围栏按"一直到文末都是代码"处理，并告警，不能静默丢掉区间。"""
    text = "## 标题\n\n```python\nprint(1)\n# 这行不该是标题\n"
    headings, code_ranges = _scan_structure(text)

    assert len(headings) == 1
    assert code_ranges == [(text.index("```"), len(text))]


def test_scan_closing_fence_with_info_string_is_not_closing():
    """带语言标记的围栏行（```python）不能当关闭标记。

    【为什么】CommonMark 规定关闭围栏行只能是纯围栏。若把代码块内容里出现的
    ```python 误判为关闭，代码区间会中途截断，后面所有 # 注释都会被当成标题。
    """
    text = "```\n```python\nx = 1\n```\n"
    _, code_ranges = _scan_structure(text)

    # 整段（从第一个 ``` 到最后一个 ```）都应是代码
    assert len(code_ranges) == 1
    assert code_ranges[0] == (0, len(text))


def test_scan_atx_heading_requires_space():
    """# 后必须有空格才是标题；`#include <stdio.h>` 不是。"""
    headings, _ = _scan_structure("#include <stdio.h>\n# 真标题\n")
    assert [h["title"] for h in headings] == ["真标题"]


def test_scan_closing_hash_decorator_stripped():
    """尾部 `###` 装饰要剥掉：`## 标题 ###` 的 title 应是「标题」。"""
    headings, _ = _scan_structure("## 标题 ###\n")
    assert headings[0]["title"] == "标题"


# ============================================================================
# 二、父块构建 _slice_by_level / _build_parents
# ============================================================================
def test_slice_by_level_returns_preamble_block():
    """第一个标题之前的内容要单独成块（前言块），不能丢。"""
    headings, _ = _scan_structure("前言文字\n\n# 标题\n\n正文\n")
    ranges = _slice_by_level(headings, 2, 0, 20)

    assert ranges[0][0] == 0                      # 从 0 开始 = 前言
    assert ranges[0][1] == headings[0]["start"]   # 到第一个标题为止


def test_slice_by_level_includes_shallower_headings():
    """切 ## 时 # 也算边界，避免一个 ## 块跨越到另一个 # 章节里。"""
    text = "# 第一章\n\n## A\n\n甲\n\n# 第二章\n\n## B\n\n乙\n"
    headings, _ = _scan_structure(text)
    ranges = _slice_by_level(headings, 2, 0, len(text))

    starts = [text[s:s + 8] for s, _ in ranges]
    assert any(s.startswith("# 第二章") for s in starts)


def test_build_parents_default_level_is_h2():
    """默认按 ## 切父块：2 个 ## 章节 + 1 个 "# 章标题本体" 的前言块 = 3 块。

    【为什么是 3 而不是 2】# 也是边界（级别 <= 2 都算），所以 `# 安装指南`
    到第一个 `##` 之间的"章节引子"自成一块。这是刻意的：引子写的是整章
    概要，和具体小节不该混在一个父块里。
    """
    headings, _ = _scan_structure(DOC)
    parents = _build_parents(headings, len(DOC), settings.PARENT_CHUNK_SIZE)

    assert len(parents) == 3
    assert DOC[parents[0][0]:parents[0][1]].startswith("# 安装指南")
    assert DOC[parents[1][0]:parents[1][1]].startswith("## 环境准备")


def test_build_parents_no_duplicate_ranges():
    """父块区间不能重复。

    【回归，最严重】降级循环里少一个 continue 时，未超限的块会先被 append、
    再被 if/else 追加一次，父块整体重复一遍——子块数量凭空翻倍，
    而且同一个 chunk 在向量库里出现两次，检索结果重复。
    """
    headings, _ = _scan_structure(DOC)
    parents = _build_parents(headings, len(DOC), settings.PARENT_CHUNK_SIZE)

    assert len(parents) == len(set(parents))
    # 区间必须按文中顺序严格递增
    assert parents == sorted(parents)


def test_build_parents_downgrades_when_over_limit():
    """父块超限时用 ### 再拆，而不是直接硬切。"""
    body = "正文内容" * 400                                   # 1600 字/小节
    text = "# 章\n\n" + "".join(f"## 小节{i}\n\n{body}\n\n" for i in range(3))
    headings, _ = _scan_structure(text)
    parents = _build_parents(headings, len(text), max_size=2048)

    # 三个 ## 各 1600 字，都未超限 → 仍是 3 个父块（外加 #章 的前言）
    assert len(parents) == 4
    assert all(e - s <= 2048 for s, e in parents)


def test_build_parents_hard_cut_without_headings():
    """无标题长文：逐级降级都拆不动，最后按 max_size 硬切，且不丢内容。"""
    text = "无标题的长文内容" * 1000                            # 8000 字
    headings, _ = _scan_structure(text)
    parents = _build_parents(headings, len(text), max_size=2048)

    assert len(parents) == 4                                  # ceil(8000/2048)
    assert parents[0][0] == 0
    assert parents[-1][1] == len(text)
    assert all(e - s <= 2048 for s, e in parents)


def test_build_parents_downgrade_to_h3():
    """一个 ## 超过 2048 且内部有多个 ### 时，应降级成 ### 级别的父块。"""
    section = "".join(f"### 小节{i}\n\n{'内容' * 300}\n\n" for i in range(3))
    text = f"## 大章节\n\n{section}"
    headings, _ = _scan_structure(text)
    parents = _build_parents(headings, len(text), max_size=1024)

    assert len(parents) >= 2
    assert all(e - s <= 1024 for s, e in parents)


# ============================================================================
# 三、段落 / 滑窗
# ============================================================================
def test_paragraph_ranges_split_on_blank_line():
    """空行（含行尾带空格的脏空行）都是段落分隔符。"""
    text = "第一段\n\n第二段\n  \n第三段"
    ranges = _paragraph_ranges(text, 0, len(text))

    assert [text[s:e] for s, e in ranges] == ["第一段", "第二段", "第三段"]


def test_paragraph_ranges_drop_blank_only():
    """连续空行不能产出空段落。"""
    text = "\n\n\n甲\n\n\n\n乙\n\n\n"
    ranges = _paragraph_ranges(text, 0, len(text))
    assert [text[s:e] for s, e in ranges] == ["甲", "乙"]


def test_split_long_text_respects_chunk_size():
    """滑窗切出的每一块都不能超过 chunk_size。"""
    text = "甲" * 2000
    ranges = _split_long_text(text, 0, len(text), chunk_size=512, overlap=64)

    assert all(e - s <= 512 for s, e in ranges)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(text)


def test_split_long_text_has_overlap():
    """相邻块必须有重叠（否则断口处的句子两边都不完整）。"""
    text = "甲" * 2000
    ranges = _split_long_text(text, 0, len(text), chunk_size=512, overlap=64)

    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:], strict=False):
        assert next_start < prev_end, "相邻块之间没有重叠"


def test_split_long_text_prefers_sentence_boundary():
    """有标点时优先切在句末，切出的块应以句末标点结尾。

    【回归】窗口写成 text[search_from]（少了切片）时，窗口只有 1 个字符，
    永远找不到标点，退化成纯字符硬切；同时 cut 恒等于 search_from+1，
    每块只产出 1~2 个字符，5000 字文档能切出上千块。
    """
    text = "这是一个完整的句子。" * 100                          # 1000 字、100 个句号
    ranges = _split_long_text(text, 0, len(text), chunk_size=512, overlap=64)

    assert len(ranges) < 10                                   # 旧 bug 会产生几十上百块
    for s, e in ranges[:-1]:                                  # 最后一块允许是残句
        assert text[s:e].endswith("。")


def test_split_long_text_hard_cut_without_punctuation():
    """完全没有标点时，按字符硬切，但块长仍受上限约束。"""
    text = "无标点" * 500
    ranges = _split_long_text(text, 0, len(text), chunk_size=512, overlap=64)

    assert all(e - s <= 512 for s, e in ranges)
    # 硬切是兜底：块长应该贴着上限，而不是碎成小片
    assert all(e - s >= 512 // 2 for s, e in ranges[:-1])


def test_split_long_text_no_infinite_loop_when_overlap_large():
    """overlap 接近块长时仍必须前进，不能死循环（靠 max(cursor+1) 兜住）。"""
    text = "甲" * 1000
    ranges = _split_long_text(text, 0, len(text), chunk_size=100, overlap=99)
    assert len(ranges) < 1000


# ============================================================================
# 四、父块 → 子块
# ============================================================================
def test_children_merge_short_paragraphs():
    """多个短段落应被合并进一个块，而不是一段一块。"""
    text = "\n\n".join(f"第{i}段" for i in range(20))          # 20 个 3 字小段
    pieces = _split_parent_to_children(text, 0, len(text), 512, 64)

    assert len(pieces) == 1


def test_children_no_tiny_heading_fragment():
    """标题后面紧跟超长段落时，标题必须并入正文块，不能单独成为几字碎片。

    【回归】按"段落 → 超长段独立滑窗"的旧顺序，`# 标题`（4 字）会先被封口成
    独立 chunk，正文另起一块。一篇文档每个章节边界都多一条 4~10 字碎片。
    """
    text = "# 标题\n\n" + NO_PUNCT
    pieces = _split_parent_to_children(text, 0, len(text), 512, 64)

    assert all(e - s > 50 for s, e in pieces), [e - s for s, e in pieces]
    # 第一块必须以标题开头（说明标题被并进去了）
    assert text[pieces[0][0]:pieces[0][1]].startswith("# 标题")


def test_children_ranges_are_contiguous_and_bounded():
    """子块区间必须按顺序、互不重叠、不越界，且不超过 chunk_size。"""
    text = "\n\n".join(["甲" * 300, "乙" * 300, "丙" * 900, "丁" * 300])
    pieces = _split_parent_to_children(text, 0, len(text), 512, 64)

    assert all(e - s <= 512 for s, e in pieces)
    assert pieces[0][0] == 0
    assert pieces[-1][1] == len(text)
    for (s1, e1), (s2, _e2) in zip(pieces, pieces[1:], strict=False):
        assert s2 > s1
        # 段落之间的空行被裁掉，所以允许留 2 字符以内的缝（换行符），
        #   但绝不能"跳空"超过一个空行——那说明有内容没被切进任何块。
        assert s2 - e1 <= 2


# ============================================================================
# 五、辅助判定
# ============================================================================
def test_has_prose():
    """纯标题 / 纯空行不算正文；代码围栏行算正文。"""
    assert not _has_prose("## 只有标题")
    assert _has_prose("## 标题\n\n正文一句")
    assert not _has_prose("## 标题一\n\n### 标题二")
    assert not _has_prose("   \n\n  ")
    assert _has_prose("```python\n```")


def test_is_code_heavy_threshold():
    """>=50% 字符落在代码区间才算代码块（一句说明 + 一段代码不能被整体降权）。"""
    code_ranges = [(10, 20)]
    assert _is_code_heavy(10, 20, code_ranges)                # 100%
    assert _is_code_heavy(0, 20, code_ranges)                 # 50%
    assert not _is_code_heavy(0, 40, code_ranges)             # 25%
    assert not _is_code_heavy(0, 10, [])                       # 无代码


def test_heading_path_at():
    """按字符位置查章节路径；第一个标题之前返回 ([], 0)。"""
    headings, _ = _scan_structure(DOC)

    # DOC 以 "# 安装指南" 开头，所以位置 0 已属于该标题
    assert _heading_path_at(headings, 0) == (["安装指南"], 1)
    win = DOC.index("### Windows")
    assert _heading_path_at(headings, win) == (
        ["安装指南", "环境准备", "Windows"], 3)
    # 正文中间（### Windows 之后）应保持最深路径，而不是退回父级
    assert _heading_path_at(headings, win + 5)[0] == [
        "安装指南", "环境准备", "Windows"]

    # 前言（第一个标题之前）才是空路径
    text = "前言一句话\n\n# 标题\n\n正文\n"
    h2, _ = _scan_structure(text)
    assert _heading_path_at(h2, 0) == ([], 0)


# ============================================================================
# 六、对外契约 split_document
# ============================================================================
def test_split_document_field_contract():
    """返回的每个 dict 必须带齐文档约定的全部字段。"""
    expected = {
        "chunk_id", "doc_id", "index", "text", "char_start", "char_end",
        "char_len", "heading_path", "heading_level", "parent_id",
        "parent_text", "is_code", "content_hash", "metadata",
    }
    chunks = split_document(DOC, doc_id="docs/a.md", source="raw/a.md")

    assert chunks
    for c in chunks:
        assert set(c) == expected


def test_split_document_text_is_sliceable():
    """text 必须严格等于 原文[char_start:char_end]——这是回溯引用的地基。"""
    chunks = split_document(DOC, doc_id="docs/a.md")

    for c in chunks:
        assert c["text"] == DOC[c["char_start"]:c["char_end"]]
        assert c["char_len"] == c["char_end"] - c["char_start"]


def test_split_document_ids_and_index():
    """chunk_id / index 必须唯一、连续、且能被 chunk_id 反推顺序。"""
    chunks = split_document(DOC, doc_id="docs/a.md")

    assert [c["index"] for c in chunks] == list(range(len(chunks)))
    assert [c["chunk_id"] for c in chunks] == [
        f"docs/a.md:{i:04d}" for i in range(len(chunks))
    ]


def test_split_document_parent_mapping():
    """同一 parent_id 的块，parent_text 必须一致，且父区间覆盖全部子块。"""
    chunks = split_document(DOC, doc_id="docs/a.md")

    by_parent: dict[str, list[dict]] = {}
    for c in chunks:
        by_parent.setdefault(c["parent_id"], []).append(c)

    assert len(by_parent) == 3                                  # 2 个 ## + 1 个章节引子
    for parent_id, group in by_parent.items():
        assert parent_id.startswith("docs/a.md:p")
        texts = {c["parent_text"] for c in group}
        assert len(texts) == 1, "同一父块下的 parent_text 不一致"
        parent_text = texts.pop()
        # 每个子块都必须落在父块正文里
        for c in group:
            assert c["text"] in parent_text or c["text"] in DOC


def test_split_document_heading_path_per_chunk():
    """每个子块的章节路径必须等于"它起始位置"所在标题的路径。

    【为什么按起始位置而非内容】heading_path 的语义是"这块从哪一节开始"。
    一个 ## 章节只有 124 字、不会被拆开，里面即使有 ### 小节，路径也只到 ##。
    """
    headings, _ = _scan_structure(DOC)
    chunks = split_document(DOC, doc_id="docs/a.md")

    for c in chunks:
        expected = _heading_path_at(headings, c["char_start"])
        assert (c["heading_path"], c["heading_level"]) == expected


def test_split_document_marks_code_block():
    """代码为主的块必须打上 is_code=True，散文块为 False。"""
    md = "# 代码\n\n```python\n" + "\n".join(f"x{i} = {i}" for i in range(40)) + "\n```\n"
    chunks = split_document(md, doc_id="d.md")

    assert any(c["is_code"] for c in chunks)
    assert all(c["is_code"] for c in chunks if "```python" in c["text"])


def test_split_document_metadata_and_source():
    """metadata 原样带入；source 以入参为准，不被脏 metadata 覆盖。"""
    chunks = split_document(
        DOC, doc_id="docs/a.md", source="raw/a.md",
        metadata={"title": "安装指南", "source": "脏值"},
    )

    assert chunks[0]["metadata"]["title"] == "安装指南"
    assert chunks[0]["metadata"]["source"] == "raw/a.md"


def test_split_document_normalizes_crlf():
    """\\r\\n 先规范化成 \\n，char_start/char_end 基于规范化后的文本。"""
    text = "# 标题\r\n\r\n正文内容。\r\n"
    chunks = split_document(text, doc_id="d.md")
    normalized = text.replace("\r\n", "\n")

    for c in chunks:
        assert c["text"] == normalized[c["char_start"]:c["char_end"]]
        assert "\r" not in c["text"]


def test_split_document_empty_returns_empty_list():
    """空文本 / 纯空白返回 []，不抛异常（批处理里空文件很常见）。"""
    assert split_document("", doc_id="d.md") == []
    assert split_document("   \n\n \t ", doc_id="d.md") == []


def test_split_document_rejects_bad_params():
    """参数非法必须 fail fast，而不是切到一半才发现。"""
    with pytest.raises(ValueError):
        split_document(DOC, doc_id="")                          # doc_id 为空
    with pytest.raises(ValueError):
        split_document(DOC, doc_id="d", chunk_size=0)
    with pytest.raises(ValueError):
        split_document(DOC, doc_id="d", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        split_document(DOC, doc_id="d", overlap=-1)
    with pytest.raises(ValueError):
        split_document(DOC, doc_id="d", chunk_size=512, parent_size=100)


def test_split_document_defaults_follow_settings():
    """默认块大小/重叠/父块上限必须来自 settings，不能硬编码。"""
    md = "## 章节\n\n" + NO_PUNCT
    chunks = split_document(md, doc_id="d.md")

    assert max(c["char_len"] for c in chunks) <= settings.CHUNK_SIZE
    # 用显式参数覆盖，结果应与默认一致
    chunks2 = split_document(
        md, doc_id="d.md",
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP,
        parent_size=settings.PARENT_CHUNK_SIZE,
    )
    assert [c["char_len"] for c in chunks] == [c["char_len"] for c in chunks2]


def test_split_document_no_heading_only_chunk():
    """整篇里不允许出现"只有标题没有正文"的块（纯标题信息已在 heading_path）。"""
    md = "# 大章节\n\n## 小节一\n\n" + "正文" * 300 + "\n\n## 小节二\n\n" + "正文" * 300
    chunks = split_document(md, doc_id="d.md")

    for c in chunks:
        assert _has_prose(c["text"]), c["text"][:30]


def test_split_document_single_chunk_for_short_doc():
    """短文档（< chunk_size）应只产出一个块，不能碎成多块。"""
    md = "## 一节\n\n这是一段很短的中文正文，用来验证短文档不会被切碎。"
    chunks = split_document(md, doc_id="d.md")

    assert len(chunks) == 1
    assert chunks[0]["text"] == md.rstrip()


def test_split_document_total_length_not_shorter_than_source():
    """块长总和不能小于原文实质内容长度（防止漏切丢字）。"""
    for md in (DOC, "# 标题\n\n" + NO_PUNCT, "# 标题\n\n" + WITH_PUNCT):
        chunks = split_document(md, doc_id="d.md")
        covered = sum(c["char_len"] for c in chunks)
        assert covered >= len(md.strip()) * 0.95, (covered, len(md))
