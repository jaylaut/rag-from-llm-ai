# ============================================================================
# test_parser.py —— 文档解析（parser）的单元测试
#
# 【测什么】
#   验证 parse_file 能按格式分流，正确解析 md/html/ipynb/xml，
#   以及不支持的格式返回 None（跳过）。
# ============================================================================

import json

from rag.ingestion.parser import parse_file


# ----------------------------------------------------------------------------
# 测试 1：Markdown 直读
# ----------------------------------------------------------------------------
def test_parse_markdown(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 标题\n\n正文内容", encoding="utf-8")
    text = parse_file(f)
    assert "# 标题" in text          # 保留 Markdown 结构
    assert "正文内容" in text


def test_parse_txt(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("纯文本内容", encoding="utf-8")
    assert parse_file(f) == "纯文本内容"


# ----------------------------------------------------------------------------
# 测试 2：HTML 正文提取（去掉导航，保留正文）
# ----------------------------------------------------------------------------
def test_parse_html_extracts_main_content(tmp_path):
    f = tmp_path / "c.html"
    html = (
        "<html><body>"
        "<nav>这是导航栏不该被提取</nav>"
        "<article>"
        "<h1>文章标题</h1>"
        "<p>这是正文第一段，内容是技术说明，用来测试正文提取是否准确。</p>"
        "<p>这是正文第二段，包含足够多的中文，确保长度超过一百个字符的阈值。</p>"
        "</article>"
        "<footer>版权信息不该出现</footer>"
        "</body></html>"
    )
    f.write_text(html, encoding="utf-8")
    text = parse_file(f)
    assert "文章标题" in text            # 正文标题被提取
    assert "这是正文第一段" in text      # 正文被提取
    assert "这是导航栏不该被提取" not in text   # 导航被剔除
    assert "版权信息不该出现" not in text        # 页脚被剔除


# ----------------------------------------------------------------------------
# 测试 3：ipynb 解析（区分 markdown 和 code cell）
# ----------------------------------------------------------------------------
def test_parse_ipynb(tmp_path):
    f = tmp_path / "d.ipynb"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# 说明", "\n", "这是一段说明"]},
            {"cell_type": "code", "source": ["print('hello')"]},
        ]
    }
    f.write_text(json.dumps(notebook), encoding="utf-8")
    text = parse_file(f)
    assert "这是一段说明" in text              # markdown cell 被提取
    assert "```python" in text                 # code cell 被标记为代码块
    assert "print('hello')" in text


# ----------------------------------------------------------------------------
# 测试 4：XML/RSS 解析（保护 LaTeX 公式）
# ----------------------------------------------------------------------------
def test_parse_rss(tmp_path):
    f = tmp_path / "e.xml"
    xml = (
        '<?xml version="1.0"?>'
        '<rss><channel><item>'
        '<title>文章</title>'
        '<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        '这里是正文，包含公式 $E=mc^2$'
        '</content:encoded>'
        '</item></channel></rss>'
    )
    f.write_text(xml, encoding="utf-8")
    text = parse_file(f)
    assert "这里是正文" in text
    assert "$E=mc^2$" in text            # 公式原样保留，不被破坏


# ----------------------------------------------------------------------------
# 测试 5：不支持的格式返回 None
# ----------------------------------------------------------------------------
def test_parse_unsupported_returns_none(tmp_path):
    f = tmp_path / "f.tex"
    f.write_text("\\documentclass{article}", encoding="utf-8")
    assert parse_file(f) is None         # .tex 暂缓，返回 None 表示跳过
