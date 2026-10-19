# ============================================================================
# test_parser.py —— 文档解析（parse）的单元测试
#
# 【测什么】
#   验证 parse_file 能按格式分流，正确解析 md/html/ipynb/xml，
#   以及不支持的格式返回 None（跳过）。
#
# 【怎么跑】
#   .venv/Scripts/python.exe -m pytest tests/test_parser.py -v
#
#   注意必须指定本文件路径，不要裸跑 pytest。因为其余 14 个实现模块
#   （cleaner/splitter/embedder/...）已被手动删除、待重写，裸跑 pytest 会
#   连带 import 它们，产生 14 个 collection error。显式传路径时
#   testpaths 被覆盖，坏文件根本不会被 import，因此不受影响。
# ============================================================================
import json

from rag.ingestion.parse import parse_file

# test01:直接读Markdown
def test_parse_markdown(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 标题\n\n正文内容",encoding="utf-8")
    text = parse_file(f)
    assert "# 标题" in text
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


# ============================================================================
# 以下是 bug 回归与盲区补齐用例
# ============================================================================

# ----------------------------------------------------------------------------
# 测试 6：HTML 候选容器长度并列时不应崩溃
# ----------------------------------------------------------------------------
def test_parse_html_tie_length_no_crash(tmp_path):
    """两个候选容器的文本长度完全相同时，不应抛异常。

    【为什么要有这个用例】
        candidates 里每个元素是 (长度, Tag) 元组。直接写 max(candidates) 时，
        Python 先比第0 项长度；一旦两个候选长度相同，就会转去比较第 1 项 Tag，
        而 BeautifulSoup 的 Tag 对象不支持比较，于是抛
            TypeError: '>' not supported between instances of 'Tag' and 'Tag'
        实测真实语料 1697 篇 HTML 里约 7.3%（约 124 篇）存在长度并列，
        跑全量索引时必然踩到。正确写法是 max(candidates, key=lambda x: x[0])。
    """
    f = tmp_path / "tie.html"
    # 两个容器都是 200 个字符，故意制造"长度并列"
    html = (
        "<html><body>"
        f"<article>{'甲' * 200}</article>"
        f"<div class='content'>{'乙' * 200}</div>"
        "</body></html>"
    )
    f.write_text(html, encoding="utf-8")

    text = parse_file(f)      # 旧写法在这一行抛 TypeError
    assert text.strip()       # 修好后：正常返回非空文本


# ----------------------------------------------------------------------------
# 测试 7：HTML 多选择器取最长（核心逻辑，此前零覆盖）
# ----------------------------------------------------------------------------
def test_parse_html_picks_longest_candidate(tmp_path):
    """多个候选容器并存时，取文本最长的那个。

    【为什么要有这个用例】
        测试 2 里 <article> 的文本长度只有 65 字符，没超过代码里的 100 字阈值，
        根本没走进"多选择器取最长"这个分支，而是靠"没有候选就用整个 soup"的
        回退侥幸通过的。也就是说_parse_html 最核心的取最长逻辑一直未被验证。
        这里用 article(150字) 与 .content(400字) 制造明显的长度差，
        确保走进分支并验证真的取了最长的那个。
    """
    f = tmp_path / "longest.html"
    html = (
        "<html><body>"
        f"<article>{'文' * 150}</article>"# 较短
        f"<div class='content'>{'章' * 400}</div>"       # 较长，应被选中
        "</body></html>"
    )
    f.write_text(html, encoding="utf-8")
    text = parse_file(f)
    assert "章" in text                # 更长的 .content 被选中
    assert "文" not in text            # 较短的 article 被淘汰


# ----------------------------------------------------------------------------
# 测试 8：ipynb 代码 cell 必须用 ```python 围栏
# ----------------------------------------------------------------------------
def test_parse_ipynb_code_fence_is_markdown(tmp_path):
    """代码 cell 必须用 Markdown 围栏包裹，且语言标记拼写正确。

    【为什么要有这个用例】
        旧代码写成 "''' phthon\\n" + text + "\\n'''"：三单引号不是 Markdown 语法，
        下游 cleaner/splitter 无法按围栏识别代码段并降权；且 phthon 拼错了。
        git 历史里的原版就是 ```python 围栏。
    """
    f = tmp_path / "fence.ipynb"
    notebook = {"cells": [{"cell_type": "code", "source": ["print('hi')"]}]}
    f.write_text(json.dumps(notebook), encoding="utf-8")
    text = parse_file(f)
    assert text.startswith("```python\n")  # 开头围栏带语言标记
    assert text.rstrip().endswith("```")  # 结尾是闭合围栏
    assert "'''" not in text  # 旧的三单引号写法必须已被排除
    assert "phthon" not in text  # 拼写错误必须已被排除


# ----------------------------------------------------------------------------
# 测试 9：ipynb 多个 cell 全部被解析
# ----------------------------------------------------------------------------
def test_parse_ipynb_multiple_cells_all_parsed(tmp_path):
    """多个 cell 必须全部解析出来，防止将来误改缩进导致只取第一个 cell。"""
    f = tmp_path / "multi.ipynb"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["MARKER_A"]},
            {"cell_type": "code", "source": ["MARKER_B = 1"]},
            {"cell_type": "markdown", "source": ["MARKER_C"]},
            {"cell_type": "code", "source": ["MARKER_D = 2"]},
        ]
    }
    f.write_text(json.dumps(notebook), encoding="utf-8")
    text = parse_file(f)
    # 四个标记必须全部出现。若只出现 MARKER_A，说明循环体内有提前 return
    for marker in ["MARKER_A", "MARKER_B", "MARKER_C", "MARKER_D"]:
        assert marker in text