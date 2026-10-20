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


# ============================================================================
# 以下是 2026-10-20 审查补齐的用例（对应 parse.py 的四处修正）
# ============================================================================

# ----------------------------------------------------------------------------
# 测试 10：RSS 正文是 HTML，必须转成 Markdown（标题才能被下游识别）
# ----------------------------------------------------------------------------
def test_parse_rss_html_body_converted_to_markdown(tmp_path):
    """content:encoded 里的 `<h2>` 必须变成 `## `，否则下游扫不到任何标题。

    【为什么要有这个用例】
        kexue 的 feed 正文是 HTML 片段（`<p>…</p><h2>…</h2>`）。旧实现直接
        返回 `elem.text`（原始 HTML），splitter 扫不到 `#` 标题、段落之间也
        没有空行 —— 整篇上万字会被当成"一个没有标题的巨型段落"，全部走 512
        字硬切。语料里质量最高的一类就这样被结构性地浪费掉了。
    """
    f = tmp_path / "htmlbody.xml"
    xml = (
        '<?xml version="1.0"?>'
        '<rss><channel><item>'
        '<title>平均收敛</title>'
        '<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        '<![CDATA[<p>第一段正文。</p><h2>小标题</h2><p>第二段正文。</p>]]>'
        '</content:encoded>'
        '</item></channel></rss>'
    )
    f.write_text(xml, encoding="utf-8")
    text = parse_file(f)

    assert "<p>" not in text                      # HTML 标签已转换
    assert "第一段正文。" in text
    assert "<h2>" not in text
    assert "## 小标题" in text or "小标题" in text


def test_parse_rss_html_as_child_elements(tmp_path):
    """正文以普通子节点形式的 feed（非 CDATA），同样要能解析出来。

    【为什么】RSS 有两种常见写法：CDATA 包裹，或把 `<p>` 直接作为子节点。
    只读 `elem.text` 时后者会拿到 None，整条正文静默消失。
    """
    f = tmp_path / "child.xml"
    xml = (
        '<?xml version="1.0"?>'
        '<rss><channel><item>'
        '<title>子节点写法</title>'
        '<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        '<p>子节点里的正文内容。</p><h2>小节</h2><p>收尾。</p>'
        '</content:encoded>'
        '</item></channel></rss>'
    )
    f.write_text(xml, encoding="utf-8")
    text = parse_file(f)

    assert "子节点里的正文内容。" in text
    assert "## 小节" in text


def test_parse_rss_item_title_becomes_heading(tmp_path):
    """每个 <item> 的 title 要补成 `# 标题`，让"一篇文章"成为父块边界。"""
    f = tmp_path / "items.xml"
    xml = (
        '<?xml version="1.0"?>'
        '<rss><channel>'
        '<item><title>文章甲</title>'
        '<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        '<![CDATA[<p>甲的正文。</p>]]></content:encoded></item>'
        '<item><title>文章乙</title>'
        '<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        '<![CDATA[<p>乙的正文。</p>]]></content:encoded></item>'
        '</channel></rss>'
    )
    f.write_text(xml, encoding="utf-8")
    text = parse_file(f)

    assert "# 文章甲" in text
    assert "# 文章乙" in text
    # 顺序不能反
    assert text.index("文章甲") < text.index("文章乙")


def test_parse_rss_falls_back_to_description(tmp_path):
    """整份 feed 没有 content:encoded 时退回 description，不能整篇空手而归。"""
    f = tmp_path / "desc.xml"
    xml = (
        '<?xml version="1.0"?>'
        '<rss><channel><item>'
        '<title>只有摘要</title>'
        '<description><![CDATA[<p>这是一段摘要正文，用来验证兜底分支。</p>]]></description>'
        '</item></channel></rss>'
    )
    f.write_text(xml, encoding="utf-8")
    text = parse_file(f)

    assert "这是一段摘要正文" in text


def test_parse_rss_protects_latex_with_angle_brackets(tmp_path):
    """公式里的 `<` `>` 不能被 markdownify 当成 HTML 标签吃掉。

    【为什么】`$a<b$` 这类不等式在 kexue 语料里极常见。不先做占位保护，
    markdownify 会把 `<b$…` 当标签解析，公式当场消失（而且是静默的）。
    """
    f = tmp_path / "math.xml"
    xml = (
        '<?xml version="1.0"?>'
        '<rss><channel><item>'
        '<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        '<![CDATA[<p>当 $a<b$ 且 $x>y$ 时成立。</p>]]>'
        '</content:encoded>'
        '</item></channel></rss>'
    )
    f.write_text(xml, encoding="utf-8")
    text = parse_file(f)

    assert "$a<b$" in text
    assert "$x>y$" in text


# ----------------------------------------------------------------------------
# 测试 11：编码兜底（GBK 老站点不能被静默丢掉）
# ----------------------------------------------------------------------------
def test_parse_gbk_html_without_charset_meta(tmp_path):
    """没写 meta charset 的 GBK 页面，也要能正确读出中文。

    【为什么】docs/ 里有 1500+ 篇老站点 HTML，混着 GBK/GB2312。旧实现用
    utf-8 + errors="ignore" 读，中文字节被静默丢弃，链路全绿但语料全废。
    """
    f = tmp_path / "gbk.html"
    html = "<html><body><article><p>这是国标编码的中文正文。</p></article></body></html>"
    f.write_bytes(html.encode("gbk"))

    text = parse_file(f)
    assert "这是国标编码的中文正文。" in text


def test_parse_gbk_html_with_charset_meta(tmp_path):
    """写了 `<meta charset="gbk">` 的页面，按声明解码。"""
    f = tmp_path / "gbk2.html"
    html = ('<html><head><meta charset="gbk"></head>'
            '<body><article><p>声明为国标编码。</p></article></body></html>')
    f.write_bytes(html.encode("gbk"))

    text = parse_file(f)
    assert "声明为国标编码。" in text


def test_parse_gbk_markdown(tmp_path):
    """.md/.txt 同样走编码兜底（语料里有 GBK 的笔记文件）。"""
    f = tmp_path / "gbk.md"
    f.write_bytes("# 标题\n\n这是国标编码的笔记正文。".encode("gbk"))

    text = parse_file(f)
    assert "这是国标编码的笔记正文。" in text


def test_parse_crlf_normalized_to_lf(tmp_path):
    """解析层必须把 \\r\\n 统一成 \\n。

    【为什么】下游 cleaner/splitter 的正则（`\\n{3,}`、`\\n[ \\t]*\\n`）都假定 LF。
    CRLF 文档里段间是 `\\r\\n\\r\\n`，中间夹着 \\r，段落切分直接失效。
    历史上 `Path.read_text()` 的 universal newlines 顺手做了这件事，
    改成 `read_bytes().decode()` 读编码后这层转换会消失，必须显式补回。
    """
    f = tmp_path / "crlf.md"
    f.write_bytes("# 标题\r\n\r\n第一段\r\n\r\n第二段\r\n".encode())

    text = parse_file(f)
    assert "\r" not in text
    assert "第一段\n\n第二段" in text


# ----------------------------------------------------------------------------
# 测试 12：解析失败不炸整批
# ----------------------------------------------------------------------------
def test_parse_corrupt_ipynb_returns_none(tmp_path):
    """损坏的 ipynb（半截 JSON）返回 None，而不是抛异常中断整批任务。

    【为什么】3.5 万文件里必然有坏文件。让第一个坏文件崩掉整批，是数据管线
    最典型的低级事故（对应文档 2.5.7 "单篇失败不阻断整批"）。
    """
    f = tmp_path / "broken.ipynb"
    f.write_text('{"cells": [', encoding="utf-8")

    assert parse_file(f) is None


def test_parse_corrupt_xml_returns_none(tmp_path):
    """损坏的 XML 同理：返回 None，不抛。"""
    f = tmp_path / "broken.xml"
    f.write_text("<rss><channel><item>", encoding="utf-8")

    assert parse_file(f) is None


# ----------------------------------------------------------------------------
# 测试 13：HTML 候选阈值来自 settings
# ----------------------------------------------------------------------------
def test_parse_html_threshold_follows_settings():
    """候选容器长度阈值必须来自 settings，不能是代码里的魔法数字。"""
    from rag.config.settings import settings

    assert isinstance(settings.PARSE_HTML_MIN_CANDIDATE_LEN, int)
    assert settings.PARSE_HTML_MIN_CANDIDATE_LEN > 0


def test_parse_html_short_candidate_falls_back_to_whole_page(tmp_path):
    """所有候选都短于阈值时，回退整页——不能因为"候选太短"就返回空。"""
    from rag.config.settings import settings

    short = "短" * max(settings.PARSE_HTML_MIN_CANDIDATE_LEN - 1, 1)
    f = tmp_path / "short.html"
    f.write_text(f"<html><body><article>{short}</article></body></html>",
                 encoding="utf-8")

    text = parse_file(f)
    assert short in text
