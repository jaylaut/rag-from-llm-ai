# ============================================================================
# parse.py —— 文档解析（Document Parsing）：格式 → 文本
#
# 位置：src/rag/ingestion/parse.py
#
# 【本文件是干什么的】
#   把抓来的各种格式文件（md/html/ipynb/xml/RSS），提取成"干净、可检索"的
#   纯文本/Markdown。这是索引构建五步流水线的第一步（对应《P1-索引构建.md》
#   2.1 节）。
#
# 【真实语料有哪些格式（别按"想象"来）】
#   .md/.mdx/.txt/.rst → 直接读文本（保留 Markdown 结构）
#   .html/.htm         → 多选择器提取正文，去掉导航/广告
#   .xml（RSS）        → 解析 <content:encoded>，**HTML 正文转 Markdown**，保护 LaTeX 公式
#   .ipynb             → 它是 JSON，取 cells 里的内容
#   .tex               → LaTeX 源码，中文场景价值低，暂缓
#
# 【设计原则】
#   1. 本模块是"无状态函数"：一个文件路径进，一段文本出。不碰目录遍历、不碰
#      数据库。这样全量和增量能复用同一套内核（对应文档 3.2 节）。
#   2. **输出统一是 Markdown**：下游 splitter 靠 `#` 标题和空行切块，所以
#      HTML（含 RSS 里的 HTML 正文）必须转成 Markdown，扔原始标签过去等于
#      让整篇文章退化成"一个没有标题的巨型段落"。
#   3. **单篇失败不炸整批**：解析异常统一在 parse_file 兜住，返回 None 表示
#      跳过，由上游计入 dead-letter（对应文档 2.5.7）。
# ============================================================================

# 导入区
import json
import re
from pathlib import Path

import markdownify
from bs4 import BeautifulSoup

from rag.config.settings import settings
from rag.utils.logger import setup_logger

log = setup_logger()

# ============================================================================
# 文本读取：带编码兜底
# ============================================================================
# HTML 里声明的字符集：<meta charset="gbk"> / <meta http-equiv=... content="text/html; charset=gb2312">
_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)

# 尝试顺序：页面声明 > utf-8 > gb18030 > big5
#   gb18030 是 GBK/GB2312 的超集，直接用它兜底，不必逐个试
_FALLBACK_ENCODINGS = ("utf-8", "gb18030", "big5")

# GB 系列统一映射到 gb18030（超集，能解 gbk/gb2312 的全部字节）
_GB_FAMILY = {"gbk", "gb2312", "gb18030", "gb_2312-80", "cp936", "euc-cn"}


def _read_text(path: Path) -> str:
    """按"页面声明 → utf-8 → gb18030 → big5"的顺序解码文件，尽最大努力不丢字。

        【为什么不能只写 encoding="utf-8", errors="ignore"】
            docs/ 下的 1500+ 篇 HTML 里有一批老站点是 GBK/GB2312 编码。用
            utf-8 + ignore 读，中文字节会被**静默丢掉**——不报错、不告警，
            读出来是一串残缺的乱码，下游清洗还以为"这文档就这么短"。
            这是最难查的一类数据质量事故：链路全绿，语料全废。

        【为什么先看 meta charset】
            同一份语料里 gbk / gb2312 / utf-8 混杂，靠"试"也能过，但会先把
            GBK 文件按 utf-8 解一次（必然失败，白解一遍 5MB）。先读声明既快
            又准确。声明不可信时仍有兜底链，不会因为写错 charset 就崩。

        【参数】
            path : Path —— 文件路径

        【返回】
            str —— 解码后的文本，**换行符已统一为 \\n**（全部编码都失败时用
                   utf-8+ignore 兜底，保证不抛）
        """
    raw = path.read_bytes()

    tried: list[str] = []
    m = _CHARSET_RE.search(raw[:4096])          # 声明只可能出现在头部
    if m:
        declared = m.group(1).decode("ascii", "ignore").strip().lower()
        if declared in _GB_FAMILY:
            declared = "gb18030"
        tried.append(declared)

    for enc in _FALLBACK_ENCODINGS:
        if enc not in tried:
            tried.append(enc)

    for enc in tried:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        # 统一换行符：\r\n / \r → \n
        #   【为什么必须在解析边界做】Path.read_text() 用文本模式打开，默认会做
        #   universal newlines 转换；改用 read_bytes().decode() 后这层转换消失了。
        #   而下游 cleaner/splitter 的正则（"\n{3,}"、"\n[ \t]*\n"）都假定 LF：
        #   CRLF 文档里段间是 "\r\n\r\n"，中间夹着 \r，段落切分直接失效。
        #   在这里统一，比让每个下游模块各自处理要安全得多。
        return text.replace("\r\n", "\n").replace("\r", "\n")

    # 兜底：理论上 gb18030 能解绝大多数中文语料，走到这里说明是混合脏编码
    log.warning("编码全部失败，按 utf-8 ignore 兜底: %s", path)
    return raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")


# ============================================================================
# LaTeX 保护：HTML → Markdown 时别把公式当标签吃掉
# ============================================================================
# 匹配 $$...$$（块级，允许跨行）与 $...$（行内，不跨行）
_MATH_RE = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$)")

# 占位符前缀：用不可能出现在正文里的控制字符组合，避免与真实文本冲突
_MATH_SLOT = "\x00MATH{}\x00"


def _html_to_markdown(html: str) -> str:
    """HTML → Markdown，并把 `$...$` 公式先"藏起来"、转换完再还原。

        【为什么需要这层保护】
            markdownify 会按 HTML 标签树重排文本。数学公式里出现 `<` `>` 时
            （如 `$a<b$`、`$x>0$`），它会被当成未闭合标签处理，公式当场被吃掉。
            kexue 的正文全是带公式的技术长文，这一步不做，等于系统性损坏。

        【做法】
            1. 用正则把 `$$...$$` / `$...$` 换成不可能出现在正文里的占位符
            2. markdownify 正常转换
            3. 按序号把占位符换回原公式（公式原文一字不改）

        【局限（如实记录）】
            "两个美元符号之间就是公式"是启发式。像 `价格是 $10 到 $20` 这种
            一句话里出现两个美元金额的文本，会被误当成一个公式保护起来——
            好在保护只是"原样保留"，不会损坏内容，最多是少转一层格式。

        【参数】
            html : str —— HTML 片段（整页或 RSS 正文）

        【返回】
            str —— Markdown 文本
        """
    if not html:
        return ""

    stash: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        stash.append(m.group(1))
        return _MATH_SLOT.format(len(stash) - 1)

    guarded = _MATH_RE.sub(_stash, html)
    md = markdownify.markdownify(guarded, heading_style="ATX")

    # 还原公式：用 replace 而不是 format，公式里的 {} 不会被二次解释
    for i, expr in enumerate(stash):
        md = md.replace(_MATH_SLOT.format(i), expr)
    return md


# ============================================================================
# 核心入口：按扩展名分发到对应的解析函数
# ============================================================================
def parse_file(path: Path) -> str | None:
    """解析单个文件，返回 Markdown 文本。

        【原理】
            先看文件的后缀名（扩展名），判断它是哪类格式，再调用对应的解析函数。
            这叫"按格式分流"。

        【为什么整个分发都包在 try 里】
            真实语料里必然有损坏文件（半个 JSON 的 ipynb、编码错乱的 xml、
            lxml 解不开的残缺 HTML）。让第一个坏文件把整批 3.5 万篇的任务
            崩掉，是数据管线最典型的低级事故。这里统一兜住、告警、返回 None，
            由上游记入 dead-letter 后继续处理下一篇。

        【参数】
            path : Path —— 要解析的文件路径

        【返回】
            str  ：解析出的 Markdown 文本（可能是空串，如空文件）
            None ：格式不支持（如 .tex）或解析失败（损坏文件）——调用方按"跳过"处理
        """
    # 获取文件扩展名
    suffix = path.suffix.lower()

    try:
        # .md / .mdx / .txt / .rst 本身就是文本，不需要解析库，直接读。
        if suffix in {".md", ".mdx", ".txt", ".rst"}:
            return _read_text(path)

        # html格式
        if suffix in {".html", ".htm"}:
            return _parse_html(path)

        # ipynb 格式
        if suffix == ".ipynb":
            return _parse_ipynb(path)

        # xml格式（RSS订阅源）
        if suffix == ".xml":
            return _parse_rss(path)
    except Exception as exc:                       # noqa: BLE001 —— 单篇失败不阻断整批
        log.warning("解析失败，已跳过 %s: %s: %s", path, type(exc).__name__, exc)
        return None

    # 其余格式暂不处理，返回None
    log.warning("跳过不支持的格式:%s", path)
    return None


def _parse_html(path: Path) -> str:
    """解析 HTML 文件，提取正文并转成 Markdown。

       【原理（对应文档 2.1 节）】
           HTML 里"正文文字"和"排版标签"混在一起，还夹着导航/广告/推荐位。
           BeautifulSoup 负责"找到正文那一块"，把垃圾剔除，再把正文转成 Markdown。

       【策略（对应文档 2.1.1 "多选择器取最长"）】
           不同站点的正文容器标签不一样，我们列出常见候选（<article>、<main>、
           .content 等），逐个取文字长度，选文字最多的那个——最可能是正文。

       【参数】
           path : Path —— HTML 文件路径

       【返回】
           str —— 正文的 Markdown 文本
       """
    # 读文件内容：带编码兜底（docs/ 里混着 GBK 老站点）
    html_text = _read_text(path)

    # BeautifulSoup（html_text,"lxml）
    # html_text:待解析的HTML字符串
    # lxml:指定底层解析引擎
    soup = BeautifulSoup(html_text, "lxml")

    # 第一步：删除没用的标签——非正文区域：如脚本、样式、导航、页脚、侧栏、广告等
    for tag in soup.select(
        "script,style,nav,footer,header,aside,.ad, .advert, .advertisement, .sidebar, .recommend, .related, .breadcrumb"
    ):
        # 移除这些标签
        tag.decompose()
    # 第二步：多选择器获取最长正文
    candidates = []         # 收集所有候选正文，元素为(文字长度，标签)

    # 候选的正文容器选择器——覆盖主流稳定站、技术站的结构
    for selector in ["article","main",".content",".post",".article",".markdown-body",".entry-content","#content"]:
        #select(selector):返回所有匹配该选择器的标签
        for node in soup.select(selector):
            # 提取标签内所有的文字
            text = node.get_text()
            # 只保留一定长度的候选（阈值在 settings 里，不写死）
            #   【为什么不取最短的】候选容器可能嵌在另一个候选里（<main> 里套
            #   <article>），取最长才能拿到完整正文；短的那个往往是被截断的局部。
            if len(text) > settings.PARSE_HTML_MIN_CANDIDATE_LEN:
                candidates.append((len(text),node))


    # 第三步：选定正文节点
    if candidates:
        # max(candidates, key=...):用 key 指定"按什么比",只比第0项(长度)。
        # 【为什么必须写 key】candidates 的元素是 (长度, Tag) 元组。直接写 max(candidates),
        #   Python 会先比长度;一旦两个候选长度相同(并列),就会转去比第1项 Tag,
        #   而 BeautifulSoup 的 Tag 对象不支持比较 → 抛
        #   TypeError: '>' not supported between instances of 'Tag' and 'Tag'。
        #   实测真实语料 1697 篇 HTML 里约 124 篇(7.3%)存在长度并列,必然踩雷。
        #   写 key=lambda item: item[0] 明确"只比长度",并列时不再碰 Tag,安全。
        content_node = max(candidates, key=lambda item: item[0])[1]
    else:
        # 没有明显正文容器，就用整个soup
        content_node = soup

    # 第四步 HTML 转为 markdown
    return _html_to_markdown(str(content_node))

# ============================================================================
# ipynb 解析：取 cells 里的内容
# ============================================================================
def _parse_ipynb(path: Path) -> str:
    """解析 ipynb（Jupyter 笔记本）文件。

        【原理（对应文档 2.1 节）】
            ipynb 本质是一个 JSON 文件，里面有个 "cells" 数组，每个元素是一块
            内容，用 "cell_type" 区分是 markdown 还是 code。
            代码块有价值但要标记（对应文档 2.1 决策2：is_code=True，检索时降权）。

        【参数】
            path : Path —— ipynb 文件路径

        【返回】
            str —— 拼接好的 Markdown 文本（代码块用 ``` 包裹）
        """
    # 先把json解析成 字典/列表，在读取内容，转成Python对象
    #   json 格式错误会抛 JSONDecodeError，由 parse_file 统一兜住
    notebook = json.loads(_read_text(path))

    # 用列表保存收集的每块内容，最后统一拼接
    parts = []

    for cell in notebook.get("cells",[]):
        #cell也是字典，cell_type 字段标识类型
        cell_type = cell.get("cell_type","")
        #soure可能是字符串，也可能是字符串列表
        source = cell.get("source",[])
        if isinstance(source,list):
            text = "".join(source)      # 把列表里的字符串拼接成一个字符串
        else:
            text = source               # 否则是单个字符串

        # 如果是markdown，直接保留内容
        if cell_type == "markdown":
            parts.append(text)
        # code: 用 ``` 包裹成Markdown 代码块（下面elif 属于 if/elif 链，分支互斥）
        elif cell_type == "code":
            # 代码块用三个反引号包裹并标注语言 python，供下游 cleaner/splitter 识别 is_code 降权。
            # 【为什么不能写成 '''】三单引号不是 Markdown 语法，markdownify 转换、后续
            #   按围栏切分代码段都会失效；且language 标记必须是 python 而非拼错的 phthon。
            parts.append("```python\n" + text + "\n```")
        #使用 "\n\n".join():用两个换行符把所有块连起来，块之间留空行
    return "\n\n".join(parts)

# ============================================================================
# RSS/XML 解析：HTML 正文 → Markdown，保护 LaTeX 公式
# ============================================================================
def _parse_rss(path: Path) -> str:
    """解析 RSS/XML 文件（如 kexue 科学空间的订阅源）。

       【原理（对应文档 2.1 节）】
           正文被 <content:encoded> 标签包着，里面**是 HTML 片段**
           （`<p>…</p><h2>…</h2>`），且常带 $...$ 数学公式。

       【为什么必须转成 Markdown（这条曾经踩过坑）】
           content:encoded 里是 HTML。直接 `elem.text` 返回原始 HTML 字符串，
           下游 splitter 扫不到任何 `#` 标题、段落之间也没有空行——一整篇
           上万字的文章会被当成"一个没有标题的巨型段落"，全部走 512 字硬切。
           kexue 是这批语料里质量最高的一类，这么处理等于把它的结构白扔了。

       【LaTeX 怎么办】
           转换前先把 `$...$` / `$$...$$` 替换成占位符，转换完再还原。
           markdownify 会把 `<b>` 之类的标签转掉，若公式里恰好出现 `<` `>`
           （如 $a<b$），不保护就会被当成标签解析掉。

       【参数】
           path : Path —— XML 文件路径

       【返回】
           str —— 提取并转换后的 Markdown 正文
       """
    # 导入ElementTree(函数内导入，因为只有本函数用到，模块加载更轻)
    import xml.etree.ElementTree as ET

    # 返回树形结构，.getroot()获取根节点
    #   注意：ET 自己按 XML 声明解析编码，这里不走 _read_text
    root = ET.parse(path).getroot()

    def _inner_html(node) -> str:
        """把元素的"内部 HTML"还原成字符串。

            【为什么需要它】
                feed 里正文有两种写法：
                  a) CDATA：`<content:encoded><![CDATA[<p>正文</p>]]></content:encoded>`
                     → 解析后 node.text 就是整段 HTML 字符串
                  b) 普通子节点：`<content:encoded><p>正文</p></content:encoded>`
                     → 解析后 node.text 是 None，正文在子元素里
                只读 node.text 的做法在 (b) 上会拿到空串，整条正文静默消失。
                这里对 (b) 把子元素重新序列化回 HTML，两种写法就都通了。
        """
        if len(node) == 0:
            return node.text or ""
        parts = [node.text or ""]
        for child in node:
            parts.append(ET.tostring(child, encoding="unicode"))
            parts.append(child.tail or "")
        return "".join(parts)

    def _child_html(node, names: tuple[str, ...]) -> str:
        """取子节点里第一个名字匹配（去命名空间）且非空的内部 HTML。"""
        for child in node:
            if child.tag.split("}")[-1] in names:
                html = _inner_html(child)
                if html.strip():
                    return html
        return ""

    def _plain(html: str) -> str:
        """去掉标签，取纯文本（标题用）。"""
        return re.sub(r"<[^>]+>", "", html).strip()

    # ---- 先按 <item> 分块：一个 item 就是一篇文章 ----
    #   【为什么要按 item 走，而不是把所有 encoded 平铺拼接】
    #   feed 里 10 篇文章的正文首尾相接、彼此之间没有任何标题，下游 splitter
    #   会把它当成一篇"没有章节的超长文"，整篇走滑窗硬切。把 item 的 <title>
    #   补成 `# 标题`，父块粒度就自然落在"一篇文章"上——这才是这批语料
    #   真正的语义边界（对应文档 2.2 "一个 ## 小节就是一个天然的块"）。
    items = [e for e in root.iter() if e.tag.split("}")[-1] == "item"]
    if items:
        blocks: list[str] = []
        for item in items:
            title = _plain(_child_html(item, ("title",)))
            body = _child_html(item, ("encoded", "content"))
            if not body:
                # 没有全文就退回摘要；宁可短，也别整篇丢
                body = _child_html(item, ("description",))
                if body:
                    log.warning("RSS 条目无 content:encoded，退回 description 摘要: %s", path)
            if not body:
                continue
            md = _html_to_markdown(body).strip()
            if not md:
                continue
            # 标题作为一级标题补回去（原 feed 的 title 是纯文本，不在正文里）
            blocks.append(f"# {title}\n\n{md}" if title else md)
        return "\n\n".join(blocks)

    # ---- 兼容：不是 RSS（没有 item 节点），按标签名平铺收集 ----
    encoded: list[str] = []
    described: list[str] = []
    for elem in root.iter():
        tag_name = elem.tag.split("}")[-1]
        html = _inner_html(elem)
        if not html.strip():
            continue
        if tag_name == "encoded":
            encoded.append(html)
        elif tag_name in ("content", "description"):
            described.append(html)

    pieces = encoded or described
    if not encoded and described:
        log.warning("RSS 无 content:encoded，退回 description 摘要: %s", path)

    return "\n\n".join(_html_to_markdown(p) for p in pieces)
