# ============================================================================
# parser.py —— 文档解析（Document Parsing）：格式 → 文本
#
# 位置：src/rag/ingestion/parser.py
#
# 【本文件是干什么的】
#   把抓来的各种格式文件（md/html/ipynb/xml/RSS），提取成"干净、可检索"的
#   纯文本/Markdown。这是索引构建五步流水线的第一步（对应《P1-索引构建.md》
#   2.1 节）。
#
# 【真实语料有哪些格式（别按"想象"来）】
#   .md/.mdx/.txt/.rst → 直接读文本（保留 Markdown 结构）
#   .html/.htm         → 多选择器提取正文，去掉导航/广告
#   .xml（RSS）        → 解析 <content:encoded>，保护 LaTeX 公式
#   .ipynb             → 它是 JSON，取 cells 里的内容
#   .tex               → LaTeX 源码，中文场景价值低，暂缓
#
# 【设计原则】
#   本模块是"无状态函数"：一个文件路径进，一段文本出。不碰目录遍历、不碰
#   数据库。这样全量和增量能复用同一套内核（对应文档 3.2 节）。
# ============================================================================

# ---------- 导入区 ----------
# json：Python 标准库，处理 JSON（ipynb 本质是 JSON）。
import json

# Path：Python 标准库 pathlib 的类，处理文件路径。
from pathlib import Path

# Optional：Python 类型标注工具（Optional[str] 表示"str 或 None"）。
# markdownify：第三方库，把 HTML 转成 Markdown（保留标题/代码块/表格结构）。
import markdownify

# BeautifulSoup：第三方库，解析 HTML、提取正文、剔除导航广告。
#   它是"HTML 解析"的事实标准工具。
from bs4 import BeautifulSoup

# 项目内：日志设施。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 核心入口：按扩展名分发到对应的解析函数
# ============================================================================
def parse_file(path: Path) -> str | None:
    """解析单个文件，返回文本（Markdown）。

    【原理】
        先看文件的后缀名（扩展名），判断它是哪类格式，再调用对应的解析函数。
        这叫"按格式分流"。

    【参数】
        path : Path —— 要解析的文件路径

    【返回】
        str  ：解析出的文本内容
        None ：该格式不支持（如 .tex），返回 None 表示"跳过"
    """
    # path.suffix 取扩展名（含点，如 ".md"）；.lower() 转小写（防止 ".MD" 大写）
    suffix = path.suffix.lower()

    # ---- 分支一：纯文本类格式，直接读文件 ----
    # .md/.mdx/.txt/.rst 本身就是文本，不需要解析库，直接读即可。
    if suffix in {".md", ".mdx", ".txt", ".rst"}:
        # read_text 读文件全部内容；encoding="utf-8" 防止 Windows 中文乱码
        # （对应文档 5.1 坑3）；errors="ignore" 忽略个别无法解码的字节。
        return path.read_text(encoding="utf-8", errors="ignore")

    # ---- 分支二：HTML 类格式，用 BeautifulSoup 提取正文 ----
    if suffix in {".html", ".htm"}:
        return _parse_html(path)

    # ---- 分支三：ipynb（Jupyter 笔记本），本质是 JSON ----
    if suffix == ".ipynb":
        return _parse_ipynb(path)

    # ---- 分支四：XML（RSS 订阅源）----
    if suffix == ".xml":
        return _parse_rss(path)

    # ---- 其余格式（如 .tex LaTeX 源码）暂缓处理，返回 None ----
    log.warning("跳过不支持的格式: %s", path)
    return None


# ============================================================================
# HTML 解析：提取正文，去掉导航/广告/侧栏
# ============================================================================
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
    # 读文件内容（utf-8，忽略个别编码错误字节）
    html_text = path.read_text(encoding="utf-8", errors="ignore")

    # BeautifulSoup(html_text, "lxml")：
    #   第 1 个参数：要解析的 HTML 字符串；
    #   第 2 个参数 "lxml"：指定底层解析引擎（更快、对脏 HTML 容错更好）。
    soup = BeautifulSoup(html_text, "lxml")

    # ---- 第 1 步：先删掉垃圾标签 ----
    # 常见的"非正文"区域：脚本、样式、导航、页脚、侧栏、广告。
    # select() 用 CSS 选择器批量选中；decompose() 把这些标签整个从文档移除
    # （连标签内部的内容一起删，不残留文字）。
    for tag in soup.select(
        "script, style, nav, footer, header, aside, "
        ".ad, .advert, .advertisement, .sidebar, .recommend, .related, .breadcrumb"
    ):
        tag.decompose()

    # ---- 第 2 步：多选择器取最长正文 ----
    candidates = []   # 收集所有候选正文，元素为 (文字长度, 标签)

    # 候选的正文容器选择器（覆盖主流文档站/技术站的结构）
    for selector in ["article", "main", ".content", ".post", ".article",
                     ".markdown-body", ".entry-content", "#content"]:
        # soup.select(selector) 返回所有匹配该选择器的标签（可能为空列表）
        for node in soup.select(selector):
            # node.get_text() 提取标签内所有文字（去掉 HTML 标签）
            text = node.get_text()
            # 只保留"够长"的候选（文字太短说明不是正文，可能是空容器）
            if len(text) > 100:
                candidates.append((len(text), node))

    # ---- 第 3 步：选定正文节点 ----
    if candidates:
        # max(candidates) 按元组第 1 个元素（长度）比较，返回最大者；
        # 再取 [1] 即那个标签本身。
        content_node = max(candidates)[1]
    else:
        # 没有明显正文容器，就用整个 soup（垃圾标签已在第 1 步删掉）
        content_node = soup

    # ---- 第 4 步：HTML → Markdown ----
    # markdownify.markdownify(...)：把 HTML 转成 Markdown。
    #   heading_style="ATX" 表示标题用 "#" 而非下划线风格。
    # 这样能保留标题层级、代码块、表格结构（对应文档 2.1.2 决策1）。
    return markdownify.markdownify(str(content_node), heading_style="ATX")


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
    # json.loads()：把 JSON 字符串解析成 Python 的字典/列表。
    # 先 read_text 读文件内容（字符串），再 json.loads 转成 Python 对象。
    notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    parts = []   # 用一个列表收集每块内容，最后统一拼接

    # notebook["cells"] 是所有 cell 组成的列表，遍历每一个 cell。
    # 用 .get("cells", []) 而非 ["cells"]，避免缺字段时抛 KeyError。
    for cell in notebook.get("cells", []):
        # cell 也是字典，cell_type 字段标识类型
        cell_type = cell.get("cell_type", "")

        # source 是内容，可能是字符串，也可能是字符串列表（每行一个元素）。
        source = cell.get("source", [])
        if isinstance(source, list):       # isinstance 判断类型：若是列表
            text = "".join(source)         # join 把列表里的字符串拼成一个字符串
        else:                              # 否则就是单个字符串
            text = source

        # ---- markdown cell：直接保留内容 ----
        if cell_type == "markdown":
            parts.append(text)

        # ---- code cell：用 ``` 包起来，标记为代码块 ----
        elif cell_type == "code":
            # 代码块用三个反引号包裹，前面加语言标记 python，方便后续渲染/识别
            parts.append("```python\n" + text + "\n```")

    # "\n\n".join(parts)：用两个换行符把所有块连起来，块之间留空行
    return "\n\n".join(parts)


# ============================================================================
# RSS/XML 解析：保护 LaTeX 公式
# ============================================================================
def _parse_rss(path: Path) -> str:
    """解析 RSS/XML 文件（如 kexue 科学空间的订阅源）。

    【原理（对应文档 2.1 节）】
        正文被 <content:encoded> 标签包着，且常带 $...$ 数学公式。
        用标准库 ElementTree 解析，关键是"不能弄坏公式"。

    【参数】
        path : Path —— XML 文件路径

    【返回】
        str —— 提取出的正文
    """
    # 导入 ElementTree（这里在函数内导入，因为只有本函数用到它，模块加载更轻）。
    import xml.etree.ElementTree as ET  # ElementTree：标准库，解析 XML

    # ET.parse() 解析 XML 文件，返回一棵树；.getroot() 取根节点。
    root = ET.parse(path).getroot()

    results = []   # 收集正文

    # XML 里有命名空间，标签常写成 {namespace}content:encoded 这种形式。
    # 用 .iter() 递归遍历整棵树的所有元素。
    for elem in root.iter():
        # 去掉命名空间前缀：elem.tag 可能形如 "{http://...}encoded"，
        # split("}") 按 "}" 切开取最后一段，得到纯标签名。
        tag_name = elem.tag.split("}")[-1]

        # 命中正文标签（RSS 常见 content:encoded / content）
        if tag_name in ("encoded", "content"):
            # elem.text 是标签内的文字内容。
            # 注意：这里"不替换、不修改"，原样取出，保护 $...$ 公式不被破坏。
            if elem.text:
                results.append(elem.text)

    # 用换行拼接所有正文段
    return "\n".join(results)
