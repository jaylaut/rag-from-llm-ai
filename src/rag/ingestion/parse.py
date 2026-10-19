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

# 导入区
import json
from pathlib import Path

import markdownify
from bs4 import BeautifulSoup

from rag.utils.logger import setup_logger

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
    # 获取文件扩展名
    suffix = path.suffix.lower()
    # .md / .mdx / .txt / .rst 本身就是文本，不需要解析库，直接读。
    if suffix in {".md",".mdx",".txt",".rst"}:
        #read_text 读文件全部内容，encoding = utf-8防止中文乱码
        return path.read_text(encoding="utf-8",errors="ignore")

    # html格式
    if suffix in {".html",".htm"}:
        return _parse_html(path)

    # ipynb 格式
    if suffix == ".ipynb":
        return _parse_ipynb(path)
    #xml格式（RSS订阅源）
    if suffix == ".xml":
        return _parse_rss(path)
    # 其余格式暂不处理，返回None
    log.warning("跳过不支持的格式:%s",path)
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
    # 读文件内容：utf-8
    html_text = path.read_text(encoding="utf-8",errors="ignore")

    # BeautifulSoup（html_text,"lxml）
    # html_text:待解析的HTML字符串
    # lxml:指定底层解析引擎
    soup = BeautifulSoup(html_text,"lxml")

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
            # 只保留一定长度的候选
            if len(text) > 100:
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
    return markdownify.markdownify(str(content_node),heading_style="ATX")

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
    notebook = json.loads(path.read_text(encoding="utf-8",errors="ignore"))

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
    # 导入ElementTree(函数内导入，因为只有本函数用到，模块加载更轻)
    import xml.etree.ElementTree as ET
    # 返回树形结构，.getroot()获取根节点
    root = ET.parse(path).getroot()
    #收集正文
    results = []
    #XML有命名空间，标签常写成：{namespace}content:encoded 这种形式。用iter()递归遍历整棵树的所有元素
    for elem in root.iter():
        # 去掉命名空间前缀
        tag_name = elem.tag.split("}")[-1]
        # 命中正文标签
        if tag_name in ("encoded","content"):
            #elem.text是标签内的文字内容。这里不替换，不修改，原样取出
            if elem.text:
                results.append(elem.text)
    # 用换行拼接所有正文段
    return "\n".join(results)
