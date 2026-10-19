# ============================================================================
# splitter.py —— 分块（Chunking）：按标题切 + 段落对齐 + 固定大小兜底 + 父子块
#
# 位置：src/rag/ingestion/splitter.py
# 对应：《项目架构.md》§2.6、《P1-索引构建.md》2.2 节
#
# 【本文件是干什么的】
#   把 cleaner.py 洗好的「标准 Markdown 正文」，切成一段段大小合适、语义完整的
#   文字块（chunk），供嵌入模型和向量库使用。这是索引构建流水线的第三步。
#
# 【分块策略（按优先级，严格对应文档 2.2）】
#   1. 按标题层级切（首选）：#/##/### 是天然的语义边界，先按标题断开
#   2. 语义边界对齐（次选）：无标题的段落，在空行/句末标点处切，不拦腰斩断
#   3. 固定大小兜底（最后）：只有超长段落才滑窗硬切（512 字符、重叠 64）
#      —— 注意这是"最后手段"，不是默认策略
#
# 【父子块（Parent-Child / small-to-big）—— 核心设计】
#   矛盾：检索希望块"小"（聚焦精准），生成希望块"大"（上下文完整）。
#   解法：子块（段落级）做嵌入与检索；命中后返回它所属的父块（整个 ## 章节）
#         给生成阶段。本文件同时输出 parent_id 和 parent_text，供 P3 直接取用。
#
# 【对外契约（唯一入口）】
#   split_document(text, doc_id=..., source=...) -> list[dict]
#   输入：cleaner.clean() 返回的 Markdown 字符串
#   输出：chunk 字典列表，字段见 split_document 的 docstring
#
# 【设计原则（对应文档 3.2）】
#   本模块是"无状态函数"：一篇文档进，chunk 列表出。不遍历目录、不读写文件、
#   不持有跨文档状态——全量索引和增量索引才能复用同一套内核。
# ============================================================================
# re：标准库，正则。用于扫标题、找段落分隔、找句末标点、判定代码围栏。
import re
from msilib import make_id

# settings：全局配置。块大小/重叠/父块上限都从这里取，不在本文件写死魔法数字。
from rag.config.settings import settings

# compute_content_hash：内容指纹（SHA1）。复用注册表同一套算法，保证
#   "入库去重"和"变更检测"用的是同一个指纹，避免两套口径对不上。
from rag.ingestion.registry import compute_content_hash

# setup_logger：全项目统一的日志入口（控制台 + 文件双通道）。
from rag.utils.logger import setup_logger

log = setup_logger()
# ============================================================================
# 第一部分：结构扫描（标题 + 代码块区间）
# ============================================================================
# 标题正则（严格遵循 CommonMark）：
#   ^ {0,3}  行首最多 3 个空格（缩进 4 格以上是代码块，不是标题）
#   (#{1,6}) 1~6 个 #，捕获组 1 拿到级别
#   [ \t]+   # 后面必须跟空格/Tab
#   (.+?)    捕获组 2 是标题文字（非贪婪，避免吞掉尾部装饰）
#   [ \t]*#*[ \t]*$  允许标题尾部有 "###" 这种闭合装饰
#
# 【为什么要求 # 后必须有空格】
#   C 代码里的 "#include <stdio.h>" 若被当成标题，会把整个代码块切碎。
#   CommonMark 规定标题的 # 后必须有空格，正好帮我们挡掉这类误判。
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

# 代码围栏正则：``` 或 ~~~，至少 3 个，允许行首缩进 0~3 格。
#   捕获组拿到整串标记，用来判断"开"和"关"是否是同一种围栏、长度够不够。
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

# 段间分隔：两个换行，中间允许夹空格/Tab（如 "  \n  \n"）。
#   不用 "\n\n" 硬匹配，是为了兼容从 HTML 转过来、行尾带空格的脏文本。
_PARA_SEP_RE = re.compile(r"\n[ \t]*\n")

# 句末标点：中文句号/叹号/问号/分号 + 英文对应 + 换行。
#   超长段落切分时，优先切在这些位置，避免把一句话劈成两半。
_SENTENCE_END_RE = re.compile(r"[。！？；!?;\n]")

# 代码块覆盖率阈值：一个块里有 >=50% 的字符落在代码围栏内，就判定 is_code。
#   为什么是 50% 而不是"只要碰到代码就算"：Markdown 段落常以"```python"
#   开头紧跟一句说明，整块仍以自然语言为主，不该整体降权。
_CODE_RATIO_THRESHOLD = 0.5

# 中文密度统计用的汉字正则（与 cleaner.chinese_density 保持同一区间）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _scan_structure(text: str) -> tuple[list[dict],list[tuple[int,int]]]:
    """扫描 Markdown，产出「标题列表」和「代码块区间列表」。

            【为什么必须先扫一遍、而不是边切边判断】
                Markdown 的 # 有两种身份：标题、代码里的注释。二者只能靠"当前是否
                在代码围栏内"区分。所以必须先完整走一遍文本，建立"哪些区间是代码"
                的地图；后续所有切分都查这张地图，不会误判。

            【算法】
                逐行遍历（splitlines(keepends=True) 保留换行，保证偏移量精确）：
                  · 遇到围栏行 → 翻转 in_fence 状态，并记录区间
                  · 在围栏内 → 一律不当标题
                  · 在围栏外 → 用 _HEADING_RE 匹配标题
                同时用一个"标题栈"算出每个标题的完整路径（# → ## → ###）。

            【参数】
                text : str —— 已规范换行符（\\r\\n → \\n）的 Markdown 正文

            【返回】
                headings     : list[dict]，每项含
                               level(int) / title(str) / start(int) / end(int) / path(list[str])
                               start 是标题行首字符偏移，end 是标题行末（含换行）
                code_ranges  : list[tuple[int, int]]，每个元素是 (代码块起点, 终点)，
                               左闭右开；未闭合的围栏一直算到文末
            """
    #标题收集器
    headings: list[dict] = []
    #代码区间收集器
    code_ranges:list[tuple[int,int]] = []
    in_fence = False                # 当前是否在代码围栏内
    fence_char = ""                 # 围栏用的是` 还是 ~
    fence_len = 0                   # 围栏的长度
    fence_start = 0                 # 当前代码块的起始偏移

    # 标题栈:[(级别,标题)]:用于算层级路径
    stack:list[tuple[int,str]] = []
    offset = 0                      # 当前行在全文中的起始偏移

    #keepends=True:每行保留自己的换行符，这样len(raw)就是这一行的真实长度，offset 累加才不会漂移,否则char_start/char_end 全错
    for raw in text.splitlines(keepends=True):
        # 丢掉行尾换行，便于正则匹配
        line = raw.rstrip("\n")
        fence = _FENCE_RE.match(line)
        # 判断这一行是否围栏
        if fence:
            marker = fence.group(1)
            if not in_fence:
                # 开启代码块
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                fence_start = offset
            elif marker[0] == fence_char and len(marker) >= fence_len:
                #关闭代码块，要求同种字符且长度不少于开启标记：```` 开的必须 ```` 关
                in_fence = False
                code_ranges.append(fence_start,offset + len(raw))
            offset += len(raw)
            continue

        if not in_fence:
            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                # 维护标题栈，算出完整路径
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level,title))
                headings.append({
                    "level":level,
                    "title":title,
                    "start":offset,
                    "end":offset + len(raw),
                    "path": [t for _,t in stack],
                })
        offset += len(raw)
    if in_fence:
        log.warning("检测到未闭合的代码围栏，剩余 %d 字符按代码处理", len(text) - fence_start)
        code_ranges.append((fence_start,len(text)))
    return headings,code_ranges
# ============================================================================
# 第二部分：按标题层级切"父块"
# ============================================================================
def _slice_by_level(
        headings: list[dict],
        level: int,
        start: int,
        end: int,
) -> list[tuple[int,int]]:
    """在 [start, end) 区间内，按"级别 <= level"的标题切分。

    【为什么用"级别 <= level"而不是"等于 level"】
        切 ## 时，# 也是边界（# 代表更大的新章节）。用 <= 能把 # 一起纳入，
        避免一个 ## 块跨越到另一个 # 章节里去。

    【参数】
        headings : list[dict] —— _scan_structure 的产物（必须按 start 升序）
        level    : int        —— 切分所依据的最大标题级别
        start    : int        —— 区间起点（字符偏移）
        end      : int        —— 区间终点（字符偏移，右开）

    【返回】
        list[tuple[int, int]] —— 若干 (块起点, 块终点)。
        若区间内一个标题都没有，返回整段 [(start, end)]；
        区间开头若有"第一个标题之前"的内容，单独成一块（前言块）。
    """
    marks = [h for h in headings if h["level"] <= level and start <= h["start"] < end]
    if not marks:
        # 没有任何标题 -> 整段就是一个块
        return [(start,end)] if end > start else []
    ranges: list[tuple[int,int]] = []
    # 第一个标题之前的内容
    if marks[0]["start"] > start:
        ranges.append((start,marks[0]["start"]))

    # 每个标题到下一个标题之间就是一块
    for i,h in enumerate(marks):
        block_start = h["start"]
        block_end = marks[i + 1]["start"] if i + 1 < len(marks) else end
        ranges.append((block_start,block_end))
    return ranges
