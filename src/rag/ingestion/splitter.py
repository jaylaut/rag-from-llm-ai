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
            # 围栏标记之后的内容：只有空白才算"纯围栏行"。
            #   CommonMark 规定"关闭围栏行不能带 info string"（如 ```python 是开启，
            #   不是关闭）。不判这一点，代码块里的 ```python 片段会被误认成关闭标记，
            #   导致代码区间在中间被截断、后面的 # 注释被当成标题。
            trailing = line[fence.end():]
            if not in_fence:
                # 开启代码块
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                fence_start = offset
            elif (
                not trailing.strip()
                and marker[0] == fence_char
                and len(marker) >= fence_len
            ):
                #关闭代码块，要求同种字符且长度不少于开启标记：```` 开的必须 ```` 关
                in_fence = False
                # 区间是 (起点, 终点) 一个二元组，必须整体 append；
                #   写成 append(a, b) 会抛 TypeError: append() takes exactly one argument
                code_ranges.append((fence_start, offset + len(raw)))
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

def _build_parents(headings: list[dict],text_len: int,max_size:int) -> list[tuple[int,int]]:
    """构建父块区间：默认 ## 章节，超限则逐级降级，最后硬切兜底。

    【为什么父块默认取 ## 而不是 ###】
        文档 §2.6 明确"一个 ## 小节就是一个天然的块"。### 太细，父块会失去
        "整章上下文"的价值；# 太粗，一个 # 可能上万字，超过 PARENT_CHUNK_SIZE。

    【超限怎么办（三级降级）】
        1. 父块 <= PARENT_CHUNK_SIZE → 直接采用
        2. 超限 → 用下一级标题（###、####…）再拆，能拆出多块才采用
        3. 到 ###### 仍超限 → 按 PARENT_CHUNK_SIZE 硬切
           极端情况（一篇没有任何标题的 5 万字长文）下这是唯一出路，
           日志会告警，提示上游补标题结构。

    【参数】
        headings  : list[dict] —— 标题列表
        text_len  : int        —— 全文长度
        max_size  : int        —— 父块字符上限（来自 settings.PARENT_CHUNK_SIZE）

    【返回】
        list[tuple[int, int]] —— 父块区间列表，按文中顺序排列
    """
    # 第一层：按 # 和 ## 切分
    parents = _slice_by_level(headings,2,0,text_len)
    # 第二层： 对超过限制的父块，逐级尝试更新的标题
    for level in (3,4,5,6):
        refined: list[tuple[int,int]] = []
        for s,e in parents:
            if e - s <= max_size:
                # 没超限制，原样保留
                refined.append((s,e))
                # 【必须有 continue】没超限的块到此结束。少了它，同一区间会先 append 一次
                #   再被下面的 if/else 追加一次，父块整体重复一遍（子块数量凭空翻倍）。
                continue
            subs = _slice_by_level(headings,level,s,e)
            if len(subs) > 1:
                # 成功拆出多块，采用
                refined.extend(subs)
            else:
                # 这一级不拆，留个下一级
                refined.append((s,e))
        parents = refined
    # 第三层：仍超过限制：硬切，兜底方案
    final: list[tuple[int,int]] = []
    for s,e in parents:
        if e - s <= max_size:
            final.append((s,e))
            continue
        log.warning("父块超限且无可用标题降级（%d 字符 > %d），执行硬切", e - s, max_size)
        cursor = s
        while cursor < e:
            final.append((cursor,min(cursor + max_size,e)))
            cursor += max_size
    return final

def _heading_path_at(headings: list[dict],pos: int) -> tuple[list[str],int]:
    """查某个字符位置属于哪个标题（用于给子块标注章节路径）。

   【为什么要按位置查、而不是用父块的路径】
       父块是 ## 章节，但子块可能落在更深的 ### 小节里。按位置查最近一个
       已出现的标题，能拿到最精确的层级（如 ["安装","虚拟环境","Windows"]），
       检索时可以把章节标题一起拼进 embedding，提升召回。

   【参数】
       headings : list[dict] —— 按 start 升序的标题列表
       pos      : int        —— 子块的起始字符偏移

   【返回】
       (path, level)
       path  : list[str] —— 从一级到当前级的标题链；位置在第一标题之前则为 []
       level : int       —— 当前标题级别；前言块为 0
   """
    path: list[str] = []
    level = 0
    for h in headings:
        if h["start"] <= pos:
            #持续覆盖，循环结束时就是pos之前最后一个标题
            path = h["path"]
            level = h["level"]
        else:
            break
    return list(path),level
# ============================================================================
# 第三部分：语义边界切分（段落 → 句子 → 滑窗）
# ============================================================================
def _paragraph_ranges(text: str,start: int,end: int) -> list[tuple[int,int]]:
    """把区间按空行切成段落，返回每段的字符区间。

    【原理】
        Markdown 里"空行"就是段落分隔符（两个换行）。用正则找到所有分隔位置，
        分隔位置之间的内容就是一段。返回区间而不是字符串，是为了后面能精确
        计算 char_start/char_end，让每个块都能定位回原文。

    【参数】
        text  : str —— 全文
        start : int —— 区间起点
        end   : int —— 区间终点（右开）

    【返回】
        list[tuple[int, int]] —— 每个段落的 (起点, 终点)，已剔除纯空白段
    """
    ranges: list[tuple[int,int]] = []
    #当前段的起点
    pos = start

    # 只在区间内查找，不越界
    for m in _PARA_SEP_RE.finditer(text,start,end):
        if m.start() > pos:
            ranges.append((pos,m.start()))
        pos = m.end()

    # 收尾：最后一段,后面没有空行
    if pos < end:
        ranges.append((pos,end))

    # 收尾裁剪：段首尾的空白（空行留下的换行、行尾空格）不参与成块。
    #   不裁的话，chunk 文本会以 "\n" 开头、char_start 指向空行而非正文，
    #   块首的空行还会拖低中文密度、干扰 embedding。
    trimmed: list[tuple[int,int]] = []
    for s,e in ranges:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:                      # 只剩空白的段直接丢弃
            trimmed.append((s,e))
    return trimmed

def _split_long_text(
        text: str,
        start: int,
        end: int,
        chunk_size: int,
        overlap: int,
) -> list[tuple[int,int]]:
    """超长文本滑窗切分：优先切在句末标点，切不到才按字符硬切（带重叠）。

    【对应文档】
        「固定大小兜底：目标 512 字符、重叠 64」——只有走到这里才是兜底。

    【算法（每一步为什么这么做）】
        1. 先划定一个"窗口上限" limit = cursor + chunk_size
        2. 在窗口的"后半段 ~ 上限"之间找最后一个句末标点
           —— 只在前半段找会导致每块都很小；只在上限附近找又容易切在句中
        3. 找到 → 切在标点之后（语义完整）
        4. 没找到 → 切在 limit（纯字符硬切，说明这段没有标点，如日志/公式）
        5. 下一块从 cut - overlap 开始 → 重叠 64 字，给断口打补丁
        6. 强制 cursor 至少前进 1 字符，防止 overlap >= 块长时死循环

    【参数】
        text       : str —— 全文
        start      : int —— 超长段落起点
        end        : int —— 超长段落终点（右开）
        chunk_size : int —— 单块上限
        overlap    : int —— 重叠字符数

    【返回】
        list[tuple[int, int]] —— 切好的区间，长度均 <= chunk_size
    """
    ranges: list[tuple[int,int]] = []
    # 当前块起点
    cursor = start

    while cursor < end:
        # 本块的理论终点
        limit = min(cursor + chunk_size,end)
        if limit >= end:
            # 剩余内容不足一块 -> 直接收尾，不再切
            ranges.append((cursor,end))
            break
        # 搜索区间下限：至少切到一半，避免产出过碎的块
        search_from = max(cursor + chunk_size // 2,cursor + 1)
        # 搜索窗口 = [search_from, limit)。
        #   【必须切片，不能只取一个字符】写成 text[search_from] 拿到的是"单个字符"
        #   组成的字符串，finditer 最多只能命中 1 个标点，cut 永远等于 search_from+1，
        #   结果是每块只切出 1~2 个字符，块数爆炸（实测 5000 字文档能切出上千块）。
        #   上界必须是 limit 而不是文末：否则可能切到 limit 之后，块长突破 chunk_size。
        window = text[search_from:limit]

        cut = -1
        # 按照出现顺序遍历，最后一次赋值的m.end（）就是最后一个句末标点之后
        for m in _SENTENCE_END_RE.finditer(window):
            cut = search_from + m.end()
        if cut > cursor:
            # 切在标点后，语义完整
            ranges.append((cursor,cut))
            # 下一块回退 overlap 个字符形成重叠:max 保证一定前进
            cursor = max(cut - overlap,cursor + 1)
        else:
            # 没标点 ->硬切
            ranges.append((cursor,limit))
            cursor = max(limit - overlap,cursor + 1)
    return ranges

def _split_parent_to_children(
        text: str,
        start: int,
        end: int,
        chunk_size: int,
        overlap: int,
) -> list[tuple[int,int]]:
    """把一个父块切成若干子块：段落优先，短段合并，超长段走滑窗。

    【为什么要"合并短段"】
        一个父块里可能有十几个一行的小段落（列表项、短句）。如果每段都单独
        成块，会出现大量 20 字的碎片，向量库里全是噪声。所以按顺序贪心合并，
        只要合并后总长不超过 chunk_size 就继续并。

    【为什么合并用"原文切片"而不是字符串拼接】
        直接拼会引入分隔符，导致 char_start/char_end 与 text 对不上，回溯原文
        就失效了。用 (cur_start, 新段 end) 表示合并范围，text 字段直接取这段
        原文，保证 text == 原文[char_start:char_end]，永远可回溯。

    【超长段落为什么要"并进 cur 一起滑窗"，而不是各自独立切】
        典型场景：`## 环境准备` 后面紧跟一段 800 字的长文。
        · 独立切：标题这段 7 个字先被封口成一个独立 chunk，正文另起一块
          → 向量库里多出一条 7 字碎片，且标题与正文的语义被拆散。
        · 并进 cur：从标题起点开一个 chunk_size 的窗口，标题自然落在正文块
          开头，既不超限、又保留"标题 + 正文"的完整性。
        这正是"短段合并"在长段落边界上的延伸——只要两段在原文里连续，
        就可以当成一个整体做滑窗，不需要额外的拼接逻辑。

    【参数】
        text       : str —— 全文
        start      : int —— 父块起点
        end        : int —— 父块终点
        chunk_size : int —— 子块上限（来自 settings.CHUNK_SIZE）
        overlap    : int —— 重叠字符数（来自 settings.CHUNK_OVERLAP）

    【返回】
        list[tuple[int, int]] —— 子块区间列表
    """
    pieces:list[tuple[int,int]] = []
    #当前正在累积的块区间
    cur: tuple[int,int] | None = None

    for s,e in _paragraph_ranges(text,start,end):
        if e - s > chunk_size:
            # 超长段落：把手上的 cur 一起并进来，从 cur 的起点做滑窗。
            #   cur 为空时就从本段起点开始；cur 非空时 cur[0] < s，
            #   区间仍是原文连续切片，回溯性不受影响。
            window_start = cur[0] if cur is not None else s
            cur = None
            pieces.extend(_split_long_text(text, window_start, e, chunk_size, overlap))
            continue

        if cur is None:
            # 池子空，直接开一个新块
            cur = (s,e)
        elif e - cur[0] <= chunk_size:
            # 合并后仍不超限制，扩展右边界
            cur = (cur[0],e)
        else:
            # 已满，封口；用当前段开新块
            pieces.append(cur)
            cur = (s,e)
    if cur is not None:
        # 收尾，别漏最后一块
        pieces.append(cur)
    return pieces

# ============================================================================
# 第四部分：辅助判定（代码块标记 + 纯标题碎片判定）
# ============================================================================
def _has_prose(text: str) -> bool:
    """判断一个块里是否还有"标题以外"的实质内容。

    【为什么要这个判定】
        两个 `##` 之间如果只有一行 `# 大标题`，`_slice_by_level` 会切出一个
        只有标题行的父块，最终产出 5~10 个字符的碎片。这类块对检索毫无价值
        （它的信息已经完整写在兄弟块的 heading_path 里），却会占一条向量、
        在 top-k 里挤掉真正的内容。所以"纯标题块"不产出。

    【判断方式】
        逐行看：只要有一行既不是空行、也不匹配标题正则，就算有正文。
        代码围栏行（```）也算"非标题"，因此只含代码的块会被保留。

    【参数】
        text : str —— 块正文

    【返回】
        bool —— True 表示有实质内容；False 表示全是标题/空行
    """
    for line in text.splitlines():
        if not line.strip():
            continue
        if _HEADING_RE.match(line):
            continue
        return True
    return False


def _is_code_heavy(start:int,end:int,code_ranges:list[tuple[int,int]]) -> bool:
    """判断一个块是否"以代码为主"，用于给 is_code 打标。

    【为什么要标记（对应 parse.py 的契约）】
        parse.py 把 ipynb 的代码 cell 用 ``` 包裹，并注明"供下游 splitter
        识别 is_code 降权"。检索阶段遇到 is_code=True 的块会降权或不召回——
        代码对"学习"有价值，对"问答检索"价值低。

    【判定方式】
        计算块的字符区间与代码区间的重叠长度，占块长的比例 >= 50% 即判定为代码块。
        用比例而非"是否相交"，是为了避免"一句说明 + 一大段代码"被整体降权。

    【参数】
        start, end : int —— 块区间
        code_ranges: list[tuple[int,int]] —— 代码块区间

    【返回】
        bool —— True 表示代码为主
    """
    length = end - start
    if length <= 0 or not code_ranges:
        return False
    covered = 0
    for cs,ce in code_ranges:
        # 求区间交集[max(start,cs),min(end,ce)]  长度为正才累加
        lo = max(start,cs)
        hi = min(end,ce)
        if hi > lo:
            covered += hi - lo
    return covered / length >= _CODE_RATIO_THRESHOLD

# ============================================================================
# 第五部分：唯一对外入口
# ============================================================================
def split_document(
        text: str,
        *,
        doc_id: str,
        source: str = "",
        metadata: dict | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        parent_size: int | None = None,
) -> list[dict]:
    """把一篇 Markdown 文档切成 chunk 列表（本模块唯一对外入口）。

    【处理流程（六步）】
        ① 规范化换行（\\r\\n → \\n）
        ② 扫描结构：标题列表 + 代码块区间
        ③ 按标题层级切父块（## 章节；超 2048 降级到 ###/####/#####/######；仍超限硬切）
        ④ 每个父块内部：段落切分 → 短段合并 → 超长段滑窗（512/64）
        ⑤ 给每个子块补章节路径、父块引用、is_code、内容哈希
        ⑥ 返回 dict 列表（可直接 json.dumps 落盘或入库）

    【不产出的两种块（避免噪声进向量库）】
        · 纯空白块：两个标题之间只剩空行
        · 纯标题块：一块里除了标题行没有别的内容（信息已在 heading_path 里）

    【参数】
        text        : str        —— 清洗后的 Markdown 正文（cleaner.clean 的输出）
        doc_id      : str        —— 文档唯一标识，按架构用"源文件相对路径"
        source      : str        —— 来源标识（URL / 原始路径），写进元数据
        metadata    : dict|None  —— 额外元数据（title / category / url 等），原样带入
        chunk_size  : int|None   —— 子块上限；None 用 settings.CHUNK_SIZE (512)
        overlap     : int|None   —— 重叠字符数；None 用 settings.CHUNK_OVERLAP (64)
        parent_size : int|None   —— 父块上限；None 用 settings.PARENT_CHUNK_SIZE (2048)

    【返回】
        list[dict]，每个 dict 的字段：
          chunk_id      str        块 ID，格式 "{doc_id}:{四位序号}"，供注册表 chunk_ids 使用
          doc_id        str        所属文档
          index         int        全文档内从 0 开始的序号
          text          str        块正文，严格满足 text == 原文[char_start:char_end]
          char_start    int        在规范化后全文中的起点
          char_end      int        终点（右开）
          char_len      int        块长度，便于对账与调试
          heading_path  list[str]  章节路径，如 ["安装","Windows"]；前言为 []
          heading_level int        章节层级；前言为 0
          parent_id     str        父块 ID，格式 "{doc_id}:p{三位序号}"
          parent_text   str        父块全文（small-to-big：命中子块后直接返回它）
          is_code       bool       是否以代码为主（检索降权依据）
          content_hash  str        块内容的 SHA1（去重/增量更新判据）
          metadata      dict       传入的 metadata + source

    【异常】
        不抛业务异常。空文本返回 []；参数非法抛 ValueError（属于调用方 bug，
        应在开发期暴露，而不是静默产出错误结果）。
    """
    # ---- 参数解析：None 表示"用全局配置"，与 cleaner.py 的风格保持一致 ----
    resolved_chunk = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    resolved_overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    resolved_parent = settings.PARENT_CHUNK_SIZE if parent_size is None else parent_size

    # ---- 参数校验：fail fast，别切到一半才发现配置错了 ----
    if resolved_chunk <= 0:
        raise ValueError(f"chunk_size 必须为正数，收到 {resolved_chunk}")
    if resolved_overlap < 0:
        raise ValueError(f"overlap 不能为负，收到 {resolved_overlap}")
    if resolved_overlap >= resolved_chunk:
        # 重叠 >= 块长时，滑窗每次前进不足 1 字符，会退化成死循环
        raise ValueError(
            f"overlap({resolved_overlap}) 必须小于 chunk_size({resolved_chunk})"
        )
    if resolved_parent < resolved_chunk:
        # 父块比子块还小，父子关系就没意义了，属于配置错误
        raise ValueError(
            f"parent_size({resolved_parent}) 不能小于 chunk_size({resolved_chunk})"
        )
    if not doc_id:
        raise ValueError("doc_id 不能为空：它是 chunk_id 的前缀，也是注册表的索引键")

    # ---- ① 规范化换行 ----
    # 为什么不用 splitlines 处理：Windows 的 \r\n 在正则里会干扰 "$" 的匹配，
    # 先统一成 \n，后面所有偏移量都基于这份规范化文本（文档里会注明）
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # 空文档 / 纯空白：返回空列表，不抛异常（批处理里空文件很常见）
    if not normalized.strip():
        log.debug("doc_id=%s 内容为空，跳过分块", doc_id)
        return []

    # ---- ② 扫描结构 ----
    headings, code_ranges = _scan_structure(normalized)

    # ---- ③ 切父块 ----
    parents = _build_parents(headings, len(normalized), resolved_parent)

    # ---- ④⑤ 父块 → 子块，并组装输出 ----
    chunks: list[dict] = []
    seq = 0  # 子块全局序号：保证 chunk_id 唯一且可按序还原全文
    parent_seq = 0  # 父块序号：只对有效父块递增，避免空块导致编号跳号

    for p_start, p_end in parents:
        parent_text = normalized[p_start:p_end]
        if not parent_text.strip():
            # 标题之间夹着纯空白的父块（如两个 ## 之间只有空行），直接跳过
            continue

        parent_id = f"{doc_id}:p{parent_seq:03d}"
        parent_seq += 1

        for c_start, c_end in _split_parent_to_children(
                normalized, p_start, p_end, resolved_chunk, resolved_overlap
        ):
            chunk_text = normalized[c_start:c_end]
            if not chunk_text.strip():
                continue  # 空白片不产出
            if not _has_prose(chunk_text):
                # 纯标题碎片（如 "# 大章节" 独占一块）：信息已在兄弟块的
                #   heading_path 里，单独入库只会污染 top-k，直接丢弃。
                log.debug("doc_id=%s 丢弃纯标题碎片: %r", doc_id, chunk_text[:40])
                continue

            heading_path, heading_level = _heading_path_at(headings, c_start)

            chunks.append({
                "chunk_id": f"{doc_id}:{seq:04d}",
                "doc_id": doc_id,
                "index": seq,
                "text": chunk_text,
                "char_start": c_start,
                "char_end": c_end,
                "char_len": c_end - c_start,
                "heading_path": heading_path,
                "heading_level": heading_level,
                "parent_id": parent_id,
                "parent_text": parent_text,
                "is_code": _is_code_heavy(c_start, c_end, code_ranges),
                "content_hash": compute_content_hash(chunk_text),
                # metadata 展开后写 source，保证 source 以入参为准（不被脏元数据覆盖）
                "metadata": {**(metadata or {}), "source": source},
            })
            seq += 1

    log.info(
        "分块完成: doc_id=%s, 父块=%d, 子块=%d, 平均块长=%.1f",
        doc_id,
        parent_seq,
        len(chunks),
        (sum(c["char_len"] for c in chunks) / len(chunks)) if chunks else 0.0,
    )
    return chunks
