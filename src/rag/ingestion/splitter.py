# ============================================================================
# splitter.py —— 分块（Chunking）：按标题切 + 段落对齐 + 固定大小兜底 + 父子块
#
# 位置：src/rag/ingestion/splitter.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.2 节。把一整篇长文章，切成一段段"大小合适"的
#   文字块，供后续嵌入和检索。
#
# 【为什么必须切（三个原因）】
#   1. 文章太长，一次性塞给模型，模型"记不住"也"抓不住重点"
#   2. 检索时想返回"最相关的那一小段"，而不是"一整篇万言书"
#   3. 切得越碎检索越精准（但要保证每块意思完整）
#
# 【分块策略（按优先级，对应文档 2.2）】
#   1. 按标题层级切（首选）：## 小节是天然的语义边界
#   2. 段落对齐（次选）：没标题的纯文本在段落/句子结束处切
#   3. 固定大小兜底（最后）：超长段落才滑窗硬切（512 字符、重叠 64）
#
# 【父子块（Parent-Child / small-to-big）—— 核心设计】
#   矛盾：检索希望块"小"（聚焦精准），生成希望块"大"（上下文完整）。
#   解法：小块（子块）做嵌入去检索，命中后返回它所属的整个章节（父块）。
# ============================================================================

# ---------- 导入区 ----------
# re：Python 标准库，正则表达式（匹配标题行）。
import re

# dataclass：Python 标准库，定义"数据容器"类。
from dataclasses import dataclass

# List / Optional：Python 类型标注工具（List[X] 是"X 组成的列表"）。
# 项目内：日志设施。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 数据类：块（Chunk）
# ============================================================================
@dataclass
class Chunk:
    """一个"块"（Chunk）—— 检索的基本单位。

    【字段】
        chunk_id  : str  块的唯一标识（"文档id#c序号" 拼成，全局唯一）
        text      : str  块的内容（子块 = 小块，用于嵌入检索）
        parent_id : str  所属父块的 id（父子块关系靠这个字段建立）
    """
    chunk_id: str
    text: str
    parent_id: str


# ============================================================================
# 策略一：按标题层级切（首选）
# ============================================================================
def split_by_heading(markdown: str) -> list[str]:
    """按标题层级切分 Markdown（首选策略）。

    【原理（对应文档 2.2）】
        Markdown 里的 #、##、### 是现成的"语义边界"。一个 ## 小节天然是
        一个完整意思，切出来语义完整、检索精准。

    【参数】
        markdown : str —— 完整 Markdown 文本

    【返回】
        List[str] —— 切成的一段段文本（每段是一个标题小节）
    """
    # 正则说明：
    #   ^#{1,6}\s  匹配"行首的 1~6 个 # 后跟一个空格"（即标题行）
    #   (?=...)     正向先行断言：只在"标题行开始的位置"切分，不消耗字符
    #   re.MULTILINE 让 ^ 匹配每一行的行首（而不只是整个字符串的开头）
    # re.split 按标题行位置把文本切开，返回一个列表。
    parts = re.split(r"^(?=#{1,6}\s)", markdown, flags=re.MULTILINE)

    # 去掉空段（标题之前的空内容、连续标题产生的空段）
    return [p.strip() for p in parts if p.strip()]


# ============================================================================
# 策略三：固定大小滑窗切（兜底，只对超长段落用）
# ============================================================================
def split_by_window(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """固定大小滑窗切分（兜底策略，只对超长段落用）。

    【原理（对应文档 2.2）】
        目标 512 字符、重叠 64 字符。滑窗 = 每次前进 (chunk_size - overlap) 步，
        让相邻两块有 64 字重叠，防止把完整意思"切在中间"。

    【参数】
        text       : str   超长文本
        chunk_size : int   每块目标字符数，默认 512
        overlap    : int   相邻块重叠字符数，默认 64

    【返回】
        List[str] —— 切好的块
    """
    chunks = []                              # 存结果
    step = chunk_size - overlap              # 每次滑动的步长（512-64=448）

    # range(0, len(text), step)：从 0 开始，每次前进 step，直到文本末尾
    for start in range(0, len(text), step):
        # 取一段：从 start 到 start+chunk_size（切片自动处理越界，不会报错）
        chunks.append(text[start:start + chunk_size])
    return chunks


# ============================================================================
# 主入口：split —— 标题切 + 段落对齐 + 兜底滑窗，产出父子块
# ============================================================================
def split(
    markdown: str,
    doc_id: str,
    chunk_size: int = 512,
    overlap: int = 64,
    parent_chunk_size: int = 2048,
) -> list[Chunk]:
    """完整分块流程，产出"父子块"结构。

    【父子块原理（对应文档 2.2）】
        检索希望块"小"（聚焦精准），生成希望块"大"（上下文完整）。
        解法：小段（子块）做嵌入去检索，命中后返回它所属的整个章节（父块）。

        实现：
        - 父块 = 一个完整的标题小节（## 章节）
        - 子块 = 父块里更小的一段（段落、或滑窗切出的片段）
        - 每个子块记录 parent_id，指向它的父块

    【参数】
        markdown          : str  完整 Markdown 文本
        doc_id            : str  文档唯一标识（源文件相对路径）
        chunk_size        : int  子块最大字符数，默认 512
        overlap           : int  滑窗重叠字符数，默认 64
        parent_chunk_size : int  父块大小上限，默认 2048（超出则父块也滑窗切）

    【返回】
        List[Chunk] —— 子块列表（检索用这些；父块信息通过 parent_id 关联）
    """
    chunks: list[Chunk] = []                            # 存所有子块
    heading_sections = split_by_heading(markdown)       # 第 1 步：按标题切成小节

    # 遍历每个标题小节，作为父块
    for sec_idx, section in enumerate(heading_sections):
        # ---- 父块也可能超长（如一个超长 ## 章节），需切分成多个父块 ----
        parent_segments = (
            split_by_window(section, parent_chunk_size, 0)   # overlap=0，父块不重叠
            if len(section) > parent_chunk_size
            else [section]
        )

        # 遍历父块（通常只有 1 个，超长时才多个）
        for parent_seg in parent_segments:
            # 父块 id：doc_id + "#p" + 序号，保证全局唯一
            parent_id = f"{doc_id}#p{sec_idx}"

            # 第 2 步：父块内部再切成更小的"段落"（按空行切）
            paragraphs = [p.strip() for p in parent_seg.split("\n\n") if p.strip()]

            # 遍历每个段落，作为子块的候选
            for paragraph in paragraphs:
                # 第 3 步：若段落太长，用滑窗兜底切；否则整段作为一个子块
                if len(paragraph) > chunk_size:
                    sub_texts = split_by_window(paragraph, chunk_size, overlap)
                else:
                    sub_texts = [paragraph]

                # 把每个子块包装成 Chunk 对象
                for sub_text in sub_texts:
                    # 子块 id 用递增序号，保证唯一
                    child_id = f"{doc_id}#c{len(chunks)}"
                    chunks.append(Chunk(
                        chunk_id=child_id,    # 子块自己的 id
                        text=sub_text,        # 子块内容（嵌入检索用）
                        parent_id=parent_id,  # 指向父块（父子块关系的关键）
                    ))

    return chunks


# ============================================================================
# 辅助：提取父块内容（检索命中子块后，返回父块给生成用）
# ============================================================================
def build_parent_map(markdown: str, doc_id: str, parent_chunk_size: int = 2048) -> dict:
    """构建 parent_id → 父块文本 的映射。

    【为什么需要它】
        检索命中"子块"后，要把它所属的"父块"（完整章节）返回给大模型。
        所以需要一张表，能通过 parent_id 快速查到父块的完整文本。

    【参数】
        markdown          : str  完整 Markdown 文本
        doc_id            : str  文档标识
        parent_chunk_size : int  父块大小上限

    【返回】
        dict —— {parent_id: 父块文本}
    """
    parent_map = {}
    heading_sections = split_by_heading(markdown)

    for sec_idx, section in enumerate(heading_sections):
        # 父块超长则切分，否则整体作为一个父块
        if len(section) > parent_chunk_size:
            segments = split_by_window(section, parent_chunk_size, 0)
        else:
            segments = [section]

        for parent_seg in segments:
            parent_id = f"{doc_id}#p{sec_idx}"
            parent_map[parent_id] = parent_seg   # 记录父块文本

    return parent_map
