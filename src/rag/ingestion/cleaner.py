# ============================================================================
# cleaner.py —— 文档清洗（Cleaning）：去噪 / 去重 / 筛选
#
# 位置：src/rag/ingestion/cleaner.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.1.2 节的"六步清洗流水线"中的去噪(③)、去重(④)、
#   筛选(⑤)三步。把 parser 输出的"脏文本"，洗成"干净、可入库"的标准文本。
#
#   六步流水线全貌：
#     ① 提取  ② 转文本  ③ 去噪  ④ 去重  ⑤ 筛选  ⑥ 输出
#   其中 ①② 由 parser.py 完成，③④⑤ 由本文件完成，⑥ 由 pipeline 完成。
#
# 【三个关键设计决策（对应文档 2.1.2）】
#   1. 保留 Markdown 结构，不压纯文本（标题/代码块/表格是结构信息）
#   2. 去重分两级：精确（SHA1）+ 近似（SimHash）——处理 github 的 fork 重复
#   3. 中文密度筛选：中文字符占比 < 0.3 判定为英文，过滤掉
# ============================================================================

# ---------- 导入区 ----------
# re：Python 标准库，正则表达式（匹配中文、匹配模式）。
import re

# Optional / Set / List：Python 类型标注工具。
#   Optional[str] 表示"str 或 None"；Set[str] 是"字符串集合"；List[int] 是"整数列表"。
# 项目内：内容哈希函数（复用 registry 的，保证全系统"同一把尺子"）。
from rag.ingestion.registry import compute_content_hash

# 项目内：日志设施。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 第一部分：去噪（denoise）—— 删导航、版权、目录页等噪音
# ============================================================================
def denoise(text: str) -> str:
    """去除文本里的噪音（导航、版权声明、纯链接目录页、多余空行）。

    【原理】
        刚解析出来的文本可能混着导航文字、版权声明、多余空行。
        这一步做"轻量清理"，让文本更干净，为后续分块打基础。

    【参数】
        text : str —— 待清理的文本

    【返回】
        str —— 清理后的文本（若是目录页则返回空字符串）
    """
    # ---- 第 1 步：去掉纯链接的目录页 ----
    # 目录索引页的特征：几乎全是链接，没什么正文。
    # re.findall(r"\[.*?\]\(.*?\)", text) 匹配 Markdown 链接 [文字](地址)。
    links = re.findall(r"\[.*?\]\(.*?\)", text)
    # 如果链接数量特别多（>20），判定为目录页 → 返回空串（后续筛选会丢弃）
    if len(links) > 20:
        return ""

    # ---- 第 2 步：去掉多余空行 ----
    # \n{3,} 匹配"3 个及以上的连续换行"，替换成 2 个换行（保留段落间距，去掉堆叠空行）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ---- 第 3 步：去掉常见的版权/声明行 ----
    # 按行处理，去掉包含关键词的行（版权、备案号等不是正文）。
    lines = []
    for line in text.split("\n"):           # split("\n") 按换行把文本切成一行一行
        stripped = line.strip()             # strip() 去掉行首行尾的空白
        # 若这一行包含版权关键词，就跳过（continue 不加入结果）
        if any(k in stripped for k in ["版权所有", "Copyright", "转载", "京ICP", "备案号"]):
            continue
        lines.append(line)

    # "\n".join(lines) 把处理过的行重新拼回一个字符串
    return "\n".join(lines)


# ============================================================================
# 第二部分：精确去重（SHA1）—— 内容哈希
# ============================================================================
# 说明：精确去重的"内容指纹"直接复用 registry.compute_content_hash，
# 不重复造轮子。下面定义的是"近似去重"所需的 SimHash。


# ============================================================================
# 第三部分：近似去重（SimHash + 汉明距离）
# ============================================================================
def _tokenize(text: str, n: int = 2) -> set[str]:
    """把文本切分成 n-gram（n 个连续字符组成的片段）集合。

    【原理】
        SimHash 需要把文本变成一组"特征"（token），再用特征算出指纹。
        n-gram 是最简单有效的切法：把文本每 n 个连续字符切一片。
        比如 "我爱北京" 的 2-gram 是 {"我爱", "爱北", "北京"}。

    【参数】
        text : str —— 待切分的文本
        n    : int —— n-gram 的 n，默认 2（二元组，中文去重常用）

    【返回】
        Set[str] —— 去重后的 n-gram 集合（集合天然去重，重复片段只算一次）
    """
    text = text.strip()                          # 去首尾空白
    if len(text) < n:                            # 文本太短（不够一个 n-gram）
        return {text} if text else set()         # 有内容就返回单元素集合，否则空集合
    # 列表推导式：text[i:i+n] 取从 i 开始的 n 个字符，i 从 0 到 len-n
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def simhash(text: str, hash_bits: int = 64) -> int:
    """计算文本的 SimHash 指纹（一个 64 位整数）。

    【原理（对应文档 2.1.2 决策、5.1 近似去重）】
        SimHash 是"局部敏感哈希"（LSH）的一种：内容相近的文本，算出的指纹
        也相近（汉明距离小）。这和 SHA1 正好相反——SHA1 是"雪崩"，改一个字
        指纹全变；SimHash 是"相似则相近"。

        算法流程：
        1. 把文本切成 n-gram；
        2. 对每个 n-gram 算一个 64 位哈希；
        3. 维护一个 64 维的"权重向量"，每个 n-gram 的哈希里，某位是 1 就 +1，
           是 0 就 -1；
        4. 最后权重向量里，正数位记 1，负数位记 0，拼成一个 64 位整数。

    【参数】
        text      : str —— 待计算文本
        hash_bits : int —— 指纹位数，默认 64

    【返回】
        int —— 64 位 SimHash 指纹
    """
    # 权重向量：64 个位置，初始都是 0
    weights = [0] * hash_bits

    # 第 1 步：遍历每个 n-gram
    for token in _tokenize(text):
        # 第 2 步：对 n-gram 算一个 64 位哈希（用内置 hash，再取绝对值避免负数）
        h = abs(hash(token))
        # 第 3 步：把哈希的每一位映射到权重向量（1 加 1，0 减 1）
        for i in range(hash_bits):
            # (h >> i) & 1 取出 h 的第 i 位（0 或 1）
            if (h >> i) & 1:
                weights[i] += 1    # 该位是 1 → 权重 +1
            else:
                weights[i] -= 1    # 该位是 0 → 权重 -1

    # 第 4 步：权重 > 0 的位置记 1，否则记 0，拼成指纹
    fingerprint = 0
    for i in range(hash_bits):
        if weights[i] > 0:
            fingerprint |= (1 << i)   # 把第 i 位置为 1（位运算）
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """计算两个整数的汉明距离（二进制表示中，不同位的个数）。

    【原理】
        SimHash 判重靠汉明距离：距离越小，文本越相似。
        计算方式：两个整数异或（XOR），结果里 1 的个数就是不同位的个数。
        （异或：相同位得 0，不同位得 1，所以结果里 1 的个数 = 不同位数量）

    【参数】
        a, b : int —— 两个 SimHash 指纹

    【返回】
        int —— 汉明距离（0~64）
    """
    return bin(a ^ b).count("1")   # a^b 异或；bin() 转二进制字符串；count("1") 数 1


# ============================================================================
# 第四部分：筛选（中文密度 + 正文长度）
# ============================================================================
def chinese_density(text: str) -> float:
    """计算文本的"中文密度"（中文字符占总字符的比例）。

    【原理（对应文档 2.1 决策1）】
        github 语料里混着纯英文仓库，英文内容对本项目价值低。
        用"中文字符占比"判断：占比低 = 英文内容 = 过滤掉。

    【参数】
        text : str —— 待统计的文本

    【返回】
        float —— 中文密度，范围 0.0 ~ 1.0
    """
    if not text:              # 空文本直接返回 0
        return 0.0
    # 统计中文字符：\u4e00-\u9fff 是 Unicode 里中文汉字的主要区间。
    # re.findall 返回所有匹配的中文字符组成的列表，len() 取数量。
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    # 密度 = 中文字符数 / 总字符数
    return len(chinese_chars) / len(text)


def filter_document(text: str, min_chars: int = 100, min_density: float = 0.3) -> str | None:
    """筛选：判断一篇文档是否"值得入库"。

    【两个淘汰条件（对应文档 2.1.2 第⑤步）】
        1. 正文过短（< min_chars 字符）：内容太少，无检索价值。
        2. 中文密度过低（< min_density）：判定为英文内容，过滤掉。

    【参数】
        text        : str   清洗后的文本
        min_chars   : int   最小正文字符数，默认 100
        min_density : float 最小中文密度，默认 0.3

    【返回】
        str  ：通过筛选，返回原文本
        None ：被过滤，返回 None
    """
    # 条件一：长度检查
    if len(text) < min_chars:
        log.debug("过滤：正文过短（%d 字符）", len(text))
        return None

    # 条件二：中文密度检查
    density = chinese_density(text)
    if density < min_density:
        log.debug("过滤：中文密度过低（%.2f）", density)
        return None

    return text   # 都通过了，原样返回


# ============================================================================
# 组合入口：clean —— 去噪 → 去重 → 筛选
# ============================================================================
def clean(text: str, seen_hashes: set[str]) -> str | None:
    """完整的清洗入口：去噪 → 精确去重 → 筛选。

    【原理】
        把前面几个小函数按顺序串起来，组成"六步清洗流水线"的后半段。

    【参数】
        text        : str       解析出的原始文本
        seen_hashes : Set[str]  已经见过的内容哈希集合（用于跨文档精确去重）

    【返回】
        str  ：清洗通过，返回清洗后的文本
        None ：被去噪/去重/筛选淘汰
    """
    # 第 1 步：去噪
    text = denoise(text)
    if not text.strip():        # strip() 后为空 = 去噪把它删空了（如目录页）
        return None

    # 第 2 步：精确去重（用哈希集合判断是否已出现过）
    h = compute_content_hash(text)   # 算内容哈希
    if h in seen_hashes:             # 已在集合里 = 重复内容
        log.debug("去重：内容重复，跳过")
        return None
    seen_hashes.add(h)               # 记录这个哈希，供后续文档判重

    # 第 3 步：筛选（长度 + 中文密度）
    filtered: str | None = filter_document(text)   # 显式 Optional，避免 mypy 推断冲突
    if filtered is None:                 # 筛选不通过，filter_document 返回 None
        return None
    text = filtered

    return text   # 返回清洗后的文本


# ============================================================================
# 近似去重工具类（SimHash 去重器，供 pipeline 在批量场景使用）
# ============================================================================
class SimHashDeduplicator:
    """基于 SimHash + 汉明距离的近似去重器。

    【为什么需要它（对应文档 2.1.2 决策3）】
        github 有大量 fork/镜像，同一份内容被复制成多个仓库，内容 99% 相同
        但 SHA1 不同（可能差一点）。精确去重抓不住这种"近似重复"。
        SimHash 通过"指纹相近"识别近似重复，用汉明距离做阈值判断。

    【用法】
        dedup = SimHashDeduplicator()
        dedup.is_duplicate(text)   # True 表示疑似重复
        dedup.add(text)            # 把一份文本的指纹记入库
    """

    def __init__(self, threshold: int = 3):
        """构造函数。

        参数：
            threshold : int —— 汉明距离阈值。两个指纹距离 <= threshold 判定为重复。
                               默认 3（64 位指纹里距离 3 以内算"近似相同"）。
        """
        self.threshold = threshold           # 汉明距离阈值
        self.fingerprints: list[int] = []    # 已入库的指纹列表（SimHash 整数指纹）

    def is_duplicate(self, text: str) -> bool:
        """判断文本是否与已入库的某篇"近似重复"。

        参数：
            text : str —— 待判断文本

        返回：
            bool —— True 表示疑似重复（汉明距离 <= 阈值）
        """
        fp = simhash(text)                        # 算该文本的 SimHash
        for existing in self.fingerprints:        # 逐个比对上库的指纹
            if hamming_distance(fp, existing) <= self.threshold:
                return True                       # 找到近似重复
        return False

    def add(self, text: str) -> None:
        """把一份文本的指纹记入去重库（后续文本会跟它比对）。

        参数：
            text : str —— 要记录的文本
        """
        self.fingerprints.append(simhash(text))
