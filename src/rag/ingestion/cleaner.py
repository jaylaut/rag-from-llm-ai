# ============================================================================
# cleaner.py —— 文档清洗（Cleaning）：去噪 / 去重 / 筛选
#
# 位置：src/rag/ingestion/cleaner.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.1.2 节的"六步清洗流水线"中的去噪(③)、去重(④)、
#   筛选(⑤)三步。把 parse.py 输出的"脏文本"，洗成"干净、可入库"的标准文本。
#
#   六步流水线全貌：
#     ① 提取   ② 转文本   ③ 去噪   ④ 去重   ⑤ 筛选   ⑥ 输出
#   其中 ①② 由 parse.py 完成，③④⑤ 由本文件完成，⑥ 由 pipeline 完成。
#
# 【五个关键设计决策（对应文档 2.1.2 + 真实语料校准）】
#   1. 保留 Markdown 结构，不压纯文本（标题/代码块/表格是结构信息）
#   2. 去重分两级：精确（SHA1）+ 近似（SimHash）——处理 github 的 fork 重复
#   3. 目录页判定用"链接数 + 剩余正文量"双条件，不再只数链接
#      实测：技术长文的"参考链接/相关阅读"动辄 20+ 条，只数链接会把整篇长文
#      误判成目录页丢掉，而日志里只表现为"通过率下降"，极难定位。
#   4. 中文密度/正文长度阈值来自 settings（MIN_CHINESE_RATIO / MIN_TEXT_LEN），
#      不在本文件里写死——settings.py 第六部分就是给 cleaner 留的旋钮。
#   5. SimHash 指纹必须跨进程稳定：用 blake2b，禁用内置 hash()
#      内置 hash() 对 str 带 PYTHONHASHSEED 随机盐，同进程稳定、换进程就漂移；
#      指纹一旦落盘，重启后会把已入库文档判成新内容。
#
# 【算法版本（落盘必读）】
#   SIMHASH_ALGO_VERSION 是"这批指纹能不能互相比"的唯一依据。切词方式、
#   哈希函数、权重算法任一变化，旧指纹全部作废，必须全量重算，不能混比。
# ============================================================================

# ---------- 导入区 ----------
# hashlib  ：标准库，blake2b 摘要算法（确定性哈希，跨进程/跨机器一致）
# re       ：标准库，正则（找链接、压空行、统计汉字）
# Counter  ：统计 n-gram 频次（SimHash 的权重来源）
# Iterator ：给生成器函数写返回值类型
# lru_cache：给 token 哈希加缓存，跨文档复用高频片段
import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache

# settings：全局配置（清洗阈值、目录页判定阈值都从这里取，避免魔法数字）
# compute_content_hash：精确去重的内容指纹（SHA1），复用注册表同一套算法，
#   保证"构建期去重"和"运行期变更检测"用的是同一个指纹，一致性成本最低。
from rag.config.settings import settings
from rag.ingestion.registry import compute_content_hash
from rag.utils.logger import setup_logger

log = setup_logger()


# ---------------------------------------------------------------------------
# 去噪相关常量
# ---------------------------------------------------------------------------
# Markdown 行内链接：[文字](地址)
# 注意：这个正则不处理"URL 里带右括号"的极端情况（如 wiki 链接），
# 真实语料里占比极低，为可读性不做兼容。
LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")

# 版权/备案/转载行的关键词（命中即整行删除）
NOISE_KEYWORDS = ("版权所有", "Copyright", "转载", "京ICP", "备案号")


# ============================================================================
# 第一部分：去噪（denoise）—— 删导航、版权、目录页等噪音
# ============================================================================
def _is_directory_page(text: str, min_links: int, max_prose: int) -> bool:
    """判断是否为"只有链接列表"的目录索引页。

        【判定规则（对应《P1-索引构建.md》2.1.2 第③步）】
            条件1：链接数 >= min_links
            条件2：去掉所有链接后，剩下的"纯正文"字符数 < max_prose
            两条同时成立才算目录页。

        【为什么不能只数链接】
            技术长文的"参考链接/相关阅读"动辄 20+ 条，只数链接会把整篇长文
            误判成目录页丢掉（旧实现 `len(links) > 20` 就 return ""）。
            真正区分目录页的是"没有正文"，不是"链接多"。

        【参数】
            text      : str —— 待判定文本
            min_links : int —— 链接数下限，低于它直接放行
            max_prose : int —— 去掉链接后正文长度上限，超过它就放行

        【返回】
            bool —— True 表示判定为目录页，应整篇丢弃
        """
    if len(LINK_RE.findall(text)) < min_links:
        return False

    # 去链接 → 再去掉空白与 Markdown 装饰符（# 标题号 / * - 列表符 / > 引用 / | 表格）
    prose = LINK_RE.sub("", text)
    prose = re.sub(r"[\s#>*\-|`]+", "", prose)
    return len(prose) < max_prose


def denoise(
    text: str | None,
    *,
    min_links: int | None = None,
    max_prose: int | None = None,
) -> str:
    """去除文本里的噪音（纯链接目录页、版权声明、多余空行）。

       【处理顺序（顺序本身是契约，测试与冒烟脚本都依赖它）】
           1. 目录页 → 整篇丢弃（返回空串）
           2. 3 个及以上连续换行 → 压成 2 个换行（保留段落间距）
           3. 版权/备案/转载行 → 整行删除

       【参数】
           text      : str | None —— 待清理文本；None/空串返回空串
           min_links : int | None —— 目录页链接数下限，None 用 settings 默认
           max_prose : int | None —— 目录页正文长度上限，None 用 settings 默认

       【返回】
           str —— 清理后的文本；判定为目录页时返回空字符串
       """
    if not text:
        return ""

    resolved_min_links = (
        settings.CLEAN_DIR_PAGE_MIN_LINKS if min_links is None else min_links
    )
    resolved_max_prose = (
        settings.CLEAN_DIR_PAGE_MAX_PROSE if max_prose is None else max_prose
    )

    # ---- 第 1 步：目录页判定（链接多 + 正文少）----
    if _is_directory_page(text, resolved_min_links, resolved_max_prose):
        log.debug("去噪: 判定为目录索引页, 整篇丢弃")
        return ""

    # ---- 第 2 步：压掉堆叠空行 ----
    # \n{3,} 匹配"3 个及以上连续换行"，替换成 2 个换行（保留段落间距）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ---- 第 3 步：逐行剔除版权/声明行 ----
    lines = []
    for line in text.split("\n"):
        if any(k in line.strip() for k in NOISE_KEYWORDS):
            continue
        lines.append(line)
    return "\n".join(lines)


# ============================================================================
# 第二部分：精确去重（SHA1）—— 内容指纹
# ============================================================================
# 说明：精确去重的"内容指纹"直接复用 registry.compute_content_hash，
# 不重复造轮子。下面定义的是"近似去重"所需的 SimHash。


# ============================================================================
# 第三部分：近似去重（SimHash + 汉明距离）
# ============================================================================
def _ngrams(text: str, n: int = 2) -> Iterator[str]:
    """按 n 个连续字符切分文本，逐个产出 n-gram（不去重、不丢频次）。

        【为什么抽成生成器】
            set 版（_tokenize）和频次版（_token_frequencies）都要"怎么切"，
            切法只能有一份实现，否则两处一旦改歪，指纹就静默漂移。
            生成器本身不占内存，交给上层决定用 set 还是 Counter 收敛。

        【参数】
            text : str —— 待切分文本
            n    : int —— n-gram 的 n，默认 2（二元组，中文去重常用）

        【产出】
            str —— 每个 n-gram；文本短于 n 时产出整串一次
        """
    text = text.strip()
    if not text:
        return
    if len(text) < n:
        yield text
        return
    for i in range(len(text) - n + 1):
        yield text[i : i + n]


def _tokenize(text: str, n: int = 2) -> set[str]:
    """n-gram 集合（频次不敏感的老语义，供测试/调试使用）。

        simhash 已改用 _token_frequencies（频次加权），本函数保留是为了
        "想看一眼切出来哪些片段"时有个直观入口，且语义与旧版一致。
        """
    return set(_ngrams(text, n))


def _token_frequencies(text: str, n: int = 2) -> Counter[str]:
    """n-gram 频次表：同一个片段出现几次就记几次。

        【为什么要频次】
            纯 set 把"出现 1 次"和"出现 20 次"视为等价，主题词权重被抹平，
            两篇只是同领域、措辞不同的文档容易算出相近指纹 → 误杀。
            频次加权是 SimHash 的标准做法：重复出现的片段才代表主题。
        """
    return Counter(_ngrams(text, n))


# ---------------------------------------------------------------------------
# 稳定哈希：token → 确定性的 hash_bits 位整数
# ---------------------------------------------------------------------------
# 为什么不用内置 hash()：str 的 hash 带进程级随机盐（PYTHONHASHSEED），
#   同进程稳定、跨进程漂移。指纹要落盘/跨进程比对，必须与进程无关。
# 为什么用 blake2b：纯标准库、比 md5/sha1 快、digest_size 可精确控字节数。
_BLAKE2B_MAX_BITS = 512          # blake2b 摘要最长 64 字节 = 512 位

# 算法版本号：切词方式 / 哈希函数 / 权重算法任一变化，必须改这个串。
SIMHASH_ALGO_VERSION = "blake2b-64-2gram-freq-v2"


@lru_cache(maxsize=1 << 16)
def _stable_hash(token: str, hash_bits: int) -> int:
    """把 token 映射成跨进程、跨机器一致的 hash_bits 位整数。

        【为什么要 lru_cache】
            一份 5000 字文档约 5000 个 n-gram，但跨文档的高频片段（"的""我们"
            "install""python"）会反复命中。缓存后每篇只为"新出现"的片段付哈希
            成本，多文档批量场景实测可省 30%~50%。
            缓存键含 hash_bits，不同位宽不会串味；maxsize=65536 约占 6~8MB，
            语料更大时可调到 1<<18。

        【参数】
            token     : str —— 一个 n-gram 片段
            hash_bits : int —— 需要的位数（1~512）

        【返回】
            int —— 非负整数，低 hash_bits 位可用
        """
    if not 1 <= hash_bits <= _BLAKE2B_MAX_BITS:
        raise ValueError(f"hash_bits 必须在 1~{_BLAKE2B_MAX_BITS} 之间，收到 {hash_bits}")

    digest_size = (hash_bits + 7) // 8          # 向上取整到字节
    digest = hashlib.blake2b(
        # surrogatepass：真实语料偶有孤立代理字符，不加会抛 UnicodeEncodeError
        # 把整篇文档炸掉（parse.py 读文件用了 errors="ignore"，概率低但存在）
        token.encode("utf-8", errors="surrogatepass"),
        digest_size=digest_size,
    ).digest()
    # 摘要按字节对齐（可能比 hash_bits 宽），这里把高位截掉，
    # 保证返回值严格落在 hash_bits 位以内（调用方可以放心当"定长指纹"用）。
    return int.from_bytes(digest, "big") & ((1 << hash_bits) - 1)


def simhash(text: str, hash_bits: int = 64) -> int:
    """计算文本的 SimHash 指纹（hash_bits 位整数）。

       【原理（对应文档 2.1.2 决策、5.1 近似去重）】
           SimHash 是"局部敏感哈希"（LSH）的一种：内容相近的文本，算出的指纹
           也相近（汉明距离小）。这和 SHA1 正好相反——SHA1 是"雪崩"，改一个字
           指纹全变；SimHash 是"相似则相近"。

       【算法流程（频次加权版）】
           1. 统计 n-gram 频次（Counter，重复片段不再被压成 1 次）
           2. 每个片段算一个稳定哈希
           3. 权重向量：哈希第 i 位是 1 就 +freq，是 0 就 -freq
           4. 权重 > 0 的位置记 1，拼成指纹

       【参数】
           text      : str —— 待计算文本
           hash_bits : int —— 指纹位数，默认 64（1~512）

       【返回】
           int —— 指纹，范围 [0, 1 << hash_bits)
       """
    if not 1 <= hash_bits <= _BLAKE2B_MAX_BITS:
        raise ValueError(f"hash_bits 必须在 1~{_BLAKE2B_MAX_BITS} 之间，收到 {hash_bits}")

    weights = [0] * hash_bits

    # 第 1 步 + 第 2 步：遍历频次表，算稳定哈希
    for token, freq in _token_frequencies(text).items():
        h = _stable_hash(token, hash_bits)
        # 第 3 步：把哈希的每一位映射到权重向量（按频次加权）
        for i in range(hash_bits):
            if (h >> i) & 1:
                weights[i] += freq      # 该位是 1 → 权重 +freq
            else:
                weights[i] -= freq      # 该位是 0 → 权重 -freq

    # 第 4 步：正权重记 1，否则记 0，拼成指纹
    fingerprint = 0
    for i in range(hash_bits):
        if weights[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """计算两个整数的汉明距离（二进制表示中不同位的个数）。

       【原理】
           SimHash 判重靠汉明距离：距离越小，文本越相似。
           计算方式：两个整数异或（XOR），结果里 1 的个数就是不同位的个数。
           （异或：相同位得 0，不同位得 1，所以结果里 1 的个数 = 不同位数量）

       【参数】
           a, b : int —— 两个 SimHash 指纹

       【返回】
           int —— 汉明距离（0 ~ hash_bits）
       """
    # a ^ b 异或；bin() 转二进制字符串；count("1") 数 1
    return bin(a ^ b).count("1")


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
    if not text:
        return 0.0
    # 统计中文字符：\u4e00-\u9fff 是 Unicode 里中文汉字的主要区间。
    # re.findall 返回所有匹配的中文字符组成的列表，len() 取数量。
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(chinese_chars) / len(text)


def filter_document(
    text: str,
    min_chars: int | None = None,
    min_density: float | None = None,
) -> str | None:
    """筛选：判断一篇文档是否"值得入库"。

        【两个淘汰条件（对应文档 2.1.2 第⑤步）】
            1. 正文过短（< min_chars 字符）：内容太少，无检索价值。
            2. 中文密度过低（< min_density）：判定为英文内容，过滤掉。

        【参数】
            text        : str         清洗后的文本
            min_chars   : int | None  最小正文字符数；None 用 settings.MIN_TEXT_LEN
            min_density : float | None 最小中文密度；None 用 settings.MIN_CHINESE_RATIO

        【返回】
            str  ：通过筛选，返回原文本
            None ：被过滤，返回 None
        """
    resolved_min_chars = settings.MIN_TEXT_LEN if min_chars is None else min_chars
    resolved_min_density = (
        settings.MIN_CHINESE_RATIO if min_density is None else min_density
    )

    # 条件1：长度检查
    if len(text) < resolved_min_chars:
        log.debug("过滤: 正文过短(%d < %d)字符", len(text), resolved_min_chars)
        return None

    # 条件2：中文密度检查
    density = chinese_density(text)
    if density < resolved_min_density:
        log.debug("过滤: 中文密度过低(%.2f < %.2f)", density, resolved_min_density)
        return None

    return text


# ============================================================================
# 第五部分：组合入口 clean —— 去噪 → 去重 → 筛选
# ============================================================================
def clean(text: str | None, seen_hashes: set[str]) -> str | None:
    """完整的清洗入口：去噪 → 精确去重（查） → 筛选 → 登记指纹。

       【处理顺序（顺序本身是契约，测试和冒烟都依赖它）】
           1. 去噪：目录页在这里被整篇丢弃
           2. 精确去重：哈希命中则跳过
           3. 筛选：长度 + 中文密度（阈值取自 settings）
           4. 登记指纹：**只有通过筛选的文档才占位**
              被淘汰的文档不写 seen_hashes —— 它们没入库，不该算"见过"。

       【参数】
           text        : str | None —— 原始文本；None/空串/非字符串一律跳过
           seen_hashes : Set[str]   —— 已入库文档的 SHA1 集合（调用方跨文档复用）

       【返回】
           str  ：清洗通过，返回清洗后的文本
           None ：被去噪/去重/筛选淘汰，或入参非法
       """
    # ---- 第 0 步：入参防御 ----
    # parse_file 对不支持的格式（.tex 等）返回 None，pipeline 可能原样传进来。
    if not isinstance(text, str):
        if text is not None:
            log.warning("clean 收到非字符串输入(%s), 已跳过", type(text).__name__)
        return None

    # ---- 第 1 步：去噪 ----
    text = denoise(text)
    # strip() 后为空 = 去噪把它删空了（如目录页）
    if not text.strip():
        return None

    # ---- 第 2 步：精确去重（只查，先不登记）----
    h = compute_content_hash(text)
    if h in seen_hashes:
        log.debug("去重: 内容重复, 跳过")
        return None

    # ---- 第 3 步：筛选（长度 + 中文密度）----
    filtered = filter_document(text)
    if filtered is None:
        return None

    # ---- 第 4 步：登记指纹（只有真正入库的文档才占用去重位）----
    seen_hashes.add(h)
    return filtered


# ============================================================================
# 第六部分：近似去重工具类（SimHash 去重器，供 pipeline 在批量场景使用）
# ============================================================================
class SimHashDeduplicator:
    """基于 SimHash + 汉明距离的近似去重器。

        【为什么需要它（对应文档 2.1.2 决策3）】
            github 有大量 fork/镜像，同一份内容被复制成多个仓库，内容 99% 相同
            但 SHA1 不同（可能差一点）。精确去重抓不住这种"近似重复"。
            SimHash 通过"指纹相近"识别近似重复，用汉明距离做阈值判断。

        【用法（顺序不能反：先判后加）】
            dedup = SimHashDeduplicator()
            if dedup.is_duplicate(text):
                skip()
            dedup.add(text)

        【hash_bits 必须与写入端一致】
            指纹的位宽是"能不能互相比"的前提。本类把 hash_bits 存成实例属性，
            is_duplicate / add 都用它，避免调用方两处传参传岔。
        """

    def __init__(self, threshold: int = 3, hash_bits: int = 64) -> None:
        """构造函数。

        【参数】
            threshold : int —— 汉明距离阈值。两个指纹距离 <= threshold 判定为重复。
                               默认 3（64 位指纹里距离 3 以内算"近似相同"）。
                               注意：位宽减半时阈值要跟着降（32 位建议 1~2）。
            hash_bits : int —— 指纹位数，必须与写入端一致，默认 64。
        """
        if threshold < 0:
            raise ValueError(f"threshold 不能为负，收到 {threshold}")
        if not 1 <= hash_bits <= _BLAKE2B_MAX_BITS:
            raise ValueError(f"hash_bits 必须在 1~{_BLAKE2B_MAX_BITS} 之间，收到 {hash_bits}")

        # 汉明距离阈值
        self.threshold = threshold
        # 指纹位宽（is_duplicate / add 都用它，保证两端一致）
        self.hash_bits = hash_bits
        # 已入库的指纹列表（只有通过筛选的文档才应该进来）
        self.fingerprints: list[int] = []

    def __len__(self) -> int:
        """已登记的指纹数量（冒烟脚本/监控用来观察去重规模）。"""
        return len(self.fingerprints)

    def is_duplicate(self, text: str) -> bool:
        """判断文本是否与已入库的某篇"近似重复"。

        【参数】
            text : str —— 待判断文本（传原文，不要传 simhash 结果）

        【返回】
            bool —— True 表示疑似重复（汉明距离 <= threshold）
        """
        # 计算该文本的指纹（位宽与 add 时保持一致）
        fp = simhash(text, hash_bits=self.hash_bits)
        # 逐个比对已入库的指纹
        for existing in self.fingerprints:
            if hamming_distance(fp, existing) <= self.threshold:
                return True
        return False

    def add(self, text: str) -> None:
        """把一份文本的指纹记入去重库（后续文本会跟它比对）。

        【参数】
            text : str —— 要记录的文本（必须是"已通过筛选"的文本）
        """
        self.fingerprints.append(simhash(text, hash_bits=self.hash_bits))
