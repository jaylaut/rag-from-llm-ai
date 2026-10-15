# ============================================================================
# bm25.py —— BM25 关键词检索（稀疏检索，混合检索的另一路）
#
# 位置：src/rag/ingestion/bm25.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.4 节提到的"还有一路：BM25（关键词检索）"。
#   除了"按意思搜"（向量），还要一路"按关键词搜"（BM25），两路配合叫混合检索。
#
#   P1 只负责"建" BM25 索引，详细检索原理放 P2 讲。但这里把索引构建和
#   打分算法都实现了，因为它们是"建索引"这个动作不可分割的一部分。
#
# 【BM25 原理（人话版）】
#   给一个查询，对每篇文档打一个"相关分"。打分考虑三个因素：
#     1. 词频（TF）：这个词在这篇文档里出现几次——出现多分高，但有"饱和"（k1）
#     2. 逆文档频率（IDF）：这个词在全库多罕见——越罕见越值钱
#     3. 文档长度归一化（b）：长文档天然词多，要压一压
#
# 【为什么不直接用 rank_bm25 库】
#   项目定位是"学底层、不依赖框架"（见 requirements.txt 顶部原则）。
#   BM25 公式不复杂，手写能真正理解"倒排索引 + 打分"的原理。
#   分词用 jieba（中文分词，requirements 里已声明）。
# ============================================================================

# ---------- 导入区 ----------
# math：Python 标准库，对数运算（IDF 用 log）。
# json：Python 标准库，BM25 索引持久化（save/load）。
import json
import math

# Path：Python 标准库，处理文件路径。
from pathlib import Path

# jieba：第三方库，中文分词（把句子切成一个个词，BM25 的"词"要分词）。
import jieba

# 项目内：日志设施。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 工具函数：中文分词
# ============================================================================
def tokenize(text: str) -> list[str]:
    """把文本切分成"词"列表（BM25 的最小单位）。

    【原理】
        BM25 是基于"词"的统计。英文按空格分即可，中文没有天然空格，需要
        用 jieba 分词。jieba.lcut 返回分词后的词列表。

    参数：
        text : str —— 待分词文本

    返回：
        List[str] —— 分词后的词列表
    """
    return [w for w in jieba.lcut(text) if w.strip()]   # 去掉空白词


# ============================================================================
# BM25 索引类
# ============================================================================
class BM25Index:
    """手写 BM25 索引：倒排索引 + 打分。

    【数据结构】
        1. doc_tokens : List[List[str]] —— 每篇文档的分词结果
        2. df         : Dict[str, int]   —— 文档频率（某词出现在几篇文档里）
        3. doc_len    : List[int]        —— 每篇文档的"词数"（长度）
        4. avg_len    : float            —— 平均文档长度

    【参数】
        k1 : 词频饱和参数（默认 1.5）
        b  : 长度归一化参数（默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """构造函数。

        参数：
            k1 : float —— 词频饱和参数，越大词频影响越线性
            b  : float —— 长度归一化参数，0 表示不考虑长度，1 表示完全归一化
        """
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []   # 每篇文档的分词
        self.df: dict[str, int] = {}            # 文档频率
        self.doc_len: list[int] = []            # 每篇文档长度
        self.avg_len: float = 0.0               # 平均长度
        self.doc_ids: list[str] = []            # 每篇文档的 chunk_id（与 doc_tokens 对齐）

    # ----------------------------------------------------------------------
    # 建索引：批量添加文档
    # ----------------------------------------------------------------------
    def add_documents(self, documents: list[str], ids: list[str] | None = None) -> None:
        """批量添加文档，构建倒排索引。

        【原理】
            对每篇文档分词，统计每个词的"文档频率"（df：出现在几篇文档里）。
            这是后续算 IDF 的基础。

        参数：
            documents : List[str] —— 文档文本列表
            ids       : List[str] | None —— 每篇文档的 chunk_id（可选）。
                         若传入，长度必须与 documents 一致，检索时才能返回 chunk_id；
                         若省略，则用"序号"作为 id（P1 独立使用时的兼容行为）。
        """
        # 若没传 ids，用递增序号兜底（保持 P1 独立使用的兼容）
        if ids is None:
            ids = [str(i) for i in range(len(self.doc_ids), len(self.doc_ids) + len(documents))]
        if len(ids) != len(documents):   # 长度必须一致，否则对齐错乱
            raise ValueError(f"ids 长度 {len(ids)} 与 documents 长度 {len(documents)} 不一致")

        # strict=True：documents 与 ids 长度已在上方校验一致，这里用 strict 兜底
        for doc, doc_id in zip(documents, ids, strict=True):
            tokens = tokenize(doc)              # 分词
            self.doc_tokens.append(tokens)      # 存分词结果
            self.doc_len.append(len(tokens))    # 存文档长度
            self.doc_ids.append(doc_id)         # 存 chunk_id
            # 统计文档频率：一篇文档里同一个词只算一次（用 set 去重）
            for word in set(tokens):
                self.df[word] = self.df.get(word, 0) + 1

        # 计算平均文档长度
        if self.doc_len:
            self.avg_len = sum(self.doc_len) / len(self.doc_len)

    # ----------------------------------------------------------------------
    # 算某词在一篇文档里的 BM25 权重
    # ----------------------------------------------------------------------
    def _score_term(self, tf: int, df: int, doc_len: int, num_docs: int) -> float:
        """计算单个词对某篇文档的 BM25 打分。

        【公式】BM25 = IDF * TF项
          IDF = log(1 + (N - df + 0.5) / (df + 0.5))   —— 词越罕见越大
          TF项 = tf*(k1+1) / (tf + k1*(1 - b + b*doc_len/avg_len))  —— 词频饱和+长度归一

        参数：
            tf       : int   词频（该词在这篇文档出现几次）
            df       : int   文档频率（该词出现在几篇文档里）
            doc_len  : int   这篇文档的长度
            num_docs : int   文档总数

        返回：
            float —— 该词对该文档的贡献分
        """
        # IDF：逆文档频率。df 越大（词越常见）IDF 越小。
        idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))

        # TF 项：词频饱和 + 长度归一化
        #   tf*(k1+1) 是分子；分母里 b*doc_len/avg_len 做长度惩罚
        denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
        tf_term = tf * (self.k1 + 1) / denom

        return idf * tf_term

    # ----------------------------------------------------------------------
    # 检索：对查询打分，返回最相关的文档序号
    # ----------------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[tuple]:
        """检索：返回与查询最相关的 top_k 篇文档。

        【原理】
            把查询分词，对每篇文档累加"查询里每个词"的 BM25 权重，
            得到文档总得分，按分数降序返回 top_k。

        参数：
            query : str —— 查询文本
            top_k : int —— 返回前几名

        返回：
            List[tuple] —— [(chunk_id, 得分), ...]，得分降序。
                          chunk_id 来自 add_documents 传入的 ids（或序号兜底）。
        """
        query_tokens = tokenize(query)            # 查询分词
        num_docs = len(self.doc_tokens)
        scores = [0.0] * num_docs                 # 每篇文档的得分，初始 0

        # 对查询里的每个词（去重，重复词不重复累加）
        for word in set(query_tokens):
            df = self.df.get(word, 0)             # 该词的文档频率
            if df == 0:                           # 词不在任何文档里，跳过
                continue
            # 遍历每篇文档，累加该词的贡献
            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(word)           # 词频
                if tf == 0:
                    continue
                scores[i] += self._score_term(tf, df, self.doc_len[i], num_docs)

        # 按得分降序排列，返回前 top_k 的 (chunk_id, 得分)
        ranked = sorted(
            ((self.doc_ids[i], scores[i]) for i in range(num_docs) if scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # ----------------------------------------------------------------------
    # 持久化：保存索引到磁盘
    # ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """把 BM25 索引保存到磁盘（JSON 格式）。

        【为什么需要持久化】
            P2 混合检索需要 BM25 索引。索引建好（P1）后必须存盘，
            否则每次重启都要重新分词建索引（几十万块会非常慢）。

        【存什么】
            doc_tokens（每篇文档的分词）、doc_ids（chunk_id）、
            df（文档频率）、doc_len（文档长度）、avg_len、k1、b。
            存这些就能完整恢复 BM25 索引，无需重新分词。

        参数：
            path : Path —— 索引文件保存路径（如 data/index/bm25.json）
        """
        path.parent.mkdir(parents=True, exist_ok=True)   # 确保目录存在
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_tokens": self.doc_tokens,
            "df": self.df,
            "doc_len": self.doc_len,
            "avg_len": self.avg_len,
        }
        # ensure_ascii=False 让中文正常显示；indent=2 缩进可读
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.debug("BM25 索引已保存：%s（%d 篇）", path, len(self.doc_ids))

    # ----------------------------------------------------------------------
    # 持久化：从磁盘加载索引
    # ----------------------------------------------------------------------
    def load(self, path: Path) -> None:
        """从磁盘加载 BM25 索引。

        【原理】
            与 save 对应，从 JSON 恢复所有字段，无需重新分词。
            这是"离线建索引、在线加载"的关键——启动时直接 load，秒级就绪。

        参数：
            path : Path —— 索引文件路径
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        self.k1 = data["k1"]
        self.b = data["b"]
        self.doc_ids = data["doc_ids"]
        self.doc_tokens = data["doc_tokens"]
        self.df = data["df"]
        self.doc_len = data["doc_len"]
        self.avg_len = data["avg_len"]
        log.debug("BM25 索引已加载：%s（%d 篇）", path, len(self.doc_ids))
