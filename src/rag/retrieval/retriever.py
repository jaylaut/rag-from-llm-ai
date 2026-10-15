# ============================================================================
# retriever.py —— 检索（Retrieval）：混合检索 + RRF 融合
#
# 位置：src/rag/retrieval/retriever.py
#
# 【本文件是干什么的】
#   对应《P2-检索.md》。把用户问题转成"最相关的几段文字"（chunk）。
#   这是整个 RAG 的"心脏"——检索不准，后面生成再强也是垃圾进垃圾出。
#
#   两路检索互补（对应文档 2.1/2.2）：
#     · 稠密检索（向量）：懂"意思"，能跨同义词匹配
#     · 稀疏检索（BM25）：懂"精确词"，对专有名词/代码/错误码敏感
#   两路混合，取长补短，召回率显著提升。
#
# 【RRF 融合原理（对应文档 2.3）】
#   两路分数量纲不同（向量相似度 0~1，BM25 分数 0~30），不能直接相加。
#   改成看"排名"：每个 chunk 在两路各排第几，按 1/(k+排名) 给分相加。
#   好处：不用调权重，天生鲁棒。
# ============================================================================

# ---------- 导入区 ----------
# 类型标注
from typing import Protocol, runtime_checkable

from rag.ingestion.bm25 import BM25Index
from rag.ingestion.embedder import Embedder

# 项目内：向量存储（稠密检索的载体）、BM25（稀疏检索）、嵌入器
from rag.ingestion.vectorstore import VectorStore

# 项目内：日志
from rag.utils.logger import setup_logger

# 模块级日志对象
log = setup_logger()


# ============================================================================
# 第一部分：Retriever 协议（接口定义，对应 P5 编排层的"可插拔"）
# ============================================================================
@runtime_checkable
class Retriever(Protocol):
    """检索器协议：定义"能做什么"（能力约定）。

    【为什么定义协议】
        P5 编排层要对检索器做"策略模式"式的插拔（换检索算法不改上层）。
        用 Protocol 约定接口，任何实现了 search 方法的对象都能当检索器用。
    """

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """检索最相关的 top_k 个 chunk。

        参数：
            query : str —— 查询文本
            top_k : int —— 返回前几名

        返回：
            List[Tuple[str, float]] —— [(chunk_id, 相似度分数), ...]
        """
        ...


# ============================================================================
# 第二部分：RRF 融合函数（本阶段核心知识点）
# ============================================================================
def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """RRF（倒数排名融合）：把多路检索的排名融合成一份最终排名。

    【原理（对应文档 2.3）】
        不直接相加两路的"分数"（量纲不同），而是看"排名"：
        每个 chunk 在某一路里排第 r 名，就贡献 1/(k+r) 分（k 是平滑常数，默认 60）。
        多路的贡献相加，得到该 chunk 的 RRF 总分，再按总分降序。

    【为什么 k=60】
        k 是平滑常数，避免"排第 1 名"的分数过高（1/(0+1)=1 太陡）。
        k=60 是业界常用默认值，让分数曲线更平滑。

    【参数】
        ranked_lists : List[List[Tuple[str, float]]] —— 多路检索的结果列表。
                       每路都是 [(chunk_id, 分数), ...]（已按分数降序）。
        k            : int —— 平滑常数，默认 60

    【返回】
        List[Tuple[str, float]] —— 融合后的 [(chunk_id, rrf分数), ...]，降序
    """
    # rrf_scores: chunk_id → rrf 累计分数
    rrf_scores: dict[str, float] = {}

    # 遍历每一路检索结果
    for ranked in ranked_lists:
        # enumerate 从 0 开始，排名 r = 索引 + 1（第 1 名索引 0）
        for rank, (chunk_id, _score) in enumerate(ranked):
            # 该 chunk 在这一路的排名 r=rank+1，贡献 1/(k+r)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    # 按 RRF 分数降序排列
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ============================================================================
# 第三部分：混合检索器（把两路 + 融合串起来）
# ============================================================================
class HybridRetriever:
    """混合检索器：稠密（向量）+ 稀疏（BM25）+ RRF 融合。

    【职责】
        把 P1 建好的向量索引（FAISS）和 BM25 索引组合起来，
        提供统一的 search 入口：一次查询 → 两路召回 → RRF 融合 → 返回。

    【参数】
        embedder    : 嵌入器（用于把查询转向量）
        vectorstore : 向量存储（稠密检索）
        bm25        : BM25 索引（稀疏检索）
        rrf_k       : RRF 平滑常数
    """

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        bm25: BM25Index,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.rrf_k = rrf_k

    # ----------------------------------------------------------------------
    # 稠密检索（向量）
    # ----------------------------------------------------------------------
    def dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """稠密检索：把查询转向量，在 FAISS 里找最近的 chunk。

        【关键：查询侧加指令前缀（对应 P2 坑5）】
            qwen3-embedding 查询侧要加 Instruct 前缀（非对称检索），
            所以这里用 embedder.embed_query 而非 embed（embed_query 内部已加前缀）。

        参数：
            query : str —— 查询文本
            top_k : int —— 返回前几名

        返回：
            List[Tuple[str, float]] —— [(chunk_id, 相似度), ...]，降序
        """
        # embed_query：加指令前缀后转向量（非对称检索）
        query_vector = self.embedder.embed_query(query)
        # vectorstore.search：在 FAISS 里找最相近的（内部已过滤软删除）
        return self.vectorstore.search(query_vector, top_k)

    # ----------------------------------------------------------------------
    # 稀疏检索（BM25）
    # ----------------------------------------------------------------------
    def sparse_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """稀疏检索：用 BM25 按关键词打分。

        参数：
            query : str —— 查询文本
            top_k : int —— 返回前几名

        返回：
            List[Tuple[str, float]] —— [(chunk_id, bm25分数), ...]，降序
        """
        # bm25.search：内部已用 jieba 分词 + 打分
        return self.bm25.search(query, top_k)

    # ----------------------------------------------------------------------
    # 混合检索（两路 + RRF 融合）
    # ----------------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """混合检索：稠密 + 稀疏两路召回，RRF 融合后返回 top_k。

        参数：
            query : str —— 查询文本
            top_k : int —— 返回前几名

        返回：
            List[Tuple[str, float]] —— [(chunk_id, rrf分数), ...]，降序
        """
        # 两路都多召回一些（融合后取前 top_k，多召保证融合有足够候选）
        recall_k = top_k * 3   # 每路召回 3 倍，融合后截断

        dense_ranked = self.dense_search(query, recall_k)
        sparse_ranked = self.sparse_search(query, recall_k)

        # RRF 融合
        fused = rrf_fuse([dense_ranked, sparse_ranked], k=self.rrf_k)

        # 截断到 top_k
        return fused[:top_k]
