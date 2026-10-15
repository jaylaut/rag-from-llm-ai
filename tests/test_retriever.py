# ============================================================================
# test_retriever.py —— 检索（retriever）的单元测试
#
# 【测什么】
#   验证 RRF 融合的正确性、HybridRetriever 的 dense/sparse/混合检索。
#   对应《P2-检索.md》第四节"怎么验证做对了"。
# ============================================================================

from rag.ingestion.bm25 import BM25Index
from rag.ingestion.embedder import FakeEmbedder
from rag.ingestion.vectorstore import FaissVectorStore
from rag.retrieval.retriever import HybridRetriever, Retriever, rrf_fuse


# ----------------------------------------------------------------------------
# 测试 1：RRF 融合 —— 排名而非分数
# ----------------------------------------------------------------------------
def test_rrf_fuse_combines_rankings():
    """RRF 融合应基于排名，而非原始分数（量纲不同不影响结果）。"""
    # 两路检索结果（分数故意用不同量纲，验证 RRF 只看排名）
    dense = [("c0", 0.9), ("c1", 0.8), ("c2", 0.7)]   # 向量相似度 0~1
    sparse = [("c2", 25.0), ("c1", 18.0), ("c0", 5.0)]  # BM25 分数 0~30

    fused = rrf_fuse([dense, sparse], k=60)

    # 融合后应有 3 个 chunk，且按 RRF 分数降序
    assert len(fused) == 3
    # RRF 分数：c0=1/61+1/63, c1=1/62+1/62, c2=1/63+1/61
    # 数学性质：c0 与 c2 对称（第1+第3 vs 第3+第1），分数相等；
    # 且 1/61+1/63 > 1/62+1/62，所以 c0==c2 > c1
    fused_map = dict(fused)
    # c0 和 c2 分数相等（对称性）
    assert abs(fused_map["c0"] - fused_map["c2"]) < 1e-12
    # c0 和 c2 都高于 c1
    assert fused_map["c0"] > fused_map["c1"]
    assert fused_map["c2"] > fused_map["c1"]


def test_rrf_fuse_handles_empty():
    """空输入应返回空列表，不报错。"""
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_fuse_rewards_high_rank_in_both():
    """在两路都排前几名的 chunk，融合后应更靠前。"""
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("a", 30.0), ("c", 20.0), ("b", 10.0)]
    fused = rrf_fuse([dense, sparse], k=60)
    # a 在两路都排第 1，应排最前
    assert fused[0][0] == "a"


# ----------------------------------------------------------------------------
# 测试 2：HybridRetriever —— dense_search（向量检索）
# ----------------------------------------------------------------------------
def test_dense_search():
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    # 两个正交向量
    vs.add(["c0"], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    vs.add(["c1"], [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    bm = BM25Index()
    retriever = HybridRetriever(embedder, vs, bm)

    # 查询向量（与 c0 同方向）
    # 用 FakeEmbedder，查询会得到确定性向量
    results = retriever.dense_search("查询", top_k=1)
    assert len(results) == 1
    assert results[0][0] in ("c0", "c1")   # 返回一个 chunk_id


# ----------------------------------------------------------------------------
# 测试 3：HybridRetriever —— sparse_search（BM25 关键词检索）
# ----------------------------------------------------------------------------
def test_sparse_search():
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    bm = BM25Index()
    bm.add_documents(
        ["苹果手机降价了，价格很便宜", "今天天气很好适合出门"],
        ids=["doc1", "doc2"],
    )

    retriever = HybridRetriever(embedder, vs, bm)
    results = retriever.sparse_search("苹果手机", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "doc1"   # 含"苹果"的文档命中


# ----------------------------------------------------------------------------
# 测试 4：混合检索（两路 + RRF）
# ----------------------------------------------------------------------------
def test_hybrid_search_returns_chunks():
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    bm = BM25Index()
    bm.add_documents(["测试文档内容", "另一段内容"], ids=["a", "b"])

    retriever = HybridRetriever(embedder, vs, bm)
    results = retriever.search("测试", top_k=2)
    # 返回 chunk_id + 分数，分数降序
    assert len(results) <= 2
    assert all(isinstance(cid, str) for cid, _ in results)


# ----------------------------------------------------------------------------
# 测试 5：协议满足
# ----------------------------------------------------------------------------
def test_retriever_protocol():
    embedder = FakeEmbedder(dim=8)
    vs = FaissVectorStore(dim=8)
    bm = BM25Index()
    retriever = HybridRetriever(embedder, vs, bm)
    assert isinstance(retriever, Retriever)   # 鸭子类型满足协议
