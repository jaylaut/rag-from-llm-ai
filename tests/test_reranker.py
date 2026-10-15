# ============================================================================
# test_reranker.py —— 重排（reranker）的单元测试
#
# 【测什么】
#   验证 Reranker 协议、EmbeddingReranker 占位实现的正确性、
#   CrossEncoderReranker 的"未实现"保护。
# ============================================================================

import pytest

from rag.ingestion.embedder import FakeEmbedder
from rag.rerank.reranker import CrossEncoderReranker, EmbeddingReranker, Reranker


# ----------------------------------------------------------------------------
# 测试 1：EmbeddingReranker —— 相关性重排
# ----------------------------------------------------------------------------
def test_embedding_reranker_orders_by_similarity():
    """重排后，与查询语义更近的 chunk 应排更前。"""
    embedder = FakeEmbedder(dim=8)
    reranker = EmbeddingReranker(embedder)

    chunks = [
        ("c0", "机器学习是人工智能的核心"),
        ("c1", "今天天气很好适合散步"),
        ("c2", "深度学习是机器学习的重要方法"),
    ]

    results = reranker.rerank("机器学习", chunks)
    # 返回 3 个结果，按相似度降序
    assert len(results) == 3
    # 分数降序排列
    scores = [s for _cid, s in results]
    assert scores == sorted(scores, reverse=True)


def test_embedding_reranker_empty():
    """空候选应返回空列表。"""
    reranker = EmbeddingReranker(FakeEmbedder(dim=8))
    assert reranker.rerank("查询", []) == []


# ----------------------------------------------------------------------------
# 测试 2：协议满足
# ----------------------------------------------------------------------------
def test_reranker_protocol():
    """EmbeddingReranker 应满足 Reranker 协议（鸭子类型）。"""
    assert isinstance(EmbeddingReranker(FakeEmbedder(dim=8)), Reranker)
    assert isinstance(CrossEncoderReranker(), Reranker)


# ----------------------------------------------------------------------------
# 测试 3：CrossEncoderReranker —— 未实现保护
# ----------------------------------------------------------------------------
def test_cross_encoder_not_implemented():
    """vLLM 未就绪时，CrossEncoderReranker 应明确报错提示。"""
    reranker = CrossEncoderReranker()
    with pytest.raises(NotImplementedError):
        reranker.rerank("查询", [("c0", "内容")])
