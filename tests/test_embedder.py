# ============================================================================
# test_embedder.py —— 向量化（embedder）的单元测试
#
# 【测什么】
#   1. normalize 的 L2 归一化正确性；
#   2. FakeEmbedder 的确定性（同文本同向量）；
#   3. Embedder 协议（鸭子类型）—— FakeEmbedder 满足协议；
#   4. OllamaEmbedder 的集成测试（用 integration 标记，默认跳过，需要时跑）。
# ============================================================================

import math

import pytest

from rag.ingestion.embedder import (
    Embedder,
    FakeEmbedder,
    OllamaEmbedder,
    normalize,
)


# ----------------------------------------------------------------------------
# 测试 1：normalize —— L2 归一化
# ----------------------------------------------------------------------------
def test_normalize_unit_length():
    """归一化后向量长度应为 1。"""
    v = normalize([3.0, 4.0])   # 3-4-5 勾股数，模长为 5
    length = math.sqrt(sum(x * x for x in v))
    assert abs(length - 1.0) < 1e-9


def test_normalize_zero_vector():
    """全 0 向量归一化不除零，原样返回。"""
    assert normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_normalize_preserves_direction():
    """归一化只改变长度，不改变方向（各分量比例不变）。"""
    v = normalize([2.0, 4.0])
    # 第二分量是第一分量的 2 倍
    assert abs(v[1] - 2 * v[0]) < 1e-9


# ----------------------------------------------------------------------------
# 测试 2：FakeEmbedder —— 确定性
# ----------------------------------------------------------------------------
def test_fake_embedder_deterministic():
    """同文本多次嵌入，结果应完全一致（可复现）。"""
    emb = FakeEmbedder(dim=8)
    text = "测试文本"
    assert emb.embed([text]) == emb.embed([text])


def test_fake_embedder_dimension():
    """假向量维度应符合配置。"""
    emb = FakeEmbedder(dim=8)
    vecs = emb.embed(["a", "b", "c"])
    assert len(vecs) == 3                    # 3 条文本 → 3 个向量
    assert all(len(v) == 8 for v in vecs)    # 每个向量 8 维


def test_fake_embedder_normalized():
    """假向量也应已归一化（行为与真实一致）。"""
    emb = FakeEmbedder(dim=8)
    for v in emb.embed(["文本一", "文本二"]):
        length = math.sqrt(sum(x * x for x in v))
        assert abs(length - 1.0) < 1e-6


# ----------------------------------------------------------------------------
# 测试 3：Embedder 协议（鸭子类型）
# ----------------------------------------------------------------------------
def test_fake_embedder_satisfies_protocol():
    """FakeEmbedder 应满足 Embedder 协议（有 embed 方法即可，无需继承）。"""
    fake = FakeEmbedder()
    assert isinstance(fake, Embedder)   # runtime_checkable 使 isinstance 生效


def test_ollama_embedder_satisfies_protocol():
    """OllamaEmbedder 也应满足 Embedder 协议。"""
    real = OllamaEmbedder()
    assert isinstance(real, Embedder)


# ----------------------------------------------------------------------------
# 测试 4：OllamaEmbedder 集成测试（默认跳过，需要时 --run-integration）
# ----------------------------------------------------------------------------
@pytest.mark.integration
def test_ollama_embed_real():
    """真实调用 Ollama 嵌入（需要本机 Ollama 运行 + qwen3-embedding 已拉取）。

    用法：pytest --run-integration 才会执行；默认被跳过（不依赖外部服务）。
    """
    emb = OllamaEmbedder(model="qwen3-embedding:0.6b")
    vectors = emb.embed(["苹果手机降价了", "今天天气很好"])
    assert len(vectors) == 2                          # 2 条文本 → 2 个向量
    assert all(len(v) == 1024 for v in vectors)       # qwen3-embedding 输出 1024 维
    # 每个向量已归一化
    for v in vectors:
        length = math.sqrt(sum(x * x for x in v))
        assert abs(length - 1.0) < 1e-3
