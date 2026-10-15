# ============================================================================
# test_vectorstore.py —— 向量存储（FAISS）的单元测试
#
# 【测什么】
#   验证 FaissVectorStore 的追加、检索、软删除过滤、对账统计、持久化。
#   对应文档 2.4（FAISS 存储）和 2.5.5（软删除）的核心行为。
# ============================================================================


from rag.ingestion.vectorstore import FaissVectorStore, VectorStore


# ----------------------------------------------------------------------------
# 测试 1：追加 + 检索（内积相似度）
# ----------------------------------------------------------------------------
def test_add_and_search():
    vs = FaissVectorStore(dim=4)
    # 两个正交向量（内积为 0，语义无关）
    vs.add(["c0"], [[1.0, 0.0, 0.0, 0.0]])
    vs.add(["c1"], [[0.0, 1.0, 0.0, 0.0]])

    # 查询向量与 c0 相同方向（已归一化，内积=1）
    results = vs.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "c0"          # 最相似的是 c0
    assert abs(results[0][1] - 1.0) < 1e-5   # 内积（相似度）≈ 1.0


# ----------------------------------------------------------------------------
# 测试 2：软删除过滤
# ----------------------------------------------------------------------------
def test_soft_delete_filters_out():
    vs = FaissVectorStore(dim=4)
    vs.add(["c0"], [[1.0, 0.0, 0.0, 0.0]])
    vs.add(["c1"], [[0.0, 1.0, 0.0, 0.0]])

    # 软删除 c0
    vs.mark_deleted(["c0"])

    # 即使查询向量最像 c0，也应被过滤，返回 c1
    results = vs.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    returned_ids = [cid for cid, _ in results]
    assert "c0" not in returned_ids       # 软删除的块不返回
    assert "c1" in returned_ids           # 未删除的块正常返回


# ----------------------------------------------------------------------------
# 测试 3：对账统计（count_active）
# ----------------------------------------------------------------------------
def test_count_active():
    vs = FaissVectorStore(dim=4)
    vs.add(["c0", "c1"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert vs.count_active() == 2         # 初始 2 个 active

    vs.mark_deleted(["c0"])
    assert vs.count_active() == 1         # 软删除后剩 1 个 active


# ----------------------------------------------------------------------------
# 测试 4：持久化 + 加载
# ----------------------------------------------------------------------------
def test_save_and_load(tmp_path):
    vs = FaissVectorStore(dim=4)
    vs.add(["c0", "c1"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    vs.mark_deleted(["c0"])               # 造一个 deleted 状态

    index_path = tmp_path / "faiss.index"
    vs.save(index_path)

    # 重新加载
    vs2 = FaissVectorStore(dim=4)
    vs2.load(index_path)

    assert vs2.count_active() == 1        # 软删除状态被正确持久化
    results = vs2.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert "c0" not in [cid for cid, _ in results]   # 加载后软删除仍生效


# ----------------------------------------------------------------------------
# 测试 5：协议满足
# ----------------------------------------------------------------------------
def test_faiss_satisfies_protocol():
    vs = FaissVectorStore(dim=4)
    assert isinstance(vs, VectorStore)


# ----------------------------------------------------------------------------
# 测试 6：空索引检索不报错
# ----------------------------------------------------------------------------
def test_search_empty_index():
    vs = FaissVectorStore(dim=4)
    assert vs.search([1.0, 0.0, 0.0, 0.0], top_k=5) == []   # 空索引返回空
