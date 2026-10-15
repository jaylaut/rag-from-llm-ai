# ============================================================================
# test_compactor.py —— Compaction（压缩整理）的单元测试
#
# 【测什么】
#   验证 should_compact（僵尸占比判断）和 compact（重建索引）的正确性。
#   对应文档 2.5.6 节。
# ============================================================================

from rag.ingestion.compactor import compact, should_compact
from rag.ingestion.registry import DocumentRegistry
from rag.ingestion.vectorstore import FaissVectorStore


# ----------------------------------------------------------------------------
# 测试 1：僵尸占比判断
# ----------------------------------------------------------------------------
def test_should_compact_below_threshold(tmp_path):
    reg = DocumentRegistry(path=tmp_path / "reg.json")
    reg.upsert("d1", "h1", ["c0", "c1", "c2", "c3"], "m1")   # 4 块 active
    reg.upsert("d2", "h2", ["c4"], "m1")                     # 1 块，之后软删

    vs = FaissVectorStore(dim=4)
    # 僵尸占比 = 1/5 = 20% > 10%，应触发
    reg.mark_deleted("d2")

    assert should_compact(reg, vs, threshold=0.10) is True


def test_should_compact_no_zombie(tmp_path):
    reg = DocumentRegistry(path=tmp_path / "reg.json")
    reg.upsert("d1", "h1", ["c0", "c1"], "m1")

    vs = FaissVectorStore(dim=4)
    # 没有僵尸（0/2 = 0%），不应触发
    assert should_compact(reg, vs, threshold=0.10) is False


def test_should_compact_empty(tmp_path):
    reg = DocumentRegistry(path=tmp_path / "reg.json")   # 空注册表
    vs = FaissVectorStore(dim=4)
    assert should_compact(reg, vs, threshold=0.10) is False   # 无数据不触发


# ----------------------------------------------------------------------------
# 测试 2：compact 重建索引（只保留 active，僵尸被清理）
# ----------------------------------------------------------------------------
def test_compact_rebuilds(tmp_path):
    reg = DocumentRegistry(path=tmp_path / "reg.json")
    # 两篇 active，一篇 deleted
    reg.upsert("d1", "h1", ["c0", "c1"], "m1")
    reg.upsert("d2", "h2", ["c2"], "m1")
    reg.mark_deleted("d2")   # d2 的 c2 变成僵尸

    vs = FaissVectorStore(dim=4)
    # 入库 3 个向量（c0, c1, c2）
    vs.add(["c0", "c1"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    vs.add(["c2"], [[0.0, 0.0, 1.0, 0.0]])
    vs.mark_deleted(["c2"])   # 向量侧也软删除 c2

    index_path = tmp_path / "faiss.index"
    rebuilt = compact(reg, vs, index_path)

    # 只保留 active 的 c0, c1（2 个），僵尸 c2 被清理
    assert rebuilt == 2

    # 加载新索引验证
    vs2 = FaissVectorStore(dim=4)
    vs2.load(index_path)
    assert vs2.count_active() == 2
