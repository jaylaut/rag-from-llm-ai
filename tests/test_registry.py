# ============================================================================
# test_registry.py —— 文档注册表的单元测试
#
# 【测什么】
#   验证 DocumentRegistry 的核心行为：upsert 版本号递增、反向索引维护、
#   软删除、序列化/反序列化（save + 重新加载）、对账统计、幂等性质。
#   对应《P1-索引构建.md》第四节"怎么验证做对了"里的增量闭环验证项。
# ============================================================================

import json

from rag.ingestion.registry import (
    DocumentRegistry,
    compute_content_hash,
)


# ----------------------------------------------------------------------------
# 测试 1：compute_content_hash —— 相同内容哈希相同，不同内容哈希不同
# ----------------------------------------------------------------------------
def test_compute_content_hash_stable():
    """同一内容多次计算，哈希结果必须完全一致。"""
    text = "这是一段用于测试的内容"
    assert compute_content_hash(text) == compute_content_hash(text)


def test_compute_content_hash_differs():
    """内容哪怕改一个字符，哈希也必须不同（指纹的"雪崩"性质）。"""
    assert compute_content_hash("苹果手机降价了") != compute_content_hash("苹果手机降价了吗")


def test_compute_content_hash_is_sha1():
    """哈希结果应是 40 位十六进制字符串（SHA1 的标准长度）。"""
    h = compute_content_hash("hello")
    assert len(h) == 40                      # SHA1 固定 40 个十六进制字符
    assert all(c in "0123456789abcdef" for c in h)   # 全是十六进制字符


# ----------------------------------------------------------------------------
# 测试 2：upsert —— 新增时 version=1，重复 upsert 时 version 递增
# ----------------------------------------------------------------------------
def test_upsert_new_document(tmp_registry_path):
    """首次 upsert 一个 doc_id，应得到 version=1 的记录。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    rec = reg.upsert(
        doc_id="docs/a.md",
        content_hash="hash_a",
        chunk_ids=["docs/a.md#c0", "docs/a.md#c1"],
        embedding_model="qwen3-embedding:0.6b",
    )
    assert rec.version == 1                  # 新文档版本从 1 开始
    assert rec.status == "active"            # 新文档是 active
    assert rec.doc_id == "docs/a.md"


def test_upsert_same_doc_increments_version(tmp_registry_path):
    """对同一 doc_id 再次 upsert，version 应 +1（幂等键的核心）。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("docs/a.md", "hash_v1", ["c0"], "m1")
    rec2 = reg.upsert("docs/a.md", "hash_v2", ["c1", "c2"], "m1")
    assert rec2.version == 2                 # 第二次入库版本 +1
    # 主表里只应有这一条记录（不会产生重复条目）
    assert len(reg.records) == 1


# ----------------------------------------------------------------------------
# 测试 3：hash_index 反向索引 —— 按内容判重
# ----------------------------------------------------------------------------
def test_find_by_hash(tmp_registry_path):
    """upsert 后，应能通过 content_hash 反查到 doc_id。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("uploads/tmp_123.md", "hash_content", ["c0"], "m1")
    assert reg.find_by_hash("hash_content") == "uploads/tmp_123.md"
    assert reg.find_by_hash("不存在的哈希") is None   # 未命中返回 None


def test_hash_index_survives_soft_delete(tmp_registry_path):
    """软删除后，反向索引仍应保留（历史 hash 仍需去重，防止已删内容重新入库）。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("docs/a.md", "hash_keep", ["c0"], "m1")
    reg.mark_deleted("docs/a.md")            # 软删除
    # 关键断言：hash_index 不被移除
    assert reg.find_by_hash("hash_keep") == "docs/a.md"


# ----------------------------------------------------------------------------
# 测试 4：软删除 —— 逻辑上立即失效
# ----------------------------------------------------------------------------
def test_mark_deleted(tmp_registry_path):
    """软删除后，is_active 应返回 False，且 status 变为 deleted。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("docs/a.md", "h", ["c0"], "m1")
    assert reg.is_active("docs/a.md") is True   # 删除前是 active

    reg.mark_deleted("docs/a.md")
    assert reg.is_active("docs/a.md") is False  # 删除后不是 active
    assert reg.records["docs/a.md"].status == "deleted"


# ----------------------------------------------------------------------------
# 测试 5：序列化 —— save 后重新加载，数据不丢失
# ----------------------------------------------------------------------------
def test_save_and_reload(tmp_registry_path):
    """save 到磁盘后，新建一个实例重新 load，数据应完整恢复。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("docs/a.md", "hash_a", ["c0", "c1"], "m1")
    reg.upsert("docs/b.md", "hash_b", ["c2"], "m1")
    reg.mark_deleted("docs/b.md")            # 造一个 deleted 状态
    reg.save()

    # 重新加载：新建实例，走 _load 从磁盘恢复
    reg2 = DocumentRegistry(path=tmp_registry_path)
    assert reg2.records["docs/a.md"].content_hash == "hash_a"
    assert reg2.records["docs/a.md"].chunk_ids == ["c0", "c1"]
    assert reg2.records["docs/b.md"].status == "deleted"
    # 反向索引也恢复
    assert reg2.find_by_hash("hash_a") == "docs/a.md"


def test_save_is_valid_json(tmp_registry_path):
    """save 产出的文件应是合法 JSON（能 json.loads 解析），且含两张表。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("docs/a.md", "h", ["c0"], "m1")
    reg.save()
    data = json.loads(tmp_registry_path.read_text(encoding="utf-8"))
    assert "records" in data               # 含主表
    assert "hash_index" in data            # 含反向索引


# ----------------------------------------------------------------------------
# 测试 6：对账统计 —— count_active_chunks
# ----------------------------------------------------------------------------
def test_count_active_chunks(tmp_registry_path):
    """active 文档的 chunk 总数统计应正确，且软删除的文档不计入。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("docs/a.md", "ha", ["c0", "c1", "c2"], "m1")   # 3 块
    reg.upsert("docs/b.md", "hb", ["c3"], "m1")               # 1 块
    assert reg.count_active_chunks() == 4

    reg.mark_deleted("docs/b.md")           # 软删除 b，其 1 块不计入
    assert reg.count_active_chunks() == 3


# ----------------------------------------------------------------------------
# 测试 7：get_chunk_ids —— 删除/重组时定位块
# ----------------------------------------------------------------------------
def test_get_chunk_ids(tmp_registry_path):
    """能取出某文档的全部块 ID；不存在的文档返回空列表。"""
    reg = DocumentRegistry(path=tmp_registry_path)
    reg.upsert("docs/a.md", "h", ["c0", "c1"], "m1")
    assert reg.get_chunk_ids("docs/a.md") == ["c0", "c1"]
    assert reg.get_chunk_ids("不存在的文档") == []
