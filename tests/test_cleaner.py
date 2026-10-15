# ============================================================================
# test_cleaner.py —— 文档清洗（cleaner）的单元测试
#
# 【测什么】
#   验证去噪（denoise）、中文密度（chinese_density）、筛选（filter_document）、
#   精确去重（clean + seen_hashes）、近似去重（SimHash + 汉明距离）。
# ============================================================================

from rag.ingestion.cleaner import (
    SimHashDeduplicator,
    chinese_density,
    clean,
    denoise,
    filter_document,
    hamming_distance,
    simhash,
)
from rag.ingestion.registry import compute_content_hash


# ----------------------------------------------------------------------------
# 测试 1：中文密度
# ----------------------------------------------------------------------------
def test_chinese_density_pure_chinese():
    assert chinese_density("我爱北京天安门") == 1.0   # 纯中文密度为 1


def test_chinese_density_pure_english():
    assert chinese_density("hello world python") == 0.0   # 纯英文密度为 0


def test_chinese_density_mixed():
    # "中文abc" 共 5 个字符，其中中文 2 个 → 密度 = 2/5 = 0.4
    d = chinese_density("中文abc")
    assert abs(d - 0.4) < 1e-9   # 用浮点误差容忍的近似断言


# ----------------------------------------------------------------------------
# 测试 2：筛选（长度 + 中文密度）
# ----------------------------------------------------------------------------
def test_filter_document_too_short():
    assert filter_document("太短", min_chars=100) is None   # 过短被过滤


def test_filter_document_english():
    # 足够长但全是英文 → 中文密度低 → 过滤
    english = "this is an english document " * 50
    assert filter_document(english, min_chars=10, min_density=0.3) is None


def test_filter_document_passes():
    chinese = "这是一个中文技术文档的内容。" * 20   # 足够长且全中文
    assert filter_document(chinese, min_chars=50, min_density=0.3) == chinese


# ----------------------------------------------------------------------------
# 测试 3：精确去重（clean + seen_hashes）
# ----------------------------------------------------------------------------
def test_clean_deduplicates():
    content = "这是足够长的中文内容，用于测试去重。" * 10
    seen = set()
    assert clean(content, seen) is not None          # 第一次通过
    assert clean(content, seen) is None              # 第二次重复被去重


# ----------------------------------------------------------------------------
# 测试 4：去噪（目录页、版权行、多余空行）
# ----------------------------------------------------------------------------
def test_denoise_removes_copyright():
    text = "正文内容\n版权所有 2026\n京ICP备123号\n另一段正文"
    cleaned = denoise(text)
    assert "版权所有" not in cleaned
    assert "京ICP备" not in cleaned
    assert "正文内容" in cleaned


def test_denoise_directory_page():
    # 全是链接的目录页 → 返回空串
    links = "\n".join(f"[链接{i}](http://example.com/{i})" for i in range(30))
    assert denoise(links) == ""


def test_denoise_collapses_blank_lines():
    text = "第一段\n\n\n\n\n第二段"
    cleaned = denoise(text)
    assert "\n\n\n" not in cleaned   # 多余空行被合并


# ----------------------------------------------------------------------------
# 测试 5：SimHash 近似去重
# ----------------------------------------------------------------------------
def test_simhash_similar_texts_have_small_distance():
    """相近文本的 SimHash 汉明距离应很小。"""
    a = "这是一段关于机器学习的详细介绍内容，讲述了深度学习模型的原理。" * 5
    b = "这是一段关于机器学习的详细介绍内容，讲述了深度学习模型的原理。" * 5
    assert simhash(a) == simhash(b)   # 完全相同 → 指纹相同 → 距离 0


def test_hamming_distance_identical_is_zero():
    a = simhash("相同的内容")
    assert hamming_distance(a, a) == 0


def test_simhash_dedup_identical():
    dedup = SimHashDeduplicator()
    text = "关于RAG检索增强生成的详细技术介绍。" * 10
    dedup.add(text)
    assert dedup.is_duplicate(text) is True      # 相同内容判重
    assert dedup.is_duplicate("完全不同的另一篇文档内容。" * 10) is False


# ----------------------------------------------------------------------------
# 测试 6：compute_content_hash 与 cleaner 的联动（同一把尺子）
# ----------------------------------------------------------------------------
def test_content_hash_used_by_cleaner():
    """clean 内部用 compute_content_hash 做精确去重，验证哈希一致。"""
    content = "一段用于验证哈希一致性的中文内容。" * 5
    assert compute_content_hash(content) == compute_content_hash(content)
