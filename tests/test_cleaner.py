# ============================================================================
# test_cleaner.py —— 文档清洗（cleaner.py）的单元测试
#
# 【测什么】
#   1. 对外 5 个函数：denoise / chinese_density / filter_document /
#      hamming_distance / clean
#   2. SimHash 三件套：_ngrams / _stable_hash / simhash + SimHashDeduplicator
#   3. 配置接线：默认阈值必须来自 settings，而不是硬编码
#
# 【怎么跑】
#   .venv/Scripts/python.exe -m pytest tests/test_cleaner.py -v
#   .venv/Scripts/python.exe -m pytest tests/test_cleaner.py -v -rxX
#
#   注意必须指定本文件路径，不要裸跑 pytest（其余实现模块尚未全部就绪，
#   裸跑会连带 import 它们产生 collection error）。
# ============================================================================
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag.config.settings import settings
from rag.ingestion.cleaner import (
    SIMHASH_ALGO_VERSION,
    SimHashDeduplicator,
    _ngrams,
    _stable_hash,
    _token_frequencies,
    _tokenize,
    chinese_density,
    clean,
    denoise,
    filter_document,
    hamming_distance,
    simhash,
)
from rag.ingestion.registry import compute_content_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 138 字，中文密度 ≈ 0.96 —— 代表"应入库"的合格文档
LONG_CN = "本文介绍检索增强生成的基本原理与工程实现路径。" * 6
# 195 字符，中文密度 0 —— 代表"英文内容，应被过滤"
LONG_EN = "This is an English paragraph used to verify the language filter. " * 3


# ============================================================================
# 一、去噪 denoise
# ============================================================================
def test_denoise_collapses_blank_lines():
    """3 个以上连续换行压成 1 个空行（保留段落间距，去掉堆叠空行）。"""
    assert denoise("第一段\n\n\n\n\n第二段") == "第一段\n\n第二段"


def test_denoise_removes_noise_lines():
    """版权/备案/转载行整行删除，正文行原样保留。"""
    text = "\n".join([
        "正文第一行",
        "版权所有 © 2024 某某公司",
        "正文第二行",
        "Copyright 2024 example.com",
        "正文第三行",
        "京ICP备12345678号",
        "转载自某博客",
        "正文第四行",
    ])
    out = denoise(text)
    for kw in ["版权所有", "Copyright", "京ICP", "转载"]:
        assert kw not in out
    for keep in ["正文第一行", "正文第二行", "正文第三行", "正文第四行"]:
        assert keep in out


def test_denoise_drops_link_directory_page():
    """链接 50 条、正文近乎为零 —— 典型目录页，整篇丢弃。"""
    page = "".join(f"- [第{i}篇](https://example.com/{i}.html)\n" for i in range(50))
    assert denoise(page) == ""


def test_denoise_keeps_long_article_with_many_links():
    """21 条链接 + 大段正文 = 正常技术长文，绝不能整篇丢掉。

    【为什么要有这个用例（回归旧 bug）】
        旧实现 `len(links) > 20` 直接 return ""，带"参考链接/相关阅读"的
        技术长文会整篇蒸发，而且日志里只看到"通过率下降"，极难定位。
        新规则要求"链接多" AND "正文少"两个条件同时成立。
    """
    body = "正文内容" * 200                                  # 800 字，远超 prose 阈值
    links = "\n".join(f"- [参考{i}](https://example.com/{i})" for i in range(21))
    out = denoise(body + "\n\n" + links)

    assert out != ""
    assert "正文内容" in out


def test_denoise_keeps_page_with_links_and_prose():
    """恰好 20 条链接但有大段正文：不是目录页，必须保留（边界）。"""
    page = (
        "".join(f"- [第{i}篇](https://example.com/{i}.html)\n" for i in range(20))
        + "本文正文说明" * 100                               # 600 字正文
    )
    assert denoise(page, min_links=20, max_prose=200) != ""


def test_denoise_directory_thresholds_are_tunable():
    """两个阈值都能覆盖，便于按真实语料校准。"""
    page = "".join(f"- [第{i}篇](https://e.com/{i})\n" for i in range(6)) + "说明"

    assert denoise(page) != ""                               # 默认 min_links=20，6 条不算
    assert denoise(page, min_links=5) == ""                  # 调到 5 → 判目录页
    assert denoise(page, min_links=5, max_prose=1) != ""     # prose 条件收紧后放行


def test_directory_thresholds_follow_settings():
    """默认阈值必须来自 settings，不是硬编码（否则调参不生效）。"""
    min_links = settings.CLEAN_DIR_PAGE_MIN_LINKS
    max_prose = settings.CLEAN_DIR_PAGE_MAX_PROSE

    exactly = "".join(f"- [{i}](https://e.com/{i})\n" for i in range(min_links))
    below = "".join(f"- [{i}](https://e.com/{i})\n" for i in range(min_links - 1))

    assert len(exactly) > 0 and max_prose > 0
    assert denoise(exactly) == ""            # 链接数刚好达到下限 + 无正文 → 目录页
    assert denoise(below) != ""              # 差一条 → 放行


def test_denoise_preserves_markdown_structure():
    """标题/代码块/表格必须原样保留（对应设计决策 1：不压纯文本）。"""
    text = "# 一级标题\n\n## 二级小节\n\n```python\nprint(1)\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    out = denoise(text)
    assert "# 一级标题" in out
    assert "```python" in out
    assert "| a | b |" in out


def test_denoise_is_idempotent():
    """幂等：重复清洗不应继续变化（增量重跑的前提）。"""
    text = "第一段\n\n\n\n第二段\n版权所有\n第三段"
    once = denoise(text)
    assert denoise(once) == once


def test_denoise_empty_and_none():
    """空串 / None 都返回空串，不抛异常。"""
    assert denoise("") == ""
    assert denoise(None) == ""


# ============================================================================
# 二、中文密度 chinese_density
# ============================================================================
def test_chinese_density_pure_chinese():
    assert chinese_density("机器学习") == pytest.approx(1.0)


def test_chinese_density_pure_english():
    assert chinese_density("machine learning") == pytest.approx(0.0)


def test_chinese_density_mixed():
    """分母是总字符数（含英文/空格），2/5 = 0.4。"""
    assert chinese_density("中文abc") == pytest.approx(0.4)


def test_chinese_density_empty_returns_zero():
    """空串不能除零。"""
    assert chinese_density("") == 0.0


def test_chinese_density_no_chinese_chars():
    assert chinese_density("1234567890 !@#$%") == 0.0


# ============================================================================
# 三、筛选 filter_document
# ============================================================================
def test_filter_document_boundary_exactly_min_chars_passes():
    """边界：len == 100 必须通过（代码是 <100 才淘汰）。"""
    text = "中" * 40 + "a" * 60
    assert len(text) == 100
    assert filter_document(text) == text


def test_filter_document_too_short_returns_none():
    text = "中" * 40 + "a" * 59          # 99 字
    assert filter_document(text) is None


def test_filter_document_low_density_returns_none():
    assert filter_document(LONG_EN) is None


def test_filter_document_custom_thresholds():
    """阈值可覆盖：默认值之外的分支也要能走通。"""
    text = "中文混排abcde"               # 4 个汉字 / 9 字符 ≈ 0.44
    assert filter_document(text, min_chars=5, min_density=0.4) == text
    assert filter_document(text, min_chars=5, min_density=0.5) is None
    assert filter_document(text, min_chars=10) is None


def test_filter_document_defaults_follow_settings():
    """不传参时阈值来自 settings（调 settings 立即生效，不用改代码）。"""
    n = settings.MIN_TEXT_LEN
    text = "中" * n                      # 恰好达标，密度 1.0
    assert filter_document(text) == text
    assert filter_document(text[:-1]) is None


# ============================================================================
# 四、汉明距离 hamming_distance
# ============================================================================
def test_hamming_distance_same_value():
    assert hamming_distance(0b1010, 0b1010) == 0
    assert hamming_distance(0, 0) == 0


def test_hamming_distance_diff_bits():
    assert hamming_distance(0b1010, 0b0101) == 4
    assert hamming_distance(0, 1 << 63) == 1


def test_hamming_distance_symmetric():
    assert hamming_distance(12345, 54321) == hamming_distance(54321, 12345)


# ============================================================================
# 五、切词与频次（_ngrams / _tokenize / _token_frequencies）
# ============================================================================
def test_ngrams_yields_all_bigrams():
    """2-gram 按"每 2 个连续字符"逐位滑动，顺序与切片一致。"""
    assert list(_ngrams("我爱北京")) == ["我爱", "爱北", "北京"]


def test_ngrams_short_text_returns_whole_string():
    assert list(_ngrams("我")) == ["我"]
    assert list(_ngrams("")) == []


def test_tokenize_returns_set():
    assert _tokenize("我爱北京") == {"我爱", "爱北", "北京"}
    assert _tokenize("aaaa") == {"aa"}          # 集合天然去重
    assert _tokenize("我") == {"我"}
    assert _tokenize("") == set()


def test_token_frequencies_keeps_repeats():
    """频次不能被 set 吃掉 —— 这是 SimHash 频次加权的全部意义。"""
    assert _token_frequencies("aaaa") == {"aa": 3}
    assert _token_frequencies("我爱北京") == {"我爱": 1, "爱北": 1, "北京": 1}
    assert len(_token_frequencies("")) == 0


# ============================================================================
# 六、稳定哈希 _stable_hash
# ============================================================================
def test_stable_hash_is_cached():
    """缓存确实生效：同一 token 第二次直接命中，不再算 blake2b。"""
    _stable_hash.cache_clear()
    _stable_hash("cache-probe", 64)
    _stable_hash("cache-probe", 64)

    info = _stable_hash.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_stable_hash_range():
    """返回值必须落在 hash_bits 位以内（否则拼指纹时高位越界）。"""
    for bits in (1, 8, 32, 64, 128):
        value = _stable_hash(f"range-{bits}", bits)
        assert 0 <= value < (1 << bits)


def test_stable_hash_rejects_invalid_bits():
    with pytest.raises(ValueError):
        _stable_hash("x", 0)
    with pytest.raises(ValueError):
        _stable_hash("x", 513)


def test_stable_hash_is_stable_across_processes():
    """同一 token 在不同 PYTHONHASHSEED 的进程里必须得到同一个值。

    【为什么必须跨进程测】
        内置 hash() 在本进程内是稳定的，只在本进程内断言毫无意义。
        指纹要落盘/跨进程比对，必须换个进程（换随机盐）还能对上。
    """
    code = (
        "from rag.ingestion.cleaner import _stable_hash;"
        "print(_stable_hash('检索增强生成的确定性测试', 64))"
    )

    def run(seed: str) -> str:
        env = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "PYTHONPATH": str(PROJECT_ROOT / "src")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    assert run("0") != ""
    assert run("0") == run("1")


# ============================================================================
# 七、SimHash 指纹 simhash
# ============================================================================
def test_simhash_empty_text_is_zero():
    """空文本没有任何 n-gram，权重全 0，指纹应为 0。"""
    assert simhash("") == 0


def test_simhash_returns_64bit_int():
    fp = simhash(LONG_CN)
    assert isinstance(fp, int)
    assert 0 <= fp < (1 << 64)


def test_simhash_respects_hash_bits():
    assert simhash(LONG_CN, hash_bits=32) < (1 << 32)
    assert simhash(LONG_CN, hash_bits=128) < (1 << 128)


def test_simhash_rejects_invalid_bits():
    with pytest.raises(ValueError):
        simhash(LONG_CN, hash_bits=0)


def test_simhash_golden_value():
    """黄金指纹：锁定"算法版本"。

    【为什么要有这个用例】
        指纹是落盘数据。切词方式 / 哈希函数 / 权重算法任一变化，这个值就会变，
        而库里所有历史指纹同时作废。让它在测试里"变红"，是唯一能在 CI 阶段
        拦住"悄悄换算法"的手段。若确属有意升级：改 SIMHASH_ALGO_VERSION、
        重算全量指纹、再更新这里的常量——三步缺一不可。

        取值命令：
            python -c "from rag.ingestion.cleaner import simhash; print(hex(simhash('检索增强生成')))"
    """
    assert SIMHASH_ALGO_VERSION == "blake2b-64-2gram-freq-v2"
    assert simhash("检索增强生成") == 0xD46132E16EAD45E0


def test_simhash_similar_texts_are_close():
    """局部敏感性：只改一个词，汉明距离应该明显小于无关文本。"""
    base = "机器学习是人工智能的一个分支，它研究如何让计算机从数据中学习规律并改进性能。" * 3
    variant = base.replace("计算机", "电脑")

    distance = hamming_distance(simhash(base), simhash(variant))
    assert distance <= 12, f"一个词的改动导致距离偏大：{distance}"


def test_simhash_unrelated_texts_are_far():
    """无关文本的指纹应显著拉开（64 位随机期望约 32）。"""
    a = "机器学习是人工智能的一个分支，它研究如何让计算机从数据中学习规律并改进性能。" * 3
    b = "红烧肉的做法：五花肉切块焯水，加冰糖老抽慢炖四十分钟收汁。"

    assert hamming_distance(simhash(a), simhash(b)) >= 30


def test_simhash_same_text_same_fingerprint():
    """同一文本连算两次必须完全一致（纯函数，无隐藏状态）。"""
    assert simhash(LONG_CN) == simhash(LONG_CN)


# ============================================================================
# 八、近似去重器 SimHashDeduplicator
# ============================================================================
def test_deduplicator_init():
    """构造函数必须真的初始化 threshold / hash_bits / fingerprints。

    【为什么要有这个用例（回归旧 bug）】
        旧代码把 `__init__` 写成了 `__int__`（拼写错误），对象创建后
        threshold 和 fingerprints 都没被赋值，一调 is_duplicate 就 AttributeError。
    """
    dedup = SimHashDeduplicator(threshold=5, hash_bits=32)
    assert dedup.threshold == 5
    assert dedup.hash_bits == 32
    assert dedup.fingerprints == []
    assert len(dedup) == 0


def test_deduplicator_rejects_invalid_args():
    with pytest.raises(ValueError):
        SimHashDeduplicator(threshold=-1)
    with pytest.raises(ValueError):
        SimHashDeduplicator(hash_bits=0)


def test_deduplicator_detects_same_text():
    dedup = SimHashDeduplicator()
    dedup.add(LONG_CN)

    assert len(dedup) == 1
    assert dedup.is_duplicate(LONG_CN) is True


def test_deduplicator_allows_distinct_text():
    dedup = SimHashDeduplicator()
    dedup.add(LONG_CN)

    assert dedup.is_duplicate(LONG_EN) is False


def test_deduplicator_empty_is_not_duplicate():
    """空库对任何文本都不应判重复（否则第一篇文章就被吃掉）。"""
    dedup = SimHashDeduplicator()
    assert dedup.is_duplicate(LONG_CN) is False


def test_deduplicator_threshold_matters():
    """阈值越小越严：同一个"改词"文本，阈值 0 判不重复、放宽后判重复。"""
    base = "机器学习是人工智能的一个分支，它研究如何让计算机从数据中学习规律并改进性能。" * 3
    variant = base.replace("计算机", "电脑")

    strict = SimHashDeduplicator(threshold=0)
    strict.add(base)
    assert strict.is_duplicate(variant) is False

    loose = SimHashDeduplicator(threshold=16)
    loose.add(base)
    assert loose.is_duplicate(variant) is True


# ============================================================================
# 九、组合入口 clean
# ============================================================================
def test_clean_passes_good_document():
    seen: set[str] = set()
    out = clean(LONG_CN, seen)

    assert out is not None
    assert out == denoise(LONG_CN)
    assert len(seen) == 1
    assert compute_content_hash(denoise(LONG_CN)) in seen


def test_clean_dedups_identical_text():
    """同一份内容第二次进来必须返回 None，且集合不重复膨胀。"""
    seen: set[str] = set()
    assert clean(LONG_CN, seen) is not None
    assert clean(LONG_CN, seen) is None
    assert len(seen) == 1


def test_clean_dedups_after_denoise_normalization():
    """去重指纹算在"去噪之后"，所以只差空行数的两份文本视为同一篇。"""
    raw_1 = LONG_CN[:50] + "\n\n\n\n" + LONG_CN[50:]
    raw_2 = LONG_CN[:50] + "\n\n" + LONG_CN[50:]

    seen: set[str] = set()
    assert clean(raw_1, seen) is not None
    assert clean(raw_2, seen) is None


def test_clean_drops_link_directory_page():
    seen: set[str] = set()
    page = "".join(f"- [第{i}篇](https://example.com/{i}.html)\n" for i in range(21))

    assert clean(page, seen) is None
    assert seen == set()             # 目录页在哈希之前就被淘汰，不污染集合


def test_clean_drops_too_short_document():
    seen: set[str] = set()
    assert clean("中" * 40 + "a" * 59, seen) is None


def test_clean_does_not_record_hash_for_rejected_doc():
    """被筛掉的文档不占去重位：同一篇再进来还是走筛选，而不是被误报为"重复"。"""
    seen: set[str] = set()
    short = "中" * 40 + "a" * 59

    assert clean(short, seen) is None
    assert clean(short, seen) is None
    assert seen == set()


def test_clean_drops_english_document():
    assert clean(LONG_EN, set()) is None


def test_clean_drops_empty_input():
    assert clean("", set()) is None
    assert clean("   \n\n  ", set()) is None


def test_clean_accepts_none_without_crashing():
    """.tex 这类不支持格式 parse_file 返回 None，clean 必须吃掉它。"""
    assert clean(None, set()) is None


def test_clean_rejects_non_str_input():
    """bytes / int 等非法输入直接跳过，不让它在 re.sub 里炸掉整批任务。"""
    assert clean(b"bytes", set()) is None       # type: ignore[arg-type]
    assert clean(123, set()) is None            # type: ignore[arg-type]


def test_clean_keeps_markdown_structure():
    text = "# 标题\n\n" + LONG_CN + "\n\n```python\nprint(1)\n```"
    out = clean(text, set())

    assert out is not None
    assert "# 标题" in out
    assert "```python" in out


# ============================================================================
# 十、去噪不能破坏代码块 / 近似去重接入主入口（2026-10-20 补）
# ============================================================================
def test_denoise_keeps_copyright_line_inside_code_fence():
    """代码围栏里的 Copyright 行是代码内容（许可头），不能被当页脚噪音删掉。

    【为什么要有这个用例】
        NOISE_KEYWORDS 含 "Copyright" 与 "转载"。真实语料里大量代码块开头是
        Apache/MIT 许可声明，旧实现逐行删除会把作者署名和许可条款从代码块里
        抹掉——结构没坏，但内容永久丢失，且没有任何日志提示。
    """
    text = "\n".join([
        "正文第一行，说明这段示例的用途，内容足够长以避免被当成短文档。",
        "```python",
        "# Copyright (c) 2024 The Authors",
        "# 版权所有 某某公司",              # 代码块内的中文版权行同样要保留
        "print('hello')",
        "```",
        "版权所有 © 2024 某某公司",        # 代码块外的页脚行，应删除
        "正文第二行，收尾。",
    ])
    out = denoise(text)

    assert "# Copyright (c) 2024 The Authors" in out
    assert "# 版权所有 某某公司" in out
    assert "print('hello')" in out
    assert "版权所有 © 2024 某某公司" not in out


def test_denoise_keeps_unclosed_fence_content():
    """未闭合围栏：其后内容一律按代码处理，不做噪音删除（结构性保守）。"""
    text = "```python\n# Copyright 2024\nprint(1)"
    out = denoise(text)

    assert "# Copyright 2024" in out


def test_clean_uses_simhash_deduplicator_when_provided():
    """传入 deduplicator 时，近似重复的文档要在主入口被拦下。

    【为什么要有这个用例】
        文档 2.1.2 的 ④ 明确要求去重 = 精确 + 近似，但旧版 clean() 只查 SHA1，
        SimHashDeduplicator 写了却没人调用——"看起来实现了，实际没接上"。
    """
    dedup = SimHashDeduplicator(threshold=16)      # 放宽阈值，模拟改写版
    seen: set[str] = set()

    base = "检索增强生成通过外挂知识库降低大模型幻觉。" * 6
    variant = "检索增强生成通过外挂知识库减少大模型幻觉。" * 6   # 改了一个词

    assert clean(base, seen, deduplicator=dedup) is not None
    assert len(dedup) == 1
    assert clean(variant, seen, deduplicator=dedup) is None        # 近似重复被拦
    assert len(dedup) == 1


def test_clean_without_deduplicator_keeps_old_behavior():
    """不传 deduplicator 时行为与旧版一致（只按 SHA1 判重），保证向后兼容。"""
    seen: set[str] = set()
    base = "检索增强生成通过外挂知识库降低大模型幻觉。" * 6
    variant = "检索增强生成通过外挂知识库减少大模型幻觉。" * 6

    assert clean(base, seen) is not None
    assert clean(variant, seen) is not None        # 只有精确去重，不拦近似重复


def test_clean_does_not_register_rejected_doc_in_deduplicator():
    """被筛掉的文档不能登记进 deduplicator，否则同内容的下一篇会被误判成重复。"""
    dedup = SimHashDeduplicator(threshold=3)
    seen: set[str] = set()
    short = "中" * 40 + "a" * 59                   # 长度不足，会被筛掉

    assert clean(short, seen, deduplicator=dedup) is None
    assert len(dedup) == 0
    assert seen == set()
