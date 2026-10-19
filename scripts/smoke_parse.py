# ============================================================================
# smoke_parse.py —— 文档解析（parse.py）的真实语料冒烟测试
#
# 位置：scripts/smoke_parse.py
#
# 【本文件是干什么的】
#   单元测试只能证明"逻辑符合我写的断言"，证明不了"在真实数据上能用"。
#   真实语料 15658 个文件千奇百怪（HTML 结构各异、ipynb 格式不统一、
#   甚至有伪装成 .ipynb 的 HTML），必须跑一遍真实数据才能暴露问题。
#
# 【为什么抽样而不全量】
#   实测各格式单篇耗时：md 约 1.2ms，而 html 约 294ms（lxml 解析 +
#   markdownify 转换是大头）。1697 篇 HTML 全量要8 分钟，对一次性验证太慢。
#   所以按格式抽样：快的格式多抽，html 少抽，总时长控制在 1 分钟内。
#
# 【为什么必须 try/except 包住每个文件】
#   异常清单本身就是这次冒烟的产出物。如果让第一个异常中断脚本，
#   就只能看到"崩了"，看不到"哪些崩了、崩的比例是多少"。所以按异常类型
#   归类统计，最后一并输出。
#
# 【用法】
#   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/smoke_parse.py
#
#   注意 PYTHONIOENCODING=utf-8 必须加：Windows 控制台默认 GBK，
#   打印中文表格会乱码。
# ============================================================================

# ---------- 导入区 ----------
# json：落盘报告用
# random：抽样用（固定种子保证可复现）
# statistics：中位数
# time：计时
# Counter：异常类型计数
# defaultdict：按后缀分组
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from rag.config.settings import settings
from rag.ingestion.parse import parse_file
from rag.utils.logger import setup_logger
log = setup_logger(log_dir=settings.LOG_DIR)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# 每种后缀抽多少个文件。html 慢，少抽；md 快，多抽。
# 抽样量依据实测性能定的，目标总时长 < 60 秒。
SAMPLE_SIZE = {
    ".html": 120,     # 294ms/篇，必须少抽；全量 1697 篇要 8 分钟
    ".ipynb": 200,
    ".md": 200,
    ".mdx": 200,
    ".txt": 150,
    ".rst": 150,
    ".xml": 7,        # 总共只有 7 个，全跑
}

# 固定随机种子，保证每次抽样结果一致、可复现、可对比
RANDOM_SEED = 42

# 报告落盘位置
REPORT_PATH = settings.LOG_DIR / "smoke_parse_report.json"


# ---------------------------------------------------------------------------
# 收集文件：按后缀分组
# ---------------------------------------------------------------------------
def collect_files(root: Path) -> dict[str, list[Path]]:
    """遍历语料目录，按后缀把文件分组。

    【为什么要过滤 __MACOSX】
        macOS 解压 zip 时会生成 __MACOSX 目录，里面是 ._开头的资源文件，
        全是垃圾。实测 raw/ 残留里有 19692 个，不过滤会严重干扰统计。
       正常语料在 data/raw（settings.RAW_DIR）里本来就没有，
        这里加过滤是双保险。

    【为什么要用 Path 对象而不是拼字符串】
        真实语料里有 3884 个中文文件名、1341 个含空格的文件名。
        Path 对象能正确处理，绝不能手拼字符串传给 shell。

    参数：
        root : Path —— 语料根目录

    返回：
        dict[str, list[Path]] —— {后缀: 文件列表}
    """
    groups: dict[str, list[Path]] = defaultdict(list)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts: # 跳过 macOS 垃圾目录
            continue
        # 用小写后缀归组，这样 AGENT.MD 和 x.md 会被分到同一个 .md 组
        groups[path.suffix.lower()].append(path)

    return groups


# ---------------------------------------------------------------------------
# 探测一批文件
# ---------------------------------------------------------------------------
def probe(files: list[Path]) -> dict:
    """对一批文件逐个跑 parse_file，统计成功率和长度分布。

    【核心原则】每个文件都用 try/except 包住，一个都不能让异常中断整个脚本。

    参数：
        files : list[Path] —— 要跑的文件

    返回：
        dict —— 统计结果（成功数/空串数/异常分布/长度分布/耗时）
    """
    ok = 0              # 解析成功且有内容
    empty = 0           # 解析成功但内容为空串
    error = 0# 抛了异常
    lengths: list[int] = [] # 有内容的文本长度，用于算分布
    error_types: Counter = Counter()  # 异常类型 -> 次数
    error_samples: dict[str, str] = {}  # 异常类型 -> 一个示例文件路径

    start = time.time()

    for path in files:
        try:
            text = parse_file(path)
        except Exception as exc:                # noqa: BLE001 —— 故意捕获全部
            error += 1
            name = type(exc).__name__
            error_types[name] += 1
            # 每种异常类型只记一个示例，方便回头排查
            error_samples.setdefault(name, str(path))
            continue

        if text is None:
            # 不支持的后缀，返回 None。不算错误。
            ok += 1
        elif not text.strip():
            empty += 1
        else:
            ok += 1
            lengths.append(len(text))

    elapsed = time.time() - start

    return {
        "total": len(files),
        "ok": ok,
        "empty": empty,
        "error": error,
        "error_types": dict(error_types),
        "error_samples": error_samples,
        "min_len": min(lengths) if lengths else 0,
        "median_len": int(statistics.median(lengths)) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "elapsed_sec": round(elapsed, 1),
        "ms_per_file": round(elapsed / len(files) * 1000, 1) if files else 0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    """抽样跑真实语料，打印报告表并落盘 JSON。"""
    root = settings.RAW_DIR
    print(f"语料根目录：{root}")

    if not root.exists():
        print(f"错误：语料目录不存在 -> {root}")
        return

    groups = collect_files(root)
    print(f"共发现 {sum(len(v) for v in groups.values())} 个文件，"
          f"分 {len(groups)} 种后缀\n")

    # 抽样 + 逐类探测
    random.seed(RANDOM_SEED)
    report: dict[str, dict] = {}
    for suffix, size in SAMPLE_SIZE.items():
        available = groups.get(suffix, [])
        if not available:
            print(f"[跳过] {suffix} —— 语料里没有这类文件")
            continue

        # 样本量不超过实际数量
        n = min(size, len(available))
        sample = random.sample(available, n)
        print(f"[探测] {suffix} 抽取 {n}/{len(available)} 个 ...")

        report[suffix] = probe(sample)

    # ---- 控制台报告表 ----
    print("\n" + "=" * 78)
    print("解析冒烟测试报告")
    print("=" * 78)
    header = f"{'后缀':<8}{'样本':>6}{'成功':>6}{'空串':>6}{'异常':>6}" \
             f"{'最短':>8}{'中位':>8}{'最长':>8}{'ms/篇':>8}"
    print(header)
    print("-" * 78)

    for suffix, r in report.items():
        # 成功率 = 成功数/ 总数（异常算失败）
        rate = f"{r['ok'] / r['total'] * 100:.1f}%" if r["total"] else "-"
        print(f"{suffix:<8}{r['total']:>6}{rate:>6}"
              f"{r['empty']:>6}{r['error']:>6}"
              f"{r['min_len']:>8}{r['median_len']:>8}{r['max_len']:>8}"
              f"{r['ms_per_file']:>8}")

    # ---- 异常明细 ----
    print("\n" + "=" * 78)
    print("异常明细")
    print("=" * 78)
    has_error = False
    for suffix, r in report.items():
        if r["error_types"]:
            has_error = True
            print(f"\n[{suffix}] 共 {r['error']} 个异常：")
            for exc_name, count in r["error_types"].items():
                sample_path = r["error_samples"].get(exc_name, "")
                print(f"    {exc_name:<20} {count:>4} 次")
                print(f"      示例：{sample_path}")

    if not has_error:
        print("\n无异常。")

    # ---- 落盘 JSON ----
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_dir": str(root),
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "results": report,
    }
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已保存：{REPORT_PATH}")


if __name__ == "__main__":
    main()
