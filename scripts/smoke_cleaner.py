# ============================================================================
# smoke_cleaner.py —— 文档清洗（cleaner.py）的真实语料冒烟测试
#
# 位置：scripts/smoke_cleaner.py
#
# 【本文件是干什么的】
#   单元测试只能证明"逻辑符合我写的断言"，证明不了"在真实数据上能用、
#   而且去噪/去重/筛选三个分支真的被走到了"。本脚本用 data/raw 下的真实
#   语料跑 parse → clean，输出统计并做硬性断言。
#
# 【和 smoke_parse.py 的分工】
#   smoke_parse：只管"解析出来没有"，统计成功率与耗时。
#   smoke_cleaner：管"洗得对不对"——通过率、淘汰原因分布、输出不变量、
#                  去重是否真的生效、近似去重（可选）能否跑通。
#
# 【为什么按后缀抽样】
#   实测 html 单篇约 294ms（lxml + markdownify），全量 1697 篇要 8 分钟。
#   按后缀分配样本量：快的多抽、慢的少抽，总时长控制在 1 分钟内。
#
# 【为什么每个文件都要 try/except】
#   异常清单本身就是产出物。让第一个异常中断脚本，就只知道"崩了"，
#   不知道"崩了几篇、崩在哪类"。所以按异常类型归类统计，最后统一输出。
#
# 【用法】
#   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/smoke_cleaner.py
#   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/smoke_cleaner.py --with-simhash
#   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/smoke_cleaner.py --suffix .md --limit 100
#
#   注意 PYTHONIOENCODING=utf-8 必须加：Windows 控制台默认 GBK，
#   打印中文表格会乱码。
#
# 【退出码】
#   0 = 通过；1 = 断言失败（链路异常/通过率越界/不变量破坏/超时）；2 = 环境缺失
# ============================================================================

# ---------- 导入区 ----------
# argparse：命令行参数
# json：报告落盘
# random：抽样（固定种子保证可复现）
# sys：退出码
# time：计时
# Counter：统计计数
# defaultdict：按后缀分组
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rag.config.settings import settings
from rag.ingestion.cleaner import (
    SIMHASH_ALGO_VERSION,
    SimHashDeduplicator,
    chinese_density,
    clean,
    denoise,
)
from rag.ingestion.parse import parse_file
from rag.ingestion.registry import compute_content_hash
from rag.utils.logger import setup_logger

log = setup_logger(log_dir=settings.LOG_DIR)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# 每种后缀抽多少个。依据 smoke_parse.py 的实测耗时分配，目标总时长 < 60 秒。
SAMPLE_SIZE = {
    ".md": 200,
    ".mdx": 150,
    ".txt": 100,
    ".rst": 50,
    ".html": 40,      # 294ms/篇，必须少抽
    ".ipynb": 80,
    ".xml": 7,        # 语料里只有 7 个，全跑
}

# 固定随机种子：每次抽到同一批文件，结果可复现、可对比
RANDOM_SEED = 42

# 报告落盘位置
REPORT_PATH = settings.LOG_DIR / "smoke_cleaner_report.json"


# ---------------------------------------------------------------------------
# 收集文件：按后缀分组
# ---------------------------------------------------------------------------
def collect_files(root: Path) -> dict[str, list[Path]]:
    """遍历语料目录，按后缀把文件分组。

    【为什么过滤 __MACOSX】
        macOS 解压 zip 会生成 __MACOSX 目录，里面全是 `._` 开头的资源垃圾，
        实测 raw/ 残留里近 2 万个，不过滤会严重干扰统计。

    参数：
        root : Path —— 语料根目录

    返回：
        dict[str, list[Path]] —— {后缀: 文件列表}
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts:
            continue
        groups[path.suffix.lower()].append(path)
    return groups


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------
def pick_samples(
    groups: dict[str, list[Path]],
    suffixes: set[str] | None,
    limit: int,
    scale: float = 1.0,
) -> tuple[list[Path], dict[str, int]]:
    """按后缀固定比例抽样，返回样本列表与实际抽样数。

    参数：
        groups   : dict —— 按后缀分组后的文件
        suffixes : set | None —— 只跑这些后缀；None 表示用默认 SAMPLE_SIZE
        limit    : int —— 总样本上限，0 表示不限制
        scale    : float —— 每类样本量放大倍数（实测精确重复率约 1.8%，
                   默认样本量很难触达该分支，需要时用 --scale 放大）

    返回：
        (样本列表, {后缀: 抽样数})
    """
    random.seed(RANDOM_SEED)
    picked: list[Path] = []
    plan: dict[str, int] = {}

    for suffix, size in SAMPLE_SIZE.items():
        if suffixes and suffix not in suffixes:
            continue
        available = groups.get(suffix, [])
        if not available:
            continue
        n = min(max(int(size * scale), 1), len(available))
        picked.extend(random.sample(available, n))
        plan[suffix] = n

    if limit > 0:
        picked = picked[:limit]

    return picked, plan


# ---------------------------------------------------------------------------
# 跑一批文件
# ---------------------------------------------------------------------------
def run(files: list[Path], use_simhash: bool) -> dict:
    """对一批文件逐个跑 parse → clean，统计各分支命中情况。

    参数：
        files        : list[Path] —— 样本文件
        use_simhash  : bool —— 是否额外跑 SimHash 近似去重

    返回：
        dict —— counts / problems / warnings / samples / dedup_size / elapsed
    """
    counts: Counter = Counter()
    problems: list[str] = []
    warnings: list[str] = []
    samples: list[tuple[str, str]] = []

    # 跨文档共享：精确去重集合（SHA1）
    seen: set[str] = set()
    # 近似去重器（可选）：只在文档真正通过 clean 之后才登记指纹
    dedup = SimHashDeduplicator(threshold=3) if use_simhash else None

    started = time.time()

    for path in files:
        counts["files"] += 1

        # ---- 解析 ----
        try:
            raw = parse_file(path)
        except Exception as exc:                       # noqa: BLE001 —— 故意全捕
            counts["parse_error"] += 1
            problems.append(f"解析异常 {path}: {type(exc).__name__}: {exc}")
            continue

        if not raw or not raw.strip():
            counts["parse_empty"] += 1
            continue

        # ---- 清洗 ----
        # snapshot 必须在 clean 之前取：clean 会往 seen 里写指纹，
        # 用写后的集合判断"是否重复"会永远为 True。
        snapshot = set(seen)
        try:
            out = clean(raw, seen)
        except Exception as exc:                       # noqa: BLE001
            counts["clean_error"] += 1
            problems.append(f"清洗异常 {path}: {type(exc).__name__}: {exc}")
            continue

        if out is None:
            # 按 clean 内部的判定顺序（去噪 → 精确去重 → 筛选）反推淘汰原因
            denoised = denoise(raw)
            if not denoised.strip():
                counts["drop_empty"] += 1              # 目录页 / 全被噪音行吃空
            elif compute_content_hash(denoised) in snapshot:
                counts["drop_duplicate"] += 1          # 精确重复
            elif len(denoised) < settings.MIN_TEXT_LEN:
                counts["drop_short"] += 1
            else:
                counts["drop_low_density"] += 1
            continue

        counts["passed"] += 1

        # ---- 输出不变量（接线自检）----
        # ① clean 的返回值必须等于"去噪后的原文"，不允许偷偷改内容
        if out != denoise(raw):
            problems.append(f"不变量破坏(输出≠去噪结果) {path}")
        # ② 通过筛选的文档必须已登记指纹（否则下一轮会重复入库）
        if compute_content_hash(out) not in seen:
            problems.append(f"不变量破坏(指纹未登记) {path}")
        # ③ 阈值必须真的被遵守（防"clean 忘了传参"这类接线错误）
        if len(out) < settings.MIN_TEXT_LEN:
            problems.append(f"不变量破坏(长度) {path}: {len(out)}")
        if chinese_density(out) < settings.MIN_CHINESE_RATIO:
            problems.append(f"不变量破坏(密度) {path}: {chinese_density(out):.3f}")

        if len(samples) < 3:
            samples.append((path.name, out[:80].replace("\n", " ")))

        # ---- 近似去重（可选）----
        if dedup is not None:
            try:
                if dedup.is_duplicate(out):
                    counts["simhash_duplicate"] += 1
                else:
                    dedup.add(out)
            except Exception as exc:                   # noqa: BLE001
                warnings.append(f"SimHash 不可用，已降级：{type(exc).__name__}: {exc}")
                dedup = None

    elapsed = time.time() - started

    return {
        "counts": counts,
        "problems": problems,
        "warnings": warnings,
        "samples": samples,
        "dedup_size": len(dedup) if dedup is not None else 0,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cleaner.py 真实语料冒烟测试")
    parser.add_argument("--dir", default=str(settings.RAW_DIR), help="语料根目录")
    parser.add_argument(
        "--suffix",
        action="append",
        default=None,
        help="只跑指定后缀（可重复，如 --suffix .md --suffix .html）",
    )
    parser.add_argument("--limit", type=int, default=0, help="总样本上限，0 表示不限")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="每类样本量放大倍数（默认 1.0；实测精确重复率约 1.8%%，需放大才能触达去重分支）",
    )
    parser.add_argument("--with-simhash", action="store_true", help="额外跑 SimHash 近似去重")
    parser.add_argument("--min-pass-rate", type=float, default=0.01, help="通过率下限")
    parser.add_argument(
        "--max-pass-rate", type=float, default=0.99, help="通过率上限（防筛选形同虚设）"
    )
    parser.add_argument("--max-seconds", type=float, default=120.0, help="耗时上限（秒）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.dir)

    print(f"语料根目录：{root}")
    print(f"清洗阈值：MIN_TEXT_LEN={settings.MIN_TEXT_LEN}  "
          f"MIN_CHINESE_RATIO={settings.MIN_CHINESE_RATIO}  "
          f"DIR_PAGE_MIN_LINKS={settings.CLEAN_DIR_PAGE_MIN_LINKS}  "
          f"DIR_PAGE_MAX_PROSE={settings.CLEAN_DIR_PAGE_MAX_PROSE}")
    print(f"SimHash 算法：{SIMHASH_ALGO_VERSION}\n")

    if not root.is_dir():
        print(f"[SMOKE] 语料目录不存在 -> {root}")
        return 2

    groups = collect_files(root)
    print(f"共发现 {sum(len(v) for v in groups.values())} 个文件，分 {len(groups)} 种后缀")

    suffixes = set(args.suffix) if args.suffix else None
    files, plan = pick_samples(groups, suffixes, args.limit, args.scale)
    if not files:
        print("[SMOKE] 没有抽到可解析样本（检查 --suffix / 语料目录）")
        return 2

    print(f"抽样计划：{plan}")
    limit_note = f"，--limit={args.limit}" if args.limit else ""
    print(f"实际样本：{len(files)} 个{limit_note}\n")
    print(f"[探测] 开始清洗，近似去重={'开' if args.with_simhash else '关'} ...")

    result = run(files, args.with_simhash)
    counts: Counter = result["counts"]
    problems: list[str] = result["problems"]
    warnings: list[str] = result["warnings"]
    elapsed: float = result["elapsed"]

    pass_rate = counts["passed"] / counts["files"] if counts["files"] else 0.0

    # ---- 报告表 ----
    print("\n" + "=" * 78)
    print("清洗冒烟测试报告")
    print("=" * 78)
    rows = [
        ("样本文件数", counts["files"]),
        ("解析为空", counts["parse_empty"]),
        ("清洗通过", counts["passed"]),
        ("丢弃·空文档/目录页", counts["drop_empty"]),
        ("丢弃·精确重复", counts["drop_duplicate"]),
        ("丢弃·正文过短", counts["drop_short"]),
        ("丢弃·中文密度低", counts["drop_low_density"]),
        ("异常（解析+清洗）", counts["parse_error"] + counts["clean_error"]),
    ]
    if args.with_simhash:
        rows.append(("丢弃·近似重复", counts["simhash_duplicate"]))
        rows.append(("已登记指纹数", result["dedup_size"]))

    for label, value in rows:
        print(f"  {label:<22}{value:>6}")

    print(f"  {'通过率':<22}{pass_rate:>6.1%}")
    print(f"  {'耗时':<22}{elapsed:>6.2f}s   （{elapsed / max(counts['files'], 1) * 1000:.1f} ms/篇）")

    # ---- 通过样本 ----
    if result["samples"]:
        print("\n通过样本（前 80 字）：")
        for name, preview in result["samples"]:
            print(f"  - {name}: {preview}")

    # ---- 软告警：分支触达（样本量小时可能天然缺失，不判失败）----
    for key, label in (
        ("drop_empty", "空文档/目录页"),
        ("drop_duplicate", "精确重复"),
        ("drop_short", "正文过短"),
        ("drop_low_density", "中文密度过低"),
    ):
        if counts[key] == 0:
            warnings.append(f"{label}分支未被样本触达（建议加大样本或换 --suffix）")

    if warnings:
        print("\n告警：")
        for item in warnings:
            print(f"  ! {item}")

    # ---- 硬断言 ----
    failures = list(problems)
    if counts["parse_error"] or counts["clean_error"]:
        failures.append("存在解析/清洗异常，链路不稳定")
    if counts["passed"] == 0:
        failures.append("没有任何文档通过清洗：阈值过严或语料选取有误")
    if not (args.min_pass_rate <= pass_rate <= args.max_pass_rate):
        failures.append(
            f"通过率 {pass_rate:.1%} 不在 [{args.min_pass_rate:.0%}, {args.max_pass_rate:.0%}] 区间"
        )
    if elapsed > args.max_seconds:
        failures.append(f"耗时 {elapsed:.1f}s 超过上限 {args.max_seconds:.0f}s")

    if failures:
        print("\n失败：")
        for item in failures[:20]:
            print(f"  x {item}")
        if len(failures) > 20:
            print(f"  ... 另有 {len(failures) - 20} 条未列出")
        print("\n[SMOKE] FAILED")
        verdict = "FAILED"
    else:
        print("\n[SMOKE] PASSED")
        verdict = "PASSED"

    # ---- 落盘 JSON（便于与历史结果对比）----
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": verdict,
        "raw_dir": str(root),
        "simhash_algo": SIMHASH_ALGO_VERSION,
        "with_simhash": args.with_simhash,
        "sample_plan": plan,
        "scale": args.scale,
        "thresholds": {
            "MIN_TEXT_LEN": settings.MIN_TEXT_LEN,
            "MIN_CHINESE_RATIO": settings.MIN_CHINESE_RATIO,
            "CLEAN_DIR_PAGE_MIN_LINKS": settings.CLEAN_DIR_PAGE_MIN_LINKS,
            "CLEAN_DIR_PAGE_MAX_PROSE": settings.CLEAN_DIR_PAGE_MAX_PROSE,
        },
        "counts": dict(counts),
        "pass_rate": round(pass_rate, 4),
        "elapsed_sec": round(elapsed, 2),
        "problems": problems,
        "warnings": warnings,
    }
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已保存：{REPORT_PATH}")

    return 0 if verdict == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
