# ============================================================================
# smoke_splitter.py —— 分块器（splitter.py）的真实语料冒烟测试
#
# 位置：scripts/smoke_splitter.py
#
# 【本文件是干什么的】
#   单元测试证明"逻辑符合断言"，证明不了"在 3.5 万个真实文件上不会崩、块长得
#   像人话"。本脚本用 data/raw 下的真实语料跑 parse → clean → split，输出
#   分块质量报表，并对"结构性不变量"做硬断言。
#
# 【和 smoke_cleaner.py 的分工】
#   smoke_cleaner：管"洗得对不对"（通过率、淘汰原因、去重是否生效）
#   smoke_splitter：管"切得对不对"——块长分布、可回溯性、父子映射、
#                   代码块标记、纯标题碎片是否为 0
#
# 【硬断言 vs 软告警】
#   硬断言只放"错了就一定是 bug"的不变量（可回溯、不超限、ID 唯一、不丢内容）。
#   块长均值、碎片比例这类"质量指标"放软告警——它们受语料构成影响，
#   不该让脚本因为一批全是短文档的样本而失败。
#
# 【用法】
#   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/smoke_splitter.py
#   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/smoke_splitter.py --limit 100
#   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/smoke_splitter.py --suffix .md
#
#   注意 PYTHONIOENCODING=utf-8 必须加：Windows 控制台默认 GBK，中文报表会乱码。
#
# 【退出码】
#   0 = 通过；1 = 断言失败；2 = 环境缺失（语料目录/样本不足）
# ============================================================================

# ---------- 导入区 ----------
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rag.config.settings import settings
from rag.ingestion.cleaner import clean
from rag.ingestion.parse import parse_file
from rag.ingestion.splitter import _has_prose, split_document
from rag.utils.logger import setup_logger

log = setup_logger(log_dir=settings.LOG_DIR)

# 每种后缀抽多少个：目标总时长 < 60 秒（html 慢，少抽）
SAMPLE_SIZE = {
    ".md": 120,
    ".mdx": 80,
    ".txt": 60,
    ".rst": 30,
    ".html": 20,
    ".ipynb": 40,
    ".xml": 7,
}

RANDOM_SEED = 42
REPORT_PATH = settings.LOG_DIR / "smoke_splitter_report.json"

# 质量软告警阈值
MIN_AVG_CHUNK = 120      # 平均块长低于此值：切得太碎
MAX_SMALL_RATIO = 0.05   # 20 字以下碎片的占比上限
MAX_FLAT_RATIO = 0.50    # 无标题文档占比上限（标题切分命中率过低）


def collect_files(root: Path) -> dict[str, list[Path]]:
    """遍历语料目录，按后缀分组（过滤 macOS 解压垃圾）。"""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts:
            continue
        groups[path.suffix.lower()].append(path)
    return groups


def pick_samples(groups: dict[str, list[Path]], suffixes: set[str] | None) -> list[Path]:
    """按后缀固定比例抽样（固定种子，结果可复现）。"""
    random.seed(RANDOM_SEED)
    picked: list[Path] = []
    for suffix, size in SAMPLE_SIZE.items():
        if suffixes and suffix not in suffixes:
            continue
        available = groups.get(suffix, [])
        if available:
            picked.extend(random.sample(available, min(size, len(available))))
    return picked


def run(files: list[Path]) -> dict:
    """对一批文件跑 parse → clean → split，并逐块校验不变量。"""
    counts: Counter = Counter()
    problems: list[str] = []
    warnings: list[str] = []
    samples: list[tuple[str, str]] = []
    chunk_lens: list[int] = []
    per_doc: list[int] = []

    seen: set[str] = set()
    started = time.time()

    for path in files:
        counts["files"] += 1

        try:
            raw = parse_file(path)
        except Exception as exc:                        # noqa: BLE001 —— 故意全捕
            counts["parse_error"] += 1
            problems.append(f"解析异常 {path}: {type(exc).__name__}: {exc}")
            continue
        if not raw or not raw.strip():
            counts["parse_empty"] += 1
            continue

        try:
            cleaned = clean(raw, seen)
        except Exception as exc:                        # noqa: BLE001
            counts["clean_error"] += 1
            problems.append(f"清洗异常 {path}: {type(exc).__name__}: {exc}")
            continue
        if cleaned is None:
            counts["cleaned_out"] += 1
            continue

        doc_id = path.as_posix().split("data/raw/")[-1]
        try:
            chunks = split_document(cleaned, doc_id=doc_id, source=path.as_posix())
        except Exception as exc:                        # noqa: BLE001
            counts["split_error"] += 1
            problems.append(f"分块异常 {path}: {type(exc).__name__}: {exc}")
            continue

        counts["docs"] += 1
        counts["chunks"] += len(chunks)
        per_doc.append(len(chunks))

        if not chunks:
            problems.append(f"合格文档切出 0 块 {path}")
            continue

        # ---- 逐块不变量 ----
        ids = [c["chunk_id"] for c in chunks]
        if len(ids) != len(set(ids)):
            problems.append(f"chunk_id 重复 {path}")

        if [c["index"] for c in chunks] != list(range(len(chunks))):
            problems.append(f"index 不连续 {path}")

        parent_texts: dict[str, str] = {}

        for c in chunks:
            chunk_lens.append(c["char_len"])

            # ① 可回溯：text 必须严格等于原文切片（这是引用溯源的地基）
            if c["text"] != cleaned[c["char_start"]:c["char_end"]]:
                problems.append(f"不可回溯 {path} {c['chunk_id']}")
            # ② 不超限
            if c["char_len"] > settings.CHUNK_SIZE:
                problems.append(f"块超限 {path} {c['chunk_id']}: {c['char_len']}")
            # ③ 不允许纯标题碎片
            if not _has_prose(c["text"]):
                problems.append(f"纯标题碎片 {path} {c['chunk_id']}")
            # ④ 父子映射：同一 parent_id 的 parent_text 必须一致
            if c["parent_id"] in parent_texts:
                if parent_texts[c["parent_id"]] != c["parent_text"]:
                    problems.append(f"parent_text 不一致 {path} {c['parent_id']}")
            else:
                parent_texts[c["parent_id"]] = c["parent_text"]
            # ⑤ 子块必须落在父块里
            if c["char_len"] and c["text"] not in c["parent_text"]:
                problems.append(f"子块不在父块内 {path} {c['chunk_id']}")
            # ⑥ heading_path 必须是 list（哪怕空）
            if not isinstance(c["heading_path"], list):
                problems.append(f"heading_path 类型错 {path} {c['chunk_id']}")
            # ⑦ 空块不该产出
            if not c["text"].strip():
                problems.append(f"空块 {path} {c['chunk_id']}")

            if c["is_code"]:
                counts["code_chunks"] += 1
            if c["heading_path"]:
                counts["headed_chunks"] += 1
            if c["char_len"] < 20:
                counts["tiny_chunks"] += 1

        # ---- 文档级：内容覆盖度（防漏切丢字）----
        covered = sum(c["char_len"] for c in chunks)
        if covered < len(cleaned) * 0.90:
            problems.append(
                f"覆盖度过低 {path}: 块长合计 {covered} / 正文 {len(cleaned)}"
            )

        if not any(c["heading_path"] for c in chunks):
            counts["flat_docs"] += 1

        if len(samples) < 3:
            c = max(chunks, key=lambda x: x["char_len"])
            samples.append((path.name, c["text"][:80].replace("\n", " ")))

    elapsed = time.time() - started

    return {
        "counts": counts,
        "problems": problems,
        "warnings": warnings,
        "samples": samples,
        "chunk_lens": chunk_lens,
        "per_doc": per_doc,
        "elapsed": elapsed,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="splitter.py 真实语料冒烟测试")
    parser.add_argument("--dir", default=str(settings.RAW_DIR), help="语料根目录")
    parser.add_argument(
        "--suffix", action="append", default=None,
        help="只跑指定后缀（可重复，如 --suffix .md --suffix .html）",
    )
    parser.add_argument("--limit", type=int, default=0, help="总样本上限，0 表示不限")
    parser.add_argument("--max-seconds", type=float, default=180.0, help="耗时上限（秒）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.dir)

    print(f"语料根目录：{root}")
    print(f"分块参数：CHUNK_SIZE={settings.CHUNK_SIZE}  "
          f"CHUNK_OVERLAP={settings.CHUNK_OVERLAP}  "
          f"PARENT_CHUNK_SIZE={settings.PARENT_CHUNK_SIZE}\n")

    if not root.is_dir():
        print(f"[SMOKE] 语料目录不存在 -> {root}")
        return 2

    groups = collect_files(root)
    print(f"共发现 {sum(len(v) for v in groups.values())} 个文件")

    files = pick_samples(groups, set(args.suffix) if args.suffix else None)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print("[SMOKE] 没有抽到可解析样本（检查 --suffix / 语料目录）")
        return 2
    print(f"实际样本：{len(files)} 个\n")

    print("[探测] 开始 parse → clean → split ...")
    result = run(files)
    counts: Counter = result["counts"]
    problems: list[str] = result["problems"]
    warnings: list[str] = result["warnings"]
    lens: list[int] = result["chunk_lens"]
    per_doc: list[int] = result["per_doc"]
    elapsed: float = result["elapsed"]

    avg_len = sum(lens) / len(lens) if lens else 0.0
    tiny_ratio = counts["tiny_chunks"] / counts["chunks"] if counts["chunks"] else 0.0
    flat_ratio = counts["flat_docs"] / counts["docs"] if counts["docs"] else 0.0

    print("\n" + "=" * 78)
    print("分块冒烟测试报告")
    print("=" * 78)
    rows = [
        ("样本文件数", counts["files"]),
        ("解析为空", counts["parse_empty"]),
        ("被清洗淘汰", counts["cleaned_out"]),
        ("参与分块的文档", counts["docs"]),
        ("产出块数", counts["chunks"]),
        ("平均块长", f"{avg_len:.1f}"),
        ("最长块", max(lens) if lens else 0),
        ("<20 字碎片", counts["tiny_chunks"]),
        ("is_code 块", counts["code_chunks"]),
        ("带章节路径的块", counts["headed_chunks"]),
        ("无标题文档", counts["flat_docs"]),
        ("异常（解析/清洗/分块）",
         counts["parse_error"] + counts["clean_error"] + counts["split_error"]),
    ]
    for label, value in rows:
        print(f"  {label:<22}{value!s:>8}")

    if per_doc:
        per_doc_sorted = sorted(per_doc)
        mid = len(per_doc_sorted) // 2
        print(f"  {'每篇块数(中位)':<20}{per_doc_sorted[mid]:>8}")
        print(f"  {'每篇块数(最大)':<20}{max(per_doc):>8}")
    print(f"  {'耗时':<22}{elapsed:>7.2f}s  "
          f"（{counts['chunks'] / max(elapsed, 1e-9):.0f} 块/秒）")

    if result["samples"]:
        print("\n最长块样本（前 80 字）：")
        for name, preview in result["samples"]:
            print(f"  - {name}: {preview}")

    # ---- 软告警：质量指标 ----
    if avg_len < MIN_AVG_CHUNK:
        warnings.append(f"平均块长 {avg_len:.0f} < {MIN_AVG_CHUNK}，切得过碎")
    if tiny_ratio > MAX_SMALL_RATIO:
        warnings.append(f"20 字以下碎片占比 {tiny_ratio:.1%} > {MAX_SMALL_RATIO:.0%}")
    if flat_ratio > MAX_FLAT_RATIO:
        warnings.append(f"无标题文档占比 {flat_ratio:.1%} > {MAX_FLAT_RATIO:.0%}")
    if counts["chunks"] and counts["headed_chunks"] == 0:
        warnings.append("所有块都没有 heading_path：标题切分分支未被触达")

    if warnings:
        print("\n告警：")
        for item in warnings:
            print(f"  ! {item}")

    # ---- 硬断言 ----
    failures = list(problems)
    if counts["parse_error"] or counts["clean_error"] or counts["split_error"]:
        failures.append("存在解析/清洗/分块异常，链路不稳定")
    if counts["chunks"] == 0:
        failures.append("没有任何块产出")
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

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": verdict,
        "raw_dir": str(root),
        "params": {
            "CHUNK_SIZE": settings.CHUNK_SIZE,
            "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
            "PARENT_CHUNK_SIZE": settings.PARENT_CHUNK_SIZE,
        },
        "counts": dict(counts),
        "avg_chunk_len": round(avg_len, 2),
        "tiny_ratio": round(tiny_ratio, 4),
        "flat_doc_ratio": round(flat_ratio, 4),
        "elapsed_sec": round(elapsed, 2),
        "problems": problems,
        "warnings": warnings,
    }
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已保存：{REPORT_PATH}")
    return 0 if verdict == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
