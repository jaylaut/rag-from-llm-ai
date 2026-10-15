# ============================================================================
# parallel_downloader.py —— 多连接分段并行下载器
#
# 【为什么需要它】
#   沙箱代理对单连接限速（实测约 19KB/s），627MB 要下 3 小时。
#   代理限速通常按连接计算，多连接分段并行可叠加带宽。
#
# 【原理】
#   1. 先探文件总大小；
#   2. 分成 N 段，每段一个线程，用 HTTP Range 请求下载；
#   3. 每段独立存为 .partN 文件（段内断点续传，重启不丢进度）；
#   4. 全部段完成后按顺序合并成最终文件。
#
# 【用法】
#   python scripts/parallel_downloader.py <url> <输出路径> [--conn 8]
# ============================================================================

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# 读超时（秒）：代理无数据多久算中断
READ_TIMEOUT = 60
# 段内重试次数
SEGMENT_RETRY = 20


def get_total_size(url: str) -> int:
    """探测文件总大小（发一个 Range: bytes=0-0 请求看 content-range）。

    【关键】GitHub API asset 端点必须带 Accept: application/octet-stream
    才返回文件流（否则返回 JSON 元数据，拿不到 content-range）。

    参数：
        url : str —— 下载地址

    返回：
        int —— 文件总字节数
    """
    resp = requests.get(
        url,
        headers={"Accept": "application/octet-stream", "Range": "bytes=0-0"},
        stream=True, timeout=READ_TIMEOUT,
    )
    resp.raise_for_status()
    # Content-Range: bytes 0-0/266574670 → 总大小在斜杠后
    content_range = resp.headers.get("content-range", "")
    if "/" in content_range:
        return int(content_range.split("/")[-1])
    # 不支持 Range 时退化为 content-length
    return int(resp.headers.get("content-length", 0))


def download_segment(url: str, start: int, end: int, part_path: Path,
                     progress: dict, lock) -> None:
    """下载一个分段（段内断点续传 + 重试）。

    参数：
        url       : str   —— 下载地址
        start     : int   —— 段起始字节
        end       : int   —— 段结束字节（含）
        part_path : Path  —— 段临时文件
        progress  : dict  —— 全局进度（多线程共享）
        lock      : Lock  —— 进度字典的锁
    """
    for attempt in range(1, SEGMENT_RETRY + 1):
        # 段内断点：已下载多少
        pos = part_path.stat().st_size if part_path.exists() else 0
        if pos >= (end - start + 1):
            return   # 本段已完成
        mode = "ab" if pos > 0 else "wb"
        headers = {
            "Accept": "application/octet-stream",   # GitHub API asset 端点必须带
            "Range": f"bytes={start + pos}-{end}",
        }
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=READ_TIMEOUT) as resp:
                resp.raise_for_status()
                with open(part_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        f.write(chunk)
                        with lock:
                            progress["done"] += len(chunk)
            return   # 本段完成
        except Exception:
            if attempt < SEGMENT_RETRY:
                time.sleep(1)
    raise RuntimeError(f"分段 {start}-{end} 下载失败（重试 {SEGMENT_RETRY} 次）")


def main() -> None:
    """主流程：分段并行下载并合并。"""
    parser = argparse.ArgumentParser(description="多连接分段并行下载器")
    parser.add_argument("url", help="下载地址")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--conn", type=int, default=8, help="并行连接数（默认 8）")
    args = parser.parse_args()

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 已完成则跳过
    total = get_total_size(args.url)
    if dest.exists() and dest.stat().st_size == total:
        print(f"已完整，跳过: {dest} ({total/1024/1024:.1f} MB)", flush=True)
        return

    print(f"文件总大小: {total/1024/1024:.1f} MB，分 {args.conn} 段并行下载", flush=True)

    # 分段
    seg_size = total // args.conn
    segments = []
    for i in range(args.conn):
        start = i * seg_size
        end = total - 1 if i == args.conn - 1 else (i + 1) * seg_size - 1
        segments.append((start, end, dest.with_suffix(f".part{i}")))

    # 多线程下载
    import threading
    # part 文件可能不存在（首次下载），用 exists 判断
    progress = {"done": sum(p.stat().st_size if p.exists() else 0 for _, _, p in segments)}
    lock = threading.Lock()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.conn) as pool:
        futures = [
            pool.submit(download_segment, args.url, s, e, p, progress, lock)
            for s, e, p in segments
        ]
        # 进度打印（主线程等待时周期性输出）
        done_futures = 0
        for fut in as_completed(futures):
            fut.result()   # 段失败会抛异常
            done_futures += 1
            print(f"  完成 {done_futures}/{args.conn} 段", flush=True)

    # 合并
    print("各段完成，合并中...", flush=True)
    with open(dest, "wb") as out:
        for _s, _e, part in segments:
            out.write(part.read_bytes())
            part.unlink()   # 合并后删段文件

    elapsed = time.time() - start_time
    speed = total / 1024 / 1024 / elapsed if elapsed > 0 else 0
    print(f"下载完成: {dest} ({total/1024/1024:.1f} MB, "
          f"耗时 {elapsed:.0f}s, 平均 {speed:.1f} MB/s)", flush=True)


if __name__ == "__main__":
    main()
