# ============================================================================
# download_llamacpp.py —— 下载 llama.cpp Windows CUDA 12.4 预编译包
#
# 【下载方式（2026-10-15 实测可用）】
#   沙箱代理只放行 api.github.com，不放行 github.com 主站下载。
#   解法：用 GitHub API 的 asset 端点 + Accept: application/octet-stream，
#   它会 302 重定向到 release-assets CDN，该链路代理放行。
#
# 【下载什么】
#   1. llama-b11658-bin-win-cuda-12.4-x64.zip    主包（254MB）
#   2. cudart-llama-bin-win-cuda-12.4-x64.zip    CUDA runtime（373MB，必需）
#
# 【为什么选 CUDA 12.4 而不是 13.4】
#   CUDA 13.x 已移除对 Pascal 架构（sm_61）的支持，本机 GTX 1050 Ti
#   是 Pascal，必须用 CUDA 12.4。
#
# 【安装位置】E:\CodingTools\llama.cpp（用户要求不装 C 盘，
#   与 Ollama/python3.12 等工具同目录）
# ============================================================================

import sys
import time
from pathlib import Path

import requests

# 下载临时目录（E 盘，不落 C 盘）
DOWNLOAD_DIR = Path("E:/python/RAG/tmp/llama_download")

# llama.cpp nightly release tag（预编译包在 nightly，正式 release 不带二进制）
TAG = "b11658"

# 要下载的文件名
FILES = [
    "llama-b11658-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
]

# 重试次数与读超时（秒）
MAX_RETRY = 10
READ_TIMEOUT = 60

# GitHub API：获取 release 的资产列表（api.github.com 代理放行）
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


def get_asset_urls() -> dict[str, str]:
    """从 GitHub API 获取目标文件的 asset 下载端点（API 形式）。

    返回：
        dict[str, str] —— {文件名: API asset 端点 URL}
    """
    resp = requests.get(f"{RELEASES_API}/tags/{TAG}", timeout=30)
    resp.raise_for_status()
    release = resp.json()

    wanted = set(FILES)
    urls = {}
    for asset in release.get("assets", []):
        if asset["name"] in wanted:
            # asset["url"] 是 API 端点形式：
            # https://api.github.com/repos/.../releases/assets/<id>
            urls[asset["name"]] = asset["url"]
    return urls


def download(url: str, dest: Path) -> None:
    """下载单个文件（走 API asset 端点，支持断点续传 + 重试）。

    【为什么需要断点续传】
        沙箱代理不稳定，627MB 一次性下载容易中断。用 HTTP Range 请求
        从断点继续，重试不浪费已下载部分。

    参数：
        url  : str  —— API asset 端点 URL
        dest : Path —— 保存路径
    """
    for attempt in range(1, MAX_RETRY + 1):
        # ---- 断点：已下载多少 ----
        pos = dest.stat().st_size if dest.exists() else 0
        mode = "ab" if pos > 0 else "wb"   # 有断点则追加，否则重新写

        # Range 请求头：从 pos 字节继续
        headers = {"Accept": "application/octet-stream"}
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"

        try:
            print(f"[{attempt}/{MAX_RETRY}] {'续传' if pos else '开始'}: "
                  f"{dest.name}（已有 {pos/1024/1024:.1f}MB）", flush=True)
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=READ_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                # 续传时 content-length 是"剩余量"，总大小 = pos + 剩余
                if pos > 0 and resp.status_code == 206:
                    total = pos + total
                downloaded = pos
                last_pct = -1
                with open(dest, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded * 100 / total)
                            if pct != last_pct and pct % 5 == 0:   # 每 5% 报一次
                                print(f"  进度 {pct}% ({downloaded/1024/1024:.0f}MB"
                                      f"/{total/1024/1024:.0f}MB)", flush=True)
                                last_pct = pct
            print(f"  完成: {dest.name} ({dest.stat().st_size/1024/1024:.1f} MB)", flush=True)
            return
        except Exception as e:
            print(f"  中断: {type(e).__name__} {str(e)[:120]}", flush=True)
            if attempt < MAX_RETRY:
                # 网络类错误短暂退避即可（续传不会丢进度）
                time.sleep(2)
    # 重试耗尽，退出并报错
    print(f"错误：{dest.name} 下载失败（已重试 {MAX_RETRY} 次），"
          f"已下载 {dest.stat().st_size/1024/1024:.1f}MB 可下次续传", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """主流程：下载两个包。"""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 第 1 步：从 API 拿 asset 端点
    print("正在获取 release 资产列表...", flush=True)
    urls = get_asset_urls()
    print(f"获取到 {len(urls)} 个目标资产", flush=True)

    # 同时拿到每个文件的完整大小（用于判断是否已下载完成）
    resp = requests.get(f"{RELEASES_API}/tags/{TAG}", timeout=30)
    sizes = {a["name"]: a["size"] for a in resp.json().get("assets", [])}

    # 第 2 步：逐个下载
    for fname in FILES:
        if fname not in urls:
            print(f"错误：未找到资产 {fname}", file=sys.stderr)
            sys.exit(1)
        dest = DOWNLOAD_DIR / fname
        # 已下载完整才跳过（大小与 API 的 size 一致）
        if dest.exists() and dest.stat().st_size == sizes.get(fname):
            print(f"已完整，跳过: {fname}", flush=True)
            continue
        download(urls[fname], dest)

    print("全部下载完成！", flush=True)


if __name__ == "__main__":
    main()
