# ============================================================================
# download_rerank_model.py —— 下载 Qwen3-Reranker GGUF 模型到 E 盘
#
# 【下载什么】
#   ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF 仓库的 qwen3-reranker-0.6b-q8_0.gguf
#   （Q8_0 量化，约 639MB，4GB 显存可跑）
#
# 【为什么走 hf-mirror.com】
#   沙箱代理不通 huggingface.co，但通 hf-mirror.com（国内镜像）。
#
# 【为什么下载到 E 盘】
#   用户要求不装 C 盘。llama-server 的 -hf 参数会把模型缓存到 C 盘用户目录，
#   所以这里手动下载到 E 盘，启动时用 -m 指定本地路径。
#
# 【安装位置】E:\CodingTools\llama.cpp\models\
# ============================================================================

import sys
import time
from pathlib import Path

import requests

# 模型保存目录（E 盘，不落 C 盘）
MODEL_DIR = Path("E:/CodingTools/llama.cpp/models")

# HuggingFace 镜像（沙箱代理不通 huggingface.co，走镜像）
HF_MIRROR = "https://hf-mirror.com"

# 仓库与文件名
REPO = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
FILENAME = "qwen3-reranker-0.6b-q8_0.gguf"

# 重试次数与读超时（秒）
MAX_RETRY = 10
READ_TIMEOUT = 60


def download(url: str, dest: Path) -> None:
    """下载模型文件（支持断点续传 + 重试）。

    参数：
        url  : str  —— 下载地址
        dest : Path —— 保存路径
    """
    for attempt in range(1, MAX_RETRY + 1):
        # ---- 断点：已下载多少 ----
        pos = dest.stat().st_size if dest.exists() else 0
        mode = "ab" if pos > 0 else "wb"

        headers = {}
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"

        try:
            print(f"[{attempt}/{MAX_RETRY}] {'续传' if pos else '开始'}: "
                  f"{dest.name}（已有 {pos/1024/1024:.1f}MB）", flush=True)
            with requests.get(url, headers=headers, stream=True,
                              timeout=READ_TIMEOUT) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
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
                            if pct != last_pct and pct % 5 == 0:
                                print(f"  进度 {pct}% ({downloaded/1024/1024:.0f}MB"
                                      f"/{total/1024/1024:.0f}MB)", flush=True)
                                last_pct = pct
            print(f"  完成: {dest.name} ({dest.stat().st_size/1024/1024:.1f} MB)", flush=True)
            return
        except Exception as e:
            print(f"  中断: {type(e).__name__} {str(e)[:120]}", flush=True)
            if attempt < MAX_RETRY:
                time.sleep(2)
    print(f"错误：模型下载失败（已重试 {MAX_RETRY} 次），"
          f"已下载 {dest.stat().st_size/1024/1024:.1f}MB 可下次续传", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """主流程：下载模型。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / FILENAME

    # 已完整则跳过（639MB 左右，用 >600MB 粗判完整）
    if dest.exists() and dest.stat().st_size > 600 * 1024 * 1024:
        print(f"模型已存在，跳过: {dest}", flush=True)
        return

    url = f"{HF_MIRROR}/{REPO}/resolve/main/{FILENAME}"
    print(f"下载地址: {url}", flush=True)
    print(f"保存到: {dest}", flush=True)
    download(url, dest)
    print("模型下载完成！", flush=True)


if __name__ == "__main__":
    main()
