# ============================================================================
# install_llamacpp.py —— 解压安装 llama.cpp 到 E:\CodingTools\llama.cpp
#
# 【做什么】
#   把下载的两个 zip 解压合并到安装目录：
#     1. llama-b11658-bin-win-cuda-12.4-x64.zip    主包（exe + DLL + 依赖）
#     2. cudart-llama-bin-win-cuda-12.4-x64.zip    CUDA runtime DLL
#   两个包都解压到同一目录（cudart 的 DLL 与主包 DLL 合并）。
#
# 【安装位置】E:\CodingTools\llama.cpp（用户要求不装 C 盘）
#
# 【用法】python scripts/install_llamacpp.py
# ============================================================================

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# 下载目录（E 盘）
DOWNLOAD_DIR = Path("E:/python/RAG/tmp/llama_download")

# 安装目录（E 盘，与 Ollama/python3.12 等同级）
INSTALL_DIR = Path("E:/CodingTools/llama.cpp")

# 要解压的 zip（按顺序，cudart 后解压以补齐 CUDA DLL）
ZIPS = [
    "llama-b11658-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
]


def extract(zip_path: Path, dest: Path) -> None:
    """解压 zip 到目标目录。

    参数：
        zip_path : Path —— zip 文件路径
        dest     : Path —— 解压目标目录
    """
    print(f"解压: {zip_path.name} -> {dest}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # 防止路径穿越（zip slip）：确保所有文件都解压到 dest 内
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"不安全的 zip 条目（路径穿越）: {member}")
        zf.extractall(dest)
    print(f"  完成", flush=True)


def main() -> None:
    """主流程：检查 → 解压 → 验证。"""
    # ---- 第 1 步：检查 zip 是否齐全 ----
    missing = [z for z in ZIPS if not (DOWNLOAD_DIR / z).exists()]
    if missing:
        print(f"错误：缺少下载文件 {missing}，请先运行 download_llamacpp.py", file=sys.stderr)
        sys.exit(1)

    # ---- 第 2 步：创建安装目录 ----
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 第 3 步：依次解压（后解压的覆盖先解压的）----
    for zname in ZIPS:
        extract(DOWNLOAD_DIR / zname, INSTALL_DIR)

    # ---- 第 4 步：验证安装 ----
    llama_server = INSTALL_DIR / "llama-server.exe"
    if not llama_server.exists():
        print(f"错误：未找到 {llama_server}，解压可能不完整", file=sys.stderr)
        sys.exit(1)

    print("\n=== 安装内容 ===", flush=True)
    # 列出目录下的 exe 和主要 DLL
    for f in sorted(INSTALL_DIR.glob("*.exe")):
        print(f"  {f.name}  ({f.stat().st_size/1024/1024:.1f} MB)", flush=True)
    dll_count = len(list(INSTALL_DIR.glob("*.dll")))
    print(f"  另有 {dll_count} 个 DLL", flush=True)

    # ---- 第 5 步：版本验证（确认 exe 能跑）----
    print("\n=== 版本验证 ===", flush=True)
    result = subprocess.run(
        [str(llama_server), "--version"],
        capture_output=True, text=True, timeout=60,
    )
    print(result.stdout.strip() or result.stderr.strip(), flush=True)

    print(f"\n安装完成！安装目录: {INSTALL_DIR}", flush=True)
    print("\n启动重排服务（rerank）命令：", flush=True)
    print(f'  {llama_server} -hf ggml-org/Qwen3-reranker-0.6B-Q8_0-GGUF:Q8_0 '
          f'--embedding --rerank --pooling rank --port 9931', flush=True)


if __name__ == "__main__":
    main()
