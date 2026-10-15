@echo off
REM ============================================================================
REM start_rerank_server.bat —— 启动 llama.cpp 重排服务（Windows 一键启动）
REM
REM 【用途】
REM   启动 Qwen3-Reranker-0.6B（GGUF）的 cross-encoder 重排服务，
REM   暴露 /v1/rerank 端点，供 RAG 项目的 P2 检索模块调用。
REM
REM 【前置条件】
REM   1. llama.cpp 已安装到 E:\CodingTools\llama.cpp（运行 install_llamacpp.py）
REM   2. 模型已下载到 E:\CodingTools\llama.cpp\models\（运行 download_rerank_model.py）
REM
REM 【为什么用 CUDA 12.4 版】
REM   CUDA 13.x 已移除 Pascal 架构支持，本机 GTX 1050 Ti 是 Pascal（sm_61）。
REM
REM 【技术要点】
REM   --reranking    启用 /v1/rerank 端点（Ollama 没有这个能力，这是选 llama.cpp 的原因）
REM   --pooling rank RANK 池化（reranker 必需，与嵌入的 mean 池化不同）
REM   -m             用本地模型路径（不用 -hf，避免模型缓存落到 C 盘）
REM   --n-gpu-layers 999  全部层放 GPU（0.6B Q8_0 约 639MB，4GB 显存足够）
REM ============================================================================

setlocal

REM ---- 路径配置（E 盘，符合"不装 C 盘"要求）----
set LLAMA_DIR=E:\CodingTools\llama.cpp
set MODEL_DIR=%LLAMA_DIR%\models
set MODEL_FILE=qwen3-reranker-0.6b-q8_0.gguf

REM ---- 检查前置条件 ----
if not exist "%LLAMA_DIR%\llama-server.exe" (
    echo [错误] 未找到 %LLAMA_DIR%\llama-server.exe
    echo 请先运行: python scripts/install_llamacpp.py
    pause
    exit /b 1
)
if not exist "%MODEL_DIR%\%MODEL_FILE%" (
    echo [错误] 未找到模型 %MODEL_DIR%\%MODEL_FILE%
    echo 请先运行: python scripts/download_rerank_model.py
    pause
    exit /b 1
)

REM ---- 启动服务 ----
echo 启动重排服务: http://127.0.0.1:9931
echo 端点: POST /v1/rerank  {"query": "...", "documents": [...]}
echo 按 Ctrl+C 停止服务
echo.

"%LLAMA_DIR%\llama-server.exe" ^
  -m "%MODEL_DIR%\%MODEL_FILE%" ^
  --embedding ^
  --reranking ^
  --pooling rank ^
  --ctx-size 512 ^
  --parallel 2 ^
  --n-gpu-layers 999 ^
  --host 127.0.0.1 ^
  --port 9931

endlocal
