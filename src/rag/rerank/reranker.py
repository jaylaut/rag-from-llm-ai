# ============================================================================
# reranker.py —— 重排（Rerank）：粗召回后的精排序
#
# 位置：src/rag/rerank/reranker.py
#
# 【本文件是干什么的】
#   对应《P2-检索.md》2.4 节。召回阶段为了"不漏"，会把范围放宽（20~50 条，
#   有噪声）；重排用更精细的方法把这几十条重新精排，留下最准的 3~5 条。
#   这叫"粗召回 + 精排序"两段式，是工业界通用做法。
#
# 【一个重要技术现实（2026-10-15 修订确认）】
#   文档原设计"用 Ollama 调 qwen3-reranker 做 cross-encoder 重排"，但实测：
#   Ollama 不支持重排模型（/api/chat、/api/generate 返回无效逗号，且无
#   /api/rerank 接口）。后续曾考虑 vLLM，但核查发现 vLLM 要求 GPU 计算能力
#   7.0+（Volta 起，依赖 Triton kernels）且不原生支持 Windows——本机 GTX 1050 Ti
#   为 Pascal 架构（compute_cap 6.1），vLLM 不可行。
#
# 【因此本文件的工程决策：接口隔离 + 占位实现】
#   1. 定义 Reranker 协议（能力约定），上层只依赖协议，不关心具体实现；
#   2. 先用 EmbeddingReranker（bi-encoder 近似重排）占位，保证全链路能跑通；
#   3. 预留 CrossEncoderReranker（真正生产级做法），部署方式为 llama.cpp
#      （llama-server --reranking），就绪后无缝切换，上层零改动。
#   这正是"接口隔离"的价值——换重排算法不改上层（呼应 P5 编排层）。
# ============================================================================

# ---------- 导入区 ----------
# 类型标注：Protocol 定义结构化接口，runtime_checkable 支持 isinstance 检查。
from typing import Protocol, runtime_checkable

# 项目内：嵌入器（EmbeddingReranker 用它算相似度）。
from rag.ingestion.embedder import Embedder

# 项目内：日志。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 第一部分：Reranker 协议（接口定义）
# ============================================================================
@runtime_checkable
class Reranker(Protocol):
    """重排器协议：定义"能做什么"（能力约定）。

    【为什么定义协议】
        重排算法可替换（bi-encoder 近似 / cross-encoder 精确 / 其它），
        上层（编排层）只依赖协议，换实现不改上层——接口隔离。

    【必须实现的方法】
        rerank(query, chunks) -> list[tuple[str, float]]
        其中 chunks 是 [(chunk_id, chunk_text), ...]，返回按相关性重排后的结果。
    """

    def rerank(self, query: str, chunks: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """对候选 chunk 重排，返回按相关性排序的 (chunk_id, 分数)。

        参数：
            query  : str —— 查询文本
            chunks : List[Tuple[str, str]] —— [(chunk_id, chunk_text), ...] 候选

        返回：
            List[Tuple[str, float]] —— [(chunk_id, 相关性分数), ...]，降序
        """
        ...   # 协议方法，用 ... 占位，不写实现体


# ============================================================================
# 第二部分：占位实现 —— EmbeddingReranker（bi-encoder 近似）
# ============================================================================
class EmbeddingReranker:
    """基于嵌入相似度的重排（bi-encoder 近似方案，作为占位）。

    【原理】
        把 query 和每个 chunk 分别编码成向量，算相似度（内积/余弦），
        按相似度降序重排。这是 bi-encoder 近似——查询和文档"分别"编码，
        丢失了词级交互，精度低于 cross-encoder。但作为"先跑通全链路"
        的占位足够。

    【为什么要标注"占位"】
        文档明确说这是"已被淘汰的做法"，生产级应该用 cross-encoder。
        本类存在的意义是：在 vLLM 部署好之前，让 P2~P5 全链路能端到端跑通。
        等 CrossEncoderReranker 就绪后，直接替换，上层零改动。
    """

    def __init__(self, embedder: Embedder):
        """构造函数。

        参数：
            embedder : Embedder —— 嵌入器（用于把 query 和 chunk 转向量）
        """
        self.embedder = embedder

    def rerank(self, query: str, chunks: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """按嵌入相似度重排候选 chunk。

        【原理】
            query 用 embed_query（加指令前缀，非对称检索）；
            每个 chunk 用 embed（文档侧，不加前缀）；
            算 query 向量与每个 chunk 向量的内积（都已归一化，内积=余弦相似度）。

        参数：
            query  : str —— 查询文本
            chunks : List[Tuple[str, str]] —— [(chunk_id, chunk_text), ...]

        返回：
            List[Tuple[str, float]] —— [(chunk_id, 相似度), ...]，降序
        """
        if not chunks:
            return []

        # 查询向量（embed_query 已加指令前缀 + 归一化）
        query_vec = self.embedder.embed_query(query)

        # 批量嵌入所有 chunk 文本（文档侧，不加前缀）
        chunk_texts = [text for _cid, text in chunks]
        chunk_vecs = self.embedder.embed(chunk_texts)

        # 算 query 与每个 chunk 的内积（相似度），因为向量已归一化，内积即余弦相似度
        scored = []
        # strict=True：chunks 与 chunk_vecs 长度必然一致（同一批文本嵌入而来）
        for (chunk_id, _text), vec in zip(chunks, chunk_vecs, strict=True):
            # 内积：逐元素相乘再求和；query_vec 与 vec 同维度，strict 兜底
            similarity = sum(q * v for q, v in zip(query_vec, vec, strict=True))
            scored.append((chunk_id, similarity))

        # 按相似度降序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ============================================================================
# 第三部分：生产实现 —— CrossEncoderReranker（预留，等 llama.cpp 就绪）
# ============================================================================
class CrossEncoderReranker:
    """cross-encoder 重排器（生产级方案，预留接口，等 llama.cpp 部署后启用）。

    【为什么这个是"真正生产级"】
        cross-encoder 把 query 和 chunk "拼接"后联合打分，能捕捉词级交互，
        精度远高于 bi-encoder（嵌入近似）。这是召回后精排的行业标准。

    【部署方式：llama.cpp（llama-server）】
        选型过程（详见 docs/项目架构.md §2.4.5）：
        - Ollama 不支持重排模型（无 /api/rerank 端点，调用返回无效输出）；
        - vLLM 要求 GPU 计算能力 7.0+（Volta 起）且不原生支持 Windows，
          本机 GTX 1050 Ti（Pascal 6.1）不可行；
        - llama.cpp 支持 Pascal + Windows 官方 release，是唯一适配方案。

        启动命令：
            llama-server -hf ggml-org/Qwen3-reranker-0.6B-Q8_0-GGUF:Q8_0 \\
                --embedding --rerank --pooling rank --port 9931

    【llama.cpp /v1/rerank 接口】
        请求：POST /v1/rerank
              {"model": "...", "query": "...", "documents": [...], "top_n": N}
        响应：{"results": [{"index": i, "relevance_score": s}, ...]}
        （index 是 documents 里的原始下标，relevance_score 是相关性分数）

    【当前状态：预留】
        llama.cpp 部署好之前，本类不可用。就绪后实现 _call_llama_cpp_rerank
        即可无缝替换 EmbeddingReranker，上层零改动。
    """

    def __init__(self, base_url: str = "http://localhost:9931",
                 model: str = "ggml-org/Qwen3-reranker-0.6B-Q8_0-GGUF"):
        """构造函数。

        参数：
            base_url : str —— llama-server 服务地址（默认本地 9931 端口）
            model    : str —— 模型名（GGUF 仓库名）
        """
        self.base_url = base_url.rstrip("/")
        self.model = model

    def rerank(self, query: str, chunks: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """cross-encoder 重排（当前未实现，等 llama.cpp 就绪）。

        【实现说明（就绪后按此实现）】
            1. 调用 POST {base_url}/v1/rerank，body：
               {"model": self.model, "query": query,
                "documents": [各 chunk 文本], "top_n": len(chunks)}
            2. 解析响应 results，每项 {"index": i, "relevance_score": s}
            3. 用 index 映射回 chunk_id，按 relevance_score 降序返回

        参数：
            query  : str —— 查询文本
            chunks : List[Tuple[str, str]] —— [(chunk_id, chunk_text), ...]

        返回：
            List[Tuple[str, float]] —— [(chunk_id, 相关性分数), ...]，降序
        """
        # 尚未实现：llama.cpp 部署好之前，这里不可用。
        # 抛出明确的错误信息，提示使用者改用 EmbeddingReranker 或先部署 llama.cpp。
        raise NotImplementedError(
            "CrossEncoderReranker 尚未实现：需先部署 llama.cpp 重排服务。"
            "启动命令：llama-server -hf ggml-org/Qwen3-reranker-0.6B-Q8_0-GGUF:Q8_0 "
            "--embedding --rerank --pooling rank --port 9931"
            "（Ollama 不支持重排、vLLM 需 sm_70+ 不适用本机，详见 docs/项目架构.md §2.4.5）。"
            "过渡期请使用 EmbeddingReranker。"
        )
