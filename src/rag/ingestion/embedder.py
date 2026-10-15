# ============================================================================
# embedder.py —— 向量化（Embedding）：把文字转成数字向量
#
# 位置：src/rag/ingestion/embedder.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.3 节。把一段文字变成"一串数字"（向量），这串
#   数字能"代表"这段文字的意思。这是整个 RAG 最核心的一步。
#
#   我们用的模型：qwen3-embedding:0.6b（本地 Ollama 部署，639MB，实测
#   6.6 条/秒，能整卡驻留 4GB 显存）。
#
# 【本文件的"企业级"重点：依赖注入抽象】
#   这里用 typing.Protocol 定义 Embedder 接口（"能力约定"，而非强制继承），
#   再提供两个实现：
#     · OllamaEmbedder —— 真实调用本地 Ollama（生产用）
#     · FakeEmbedder    —— 返回确定性假向量（测试用，不依赖 Ollama）
#   上层代码（pipeline/updater）只依赖 Embedder 协议，不关心具体实现。
#   测试时注入 FakeEmbedder，生产时注入 OllamaEmbedder，互不影响。
#
#   【为什么用 Protocol 而不是 abc.ABC】
#     Protocol 是"鸭子类型"：只要一个对象有 embed() 方法、签名匹配，就能
#     当 Embedder 用，不需要显式继承。这更灵活、更 Pythonic，也是 fastapi/
#     pydantic 生态的主流做法（配合构造器注入）。
# ============================================================================

# ---------- 导入区 ----------
# math：Python 标准库，数学运算（归一化用 sqrt 开方）。
import math

# List / Protocol / runtime_checkable：类型标注 + 协议定义工具。
#   Protocol：定义"结构化接口"（鸭子类型）；
#   runtime_checkable：让协议支持 isinstance 运行时检查（可选）。
from typing import Protocol, runtime_checkable

# numpy：第三方库，高性能数值计算（向量运算用，Ollama 返回的向量用 numpy 处理）。
# requests：第三方库，发 HTTP 请求（调用 Ollama 的接口）。
import requests

# 项目内：日志设施。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 第一部分：Embedder 协议（接口定义）
# ============================================================================
@runtime_checkable
class Embedder(Protocol):
    """Embedder 协议：定义"能做什么"（能力约定）。

    【协议 vs 抽象基类】
        协议不问"你是谁"（继承关系），只问"你能不能 embed(texts)"。
        任何实现了 embed 方法、且签名匹配的对象，都自动满足这个协议。

    【必须实现的方法】
        embed(texts) -> List[List[float]] ：把一批文本转成一批向量。
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成一批向量（协议方法，无需实现体）。

        参数：
            texts : List[str] —— 待向量化的文本列表

        返回：
            List[List[float]] —— 每个文本对应的向量
        """
        ...   # 协议方法用 ... 占位，不写具体实现


# ============================================================================
# 第二部分：工具函数 —— L2 归一化
# ============================================================================
def normalize(vector: list[float]) -> list[float]:
    """对向量做 L2 归一化，让它的"长度"变成 1。

    【原理（对应文档 5.1 坑6、6 章学习点）】
        归一化后，两个向量的"内积"就等于它们的"余弦相似度"，这样 FAISS 的
        IndexFlatIP（内积检索）才能准确衡量语义相近程度。

    【参数】
        vector : List[float] —— 原始向量（一串数字）

    【返回】
        List[float] —— 归一化后的向量（长度=1）
    """
    # 计算向量的"模长"（长度）：sqrt(各分量平方和)
    norm = math.sqrt(sum(v * v for v in vector))

    if norm == 0:                # 防止除零（全 0 向量）
        return vector            # 全 0 就原样返回，不归一化
    # 每个分量除以模长 → 新向量长度为 1
    return [v / norm for v in vector]


# ============================================================================
# 第三部分：真实实现 —— OllamaEmbedder
# ============================================================================
class OllamaEmbedder:
    """真实嵌入器：调用本地 Ollama 的 /api/embed 接口。

    【原理（对应文档 2.3 节）】
        Ollama 是本地模型运行工具，暴露一个 HTTP 接口 /api/embed。
        我们把文字 POST 给它，它返回对应的向量（一串数字）。

        为什么批量（对应文档 5.1 坑5）：
        一次塞太多文字给 Ollama 会慢或报错，所以要分批（batch_size=16 左右）。

    【非对称检索（对应文档 2.3 节的关键细节）】
        Qwen3-Embedding 支持"指令前缀"：查询侧加一句指令告诉模型"现在要做
        检索"，能提升精度；文档侧不加。所以本类提供 embed_documents（不加前缀）
        和 embed_query（加前缀）两个方法，区分对待。
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
        batch_size: int = 16,
        query_prefix: str = "Instruct: 检索与查询语义相关的内容\nQuery: ",
    ):
        """构造函数。

        参数：
            model        : str  嵌入模型名
            base_url     : str  Ollama 服务地址
            batch_size   : int  每批处理多少条文本
            query_prefix : str  查询侧指令前缀（非对称检索用）
        """
        self.model = model                       # 模型名
        self.base_url = base_url.rstrip("/")     # 去掉末尾斜杠，拼 URL 时统一
        self.batch_size = batch_size             # 批量大小
        self.query_prefix = query_prefix         # 查询前缀

    # ----------------------------------------------------------------------
    # 内部方法：调用 Ollama 接口（一次批）
    # ----------------------------------------------------------------------
    def _call_embed(self, batch: list[str]) -> list[list[float]]:
        """调用 Ollama 的 /api/embed 接口，处理一批文本。

        参数：
            batch : List[str] —— 一批文本

        返回：
            List[List[float]] —— 这批文本对应的向量
        """
        # 拼接口地址：base_url + "/api/embed"
        url = f"{self.base_url}/api/embed"

        # requests.post：向 Ollama 发 POST 请求。
        #   json=  请求体，Ollama 要求 {model, input} 两个字段。
        #   timeout=300  超时（秒），防止网络卡死一直等。
        response = requests.post(
            url,
            json={"model": self.model, "input": batch},
            timeout=300,
        )
        # 如果返回 HTTP 错误码（4xx/5xx），抛异常（由上层决定是否重试）
        response.raise_for_status()

        # response.json() 把返回的 JSON 转成 Python 字典。
        # Ollama 返回结构：{"embeddings": [[...], [...], ...]}
        return response.json()["embeddings"]

    # ----------------------------------------------------------------------
    # 实现协议方法：embed（文档侧，不加前缀）
    # ----------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（文档侧，不加指令前缀）。

        参数：
            texts : List[str] —— 待向量化的文本列表

        返回：
            List[List[float]] —— 每个文本对应的（已归一化的）向量
        """
        all_vectors = []   # 存所有结果

        # 分批处理：range(0, len(texts), batch_size) 每次前进 batch_size
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]   # 取当前这一批

            # 调用接口拿原始向量
            batch_vectors = self._call_embed(batch)

            # 对每个向量做 L2 归一化，再逐个加入总结果
            for vec in batch_vectors:
                all_vectors.append(normalize(vec))

            log.debug("已嵌入 %d/%d 条", len(all_vectors), len(texts))   # 进度日志

        return all_vectors

    # ----------------------------------------------------------------------
    # 查询侧向量化（加指令前缀，非对称检索）
    # ----------------------------------------------------------------------
    def embed_query(self, query: str) -> list[float]:
        """查询侧向量化（加指令前缀，非对称检索）。

        【原理】
            查询前加一句指令，告诉模型"现在是要做检索"，它就会按检索的方式
            算向量，能提升 1~5 个点精度。文档侧不加（见 embed 方法）。

        参数：
            query : str —— 单个查询文本

        返回：
            List[float] —— 查询的（已归一化）向量
        """
        # 加前缀后走单条嵌入
        vectors = self.embed([self.query_prefix + query])
        return vectors[0]   # 只返回这一条


# ============================================================================
# 第四部分：测试桩 —— FakeEmbedder
# ============================================================================
class FakeEmbedder:
    """测试用假嵌入器：返回确定性假向量，不依赖 Ollama。

    【为什么需要它】
        单测不能依赖真实 Ollama（否则测试会因服务没启动而失败，且慢、不可
        复现）。FakeEmbedder 用"文本哈希"生成一个确定性向量——同样的文本
        永远得到同样的向量，保证测试可复现、可断言。

    【生成向量的原理】
        对文本算一个哈希值，用它作为"种子"生成一个 8 维的确定性向量。
        维度小、速度快，足够测试向量化、归一化、入库等逻辑。
    """

    def __init__(self, dim: int = 8):
        """构造函数。

        参数：
            dim : int —— 假向量的维度，默认 8
        """
        self.dim = dim   # 向量维度

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成假向量（确定性：同文本同向量）。

        参数：
            texts : List[str] —— 待处理文本

        返回：
            List[List[float]] —— 假向量列表
        """
        vectors = []
        for text in texts:
            # 用 Python 内置 hash 得到文本的整数指纹
            seed = abs(hash(text))
            # 用种子生成一个 dim 维的确定性向量：
            # 第 i 维 = (seed 右移 i 位) 对 1000 取模，再归一化到 [-1, 1]
            vec = [((seed >> i) % 1000) / 1000.0 for i in range(self.dim)]
            vectors.append(normalize(vec))   # 同样做归一化，行为与真实一致
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """查询侧假向量（与文档侧逻辑一致）。"""
        return self.embed([query])[0]
