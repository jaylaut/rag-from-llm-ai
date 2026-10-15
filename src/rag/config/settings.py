# ============================================================================
# settings.py —— 全局配置中心（所有模块共享的"开关和参数"）
#
# 位置：src/rag/config/settings.py
#
# 【这个文件是干什么的】
#   把散落在各处的"魔法数字"（块大小、阈值、目录路径、模型名、API 地址）
#   集中到这一个文件。改参数只动这里，不用满项目去找。
#   打个比方：这是整个项目的一块"总控制面板"，所有模块都从这里读配置。
#
# 【为什么用 pydantic-settings（方案 A）】
#   1. 类型安全：每个配置项都声明了类型（str/int/float/bool），传错类型会
#      在启动时立刻报错，而不是跑到一半才崩；
#   2. 支持 .env 文件：敏感信息（如 API Key）写在 .env 文件里，.env 不进
#      Git 版本库（.gitignore 已排除），做到"密钥不入库"——这是工程化清单
#      §9.3 的硬性亮点；
#   3. 支持环境变量覆盖：同一个配置项，可以在运行时用系统环境变量覆盖，
#      便于不同环境（开发/生产）切换；
#   4. 优先级：环境变量 > .env 文件 > 类里写的默认值。
#
# 【配置加载的优先级（重要，理解 pydantic-settings 的核心）】
#   系统环境变量（最高） > .env 文件里的值 > 代码里定义的默认值（最低）
#   举例：DASHSCOPE_API_KEY 这个字段——
#     - 如果系统环境变量里有 DASHSCOPE_API_KEY，用它；
#     - 否则去 .env 文件里找 DASHSCOPE_API_KEY；
#     - 都没有，就用默认值（这里设为空字符串，程序会提示用户去配置）。
#
# 【怎么用（其它模块读配置的标准姿势）】
#   from rag.config.settings import settings      # 导入全局单例
#   settings.EMBEDDING_MODEL                          # 读嵌入模型名
#   settings.RAW_DIR                                  # 读原始语料目录
#   因为 settings 是一个"单例对象"（全项目唯一一份），任何模块 import 它，
#   拿到的都是同一个配置对象，保证配置一致、不漂移。
# ============================================================================

# ---------- 导入区 ----------
# Path：Python 标准库 pathlib 的类，用来表示"文件/目录路径"对象。
#       比手写字符串路径更安全（自动处理 Windows 反斜杠问题）。
from pathlib import Path

# Field：pydantic 提供的"字段描述器"，用来给字段附加说明和默认值信息。
#        我们用它给每个配置项写 description（说明）和默认值。
from pydantic import Field

# BaseSettings：pydantic-settings 库提供的"配置基类"。
#   我们继承它来定义配置类，就能自动获得"从 .env / 环境变量读取"的能力。
from pydantic_settings import BaseSettings

# ============================================================================
# 一、计算项目根目录（所有路径的"锚点"）
# ============================================================================

# 下面的几行是"找项目根目录"的固定套路，逐行解释：
#   __file__            Python 内置变量，表示"当前这个文件（settings.py）的路径"
#   Path(__file__)      把字符串路径转成 Path 对象
#   .resolve()          把路径转成"绝对路径"（从盘符开始的完整路径，如 E:/python/...）
#   .parent             取上一级目录；连续写 .parent 表示"往上走一级"
#
# settings.py 的物理位置是：src/rag/config/settings.py
#   所以：.parent          → src/rag/config/
#        .parent.parent    → src/rag/
#        .parent.parent.parent    → src/
#        .parent.parent.parent.parent → 项目根目录 rag-from-llm-ai/
# 因此要往上走 4 级才能到项目根目录。
# 注意：这里用 _PROJECT_ROOT（下划线前缀）命名，而不是 BASE_DIR。
# 原因：类体里也会定义同名字段 BASE_DIR，若模块级变量也叫 BASE_DIR，
# 在类体里引用时会被"字段名遮蔽"——拿到的是 FieldInfo 对象而非 Path，
# 导致 `BASE_DIR / "data"` 报 TypeError（pydantic 的经典坑）。
# 所以模块级用 _PROJECT_ROOT，类字段 BASE_DIR 的默认值引用它，互不冲突。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ============================================================================
# 二、配置类定义（继承 BaseSettings）
# ============================================================================

class Settings(BaseSettings):
    """全局配置类。所有配置项集中在这里，字段名全大写（业界约定俗成）。

    每个字段 = 一个配置项。字段名、类型、默认值、说明都写在类体里。
    pydantic-settings 会自动：
      1. 按字段名去 .env 文件 / 环境变量里找同名值（大小写不敏感）；
      2. 找到就用外部的值，找不到就用这里的默认值；
      3. 做类型校验（比如 EMBED_DIM 必须是 int，传了字符串会报错）。
    """

    # ---------------------------------------------------------------------
    # 第一部分：目录路径（所有文件/索引存放的位置）
    # ---------------------------------------------------------------------
    # Field(..., description="...") 里：
    #   第一个参数是"默认值"（这里用 ... 表示"必填，无默认"，但因为我们
    #   直接给了路径对象，实际是用后面的赋值）；description 是给人类看的说明。

    # 项目根目录（绝对路径）
    BASE_DIR: Path = Field(default=_PROJECT_ROOT, description="项目根目录")

    # 原始抓取的语料目录（未清洗的原始文件，可能很乱）
    RAW_DIR: Path = Field(default=_PROJECT_ROOT / "data" / "raw", description="原始语料目录")

    # 清洗后的标准文档目录（Markdown 格式 + 元数据）
    DOCS_DIR: Path = Field(default=_PROJECT_ROOT / "data" / "documents", description="清洗后标准文档目录")

    # 分块结果目录
    CHUNK_DIR: Path = Field(default=_PROJECT_ROOT / "data" / "chunks", description="分块结果目录")

    # 索引目录：FAISS 向量、BM25 索引、注册表、元数据都放这
    INDEX_DIR: Path = Field(default=_PROJECT_ROOT / "data" / "index", description="索引目录（FAISS/BM25/注册表/元数据）")

    # 日志目录
    LOG_DIR: Path = Field(default=_PROJECT_ROOT / "data" / "logs", description="运行日志目录")

    # ---------------------------------------------------------------------
    # 第二部分：嵌入模型（向量化用，P1 核心）
    # ---------------------------------------------------------------------
    # Ollama 本地服务地址（默认端口 11434，服务已在本机拉起）
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama 服务地址")

    # 嵌入模型名：qwen3-embedding:0.6b
    #   （639MB，实测 6.6 条/秒，能整卡驻留 4GB 显存）
    EMBEDDING_MODEL: str = Field(default="qwen3-embedding:0.6b", description="嵌入模型名")

    # 向量维度：qwen3-embedding 输出 1024 维
    EMBEDDING_DIM: int = Field(default=1024, description="向量维度")

    # 批量大小：一次调 Ollama 塞多少条文本（实测 16 条左右最优，太多会慢/报错）
    EMBED_BATCH_SIZE: int = Field(default=16, description="嵌入批量大小")

    # 查询侧指令前缀（Qwen3-Embedding 的"非对称检索"能力）：
    #   查询时在前面加这句指令，告诉模型"现在是要做检索"，能提升 1~5 个点精度；
    #   文档侧不加。这是新一代嵌入模型的标志性能力（详见架构文档 §2.1）。
    EMBED_QUERY_PREFIX: str = Field(
        default="Instruct: 检索与查询语义相关的内容\nQuery: ",
        description="查询侧指令前缀（非对称检索）",
    )

    # ---------------------------------------------------------------------
    # 第三部分：大语言模型（生成用，P3 用；这里先占位）
    # ---------------------------------------------------------------------
    # 阿里云百炼 DashScope 的 OpenAI 兼容接口 Base URL
    DASHSCOPE_BASE_URL: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope OpenAI 兼容接口地址",
    )

    # 生成模型名：qwen-max（可降级 qwen-plus / qwen-turbo 控成本）
    LLM_MODEL: str = Field(default="qwen-max", description="生成模型名")

    # DashScope API Key —— 敏感信息，不写死在代码里！
    #   默认空字符串；真正运行时从 .env 文件或环境变量 DASHSCOPE_API_KEY 读取。
    #   这就是"密钥不入库"的体现。
    DASHSCOPE_API_KEY: str = Field(default="", description="DashScope API Key（从 .env 读取）")

    # ---------------------------------------------------------------------
    # 第四部分：重排模型（cross-encoder，P2 用；这里先占位）
    # ---------------------------------------------------------------------
    # 重排模型名：Qwen3-Reranker-0.6B 的社区镜像。
    #   ⚠️ 官方 tag `qwen3-reranker` 不存在，必须用社区镜像名 + Q8_0 量化。
    RERANK_MODEL: str = Field(
        default="dengcao/Qwen3-Reranker-0.6B:Q8_0",
        description="重排模型（cross-encoder，社区镜像）",
    )

    # 重排候选条数：重排只对"少量候选"精排（召回后 Top-N 里再精排），
    #   不是对全库做。20~50 条是业界标准范围。
    RERANK_TOP_N: int = Field(default=20, description="重排候选条数")

    # ---------------------------------------------------------------------
    # 第五部分：分块参数（chunking）
    # ---------------------------------------------------------------------
    # 目标块大小（字符数）：只有超长段落才用它做"兜底硬切"。
    # 注意：这不是默认策略，默认是按标题层级切（见 splitter.py）。
    CHUNK_SIZE: int = Field(default=512, description="兜底块大小（字符数）")

    # 相邻块重叠的字符数：给断口打"补丁"，防止完整意思被拦腰切断。
    CHUNK_OVERLAP: int = Field(default=64, description="相邻块重叠字符数")

    # 父子块（small-to-big）：
    #   - 子块（小块）用来做嵌入和检索（语义聚焦、排序干净）
    #   - 父块（大块）命中后返回给生成（上下文完整）
    # 父块大小上限（字符数）：一个父块通常是整个 "## 章节"，这里设个上限兜底。
    PARENT_CHUNK_SIZE: int = Field(default=2048, description="父块大小上限（字符数）")

    # ---------------------------------------------------------------------
    # 第六部分：清洗筛选阈值（cleaner 用）
    # ---------------------------------------------------------------------
    # 中文密度下限：中文字符占比低于 0.3 判定为英文，过滤掉不入库
    MIN_CHINESE_RATIO: float = Field(default=0.3, description="中文密度下限")

    # 正文长度下限：清洗后少于 100 字的丢弃（太短没价值）
    MIN_TEXT_LEN: int = Field(default=100, description="正文长度下限（字符数）")

    # ---------------------------------------------------------------------
    # 第七部分：BM25 稀疏检索（P2 用；这里先占位分词相关参数）
    # ---------------------------------------------------------------------
    # BM25 的两个经典调参（k1 控制词频饱和，b 控制文档长度归一化）。
    # 先用业界默认值，P2 实测后再调。
    BM25_K1: float = Field(default=1.5, description="BM25 k1 参数（词频饱和）")
    BM25_B: float = Field(default=0.75, description="BM25 b 参数（文档长度归一化）")

    # ---------------------------------------------------------------------
    # 第八部分：FAISS 向量存储
    # ---------------------------------------------------------------------
    # 索引文件名（存在 INDEX_DIR 下）
    FAISS_INDEX_FILE: str = Field(default="faiss.index", description="FAISS 索引文件名")

    # 元数据文件名：与 FAISS 索引一一对应，记录 chunk_id / 来源 / 序号 / deleted 标记
    METADATA_FILE: str = Field(default="metadata.json", description="元数据文件名")

    # BM25 索引文件名
    BM25_FILE: str = Field(default="bm25.json", description="BM25 索引文件名")

    # 是否在入库前对向量做 L2 归一化（必须开！）：
    #   因为我们用 IndexFlatIP（内积检索），只有向量都归一化（长度=1）后，
    #   内积才等价于"余弦相似度"，检索结果才准确。
    NORMALIZE_EMBEDDINGS: bool = Field(default=True, description="向量 L2 归一化开关")

    # ---------------------------------------------------------------------
    # 第九部分：增量更新（updater / detector / compactor 用）
    # ---------------------------------------------------------------------
    # 文档注册表文件名（增量更新的"账本"）
    REGISTRY_FILE: str = Field(default="registry.json", description="文档注册表文件名")

    # 断点续传的进度文件（记录任务进度，中断后重跑能续上）
    JOB_FILE: str = Field(default="job.json", description="任务进度文件（断点续传）")

    # 僵尸向量（软删除后残留的废向量）占比超过这个比例就触发 compaction
    ZOMBIE_RATIO_THRESHOLD: float = Field(default=0.10, description="僵尸向量触发 compaction 阈值")

    # 距上次 compaction 超过多少天也触发（与上面的比例阈值是"或"关系）
    COMPACTION_MAX_DAYS: int = Field(default=30, description="距上次 compaction 天数阈值")

    # ---------------------------------------------------------------------
    # 【追加到】第九部分：增量更新（updater / detector / compactor 用）
    # ---------------------------------------------------------------------
    # 单篇文档处理失败的最大重试次数（企业级：瞬时故障要能自动重试）
    #   例：Ollama 偶发超时、网络抖动，重试 2~3 次大概率能恢复。
    MAX_RETRY: int = Field(default=3, description="单篇文档处理失败的最大重试次数")

    # 重试的退避系数（指数退避：第 n 次重试等待 base * factor^(n-1) 秒）
    #   避免"打爆"已经脆弱的服务（比如 Ollama 刚恢复又被瞬间打挂）。
    RETRY_BACKOFF_FACTOR: float = Field(default=2.0, description="重试退避系数（指数退避）")

    # dead-letter 记录文件：处理多次仍失败的文档清单，批末统一报告。
    #   企业级做法：单篇失败不阻断整批，记入"死信"，最后人工/自动兜底。
    DEAD_LETTER_FILE: str = Field(default="dead_letter.json", description="死信（失败文档）清单文件名")

    # 一致性对账开关：入库完成后校验"注册表 active chunk 数 == FAISS 向量数"
    ENABLE_RECONCILE: bool = Field(default=True, description="入库后是否做一致性对账")

    # ---------------------------------------------------------------------
    # 第十部分：允许处理的文件格式
    # ---------------------------------------------------------------------
    # 允许的文件扩展名集合（只有这些格式会被处理，其余忽略）。
    # 注意：.jsonl 是元数据文件（记录标题/URL），不当正文，所以不在列表里；
    #       .tex 是 LaTeX 源码，中文场景价值低，暂缓，也不在列表里。
    SUPPORTED_SUFFIXES: set = Field(
        default={".md", ".mdx", ".txt", ".rst", ".html", ".htm", ".xml", ".ipynb"},
        description="允许处理的文件扩展名集合",
    )


# ============================================================================
# 三、创建全局单例（供所有模块 import）
# ============================================================================

# Settings()：实例化配置类。
#   这一行执行时，pydantic-settings 会自动去读 .env 文件（默认在当前工作
#   目录找名为 .env 的文件）和环境变量，然后按"环境变量 > .env > 默认值"
#   的优先级，填充所有字段。
#
# 为什么要在这里实例化一次、而不是让每个模块自己 new？
#   → 保证"全项目唯一一份配置"。所有模块都 import 这个 settings 单例，
#     配置值绝对一致，不会出现"模块 A 读到的和模块 B 不一样"的漂移。
settings = Settings()
