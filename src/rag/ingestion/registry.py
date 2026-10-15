# ============================================================================
# registry.py —— 文档注册表（Document Registry）
#
# 位置：src/rag/ingestion/registry.py
#
# 【这个文件是干什么的】
#   增量更新（语料变了之后，只处理变化的部分，而不是全库重来）的根基是一张
#   "账本"，记录每一个文档当前的入库状态。没有这张账本，"哪些文档需要重新
#   处理"这个问题根本无法回答。
#
#   打个比方：图书管理员有一本"藏书登记册"，上面写着每本书的编号、内容指纹、
#   版本号、当前状态（在架/下架）。每天有新书、有改版、有下架，管理员只要对
#   照登记册，就知道今天该干什么——而不是把整个图书馆重新整理一遍。
#   这个登记册，就是这个文件里的 DocumentRegistry（文档注册表）。
#
# 【核心字段说明】（就是登记册上的每一列）
#   doc_id          文档唯一编号（用源文件的相对路径，稳定、能回溯到来源）
#   content_hash    清洗后正文的 SHA1 指纹（内容一变，指纹就变 → 用它判断"改没改"）
#   version         版本号，每成功入库一次 +1（配合 doc_id 组成"幂等键"，防重复入库）
#   chunk_ids       这篇文档切出的所有"块"的编号列表（删除/重组时靠它定位）
#   embedding_model 入库时用的嵌入模型名（换模型时用来触发"全量重嵌"）
#   status          状态：active（在库）/ deleted（已软删除）
#   indexed_at      最近一次成功入库的时间（后续做"新鲜度加权"排序用）
#
# 【为什么用 JSON 文件存，而不用数据库（SQLite）】
#   1. 零依赖：直接用 Python 标准库 json，不需要额外安装任何东西
#   2. 人眼可读：出问题时直接打开文件就能看，方便排查
#   3. 性能足够：几千条记录的读写是毫秒级，完全够用
#   4. 演进预留：将来条目超过 1 万条、或需要复杂查询时，再换成 SQLite——
#      但那是本文件内部的事，对外接口不变，上层代码一行都不用改
# ============================================================================
# ---------- 导入区 ----------
# json：Python 标准库，负责把 Python 字典/列表 和 JSON 字符串互相转换。
#       （JSON 是一种通用的文本数据格式，类似"写给人看的账本格式"）
# hashlib：Python 标准库，提供 SHA1、MD5 等"指纹算法"。
#          所谓指纹算法：给一段内容，算出一个固定长度的唯一字符串；
#          内容只要有一丁点变化，算出来的字符串就完全不同。
import hashlib
import json

# dataclass：Python 标准库提供的"数据类"装饰器。
#            它的作用是：帮你自动生成 __init__（初始化方法）等样板代码，
#            让你能用很少的代码定义一个"专门用来装数据的类"。
# field：dataclass 的辅助函数，用来给字段设置"默认值"。
#         list 默认值不能直接写 =[]，要用 field(default_factory=list) 这种安全写法。
from dataclasses import asdict, dataclass, field

# datetime：Python 标准库，处理日期时间。
#           这里的 datetime.now() 会返回"当前这一刻"的精确时间。
from datetime import datetime

# Path：Python 标准库 pathlib 模块里的类，专门用来表示"文件/目录路径"。
#       比手写字符串路径更安全（自动处理 Windows 的反斜杠问题）。
from pathlib import Path

# Optional / List：Python 的"类型标注"工具。
#   Optional[str] 表示"这个字段要么是字符串，要么是空（None）"
#   List[str]     表示"一个由字符串组成的列表"
# ----------------------------------------------------------------------------
# 【引入全项目统一的日志设施】
# setup_logger：项目自研的日志工具（src/rag/utils/logger.py），
#   它内部封装了"控制台 + 文件"双通道日志，并把配置做成"幂等共享"——
#   同一个名字（默认 "rag"）的 Logger，无论哪个模块调用，拿到的都是
#   同一个实例、同一套配置、同一个日志文件。
#   这样做的意义：入口脚本（如 updater.py）先配一次日志，各业务模块
#   （parser/cleaner/.../registry）用默认名调用时，自动共享这份配置，
#   所有人的日志都写进同一个文件、同一个时间线，排查问题不抓瞎。
# ----------------------------------------------------------------------------
from rag.utils.logger import setup_logger

# 模块级日志对象：本文件所有方法共用这一个 log。
# 调用 setup_logger() 时：
#   - 若入口脚本已经配置过（name="rag"），这里会"幂等命中"，直接复用已有配置；
#   - 若本模块被单独运行、还没有任何配置，这里会现场新建一套默认配置。
# 之后用 log.info / log.debug / log.warning / log.exception 打日志。
log = setup_logger()

# ============================================================================
# 第一部分：数据类（DocRecord）—— 一条"登记记录"的模板
# ============================================================================
@dataclass      # 装饰器：定义数据容器，可自动生成初始化代码
class DocRecord:
    # 文档登记记录，一条文档登记一行。每个字段都对应 "文档注册表"里的一列。
    doc_id: str                                             # 文档唯一标识：源文件相对路径
    content_hash: str                                       # 清洗后正文的SHA1指纹
    version: int = 1                                        # 本版号，默认从1开始，每次入口 + 1
    chunk_ids: list[str] = field(default_factory=list)      # 该文档切除的块ID列表
    embedding_model: str = ""                               # 入库时的嵌入模型
    status: str = "active"                                  # 状态：active(在库) / deleted(已删除)
    indexed_at: str | None = None                        # 最近入库时间

# ============================================================================
# 第二部分：工具函数 —— 计算"内容指纹"（content hash）
# ============================================================================
def compute_content_hash(text: str) -> str:
    """计算一段文本的 SHA1 指纹（本文件最核心的函数）。

        【功能原理】
            SHA1 是一种"单向指纹算法"：把任意长度的文本，映射成一个 40 个字符
            的十六进制字符串。关键性质：同样的文本永远算出同样的指纹；文本只要
            改一个字，指纹就完全不同。因此指纹是"这段内容变没变"的绝对判据。

        【为什么用它判断变更，而不用文件"修改时间"（mtime）】
            mtime 有两个坑：
            · 假变更：git checkout、解压、云盘同步会刷新 mtime，但内容没变；
            · 漏变更：某些工具改内容却不更新 mtime。
            指纹是"内容本身"算出来的，骗不了它。

        【参数】
            text : str —— 要计算指纹的文本（注意：是"清洗后正文"，而非原始文件）

        【返回】
            str —— 40 个字符的十六进制指纹字符串
        """
    hash_obj = hashlib.sha1(text.encode("utf-8"))  # 创建SHA1计算器，把字符转为字节
    return hash_obj.hexdigest()         # 生成 十六进制字符串

# ============================================================================
# 第三部分：主类（DocumentRegistry）—— 账本的"读写管家"
# ============================================================================
class DocumentRegistry:
    """文档注册表：负责账本的加载、查询、增删改、保存。

        【设计思想】
            把"账本"抽象成一个对象，外部代码只需调用 get / upsert / mark_deleted
            / save 等方法，不关心底层是 JSON 还是数据库。将来换存储（如 SQLite）
            只改本类内部实现，上层零改动——这就是"接口隔离"的价值。
        """
    # 参数 path：账本文件路径（Path 对象）。默认值设为 None，表示"没传就用
    #             settings 里的配置路径"，这样调用方能省事，也能自由覆盖。
    def __init__(self, path: Path | None = None):
        """构造函数：初始化账本，并从磁盘加载已有数据。

        【参数】
            path : Optional[Path] —— 账本文件路径。
                   · 传 None（默认）→ 用 settings 里的配置路径；
                   · 传具体路径 → 覆盖默认（测试时常用，指向临时文件，避免污染真实数据）。
        """
        # 若调用方没传路径，就去全局配置 settings 里取。
        # 这是"配置集中管理"的体现：路径不散落各处，统一从 settings 读。
        if path is None:
            from rag.config.settings import settings  # 延迟导入，避免循环依赖
            path = settings.INDEX_DIR / settings.REGISTRY_FILE

        self.path = path                            # 保存账本文件路径（Path 对象）
        self.records: dict = {}                     # 主表：doc_id → DocRecord（账本主体）
        self.hash_index: dict = {}                  # 反向索引：content_hash → doc_id
        self._load()                                # 启动时从磁盘加载已有数据

    # ----------------------------------------------------------------------
    # 内部方法：从磁盘加载账本
    # ----------------------------------------------------------------------
    def _load(self) -> None:
        """从磁盘加载账本（若文件存在）。文件不存在说明是首次运行，跳过。

        【原理】
            程序重启后，内存里的账本会丢，所以必须从 registry.json 恢复。
            加载过程：读文件 → json 解析 → 把 dict 重新还原成 DocRecord 对象。
        """
        # 文件不存在 = 首次运行，账本是空的，直接返回
        if not self.path.exists():
            return

        # 读文件内容（utf-8，防中文乱码），json.loads 解析成 dict
        raw = json.loads(self.path.read_text(encoding="utf-8"))

        # 主表：raw["records"] 是 {doc_id: {字段...}}，要把每个内层 dict
        # 还原成 DocRecord 对象。用 dict 推导式遍历。
        # DocRecord(**fields) 是"关键字参数展开"：把 dict 的键值对当作
        # 构造参数传给 DocRecord，等价于 DocRecord(doc_id=..., content_hash=...)
        self.records = {
            doc_id: DocRecord(**fields)
            for doc_id, fields in raw.get("records", {}).items()
        }

        # 反向索引：直接是 {content_hash: doc_id}，原样加载即可
        self.hash_index = raw.get("hash_index", {})

    # ----------------------------------------------------------------------
    # 内部方法：保存到磁盘（原子写入）
    # ----------------------------------------------------------------------
    def save(self) -> None:
        """把账本保存到磁盘。采用"原子写入"防止文件损坏。

        【为什么原子写入（企业级底线）】
            如果直接写目标文件，写到一半断电/崩溃，会留下一个损坏的 JSON，
            下次加载直接报错，账本就废了。
            正确做法：先写一个临时文件，全部写完后用 replace() 原子地替换
            目标文件——要么替换成功（文件完整），要么没替换（还是旧文件），
            不存在"写一半"的中间状态。
        """
        # 组装要保存的数据：主表要转回 dict 才能 json 序列化。
        # asdict(record) 把 DocRecord 对象转成 dict（dataclass 自带的能力）。
        data = {
            "records": {doc_id: asdict(record) for doc_id, record in self.records.items()},
            "hash_index": self.hash_index,
        }

        # 确保父目录存在（首次保存时目录可能还没建）
        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.path.with_suffix(".tmp")   # 临时文件：registry.tmp
        # 第 1 步：写临时文件。ensure_ascii=False 让中文正常显示；indent=2 缩进可读
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 第 2 步：原子替换（replace 在 Windows/Linux 上都是原子操作）
        tmp_path.replace(self.path)

    # ----------------------------------------------------------------------
    # upsert：插入或更新一条记录（增量写入的核心入口）
    # ----------------------------------------------------------------------
    def upsert(
        self,
        doc_id: str,
        content_hash: str,
        chunk_ids: list[str],
        embedding_model: str,
    ) -> DocRecord:
        """插入或更新一条文档记录，并同步维护反向索引。

        【功能原理】
            upsert = update + insert 的合体：
            · doc_id 已存在 → 更新，version + 1；
            · doc_id 不存在 → 新建，version 从 1 开始。
            同时把 content_hash → doc_id 写进反向索引（hash_index），
            供"按内容判重"（类型 B 入站）使用。

        【参数】
            doc_id          : str        文档唯一标识（源文件相对路径）
            content_hash    : str        清洗后正文的 SHA1 指纹
            chunk_ids       : List[str]  该文档切出的所有块 ID
            embedding_model : str        入库时用的嵌入模型名

        【返回】
            DocRecord —— 写入后的记录（调用方可能需要它的 version 等字段）
        """
        existing = self.records.get(doc_id)   # 取已有记录，不存在返回 None

        # 版本号：新文档从 1 开始，已有文档在旧版本上 +1
        version = 1 if existing is None else existing.version + 1

        # 构造新的 DocRecord 对象（字段含义见类定义注释）
        record = DocRecord(
            doc_id=doc_id,
            content_hash=content_hash,
            version=version,
            chunk_ids=list(chunk_ids),        # list() 拷贝一份，防止外部修改影响内部
            embedding_model=embedding_model,
            status="active",                  # 新写入的都是 active（在库）
            indexed_at=datetime.now().isoformat(),   # 当前时间，ISO 格式字符串
        )

        self.records[doc_id] = record              # 写入主表
        self.hash_index[content_hash] = doc_id     # 同步更新反向索引（内容→文档）

        return record

    # ----------------------------------------------------------------------
    # 按内容哈希反查（服务类型 B 入站判重）
    # ----------------------------------------------------------------------
    def find_by_hash(self, content_hash: str) -> str | None:
        """根据内容哈希查 doc_id（用于"新文档入站"判重）。

        【为什么需要它（对应文档 2.5.3 hash_index 反向索引）】
            临时上传的文件路径每次都变（如 uploads/tmp_123.md），用 doc_id
            （路径）判重会重复入库；只有按 content_hash 才能识别"同一份文档"。

        【参数】
            content_hash : str —— 内容哈希

        【返回】
            Optional[str] —— 命中返回 doc_id；未命中返回 None
        """
        return self.hash_index.get(content_hash)

    # ----------------------------------------------------------------------
    # 软删除：把 status 置为 deleted
    # ----------------------------------------------------------------------
    def mark_deleted(self, doc_id: str) -> None:
        """软删除一篇文档（把 status 置为 deleted）。

        【为什么是"软删除"而非"物理删除"（对应文档 2.5.5）】
            FAISS 的 IndexFlatIP 不支持删除单条向量。所以删除只能"逻辑上
            立即生效"（检索侧过滤掉），"物理上延迟清理"（由 compactor 干）。
            本方法只改状态标记，不动任何向量。

        【关键细节】反向索引 hash_index 不移除！
            因为历史 hash 仍需去重——若已删内容再次入站，仍要识别出"它来过"，
            避免已删文档被重新入库。

        【参数】
            doc_id : str —— 要软删除的文档标识
        """
        if doc_id in self.records:                    # 记录存在才处理
            self.records[doc_id].status = "deleted"   # 改状态字段

    # ----------------------------------------------------------------------
    # 查询：取某文档的 chunk_ids（删除/重组时定位块）
    # ----------------------------------------------------------------------
    def get_chunk_ids(self, doc_id: str) -> list[str]:
        """取某文档切出的所有块 ID。

        【用途】删除/重组文档时，靠这些 ID 找到对应的向量块。

        【参数】
            doc_id : str —— 文档标识

        【返回】
            List[str] —— 块 ID 列表（不存在则返回空列表）
        """
        record = self.records.get(doc_id)
        return record.chunk_ids if record else []

    # ----------------------------------------------------------------------
    # 查询：按状态过滤记录（active / deleted）
    # ----------------------------------------------------------------------
    def iter_by_status(self, status: str):
        """按状态过滤记录（生成器，逐个产出）。

        【为什么用生成器（yield）】
            记录可能有几万条，一次性组装成列表会占内存。生成器"用一条、
            产一条"，内存占用恒定，适合大语料。

        【参数】
            status : str —— 要过滤的状态（"active" 或 "deleted"）

        【产出】
            逐个 DocRecord 对象
        """
        for record in self.records.values():   # values() 遍历所有记录
            if record.status == status:        # 状态匹配才产出
                yield record

    # ----------------------------------------------------------------------
    # 统计：active 状态的 chunk 总数（对账用）
    # ----------------------------------------------------------------------
    def count_active_chunks(self) -> int:
        """统计所有 active 文档的 chunk 总数。

        【用途（对应文档 2.5.4 第④步对账）】
            注册表 active 文档的 chunk 总数，必须等于 FAISS 里的向量数。
            两边对不上，说明有 bug 或数据漂移，需要报警。

        【返回】
            int —— active 状态的 chunk 总数
        """
        total = 0
        for record in self.iter_by_status("active"):   # 复用上面的状态过滤
            total += len(record.chunk_ids)             # 累加每篇文档的块数
        return total

    # ----------------------------------------------------------------------
    # 查询：检查某 doc_id 是否存在且为 active
    # ----------------------------------------------------------------------
    def is_active(self, doc_id: str) -> bool:
        """判断某文档是否存在且处于 active（在库）状态。

        【参数】
            doc_id : str —— 文档标识

        【返回】
            bool —— True 表示在库；False 表示不存在或已删除
        """
        record = self.records.get(doc_id)
        return record is not None and record.status == "active"
