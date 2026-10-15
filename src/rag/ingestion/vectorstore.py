# ============================================================================
# vectorstore.py —— 向量存储（VectorStore）：接口 + FAISS 实现
#
# 位置：src/rag/ingestion/vectorstore.py
#
# 【本文件是干什么的】
#   对应《P1-索引构建.md》2.4 节。把嵌入算出来的"数字坐标"存起来，并且能
#   快速找出"离查询最近的点"。
#
#   我们选 FAISS（Meta 开源的向量检索库），理由（对应文档 2.4）：
#     · FAISS 是"一个工具库"，import 进来就能用，不占额外资源；
#     · Milvus/Qdrant 是"一套完整系统"，要单独维护服务，杀鸡用牛刀；
#     · 几十万块数据用 FAISS 绰绰有余。
#
# 【本文件的"企业级"重点：接口隔离】
#   用 Protocol 定义 VectorStore 接口，FAISS 只是其中一个实现。
#   将来换 Qdrant/LanceDB（对应文档 2.5.9 演进路径），只改实现类，
#   上层 pipeline/updater 零改动。
#
# 【最大的坑（对应文档 2.4）】
#   FAISS 的 IndexFlatIP 不支持单条删除/更新，只能增量 add()。
#   所以"删除"要靠"软删除"（在元数据里打 deleted 标记，检索时过滤）。
# ============================================================================

# ---------- 导入区 ----------
# Path：Python 标准库，处理路径。
# json：Python 标准库，元数据序列化。
import json
from pathlib import Path

# List / Tuple / Protocol / runtime_checkable / Optional：类型标注 + 协议。
from typing import Protocol, runtime_checkable

# faiss：第三方库，向量相似度检索。
import faiss

# numpy：第三方库，高性能数值计算（向量用 numpy 数组表示）。
import numpy as np

# 项目内：日志设施。
from rag.utils.logger import setup_logger

# 模块级日志对象。
log = setup_logger()


# ============================================================================
# 第一部分：VectorStore 协议（接口定义）
# ============================================================================
@runtime_checkable
class VectorStore(Protocol):
    """VectorStore 协议：定义向量存储"能做什么"。

    任何实现了以下方法的对象都可作为向量存储（FAISS、Qdrant、LanceDB 等）。
    """

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        """追加一批向量（chunk_id 与向量一一对应）。"""
        ...

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """检索最相近的 top_k 个向量，返回 (chunk_id, 相似度)。"""
        ...

    def mark_deleted(self, chunk_ids: list[str]) -> None:
        """软删除一批 chunk（在元数据打标，检索时过滤）。"""
        ...

    def count_active(self) -> int:
        """返回 active（未删除）的向量数量（对账用）。"""
        ...

    def save(self, path: Path) -> None:
        """持久化到磁盘。"""
        ...

    def load(self, path: Path) -> None:
        """从磁盘加载。"""
        ...


# ============================================================================
# 第二部分：FAISS 实现
# ============================================================================
class FaissVectorStore:
    """基于 FAISS 的向量存储实现。

    【索引类型：IndexFlatIP】
        IP = Inner Product（内积）。因为我们入库前做了 L2 归一化（向量长度=1），
        内积就等价于余弦相似度。IndexFlat 是"暴力精确检索"——挨个算距离，
        几十万级数据在内存里毫秒级出结果，完全够用（对应文档 2.4）。

    【软删除实现（对应文档 2.5.5）】
        FAISS 删不掉单条向量，所以额外维护一份"元数据"（chunk_id → 状态），
        删除时只改元数据标记，检索后在返回前过滤掉 deleted 的块。
    """

    def __init__(self, dim: int):
        """构造函数。

        参数：
            dim : int —— 向量维度（qwen3-embedding 是 1024）
        """
        self.dim = dim
        # faiss.IndexFlatIP：内积检索索引，dim 是向量维度。
        # 显式声明类型为 faiss.Index（基类）：因为 load 时 read_index 返回的
        # 是基类 Index（可能是任意索引类型），而非固定 IndexFlatIP。
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        # 元数据映射：faiss_id（索引内部序号）→ {chunk_id, deleted}
        # 因为 FAISS 只认内部序号（0,1,2,...），我们需要一张表把序号映射回 chunk_id。
        # 类型：list[dict]，每个 dict 是 {"chunk_id": str, "deleted": bool}
        self.metadata: list[dict] = []

    # ----------------------------------------------------------------------
    # 追加向量
    # ----------------------------------------------------------------------
    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        """追加一批向量到索引。

        【原理（对应文档 2.4）】
            IndexFlatIP 支持增量 add()——直接往里追加，不需要重建。
            这就是"追加是廉价的"（而删除/重组要另行设计）的依据。

        参数：
            chunk_ids : List[str]            —— 块 ID 列表
            vectors   : List[List[float]]    —— 向量列表（与 chunk_ids 一一对应）
        """
        # 把 Python 列表转成 numpy 数组（FAISS 只认 numpy）。
        # dtype="float32"：FAISS 要求 32 位浮点数（节省内存、加速计算）。
        arr = np.array(vectors, dtype="float32")

        # 先记录元数据（每个向量一条，下标从当前长度开始递增）
        # 注意：FAISS 内部会自动按追加顺序分配递增序号，无需我们手动维护起始下标。
        for chunk_id in chunk_ids:
            self.metadata.append({"chunk_id": chunk_id, "deleted": False})

        # 再往 FAISS 索引里追加向量
        self.index.add(arr)

        log.debug("FAISS 已追加 %d 条向量", len(chunk_ids))

    # ----------------------------------------------------------------------
    # 检索
    # ----------------------------------------------------------------------
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """检索最相近的 top_k 个向量。

        【原理】
            FAISS 的 index.search 一次可查多个查询向量，返回 (距离, 序号)。
            我们只查一个，取结果后把"序号"映射回"chunk_id"，并过滤掉
            deleted 的块（软删除生效点）。

        参数：
            query_vector : List[float] —— 查询向量（应已归一化）
            top_k        : int          —— 返回最相近的几条

        返回：
            List[Tuple[str, float]] —— [(chunk_id, 相似度), ...]，相似度降序
        """
        # 转 numpy，形状 (1, dim)：FAISS 期望 (查询数, 维度) 的二维数组
        q = np.array([query_vector], dtype="float32")

        # 多取一些候选（deleted 的块会占名额，多取点再过滤，保证能凑够 top_k）
        k = min(top_k * 2 + 10, len(self.metadata))   # 不能超过总数
        if k == 0:
            return []   # 空索引，直接返回

        # index.search 返回 (distances, indices)：
        #   distances 是相似度（IndexFlatIP 下是内积，越大越相似）
        #   indices   是内部序号
        distances, indices = self.index.search(q, k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):   # [0] 取第一个查询的结果
            idx = int(idx)                                # numpy 转 Python int
            if idx < 0 or idx >= len(self.metadata):      # -1 表示无效结果，跳过
                continue
            meta = self.metadata[idx]
            if meta["deleted"]:                           # 软删除过滤（关键！）
                continue
            results.append((meta["chunk_id"], float(dist)))

            if len(results) >= top_k:                     # 凑够 top_k 就停
                break

        return results   # 相似度已是降序（FAISS 默认返回降序）

    # ----------------------------------------------------------------------
    # 软删除
    # ----------------------------------------------------------------------
    def mark_deleted(self, chunk_ids: list[str]) -> None:
        """软删除一批 chunk（在元数据打标，逻辑上立即失效）。

        【原理（对应文档 2.5.5）】
            不做物理删除（FAISS 删不掉），只把元数据的 deleted 置 True。
            检索时（search 方法）会过滤掉这些块 → 逻辑上"搜不到"。

        参数：
            chunk_ids : List[str] —— 要软删除的块 ID 列表
        """
        target = set(chunk_ids)   # 转集合，查找 O(1)
        for meta in self.metadata:
            if meta["chunk_id"] in target:
                meta["deleted"] = True

    # ----------------------------------------------------------------------
    # 统计 active 数量（对账用）
    # ----------------------------------------------------------------------
    def count_active(self) -> int:
        """返回 active（未删除）的向量数量。

        【用途（对应文档 2.5.4 第④步对账）】
            应等于注册表里 active 文档的 chunk 总数。

        返回：
            int —— active 向量数
        """
        return sum(1 for meta in self.metadata if not meta["deleted"])

    # ----------------------------------------------------------------------
    # 重建支持：按 chunk_id 取出向量（compaction 用）
    # ----------------------------------------------------------------------
    def get_active_vectors(self) -> list[tuple[str, list[float]]]:
        """取出所有 active（未删除）的 (chunk_id, 向量) 对。

        【用途（对应文档 2.5.6 compaction）】
            compaction 需要"只保留 active 的向量"重建索引。FAISS 的
            IndexFlatIP 支持 reconstruct(序号) 按内部序号取出原始向量，
            结合 metadata（记录哪些是 active），就能筛选出要保留的向量。

        【返回】
            List[Tuple[str, List[float]]] —— [(chunk_id, 向量), ...]
        """
        result = []
        for faiss_id, meta in enumerate(self.metadata):
            if meta["deleted"]:                       # 跳过僵尸向量
                continue
            # index.reconstruct(序号)：FAISS 原生方法，取出该序号的向量（numpy 数组）
            vector = self.index.reconstruct(faiss_id)
            # 转回 Python list（float32 转 Python float）
            result.append((meta["chunk_id"], [float(x) for x in vector]))
        return result

    # ----------------------------------------------------------------------
    # 持久化 / 加载
    # ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """持久化到磁盘（FAISS 索引 + 元数据分开存）。

        参数：
            path : Path —— 索引文件路径（元数据存为 path 同名 .json）
        """
        path.parent.mkdir(parents=True, exist_ok=True)   # 确保目录存在
        # faiss.write_index：把索引序列化到文件
        faiss.write_index(self.index, str(path))
        # 元数据单独存 JSON
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.debug("FAISS 索引已保存：%s", path)

    def load(self, path: Path) -> None:
        """从磁盘加载（FAISS 索引 + 元数据）。

        参数：
            path : Path —— 索引文件路径
        """
        # faiss.read_index：从文件反序列化索引
        self.index = faiss.read_index(str(path))
        # 加载元数据
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        log.debug("FAISS 索引已加载：%s（%d 条）", path, len(self.metadata))
