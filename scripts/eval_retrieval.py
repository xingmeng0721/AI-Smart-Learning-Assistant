"""P10 指标验证：量化"混合检索 + Rerank"相对单路向量检索的首答正确率提升（目标 +15%）。

- 使用独立临时向量库（不污染 `.chroma`），seed 一个跨主题语料。
- 对同一评测集分别跑：单路向量 → 混合(RRF) → 混合+Rerank 三种策略。
- 指标：Recall@1 / Recall@5 / MRR（首答正确率 ≈ Recall@1）。
- Rerank 使用 components.build_reranker()（真实 bge-reranker-v2-m3 优先，Overlap 兜底）。

用法：python scripts/eval_retrieval.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "hf"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "hf", "hub"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from ingestion.chunker.chunker import StructureChunker  # noqa: E402
from ingestion.loader.markdown import MarkdownLoader  # noqa: E402
from retrieval.eval_metrics import mrr, rank_first, recall_at_k  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from service.components import build_embedder, build_reranker  # noqa: E402
from vectorstore.store import ChromaStore  # noqa: E402

# 跨主题语料（制造语义相近干扰，放大不同检索策略差异）
_CORPUS = {
    "数据结构": """# 数据结构

## 1.1 栈
栈是一种后进先出的线性数据结构，支持压栈与弹栈两种操作。

## 1.2 队列
队列遵循先进先出原则，常用于广度优先遍历与任务调度。

## 1.3 哈希表
哈希表通过散列函数把键映射到桶，实现近似 O(1) 的查找。
""",
    "操作系统": """# 操作系统

## 2.1 进程与线程
进程是资源分配与调度的基本单位，线程是进程内并发执行的最小单元。

## 2.2 虚拟内存
虚拟内存把逻辑地址映射到物理地址，通过分页管理并在缺页时换入换出。

## 2.3 死锁
死锁由互斥、占有且等待、不可剥夺、循环等待四个必要条件导致。
""",
    "计算机网络": """# 计算机网络

## 3.1 TCP 三次握手
TCP 通过 SYN、SYN-ACK、ACK 三次报文交换建立可靠连接。

## 3.2 HTTP 与 HTTPS
HTTPS 在 HTTP 之上叠加 TLS，对内容加密并校验完整性。

## 3.3 DNS 解析
DNS 把域名解析为 IP 地址，采用分级缓存与递归查询机制。
""",
    "数据库": """# 数据库系统

## 4.1 事务 ACID
事务具备原子性、一致性、隔离性与持久性四个特性。

## 4.2 索引
B+ 树索引按序组织键值，支持高效的范围查询与等值查找。

## 4.3 两阶段锁
两阶段锁协议分为加锁与解锁两阶段，保证调度的可串行化。
""",
    "机器学习基础": """# 机器学习基础

## 5.1 监督学习
监督学习使用带标签数据训练分类与回归模型。

## 5.2 过拟合
过拟合指模型在训练集上表现好但泛化到新样本时差。

## 5.3 交叉验证
交叉验证把数据分成多折，轮流作为验证集评估模型泛化能力。
""",
    "深度学习": """# 深度学习

## 6.1 神经网络
神经网络由多层神经元组成，通过反向传播更新参数。

## 6.2 注意力机制
注意力机制是 Transformer 的核心，允许模型关注输入不同位置。

## 6.3 自注意力
自注意力计算序列内部任意两位置的相关性，是 Transformer 的基础组件。
""",
    # 与上文"队列/锁/内存/哈希"产生词面重叠的易混淆章节，制造检索歧义
    "并发编程": """# 并发编程

## 7.1 加锁机制
互斥锁 读写锁 自旋锁 用于保证多个线程安全访问共享资源，避免数据竞争。

## 7.2 队列调度
任务队列 消息队列 用于解耦生产与消费，调节并发处理节奏。

## 7.3 内存屏障
内存屏障保证多核处理器缓存一致性，防止编译器和 CPU 指令重排带来错误。
""",
    # ===== 扩量语料：制造更多词面重叠与跨域歧义 =====
    "数据库实务": """# 数据库实务

## 8.1 锁的粒度
数据库锁分为表锁与行锁，行锁并发度更高但管理开销更大。

## 8.2 隔离级别
可重复读基于行锁快照实现，读已提交能避免脏读但可能出现不可重复读。

## 8.3 死锁检测
数据库通过死锁检测发现有向等待环后，与操作系统一样回滚其中一个事务。
""",
    "分布式系统": """# 分布式系统

## 9.1 一致性
线性一致性要求操作按单个全局顺序落账，最终一致性允许短暂不一致。

## 9.2 Raft 协议
Raft 通过领导者选举与日志复制维护集群状态一致，多数派投票决定提交。

## 9.3 CAP 理论
CAP 中分区容错不可舍弃，分布式系统在可用性与一致性之间取舍。
""",
    "编译原理": """# 编译原理

## 10.1 词法分析
词法分析把源程序字符流切分为 token 序列，识别关键字、标识符与常量。

## 10.2 语法分析
语法分析依据文法把 token 序列归约为抽象语法树，检测语法错误。

## 10.3 中间代码
中间代码独立于目标机器，便于做优化与跨平台移植。
""",
    "软件工程": """# 软件工程

## 11.1 敏捷开发
敏捷开发把需求拆成短迭代，频繁交付可运行软件并响应变化。

## 11.2 单元测试
单元测试针对最小函数单元验证行为，配合测试守卫防止回归。

## 11.3 代码重构
重构在不改变外部行为的前提下改善内部结构，降低维护成本。
""",
}

# 评测集：(query, 期望命中的章节片段)
_EVAL = [
    ("栈这种数据结构支持哪些操作", "1.1 栈"),
    ("哈希表为什么查找快", "1.3 哈希表"),
    ("进程和线程有什么区别", "2.1 进程与线程"),
    ("虚拟内存是如何映射地址的", "2.2 虚拟内存"),
    ("死锁产生的必要条件有哪些", "2.3 死锁"),
    ("TCP 连接是怎么建立的", "3.1 TCP 三次握手"),
    ("HTTPS 比 HTTP 安全在哪", "3.2 HTTP 与 HTTPS"),
    ("事务的四个特性是什么", "4.1 事务 ACID"),
    ("B+ 树索引有什么作用", "4.2 索引"),
    ("两阶段锁协议保证了什么", "4.3 两阶段锁"),
    ("什么是过拟合现象", "5.2 过拟合"),
    ("交叉验证怎么评估泛化", "5.3 交叉验证"),
    ("神经网络怎么更新参数", "6.1 神经网络"),
    ("Transformer 里的注意力机制是什么", "6.2 注意力机制"),
    ("自注意力是怎么计算的", "6.3 自注意力"),
    # 词面重叠的高难度查询（跨文档制造歧义）
    ("队列是怎么做到先进先出的", "1.2 队列"),
    ("加锁能避免多线程数据竞争吗", "7.1 加锁机制"),
    ("如何保证多核缓存一致性", "7.3 内存屏障"),
    ("消息队列在并发里起什么作用", "7.2 队列调度"),
    # 跨域同名术语（「锁」同时出现于数据库两阶段锁 与 并发加锁机制）
    ("数据库的两阶段锁想解决什么问题", "4.3 两阶段锁"),
    ("并发编程里自旋锁是干什么的", "7.1 加锁机制"),
    # ===== 扩量测评：新增高难度（词面/语义双重重叠） =====
    ("行锁和表锁的区别是什么", "8.1 锁的粒度"),
    ("可重复读隔离级别是怎么实现的", "8.2 隔离级别"),
    ("数据库死锁为什么也要回滚", "8.3 死锁检测"),
    ("线性一致性是什么意思", "9.1 一致性"),
    ("Raft 是怎么保证集群一致的", "9.2 Raft 协议"),
    ("CAP 理论讲的是哪三者的取舍", "9.3 CAP 理论"),
    ("词法分析输出什么序列", "10.1 词法分析"),
    ("语法分析怎么构造抽象语法树", "10.2 语法分析"),
    ("中间代码为什么跟机器无关", "10.3 中间代码"),
    ("敏捷开发是如何应对需求变化的", "11.1 敏捷开发"),
    ("单元测试如何防止回归", "11.2 单元测试"),
    ("代码重构会改变外部行为吗", "11.3 代码重构"),
    ("消息队列在并发里和数据库隔离级别有何关系", "8.2 隔离级别"),  # 词面重叠的干扰
    ("锁这种机制在数据库和并发编程里分别怎么用", "8.1 锁的粒度"),  # 混淆"锁"的归属
]


def _gold_ids(chunks, fragment: str):
    return {
        c.chunk_id
        for c in chunks
        if fragment in (c.title or "") or fragment in (c.section_path or "")
    }


def _collect(ranked, k):
    return [c["chunk_id"] for c in ranked][:k]


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="chroma_eval_"))
    try:
        embedder = build_embedder()
        store = ChromaStore(str(work), embedder=embedder)
        loader = MarkdownLoader()
        chunker = StructureChunker(settings.chunk_size)
        all_chunks = []
        for doc_id, md in _CORPUS.items():
            store.delete_doc(doc_id)
            blocks = loader.parse(md, source=doc_id)
            chunks = chunker.chunk_document(blocks, doc_id)
            store.upsert_docs(chunks)
            all_chunks.extend(chunks)
        retriever = HybridRetriever(store)
        retriever.sync_index()
        reranker = build_reranker()
        print(f"[eval] embedder={type(embedder).__name__} reranker={type(reranker).__name__} "
              f"chunks={len(all_chunks)} queries={len(_EVAL)}\n")

        strat = {"vector": [], "hybrid": [], "rerank": []}  # 每条 query: {r1,r5,mrr}
        for q, frag in _EVAL:
            g = _gold_ids(all_chunks, frag)
            v5 = [r["chunk_id"] for r in store.query(q, top_k=5)]
            hyb20 = retriever.retrieve(q, top_k=20)
            hyb5 = _collect(hyb20, 5)
            rnk5 = _collect(reranker.rerank(q, hyb20[:10]), 5)
            strat["vector"].append((recall_at_k(v5, g, 1), recall_at_k(v5, g, 5), mrr(v5, g)))
            strat["hybrid"].append((recall_at_k(hyb5, g, 1), recall_at_k(hyb5, g, 5), mrr(hyb5, g)))
            strat["rerank"].append((recall_at_k(rnk5, g, 1), recall_at_k(rnk5, g, 5), mrr(rnk5, g)))

        n = len(_EVAL)
        headers = ["策略", "Recall@1", "Recall@5", "MRR", "vs 向量(Recall@1)"]
        print(f"{headers[0]:<12}{headers[1]:>10}{headers[2]:>10}{headers[3]:>8}{headers[4]:>18}")
        base_r1 = sum(r[0] for r in strat["vector"]) / n
        for name in ("vector", "hybrid", "rerank"):
            s = strat[name]
            r1 = sum(x[0] for x in s) / n
            r5 = sum(x[1] for x in s) / n
            m = sum(x[2] for x in s) / n
            delta = (r1 - base_r1) / base_r1 * 100 if base_r1 else 0.0
            print(f"{name:<12}{r1:>10.2%}{r5:>10.2%}{m:>8.3f}{delta:>17.1f}%")
        print(f"\n基准确：单路向量 Recall@1 = {base_r1:.2%}")

        # 单条明细：vector 与 rerank 差异大的 query 展示
        print("\n[明细] 首命中 rank 差异（vector vs rerank）：")
        for (q, frag), vset, hyb, rnk in zip(
            _EVAL, strat["vector"], strat["hybrid"], strat["rerank"]
        ):
            if vset[2] != rnk[2]:  # mrr 不同
                g = _gold_ids(all_chunks, frag)
                vr = rank_first([c["chunk_id"] for c in store.query(q, top_k=5)], g)
                rr = rank_first(_collect(retriever.retrieve(q, top_k=20), 5), g)
                print(f"  「{q}」verror_rank={vr} rerank_rank={rr}  (gold={sorted(g)})")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()