# AI 智能学习助教 —— 大纲

基于 RAG（Retrieval-Augmented Generation）构建知识增强问答与主动学习系统。

> 提示：可以使用 Ctrl+F 快速查找自己想要的章节。

## 1. 项目目标

- 从文档解析 → 结构化分块 → 向量化 → 混合检索 → 重排 → 生成 + 引用溯源，构建完整知识问答链路。
- 支持多轮对话、知识库增量更新、Tool Use 智能调度。
- 用可量化指标验证链路效果（首答正确率、召回质量）。

## 2. 技术选型

| 层 | 选型 | 说明 |
|---|---|---|
| 语言/运行时 | Python 3.14 + venv + pip | 本机环境 |
| Web 服务 | FastAPI + Uvicorn | 异步 API + Streaming Response |
| Web 前端 | Vue 3 + Vite | 对话 / Agent / 知识库管理 / 会话管理 |
| LLM / Embedding / Rerank | 本地 Ollama | HTTP API `localhost:11434`，无需 CLI |
| 向量数据库 | ChromaDB | 内嵌式，元数据过滤，会话上下文存储 |
| TF-IDF 关键词检索 | scikit-learn TfidfVectorizer（自实现索引） | 双路召回之一 |
| 文档解析 | pypdf / python-docx / markdown | 分类型解析 |
| 测试 | pytest | 全程 TDD |
| 包/依赖 | requirements.txt | 轻量管理 |

## 3. 系统架构与模块划分

```
┌─────────────────────────────────────────────────────────────┐
│                       API 层 (FastAPI)                        │
│  /query(流式) /agent/query  /ingest  /doc*  /session*  /health │
│                     + 静态托管 Web 前端                        │
└───────────────┬──────────────────────────────┬────────────────┘
                │                              │
      ┌─────────▼──────────┐       ┌───────────▼───────────┐
      │   Generation 层      │       │   Retrieval 层         │
      │  · Query Rewrite    │       │  · 向量检索            │
      │  · 生成+引用溯源     │       │  · TF-IDF 关键词检索    │
      │  · 幻觉控制/拒答     │──────▶│  · RRF/加权融合        │
      │  · 多轮上下文        │       │  · Rerank 二次精排      │
      └─────────┬──────────┘       │  · 置信度阈值判定        │
                │                   └───────────┬───────────┘
                │                               │
      ┌─────────▼───────────────────────────────▼───────────┐
      │                     Core 数据层                       │
      │  · Embedding（Ollama）  · ChromaDB 向量存储            │
      │  · TF-IDF 索引          · 会话记忆                     │
      └──────────────────────────┬────────────────────────────┘
                                 │
      ┌──────────────────────────▼────────────────────────────┐
      │                  Ingestion 层                          │
      │  文档解析 → 结构化 Chunking → 递归二分超 Token           │
      │  → Embedding → 增量写入向量库 / 更新索引                 │
      └────────────────────────────────────────────────────────┘
```

### 模块职责

| 模块 | 目录 | 职责 |
|---|---|---|
| 配置 | `config/` | Ollama 地址/模型、Chroma 路径、阈值等集中管理 |
| 文档接入 | `ingestion/loader/` | PDF / DOCX / MD / TXT 解析（按后缀自动选 Loader） |
| 分块 | `ingestion/chunker/` | 章节/小节层级结构化分块、语义边界、递归二分 |
| 嵌入 | `embedding/` | Ollama Embedding 客户端 |
| 向量存储 | `vectorstore/` | ChromaDB 封装：写入/检索/增量更新/元数据过滤 |
| 检索 | `retrieval/` | 向量 + TF-IDF 双路召回、RRF/加权融合 |
| 重排 | `retrieval/rerank/` | BGE Rerank 二次精排 |
| 查询理解 | `query/` | Query Rewrite、意图拆分、多路检索融合 |
| 生成 | `generation/` | LLM 生成、引用绑定、拒答/边界检测 |
| 记忆 | `memory/` | 基于向量存储的会话上下文维护 |
| 服务 | `api/` | FastAPI 异步路由 + SSE 流式返回 |
| Agent | `agent/` | Tool Use 调度：笔记查询/资料检索/章节跳转 |
| 前端 | `web/` | Vue 3 + Vite：对话 / Agent / 知识库管理 / 会话管理（构建产物由 FastAPI 托管） |
| 测试 | `tests/` | pytest，按模块组织 |

### 核心数据流（一次问答）

1. 用户 Query 进入 → 结合历史做 **Query Rewrite**（去指代/拆多意图）。
2. 重写后的 Query 并行执行 **向量检索** + **TF-IDF 关键词检索** 双路召回。
3. 候选 Chunk 经 **RRF/加权融合** 去重排序。
4. 取 Top-N 送 **Rerank** 二次精排，注入 LLM 上下文。
5. 生成前做 **置信度阈值 / 知识库边界** 判定；低相关 Query 触发拒答。
6. LLM **流式生成**答案，每个句子绑定原始 Chunk Metadata，输出引用溯源。

## 4. 关键词汇

- **Chunking**：将长文本按结构化层级切成语义完整的 Chunk。
- **Hybrid Retrieval**：向量语义召回 + 关键词精确召回融合。
- **Rerank**：对上一步召回的候选做交叉编码精排。
- **RRF**：Reciprocal Rank Fusion，多路结果融合算法。
- **Query Rewrite**：LLM 将口语化/带指代的多轮 Query 改写为标准检索语句。
- **Streaming Response**：Token 逐块流式返回，降低感知延迟。

## 5. 量化指标（验收目标）

- 首答正确率较单路向量检索提升约 **15%**（Rerank + 融合贡献）。
- 拒答准确率：低相关 Query 正确触发拒答。
- 引用可溯源：答案可定位到原文 Chunk 与章节。

## 6. 待办（后续完善）

- [x] Ollama 服务与模型安装（qwen2.5/llama3.1 + nomic-embed-text/bge-m3 + bge-reranker-v2-m3）
- [x] 会话持久化（落盘 `.sessions/sessions.json`）
- [x] Agent Tool Use 落地清单细化
- [x] 前端对话界面与知识库/会话管理功能（Vue 3 + Vite）
- [x] PDF / DOCX / TXT 文档解析（`/upload` 文件导入）