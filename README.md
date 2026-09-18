# AI 智能学习助教 —— RAG 知识增强问答系统

基于 RAG（Retrieval-Augmented Generation）架构的智能学习助教，覆盖**文档解析 → 结构化分块 → 向量化 → 混合检索 → 重排 → 生成 + 引用溯源**完整链路，支持多轮对话与知识库增量更新。

> 仓库：https://github.com/xingmeng0721/AI-Smart-Learning-Assistant
> 提示：可以使用 Ctrl+F 快速查找自己想要的章节。

## 功能特性

- **混合检索**：向量检索（ChromaDB）+ TF-IDF 关键词检索双路召回，RRF 融合。
- **Rerank 精排**：对融合候选二次排序，把高相关上下文注入 LLM。
- **结构化 Chunking**：按章节层级切分，超长块递归二分，保留语义边界。
- **Query Rewrite**：多轮指代消解、多意图拆分（接真实 LLM 后启用）。
- **幻觉控制**：检索置信度阈值判定，低相关 Query 触发拒答。
- **引用溯源**：答案绑定原始 Chunk Metadata，可定位到章节/页码/源文件。
- **异步流式服务**：FastAPI + SSE 流式返回 Token，降低感知延迟。
- **Agent Tool Use**：ReAct 循环，LLM 决策调用工具（笔记查询 / 资料检索 / 章节跳转）。
- **增量导入**：按 doc_id 幂等更新，导入即索引；支持 Markdown/PDF/DOCX/TXT 文件上传。
- **会话持久化**：多轮对话默认落盘 `.sessions/sessions.json`，重启不丢失（可关闭）。
- **Web 前端**（Vue 3 + Vite）：对话（SSE 流式）/ Agent / 知识库管理 / 会话管理四个面板。

## 技术栈

Python 3.11 · FastAPI · Uvicorn · ChromaDB · scikit-learn · httpx

## 快速启动

### 1. 环境

```bash
conda create -y -n airag-tutor python=3.11
conda activate airag-tutor
pip install -r requirements.txt   # 依赖清单见 requirements.txt
```

> 依赖较多（含 torch + FlagEmbedding，用于 Rerank 精排），安装耗时较长。

### 2. 种入示例知识库（可选）

```bash
python scripts/seed_demo.py
```

### 3. 启动服务

```bash
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

### 4. 验证

```bash
python scripts/e2e_check.py   # 端到端：问答 / 拒答 / 增量导入
# 或浏览器打开 http://127.0.0.1:8000/ 使用 Web 界面
# API 交互文档：http://127.0.0.1:8000/docs
```

## Web 界面

启动服务后，浏览器打开 **http://127.0.0.1:8000/** 即可使用四个面板：

| 面板 | 功能 |
|---|---|
| 对话 | 多轮流式问答（SSE 实时渲染 token，展示改写意图/引用），可新建会话 |
| Agent | Agent Tool Use 问答（max_steps 可调），展示工具调用轨迹与最终答案 |
| 知识库 | 文档列表、查看分块、删除文档、粘贴 Markdown 文本导入新文档 |
| 会话管理 | 按 session 查看/清空多轮对话历史 |

> 前端源码在 `web/`（Vue 3 + Vite）。开发模式：`cd web && npm run dev`（自动代理 API 到 8000 端口）；生产构建：`npm run build` 产物 `web/dist` 由 FastAPI 自动托管。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（返回 chunk 数、模式、组件类型） |
| POST | `/query?session_id=&q=` | 流式问答（SSE） |
| POST | `/agent/query?q=&max_steps=` | Agent 问答（LLM 决策调用工具） |
| POST | `/ingest` | 增量导入 Markdown（支持 query 参数或 JSON body `{doc_id,text}`，幂等） |
| POST | `/upload` | 上传文件导入（按后缀自动选 md/pdf/docx/txt 解析器，`form-data: file`，可选 `doc_id`） |
| GET | `/documents` | 文档列表（按 doc_id 聚合 chunk 数/标题） |
| GET | `/doc/{doc_id}` | 查看文档分块列表 |
| DELETE | `/doc/{doc_id}` | 删除文档 |
| GET | `/sessions` | 会话列表 |
| GET | `/session/{session_id}` | 查看会话消息 |
| DELETE | `/session/{session_id}` | 清空会话 |

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `LLM_MODEL` | `qwen2.5:7b` | 生成模型 |
| `EMBED_MODEL` | `bge-m3` | 嵌入模型 |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | 重排模型 |
| `BGE_RERANKER_MODEL` | `models/bge-reranker-v2-m3` | 本地 BgeReranker 模型目录（项目内 `models/`） |
| `CHUNK_SIZE` | `512` | 分块目标 token 数 |
| `RETRIEVAL_TOP_K` | `20` | 召回候选数 |
| `RERANK_TOP_K` | `5` | 精排进入上下文数 |
| `RERANK_FEED_TOP_K` | `8` | 进入精排前保留候选数（限制 CPU 精排耗时） |
| `RERANK_ENABLED` | `true` | 是否启用精排（追求速度可设 `false` 走混合检索原排序） |
| `RERANKER_USE_FP16` | `auto` | BgeReranker 半精度（量化）：`true/false/auto`（auto=有 CUDA 才开） |
| `RERANKER_BATCH_SIZE` | `32` | BgeReranker 整批打分大小 |
| `EMBED_BATCH_SIZE` | `16` | OllamaEmbedder 分批向量化大小（每批独立重试） |
| `CONFIDENCE_THRESHOLD` | `0.35` | 拒答置信度阈值（真实 Embedding 默认；本地 Hash 模式可放宽到 0.15） |
| `CHROMA_PERSIST_DIR` | `.chroma` | 向量库持久化目录 |
| `SESSION_PERSIST_FILE` | `.sessions/sessions.json` | 会话持久化文件（置空 `""` 即不落盘） |

## 接入真实模型（Ollama）

模型层为**可插拔抽象 + 自动降级**：`service/components.py` 会检测 Ollama 与模型可用性，优先用真实模型，否则回退本地实现。

| 抽象 | 真实模型 | 本地备用 |
|---|---|---|
| Embedding | `OllamaEmbedder`（bge-m3） | `HashEmbedder`（确定性哈希） |
| Rerank | `BgeReranker`（BAAI/bge-reranker-v2-m3，本地 HF 模型） | `OverlapReranker`（字符重叠） |
| LLM | `OllamaLLM`（qwen2.5:7b） | `MockLLM`（规则回复） |

### 安装与模型（模型目录可用环境变量指定到任意盘符）

```bash
# 1. 安装 Ollama（winget 或官网安装包）
winget install --id Ollama.Ollama -e

# 2. 【建议】模型目录放到空间充足的盘（避免占满系统盘 C:）
#    在"用户环境变量"新建 OLLAMA_MODELS=D:\ollama\models   # 换成你的目录
#    重启 Ollama

# 3. 拉取模型（LLM 与 Embedding）
ollama pull qwen2.5:7b     # LLM（约 4.7GB）
ollama pull bge-m3         # Embedding（约 1.2GB）

# 4. 启动 / 验证
ollama serve
curl http://localhost:11434/api/tags   # 应看到上述模型

# 5. 拒答阈值（真实 Embedding 下默认 0.35，可按需调整）
set CONFIDENCE_THRESHOLD=0.35
```

### Rerank 模型（BAAI/bge-reranker-v2-m3）

Ollama 官方库没有独立 rerank 模型，本项目改用**本地加载 HF 的 Cross-Encoder 精排器**：

```bash
# 1. 安装依赖（CPU 版 PyTorch + BAAI 官方 FlagEmbedding）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install FlagEmbedding

# 2. 下载模型到项目 models/ 目录（约 2.3GB；国内可用 HF_ENDPOINT=https://hf-mirror.com）
set HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3')"
```

- 模型权重就绪后 `service/components.py` 会自动选中 `BgeReranker`（`/health` 的 `reranker` 上报该类型），否则依次回退 `OllamaReranker` → `OverlapReranker`。
- 分数经 sigmoid 归一化为 (0,1)：真实 reranker 上「注意力机制」相关命中 0.97，无关 ~0.001，区分度显著。
- 自定义目录：设置环境变量 `BGE_RERANKER_MODEL`（默认 `models/bge-reranker-v2-m3`）；`models/` 已加入 `.gitignore`，不会入库。

## 项目结构

```
config/           配置中心
ingestion/        文档解析(loader: md/pdf/docx/txt) + 结构化分块(chunker)
embedding/        Embedding 抽象 + HashEmbedder
vectorstore/      ChromaDB 封装（写入/检索/增量/删除）
retrieval/        混合检索(TF-IDF+向量) + 融合 + Rerank + 置信度
query/            Query Rewrite
memory/           会话记忆（可落盘持久化）
generation/       生成链路（上下文组装/引用）
service/          RAG 编排（检索→重排→置信度→生成）
agent/            Agent 工具（笔记查询/资料检索/章节跳转）+ ReAct 循环
api/              FastAPI 服务（SSE 流式 / /agent/query / 管理端点 / /upload / 静态托管）
web/              Web 前端（Vue 3 + Vite，构建产物 web/dist）
llm/              LLM 抽象 + Ollama/Mock 实现
scripts/          种库 / 端到端验证 / 检索评测
```

> `tests/`、`models/`、`.chroma/`、`.sessions/`、`.ollama/`、`web/dist/` 等本地/产物文件在 `.gitignore` 中，不会提交。

## 测试

```bash
python -m pytest tests/ -v
```

## 已知限制

- **Rerank 冷启动**：首次调用 BgeReranker 需加载约 2.3GB 权重（CPU，约数秒）；依赖 FlagEmbedding/torch。
- **Rerank 耗时**：CPU 上 bge-reranker-v2-m3 逐候选精排有延迟，`RERANK_TOP_K` 建议保持较小。
- **LLM 改写耗时**：接 qwen2.5:7b 后，查询会先经 LLM 改写再检索，单轮延迟有所上升。
