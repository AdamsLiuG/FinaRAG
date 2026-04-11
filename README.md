<p align="center">
  <h1 align="center">FinaRAG</h1>
  <p align="center"><strong>面向中文金融研报与年报的 RAG 问答系统</strong></p>
  <p align="center">
    <em>Retrieval-Augmented Generation for Chinese Financial Document Understanding</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
  <img src="https://img.shields.io/badge/framework-RAG-orange" alt="RAG"/>
  <img src="https://img.shields.io/badge/LLM-Qwen-purple" alt="Qwen"/>
  <img src="https://img.shields.io/badge/retrieval-FAISS%20|%20BM25%20|%20BGE--M3-009688" alt="Hybrid Retrieval"/>
</p>

---

## 📖 项目概述

FinaRAG 是一个面向 **中文金融研报 / 年报** 场景的端到端 RAG（Retrieval-Augmented Generation）系统，覆盖从 PDF 解析、结构化表格序列化、多路召回、级联重排到 LLM 生成与答案校验的完整流水线。

系统针对金融文档的特有挑战——**复杂表格抽取、数值精度要求高、跨文档比较查询、中英双语混排**——提供了工程化的解决方案，在多轮迭代优化后能够端到端地回答 `数值(number)`、`名称(name/names)`、`是非判断(boolean)` 四类问题，并生成可追溯的 citation 与推理链。

### 核心亮点

| 维度 | 技术要点 |
|------|----------|
| **文档解析** | Docling PDF 解析引擎 + RapidOCR 双路 OCR，自动回退机制 |
| **表格处理** | 结构化表格序列化 → Information Block + 数值 Grounding，支持单元格级引用 |
| **混合召回** | 四路检索（FAISS 向量 / BM25 / BGE-M3 Sparse / Tag 标签）+ RRF / Average 融合 |
| **重排序** | 支持 Single / Cascade 策略：ColBERT 一阶 → LLM/FlagEmbedding/vLLM 二阶 |
| **文档路由** | 基于元数据的 Document Catalog Router，支持多维筛选（行业、板块、标签） |
| **查询改写** | 规则驱动的金融术语同义扩展 + 多查询融合去重 |
| **HyDE** | Hypothetical Document Embedding fallback 机制，低分场景自动触发 |
| **答案校验** | 基于检索元数据的后验 Validation（币种、年份、topic flag 一致性校验） |
| **评测体系** | 内建 Exact Match / Recall@K / Precision@K / Confidence Calibration + Error Analysis |
| **交互式 Demo** | Streamlit 应用：深色金融风格 UI，支持多数据集切换、推理链可视化 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FinaRAG Pipeline                                │
├─────────┬──────────────┬──────────────┬──────────────┬──────────────────────┤
│  Stage  │   PDF 解析   │  切块 & 索引  │   检索 & 排序  │   生成 & 校验       │
├─────────┼──────────────┼──────────────┼──────────────┼──────────────────────┤
│         │              │              │              │                      │
│  输入   │  PDF 文档    │ 解析后 JSON   │   用户问题    │  检索证据 + 问题     │
│         │              │              │              │                      │
│  处理   │ Docling +    │ 结构感知切块   │ ┌──────────┐ │  LLM 答案生成        │
│         │ RapidOCR     │ (Parent/Child)│ │ 查询改写   │ │  (Pydantic Schema)  │
│         │              │              │ │ 金融术语扩展│ │                      │
│         │ 表格序列化    │ Embedding    │ └──────┬───┘ │  Table Grounding      │
│         │ (Info Blocks) │ (BAAI/bge-m3)│        │     │  (数值对齐)           │
│         │              │              │ ┌──────▼───┐ │                      │
│         │ 元数据标注    │ 四路索引构建  │ │ 四路混合召回│ │  Answer Validation   │
│         │ (行业/标签)   │ FAISS/BM25/  │ │ FAISS     │ │  (置信度校验)        │
│         │              │ Sparse/Tag   │ │ BM25      │ │                      │
│         │              │              │ │ BGE-M3    │ │  Citation 构建       │
│         │              │              │ │ Tag       │ │  (页码/表格引用)     │
│         │              │              │ └──────┬───┘ │                      │
│         │              │              │ ┌──────▼───┐ │                      │
│         │              │              │ │ 重排序     │ │                      │
│         │              │              │ │ Cascade/  │ │                      │
│         │              │              │ │ Single    │ │                      │
│         │              │              │ └──────────┘ │                      │
│         │              │              │              │                      │
│  输出   │ 结构化 JSON  │ 向量/词法索引  │ Top-K 证据   │  答案 + 推理链 +     │
│         │ + 表格 JSON  │ + 元数据存储  │ (含分数)     │  引用 + 置信度       │
└─────────┴──────────────┴──────────────┴──────────────┴──────────────────────┘
```

### 多路检索融合策略

```
               ┌─────────────────────────────────────────┐
               │          HybridRetriever                 │
               │                                         │
   Query ──────┤   ┌─────────────┐  ┌─────────────┐     │
               │   │ Vector (FAISS)│ │ BM25 (OKapi) │    │
               │   └──────┬──────┘  └──────┬──────┘     │
               │          │                │             │
               │   ┌──────┴──────┐  ┌──────┴──────┐     │
               │   │ Sparse(BGE) │  │ Tag Index   │     │
               │   └──────┬──────┘  └──────┬──────┘     │
               │          │                │             │
               │          └───── Fusion ───┘             │
               │            (RRF / Average)              │
               │                 │                       │
               │          ┌──────▼──────┐                │
               │          │  Reranker   │                │
               │          │ ┌─────────┐ │                │
               │          │ │ ColBERT  │ │  Cascade 模式  │
               │          │ │    ↓     │ │                │
               │          │ │ LLM/Flag │ │                │
               │          │ └─────────┘ │                │
               │          └─────────────┘                │
               └─────────────────────────────────────────┘
```

---

## 📁 项目结构

```
FinaRAG/
├── main.py                    # CLI 入口 (click)
├── config/                    # YAML 流水线配置
│   ├── qwen_zh_finance.yaml              # 中文金融主配置
│   ├── qwen_zh_finance_colbert_cascade_*.yaml  # ColBERT 级联重排配置
│   └── qwen_zh_finance_hyde_fallback.yaml      # HyDE fallback 配置
│
├── src/
│   ├── pipeline.py            # 流水线编排 (Pipeline + RunConfig)
│   ├── questions_processing.py # 问答核心 (QuestionsProcessor)
│   ├── retrieval.py           # 多路检索 + 混合融合 (HybridRetriever)
│   ├── reranking.py           # 重排序引擎 (CascadeReranker, LLMReranker, FlagEmbedding, vLLM)
│   ├── query_rewrite.py       # 查询改写 + 金融术语扩展 (QuestionRewriter)
│   ├── text_splitter.py       # 结构感知 Parent/Child 切块
│   ├── ingestion.py           # 索引构建 (VectorDB, BM25, Sparse, Tag)
│   ├── embedding_backend.py   # 嵌入模型后端 (多 GPU + 线程安全加载)
│   ├── table_grounding.py     # 数值对齐引擎 (TableGrounder)
│   ├── hyde.py                # HyDE 假设文档生成
│   ├── answer_validation.py   # 后验答案校验
│   ├── report_catalog.py      # 文档目录路由 (ReportCatalog)
│   ├── prompts.py             # Prompt 模板 (按 schema 分化)
│   ├── citation_formatter.py  # Citation 构建与去重
│   ├── api_requests.py        # LLM API 封装
│   ├── document_store.py      # 文档存储层
│   └── text_normalization.py  # 中英文本规范化
│
├── eval/
│   ├── metrics.py             # 评测指标 (Exact Match, Recall@K, Precision@K)
│   └── error_analysis.py      # 错误归因 (routing / parse / retrieval / generation)
│
├── demo_app/
│   └── streamlit_app.py       # 交互式 Demo (深色金融风格 UI)
│
├── tests/                     # pytest 测试套件 (20+ 测试模块)
│   ├── test_answer_validation.py
│   ├── test_cascade_reranking.py
│   ├── test_query_rewrite.py
│   ├── test_table_grounding.py
│   ├── test_parent_child_splitter.py
│   ├── test_hyde.py
│   ├── test_eval_metrics.py
│   └── ...
│
├── requirements.txt
├── setup.py
└── LICENSE (MIT)
```

---

## 🔧 技术栈

| 类别 | 技术 |
|------|------|
| 语言 & 运行时 | Python 3.11+ |
| PDF 解析 | [Docling](https://github.com/DS4SD/docling) + RapidOCR + PyPDF2 fallback |
| 向量检索 | FAISS (Flat / IVF / HNSW) + SentenceTransformers |
| 嵌入模型 | BAAI/bge-m3 (默认)，支持自定义模型 |
| 稀疏检索 | BM25 (rank-bm25) + BGE-M3 Sparse Lexical Weights (FlagEmbedding) |
| 重排序 | FlagEmbedding Reranker / ColBERT / LLM Prompt / vLLM API |
| LLM 推理 | Qwen 系列 (通过 OpenAI 兼容 API)，支持 Gemini 等多 Provider |
| 切块 | LangChain RecursiveCharacterTextSplitter (tiktoken 编码器) |
| 配置管理 | YAML + Pydantic BaseModel |
| CLI | click |
| 交互式 Demo | Streamlit |
| 测试 | pytest (20+ 测试模块) |
| token 计数 | tiktoken (o200k_base) |

---

## ⚡ 快速开始

### 1. 环境准备

```bash
git clone https://github.com/<your-username>/FinaRAG.git
cd FinaRAG

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# LLM 推理服务 (Qwen / 兼容 OpenAI 格式的本地服务)
QWEN_BASE_URL=http://localhost:8000/v1
QWEN_API_KEY=your-api-key
QWEN_MODEL=Qwen3.5-27B

# 嵌入模型
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cuda:0

# 重排序
RERANKING_BACKEND=flag_embedding        # flag_embedding / llm_prompt / vllm_api
RERANKING_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKING_DEVICE=cuda:0
```

### 3. 数据处理流水线

```bash
# Step 1: PDF 解析 → 结构化 JSON
python main.py parse-pdfs \
  --pdf-dir data/my_dataset/pdf_reports \
  --output-dir data/my_dataset/parsed_reports

# Step 2: 表格序列化 (可选但推荐)
python main.py serialize-tables \
  --parsed-dir data/my_dataset/parsed_reports \
  --output-dir data/my_dataset/serialized_tables

# Step 3: 完整流水线处理 (切块 + 索引构建 + 数据准备)
python main.py process-reports \
  --config config/qwen_zh_finance.yaml \
  --dataset-dir data/my_dataset
```

### 4. 问答执行

```bash
# 批量回答预定义问题集
python main.py answer-questions \
  --config config/qwen_zh_finance.yaml \
  --dataset-dir data/my_dataset

# 启动交互式 Demo
streamlit run demo_app/streamlit_app.py
```

---

## 🔬 核心模块详解

### 1. 结构感知 Parent/Child 切块

传统平铺切块容易打断表格和段落上下文。FinaRAG 采用 **Parent/Child 双层切块** 策略：

- **Parent Chunk**：保留完整的结构化块（段落 / 表格），作为上下文锚点
- **Child Chunk**：基于 tiktoken 编码器进一步细分（默认 320 tokens），用于精确检索匹配
- **检索时**：先匹配 Child → 回溯 Parent → 提供完整上下文给 LLM

```python
# src/text_splitter.py - 核心逻辑
class TextSplitter:
    def _split_report(self, file_content, serialized_tables_report_path, ...):
        # 1. 提取结构块 (heading-aware)
        # 2. 构建 Parent chunk (保留完整段落/表格)
        # 3. 拆分 Child chunk (320 tokens, overlap=50)
        # 4. 维护 parent_chunk_id ↔ child_chunk_ids 映射
```

**切块时同步注入的元数据字段：**

| 字段 | 说明 |
|------|------|
| `embedding_text` | 结构化嵌入文本（含公司名/代码/年份/行业/章节） |
| `search_text` | 全元数据拼接文本（供 BM25/Sparse 检索） |
| `evidence_type` | `narrative` / `table` |
| `topic_flags` | `has_leadership_changes` / `has_dividend_policy_changes` 等 |
| `business_tags` / `strategy_tags` | 业务标签 / 战略标签（出海 / 数字化转型 / AI 等） |

### 2. 四路混合检索 + 融合

```python
# src/retrieval.py
class HybridRetriever:
    # 四路检索后端，每路独立返回 top-N 候选
    vector_retriever   # FAISS (cosine similarity)
    bm25_retriever     # BM25Okapi (词频匹配)
    sparse_retriever   # BGE-M3 Sparse Lexical Weights
    tag_retriever      # 基于元数据标签的结构化检索

    # 融合策略
    fusion_method: "rrf" | "average"
    #   RRF: score = Σ 1/(k + rank_i), k=60
    #   Average: score = mean(normalized_scores)
```

**融合去重逻辑：** 同一 chunk 被多路召回时，取各路归一化分数的最高值，同时累加 RRF 分数，合并 `retrieval_sources` 和 `matched_tags`。

### 3. 级联重排序 (Cascade Reranking)

```
Candidate Pool (50+ docs)
         │
    ColBERT Reranker (一阶粗排)
         │  ─── colbert_top_n (e.g., 10)
         │
    Final Reranker (二阶精排)
    ├── LLM Prompt Reranker   (通用, 带详细 reasoning)
    ├── FlagEmbedding Reranker (快速, bge-reranker)
    └── vLLM API Reranker      (高性能, 批量推理)
         │
    Top-K Results → LLM 生成
```

```python
# src/reranking.py
class CascadeReranker:
    def rerank_documents(self, query, documents, ...):
        # Stage 1: ColBERT 粗排 → 缩小候选池
        colbert_ranked = self._colbert_rerank(query, documents)
        top_candidates = colbert_ranked[:self.colbert_top_n]

        # Stage 2: 精排 (LLM/FlagEmbedding/vLLM)
        final_ranked = self.final_reranker.rerank_documents(query, top_candidates)
        return final_ranked
```

### 4. 查询改写与金融术语扩展

```python
# src/query_rewrite.py
class QuestionRewriter:
    # 规则驱动的查询理解 + 扩展
    def rewrite(self, question, schema, company_name) -> QueryPlan:
        # 1. 文本归一化
        # 2. 提取过滤器: 年份, 币种, 交易所, 板块, 行业, 报告类型
        # 3. 金融术语同义扩展:
        #    "营业收入" → ["营收", "收入"]
        #    "归母净利润" → ["母公司股东净利润", "归属于母公司股东的净利润"]
        #    "share buyback" → ["share repurchase", "repurchase program"]
        # 4. 输出 QueryPlan (搜索查询列表 + 结构化过滤器 + 路由策略)
```

`QueryPlan` 驱动后续的检索过滤、文档路由和答案校验，是查询理解的统一数据结构。

### 5. Table Grounding (数值对齐引擎)

在金融问答中，数值类问题对精度要求极高。FinaRAG 实现了 **Table Grounding** 机制：

```python
# src/table_grounding.py
class TableGrounder:
    def ground_number_query(self, question, retrieval_results, filters):
        # 1. 从检索结果定位候选文档的 structured_tables
        # 2. 遍历每个 cell_record，计算综合匹配分:
        #    - 行首匹配 (×2.2 权重) + 列首匹配 (×1.8)
        #    - 页面重叠加分 (+1.5) + 年份匹配 (+1.5) + 币种匹配 (+0.6)
        # 3. 返回最佳匹配单元格的 normalized_value 作为 grounding 锚点
        # 4. 阈值过滤: match_score < 2.2 则放弃 grounding
```

该机制能够直接从原始表格中定位精确数值，避免 LLM 对数字的幻觉或近似。

### 6. 答案后验校验 (Answer Validation)

```python
# src/answer_validation.py
def validate_answer(answer_dict, retrieval_results, query_plan) -> ValidatedAnswer:
    # 多维度校验 → 自动降级置信度
    # ├── 币种一致性 (query vs retrieval metadata)
    # ├── 报告年份一致性
    # ├── 文档类型一致性
    # ├── 期间匹配度
    # ├── Topic flag 覆盖
    # ├── 数值 Grounding 校验 (值存在性 / 期间 / 币种)
    # └── Citation 覆盖检查
    #
    # 不通过 → final_answer 置为 "N/A"，confidence 降为 "low"
```

### 7. HyDE (Hypothetical Document Embedding)

当初始检索结果质量不佳时（top score < 阈值 或 score margin 过小），自动触发 HyDE fallback：

```python
# src/hyde.py
class HyDEGenerator:
    def generate(self, question, schema, query_plan, route_info):
        # 1. 构建 HyDE prompt (包含公司名/年份/报告类型/行业等上下文)
        # 2. LLM 生成假设性证据段落 (不回答问题，只生成检索锚点)
        # 3. 用生成的段落作为新 query 进行二次检索
        # 4. 合并初始结果 + HyDE 结果 → 重新排序
```

---

## 📊 评测体系

FinaRAG 内建完整的评测模块，支持多维度性能分析：

### 评测指标

```python
# eval/metrics.py
compare_answers(pred, ref)          # Exact Match + Page Hit + Citation Precision
compare_ranked_retrieval(pred, ref) # Recall@K + Precision@K + Hit@K

# eval/error_analysis.py
summarize_error_analysis(pred, ref) # 错误归因: routing / parse / retrieval / generation / validation
```

| 指标 | 说明 |
|------|------|
| `reference_exact_match` | 答案精确匹配率 (按 schema 类型归一化) |
| `reference_page_hit` | 引用页码命中率 |
| `citation_page_hit` | Citation 页码命中率 |
| `retrieval_hit_at_k` | 检索结果中包含正确页面的比率 |
| `macro_recall_at_K` | 检索 Top-K 的宏平均召回率 |
| `macro_precision_at_K` | 检索 Top-K 的宏平均精确率 |
| `confidence_calibration` | 按置信度分组的实际准确率 (校准分析) |

### 错误归因

`error_analysis.py` 将每个错误样本自动分类到流水线的具体阶段：

- **routing** — 文档路由错误（公司名未匹配 / 歧义路由）
- **parse** — PDF 解析失败（Docling 异常 / OCR 质量）
- **retrieval** — 检索支撑不足（弱 citation 覆盖）
- **generation** — LLM 生成错误（有检索支撑但答案不匹配）
- **validation** — 后验校验被降级（币种/年份/类型不一致）

---

## ⚙️ 配置系统

所有流水线参数通过 YAML 配置文件管理，支持运行时覆盖：

```yaml
# config/qwen_zh_finance.yaml
use_serialized_tables: true                 # 启用表格序列化
parent_document_retrieval: true             # 启用 Parent/Child 检索
parent_retrieval_mode: block                # block | page

# 检索后端
use_vector_dbs: true
use_bm25_db: true
use_sparse_lexical_db: true
use_tag_db: true

# 向量索引参数
vector_index_type: "flat"                   # flat | ivf | hnsw
vector_ivf_nlist: 32
vector_hnsw_m: 32
vector_hnsw_ef_construction: 200

# 重排序
llm_reranking: true
llm_reranking_sample_size: 8
top_n_retrieval: 5

# 文档路由
doc_router_enabled: true
candidate_doc_top_k: 5

# 数值 Grounding
numeric_grounding_enabled: true

# LLM 配置
api_provider: "qwen"
answering_model: "Qwen3.5-27B"
document_language: "zh"
```

**预置配置：**

| 配置文件 | 场景 |
|---------|------|
| `qwen_zh_finance.yaml` | 中文金融文档全功能配置 |
| `qwen_zh_finance_colbert_cascade_bge.yaml` | ColBERT+BGE 级联重排 |
| `qwen_zh_finance_colbert_cascade_qwen.yaml` | ColBERT+Qwen 级联重排 |
| `qwen_zh_finance_hyde_fallback.yaml` | 启用 HyDE fallback |
| `qwen_base.yaml` | 基础配置 (仅向量检索) |

---

## 🖥️ 交互式 Demo

```bash
streamlit run demo_app/streamlit_app.py
```

Streamlit Demo 提供深色金融风格 UI，核心功能包括：

- **多数据集切换**：自动发现 `data/` 目录下的数据集
- **实时问答**：输入自然语言问题，展示完整推理链
- **检索可视化**：展示四路召回结果、融合分数、重排序过程
- **证据追溯**：页码引用、表格引用、Citation 详情
- **配置热切换**：运行时修改检索后端、重排策略、模型参数
- **系统监控**：嵌入模型状态、检索后端连接状态

---

## 🧪 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_query_rewrite.py -v        # 查询改写
pytest tests/test_cascade_reranking.py -v    # 级联重排
pytest tests/test_table_grounding.py -v      # 数值对齐
pytest tests/test_answer_validation.py -v    # 答案校验
pytest tests/test_parent_child_splitter.py -v # 切块
pytest tests/test_hyde.py -v                 # HyDE
pytest tests/test_eval_metrics.py -v         # 评测指标
```

---

## 📎 CLI 命令参考

```bash
python main.py --help

# 主要命令
python main.py prepare-dataset   # 数据集初始化
python main.py parse-pdfs        # PDF 解析
python main.py serialize-tables  # 表格序列化
python main.py process-reports   # 完整流水线 (切块 + 索引)
python main.py answer-questions  # 批量问答
```

---

## 🔑 工程亮点小结

1. **端到端设计**：从 PDF → 结构化解析 → 多路索引 → 混合召回 → 排序 → 生成 → 校验，每一步均有模块化实现和独立测试
2. **金融场景深度适配**：表格序列化 + 数值 Grounding + 金融术语同义扩展 + 币种/年份/期间过滤
3. **可扩展的检索架构**：四路检索后端按需启用，融合策略和重排策略均通过配置切换
4. **生产级工程实践**：
   - 线程安全的嵌入模型加载（解决 meta-tensor 并发问题）
   - 多 GPU 嵌入分片 + ThreadPoolExecutor 并行推理
   - Pydantic Schema 驱动的 LLM 输出解析 + JSON 修复
   - 答案后验校验 + 置信度校准
5. **完善的评测闭环**：内建 Exact Match / Retrieval Recall / Confidence Calibration / Error Analysis，支持配置间横向对比

---

## 📄 License

[MIT License](LICENSE)
