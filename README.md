# FinaRAG

- 面向金融年报、券商研报和 PDFCrawl 语料的检索增强问答（RAG）工具链，覆盖从 PDF 解析、索引构建到问答、评测和交互式演示的完整流程。
- 项目重点不只是“把 PDF 放进向量库”，而是围绕金融文档中的表格、口径、币种、年份、文档路由和证据溯源做结构化增强。

## 项目概览摘要

| 维度 | 内容 |
| --- | --- |
| 主要入口 | `main.py`（CLI）、`demo_app/streamlit_app.py`（交互式工作台）、`eval/*.py`（评测脚本） |
| 核心流程 | PDF / PDFCrawl 语料 -> Docling 解析 -> 表格序列化 -> Parent/Child 切块 -> 向量/BM25/稀疏索引 -> 路由/检索/重排 -> 结构化答案/引用/校验 |
| 主要场景 | 中文上市公司年报问答、券商研报问答、数字类表格问答、多公司比较问答 |
| 默认回答模型 | `Qwen3.5-35B-A3B-AWQ-4bit` |
| 默认嵌入模型 | `BAAI/bge-m3` |
| 示例数据集 | `data/test_set`、`data/chinese_annual_reports_2024`、`data/erc2_set` |
| 演示方式 | Streamlit 工作台，支持配置选择、数据集切换、问题样例和 PDF 上传 |

## 项目简介

FinaRAG 是一个面向金融文档问答场景的 Python 项目，聚焦年报、研报等复杂 PDF 的解析、索引和问答。它通过结构化解析、表格语义化、混合检索、候选文档路由和答案校验，降低金融问答中常见的页码幻觉、指标口径混淆和数字回答不落地等问题。

项目适用于以下场景：

- 对上市公司年报进行问答、指标核对和出处追溯
- 对券商研报进行要点提取、布尔判断和命名实体问答
- 对数字类问题进行表格 grounding，减少“模型猜数”
- 在多份公司文档中做候选文档路由和比较问答

高层架构如下：

```text
PDF / PDFCrawl Dataset
  -> Docling Parsing + OCR
  -> Parsed Report Merge
  -> Structure-Aware Parent / Child Chunking
  -> Serialized Table Blocks + Structured Tables
  -> FAISS / BM25 / BGEM3 Sparse Indexes
  -> Query Rewrite + Metadata Filters + Document Catalog Routing
  -> Optional Rerank
  -> Structured Answer + Citations + Confidence + Validation
  -> Evaluation / Error Analysis / Streamlit Demo
```

## 功能特性

- 支持从 `PDFCrawl` 输出一键整理成 FinaRAG 可直接消费的数据集目录，自动生成 `document_manifest.csv`、`pdf_reports/` 和 `questions.json`
- 使用 `Docling` 解析 PDF，支持 `docling_rapidocr` 和 `docling_easyocr`，并对 Docling v2 的已知可恢复错误回退到旧版 backend
- 对解析结果进行结构感知切块，生成 `parent_chunks` 与 `chunks`，支持真正的 Parent/Child 检索，而不只是整页扩展
- 可选表格序列化与结构化表格抽取，数字问题可通过 `TableGrounder` 将指标定位到具体表格单元格
- 支持多路检索：FAISS 向量检索、BM25、BGE-M3 sparse lexical 检索，并支持 `rrf` / `average` 融合
- 支持查询改写、年份/币种/报告类型/证券代码过滤、候选文档路由和多公司比较问答
- 输出结构化答案，包含 `references`、`citations`、`confidence`、`validation_flags`、`route_info` 和调试信息
- 内置评测脚本，可对答案文件进行统计、参考答案对比和错误归因分析；提供 Streamlit 工作台用于交互式演示

## 技术栈

| 分类 | 技术 / 依赖 |
| --- | --- |
| 语言与基础设施 | Python，`click==8.1.7`，`python-dotenv==1.0.1`，`pydantic==2.9.2` |
| PDF 解析与 OCR | `docling[rapidocr]==2.14.0`，RapidOCR / EasyOCR 模式切换 |
| 文本切分与预处理 | `langchain==0.3.3`，`tiktoken==0.8.0`，`jieba==0.42.1` |
| 向量检索 | `faiss-cpu==1.9.0.post1`，`sentence-transformers==3.3.1` |
| 词法检索 / 稀疏检索 | `rank-bm25==0.2.2`，`FlagEmbedding==1.3.5` |
| LLM 与推理接口 | Qwen 兼容 OpenAI-style API、Gemini SDK、IBM 接口适配 |
| 数据处理 | `pandas==2.2.3`，`numpy==1.26.4`，`PyYAML==6.0.2`，`json_repair==0.35.0` |
| Web / Demo | `streamlit==1.44.1` |
| 测试与评测 | `pytest==8.3.5`，`unittest`，仓库内 `eval/` 脚本 |
| 打包方式 | `setup.py` 可编辑安装 |
| 部署形态 | 本地 CLI + 本地 Streamlit 进程；待补充：仓库未提供 Docker / K8s / CI 工作流 |

## 项目结构

```text
FinaRAG/
├── main.py                           # CLI 入口：数据集整理、PDF 解析、表格序列化、索引构建、批量问答
├── config/                           # YAML 运行配置，定义检索栈、重排、OCR、路由和问答行为
│   ├── qwen_base.yaml
│   ├── qwen_rerank.yaml
│   ├── qwen_ser_rerank.yaml
│   └── qwen_zh_finance.yaml
├── data/                             # 示例数据集、中文 benchmark 模板和预构建索引产物
│   ├── chinese_annual_reports_2024/  # 中文年报数据集，含 30 份文档 manifest、15 道标注问题和调试产物
│   ├── chinese_benchmark/            # 中文 benchmark scaffold、manifest 模板、问题模板、gold 答案模板
│   ├── erc2_set/                     # ERC2 数据集说明和样例元数据
│   └── test_set/                     # 轻量英文测试数据集，含 5 道样例问题
├── demo_app/
│   └── streamlit_app.py              # Streamlit 工作台，支持配置切换、数据集选择、问题样例、PDF 上传
├── eval/
│   ├── run_eval.py                   # 评测单个 answers 文件或运行 pipeline 后评测
│   ├── compare_configs.py            # 对多个配置批量跑分并输出对比结果
│   ├── metrics.py                    # answer rate、citation coverage、reference exact match 等指标
│   └── error_analysis.py             # routing / retrieval / generation / validation 失败归因
├── src/
│   ├── pipeline.py                   # 顶层流程编排、路径约定、运行配置定义
│   ├── pdf_parsing.py                # Docling 解析、OCR 配置、并行 PDF 处理
│   ├── tables_serialization.py       # 使用 LLM 将表格转成上下文独立的信息块
│   ├── parsed_reports_merging.py     # 将解析结果整理为后续切块可消费的页面结构
│   ├── text_splitter.py              # 结构感知 Parent/Child 切块、结构化表格记录抽取
│   ├── ingestion.py                  # FAISS / BM25 / sparse lexical 索引构建
│   ├── retrieval.py                  # 向量检索、BM25、混合检索、融合和重排
│   ├── retrieval_filters.py          # 元数据过滤和问题类型 bonus
│   ├── report_catalog.py             # 候选文档目录、公司抽取和文档路由
│   ├── query_rewrite.py              # 金融术语扩展、年份/币种/报告类型/证券代码抽取
│   ├── questions_processing.py       # 单题/批量问答、比较问答、调试产物写出
│   ├── table_grounding.py            # 数值问题的表格定位与数值归一
│   ├── citation_formatter.py         # citations / confidence 生成
│   ├── answer_validation.py          # 币种、年份、期间、table grounding 等后校验
│   ├── document_manifest.py          # 统一读取 CSV / JSON manifest
│   └── pdfcrawl_dataset.py           # PDFCrawl -> FinaRAG 数据集布局适配
├── tests/                            # Parent/Child 检索、文档路由、query rewrite、表格 grounding 等测试
├── requirements.txt                  # Python 依赖清单
├── setup.py                          # 包安装入口
├── LICENSE                           # MIT License
└── README.md
```

说明：

- 当前仓库没有独立的 `api/` 目录，外部使用入口主要是 `main.py`、`eval/` 和 `demo_app/`
- 当前仓库没有独立的 `scripts/` 目录，批处理命令由 `click` CLI 和 `eval/*.py` 提供
- 当前仓库没有独立的 `docs/` 目录，项目说明主要集中在 `README.md`、数据集子目录说明和 `Advanced_plan.md`

## 快速开始

### 环境要求

- Python 3.11+，建议 3.12
- Linux / macOS 环境优先
- 如需启用本地 GPU 嵌入或本地 reranker，请准备可用 CUDA 环境
- 如需运行表格序列化、问答和重排，请准备可访问的 LLM 接口

### 安装步骤

```bash
git clone https://github.com/AdamsLiuG/FinaRAG.git
cd FinaRAG

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 环境变量配置

仓库根目录会读取 `.env`。如果需要从零配置，可以新建 `.env` 并至少提供下列变量：

```env
QWEN_BASE_URL=https://your-qwen-compatible-endpoint
QWEN_API_KEY=your-api-key
QWEN_MODEL=Qwen3.5-35B-A3B-AWQ-4bit

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu

RERANKING_BACKEND=flag_embedding
RERANKING_MODEL=BAAI/bge-reranker-v2-m3
RERANKING_DEVICE=cpu
```

### 初始化 Docling 模型

首次运行前建议先下载 Docling 所需模型：

```bash
.venv/bin/python main.py download-models
```

### 准备数据集

如果你已经有 `PDFCrawl` 输出，可直接生成 FinaRAG 数据目录：

```bash
.venv/bin/python main.py prepare-pdfcrawl-dataset \
  --pdfcrawl-root /path/to/PDFCrawl/output \
  --dataset-dir data/my_dataset \
  --link-mode symlink \
  --currency CNY \
  --language zh
```

生成后的数据集目录至少包含：

```text
data/my_dataset/
├── document_manifest.csv
├── questions.json
└── pdf_reports/
```

### 本地启动与索引构建

注意：`parse-pdfs`、`serialize-tables`、`process-reports`、`process-questions` 默认把“当前工作目录”当作数据集根目录。

以 `data/chinese_annual_reports_2024` 为例：

```bash
cd data/chinese_annual_reports_2024

../../.venv/bin/python ../../main.py parse-pdfs \
  --parallel \
  --chunk-size 2 \
  --max-workers 4 \
  --config-path ../../config/qwen_zh_finance.yaml

../../.venv/bin/python ../../main.py serialize-tables \
  --max-workers 4 \
  --config-path ../../config/qwen_zh_finance.yaml

../../.venv/bin/python ../../main.py process-reports \
  --config-path ../../config/qwen_zh_finance.yaml
```

### 批量问答

```bash
cd data/chinese_annual_reports_2024

../../.venv/bin/python ../../main.py process-questions \
  --config-path ../../config/qwen_zh_finance.yaml
```

输出结果会写到当前数据集目录，文件名由配置决定，例如：

- `answers_qwen_zh_finance.json`
- `answers_qwen_zh_finance_debug.json`

如果目标文件已存在，系统会自动追加 `_01`、`_02` 等后缀。

### 构建命令

本项目没有传统前端/后端编译步骤；“构建”主要指索引和问答资产构建：

```bash
cd data/chinese_annual_reports_2024

../../.venv/bin/python ../../main.py process-reports \
  --config-path ../../config/qwen_zh_finance.yaml
```

### 测试命令

```bash
python -m pytest -q
```

### 启动 Streamlit Demo

```bash
cd /path/to/FinaRAG
.venv/bin/streamlit run demo_app/streamlit_app.py --server.port 8501
```

默认访问地址：

- `http://127.0.0.1:8501`

## 配置说明

### 1. YAML 配置文件

仓库当前提供以下运行配置：

| 文件 | 主要特征 |
| --- | --- |
| `config/qwen_base.yaml` | 向量检索 + Parent/Child 检索 + Query Rewrite |
| `config/qwen_rerank.yaml` | 向量 + BM25 混合召回 + LLM rerank |
| `config/qwen_ser_rerank.yaml` | 表格序列化 + 向量/BM25/sparse 混合召回 + LLM rerank |
| `config/qwen_zh_finance.yaml` | 中文金融配置，启用中文 OCR、文档路由、numeric grounding |

核心配置项来自 `RunConfig`：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `use_serialized_tables` | `false` | 是否启用表格序列化和 `serialized_table` chunk |
| `parent_document_retrieval` | `false` | 是否启用父级文档回溯 |
| `parent_retrieval_mode` | `page` | 父级回溯模式，支持 `page` / `block` |
| `use_vector_dbs` | `true` | 是否启用 FAISS 向量检索 |
| `use_bm25_db` | `false` | 是否启用 BM25 |
| `use_sparse_lexical_db` | `false` | 是否启用 BGE-M3 sparse lexical 检索 |
| `llm_reranking` | `false` | 是否启用重排 |
| `llm_reranking_sample_size` | `30` | 进入重排的候选数 |
| `top_n_retrieval` | `10` | 生成答案前保留的最终候选数 |
| `parallel_requests` | `10` | 批量问题处理时的并行线程数 |
| `full_context` | `false` | 是否跳过检索，直接给整份文档上下文 |
| `api_provider` | `qwen` | 回答模型提供方 |
| `answering_model` | `Qwen3.5-35B-A3B-AWQ-4bit` | 问答模型名 |
| `document_language` | `en` | 文档语言，影响 OCR 和切块分隔符 |
| `ocr_mode` | `docling_rapidocr` | OCR 模式，支持 `docling_rapidocr` / `docling_easyocr` |
| `doc_router_enabled` | `false` | 是否启用文档目录路由 |
| `candidate_doc_top_k` | `5` | 候选文档数量上限 |
| `numeric_grounding_enabled` | `false` | 是否启用表格 grounding |
| `reasoning_debug_enabled` | `true` | 是否保留逐步分析和调试信息 |

### 2. 环境变量

#### 2.1 LLM 与问答运行时

| 变量名 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `QWEN_BASE_URL` | 条件必填 | 无 | Qwen 兼容接口地址；未设置时回退到 `LLM_BASE_URL` |
| `QWEN_API_KEY` | 条件必填 | 无 | Qwen 接口鉴权；未设置时回退到 `LLM_API_KEY` |
| `QWEN_MODEL` | 否 | `Qwen3.5-35B-A3B-AWQ-4bit` | Qwen 提供方默认回答模型 |
| `QWEN_MAX_TOKENS` | 否 | 无 | Qwen 请求 `max_tokens` |
| `QWEN_STREAM` | 否 | `true` | 是否启用流式返回 |
| `QWEN_ENABLE_THINKING` | 否 | `false` | 是否开启 thinking 模式 |
| `LLM_BASE_URL` | 否 | 无 | 通用兼容接口地址，作为 provider 级变量的后备值 |
| `LLM_API_KEY` | 否 | 无 | 通用兼容接口密钥 |
| `LLM_MODEL` | 否 | 无 | 通用回答模型后备值 |
| `LLM_MAX_TOKENS` | 否 | 无 | 通用 `max_tokens` |
| `LLM_STREAM` | 否 | 无 | 通用流式开关 |
| `LLM_ENABLE_THINKING` | 否 | 无 | 通用 thinking 开关 |
| `LLM_PROVIDER` | 否 | 无 | 表格序列化 provider 的后备值 |
| `TABLE_SERIALIZER_PROVIDER` | 否 | `qwen` | 表格序列化使用的 provider |
| `TABLE_SERIALIZER_MODEL` | 否 | `Qwen3.5-35B-A3B-AWQ-4bit` | 表格序列化使用的模型 |
| `RAG_MAX_CONTEXT_CHARS` | 否 | `8000` | 送入回答模型的总上下文字符数上限 |
| `RAG_MAX_DOC_CHARS` | 否 | `2500` | 单个召回片段字符数上限 |
| `IBM_API_KEY` | 条件必填 | 无 | 当 `api_provider=ibm` 时使用 |
| `GEMINI_API_KEY` | 条件必填 | 无 | 当 `api_provider=gemini` 时使用 |
| `JINA_API_KEY` | 条件必填 | 无 | 当使用 Jina reranker 时使用 |

#### 2.2 Embedding、检索与重排

| 变量名 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | 否 | `BAAI/bge-m3` | 稠密向量模型 |
| `EMBEDDING_DEVICE` | 否 | `cpu` | 稠密向量设备，支持 `cpu`、`cuda:0`、`cuda:0,cuda:1` |
| `EMBEDDING_BATCH_SIZE` | 否 | `32` | 稠密向量 batch size |
| `EMBEDDING_TRUST_REMOTE_CODE` | 否 | `false` | 是否信任远端模型代码 |
| `EMBEDDING_FALLBACK_MODEL_NAME` | 否 | `BAAI/bge-small-en-v1.5` | 发生 meta-tensor 错误时的回退模型 |
| `EMBEDDING_SPARSE_MODEL_NAME` | 否 | 继承 `EMBEDDING_MODEL_NAME` | BGE-M3 sparse lexical 模型 |
| `EMBEDDING_SPARSE_DEVICE` | 否 | 继承 `EMBEDDING_DEVICE` | 稀疏检索设备 |
| `EMBEDDING_SPARSE_BATCH_SIZE` | 否 | `32` | 稀疏编码 batch size |
| `EMBEDDING_SPARSE_QUERY_MAX_LENGTH` | 否 | `256` | 稀疏 query 最大长度 |
| `EMBEDDING_SPARSE_PASSAGE_MAX_LENGTH` | 否 | `512` | 稀疏 passage 最大长度 |
| `EMBEDDING_SPARSE_USE_FP16` | 否 | CUDA 下自动启用 | 稀疏模型是否启用 FP16 |
| `RERANKING_BACKEND` | 否 | `llm_prompt` | 重排后端，支持 `llm_prompt` 或 `flag_embedding` |
| `RERANKING_MODEL` | 否 | 依后端而定 | `flag_embedding` 时默认 `BAAI/bge-reranker-v2-m3`；`llm_prompt` 时回退到 `LLM_MODEL` 或 Qwen 默认模型 |
| `RERANKING_DEVICE` | 否 | `cuda:0` | 本地 reranker 设备 |
| `RERANKING_USE_FP16` | 否 | `true` | 本地 reranker 是否启用 FP16 |
| `RERANKING_TRUST_REMOTE_CODE` | 否 | `false` | 本地 reranker 是否信任远端模型代码 |
| `HYBRID_RETRIEVAL_FUSION` | 否 | `rrf` | 混合检索融合策略，支持 `rrf` / `average` |
| `HYBRID_RETRIEVAL_RRF_K` | 否 | `60` | RRF 融合参数 |
| `VECTOR_INDEX_TYPE` | 否 | `flat` | FAISS 索引类型，支持 `flat` / `ivf` / `hnsw` |
| `VECTOR_SEARCH_K` | 否 | `0` | 向量检索候选数；`0` 表示自动推导 |
| `VECTOR_IVF_NLIST` | 否 | `32` | IVF 索引聚类中心数 |
| `VECTOR_IVF_NPROBE` | 否 | `8` | IVF 查询探测数 |
| `VECTOR_HNSW_M` | 否 | `32` | HNSW 图连接数 |
| `VECTOR_HNSW_EF_CONSTRUCTION` | 否 | `200` | HNSW 构建参数 |
| `VECTOR_HNSW_EF_SEARCH` | 否 | `64` | HNSW 查询参数 |

#### 2.3 内置 Qwen 预设覆盖项

当使用 `process-questions --config qwen_base` 这类内置预设，而不是 `--config-path` 指定 YAML 时，以下环境变量会覆盖预设行为：

- `QWEN_PARALLEL_REQUESTS`
- `QWEN_PARENT_DOCUMENT_RETRIEVAL`
- `QWEN_PARENT_RETRIEVAL_MODE`
- `QWEN_TOP_N_RETRIEVAL`
- `QWEN_LLM_RERANKING_SAMPLE_SIZE`
- `QWEN_DOCUMENT_LANGUAGE`
- `QWEN_OCR_MODE`
- `QWEN_DOC_ROUTER_ENABLED`
- `QWEN_CANDIDATE_DOC_TOP_K`
- `QWEN_NUMERIC_GROUNDING_ENABLED`
- `QWEN_REASONING_DEBUG_ENABLED`
- `QWEN_VECTOR_INDEX_TYPE`
- `QWEN_VECTOR_SEARCH_K`
- `QWEN_VECTOR_IVF_NLIST`
- `QWEN_VECTOR_IVF_NPROBE`
- `QWEN_VECTOR_HNSW_M`
- `QWEN_VECTOR_HNSW_EF_CONSTRUCTION`
- `QWEN_VECTOR_HNSW_EF_SEARCH`
- `QWEN_RETRIEVER_CACHE_ENABLED`

## 使用说明

### 1. 典型使用流程

```text
准备数据集
  -> parse-pdfs
  -> serialize-tables（如配置启用）
  -> process-reports
  -> process-questions
  -> eval/run_eval.py 或 Streamlit Demo
```

### 2. 使用内置示例数据集

#### 中文年报样例

```bash
cd data/chinese_annual_reports_2024

../../.venv/bin/python ../../main.py process-questions \
  --config-path ../../config/qwen_zh_finance.yaml

../../.venv/bin/python ../../eval/run_eval.py \
  --dataset-dir . \
  --answers-file answers_qwen_zh_finance.json
```

当前数据集中：

- `document_manifest.csv` 包含 `30` 份中文年报
- `questions.json` 包含 `15` 道标注问题
- 覆盖行业包括 `consumer`、`new_energy`、`semiconductor`

#### 英文轻量测试集

```bash
cd data/test_set

../../.venv/bin/python ../../main.py process-questions \
  --config-path ../../config/qwen_base.yaml
```

注意：`data/test_set/README.md` 已明确说明预构建的 `databases.zip` 与当前本地 embedding 管线不兼容；如果需要严格复现当前代码路径，建议从 `pdf_reports/` 重新构建索引。

### 3. 评测答案文件

单个答案文件评测：

```bash
.venv/bin/python eval/run_eval.py \
  --dataset-dir data/chinese_annual_reports_2024 \
  --answers-file data/chinese_annual_reports_2024/answers_qwen_zh_finance.json
```

运行 pipeline 后立即评测：

```bash
.venv/bin/python eval/run_eval.py \
  --dataset-dir data/test_set \
  --config qwen_base \
  --run-pipeline
```

对比多组配置：

```bash
.venv/bin/python eval/compare_configs.py \
  --dataset-dir data/test_set \
  --configs qwen_base,qwen_rerank,qwen_ser_rerank \
  --markdown-output results/config_compare.md
```

### 4. Streamlit 工作台

```bash
.venv/bin/streamlit run demo_app/streamlit_app.py --server.port 8501
```

Demo 提供的功能包括：

- 配置方案选择
- 数据集切换
- Top-K 和温度调节
- 样例问题载入
- PDF 上传到 `data/upload_workspace/pdf_reports/`
- 候选文档路由、引用、检索结果和系统状态展示

### 5. 数据集文件约定

#### `document_manifest.csv`

`prepare-pdfcrawl-dataset` 生成的 manifest 列包括：

| 字段 | 说明 |
| --- | --- |
| `doc_id` | 文档唯一 ID，通常与 PDF 文件名同名 |
| `company_name` | 公司名称 |
| `company_aliases` | 公司别名，支持 `|` 等分隔 |
| `security_code` | 证券代码 |
| `doc_source_type` | 文档来源类型，如 `annual_report`、`research_report` |
| `report_title` | 文档标题 |
| `report_date` | 报告发布日期 |
| `fiscal_year` | 财报年度 |
| `broker_name` | 券商名称 |
| `major_industry` | 行业标签 |
| `language` | 文档语言，如 `zh`、`en`、`bilingual` |
| `currency` | 币种，如 `CNY`、`USD` |
| `source_manifest` | 来源 manifest 路径 |
| `source_file_path` | 原始 PDF 路径 |
| `pdf_url` | PDF 来源 URL |

#### `questions.json`

最小可运行格式只需要：

```json
[
  {
    "text": "贵州茅台2024年年报中的法定代表人是谁？",
    "kind": "name"
  }
]
```

如果需要评测或 gold 标注，可扩展为：

```json
[
  {
    "id": "zh-ar-001",
    "text": "贵州茅台2024年年报中的营业收入是多少元？",
    "kind": "number",
    "doc_ids": ["600519_2024_20250403"],
    "gold_value": 170899152276.34,
    "gold_pages": [5],
    "evidence_type": "table",
    "should_refuse": false
  }
]
```

### 6. 程序化使用示例

使用 `Pipeline` 处理整套数据：

```python
from pathlib import Path
from src.pipeline import Pipeline, load_run_config

dataset_dir = Path("data/chinese_annual_reports_2024")
run_config = load_run_config(Path("config/qwen_zh_finance.yaml"))

pipeline = Pipeline(dataset_dir, run_config=run_config)
answers_file = pipeline.process_questions()
print(answers_file)
```

使用 `QuestionsProcessor` 处理单题：

```python
from pathlib import Path
from src.pipeline import load_run_config
from src.questions_processing import QuestionsProcessor

dataset_dir = Path("data/chinese_annual_reports_2024")
cfg = load_run_config(Path("config/qwen_zh_finance.yaml"))

processor = QuestionsProcessor(
    vector_db_dir=dataset_dir / "databases_ser_tab" / "vector_dbs",
    bm25_db_path=dataset_dir / "databases_ser_tab" / "bm25_dbs",
    sparse_db_dir=dataset_dir / "databases_ser_tab" / "sparse_dbs",
    documents_dir=dataset_dir / "databases_ser_tab" / "chunked_reports",
    subset_path=dataset_dir / "document_manifest.csv",
    parent_document_retrieval=cfg.parent_document_retrieval,
    parent_retrieval_mode=cfg.parent_retrieval_mode,
    use_vector_dbs=cfg.use_vector_dbs,
    use_bm25_db=cfg.use_bm25_db,
    use_sparse_lexical_db=cfg.use_sparse_lexical_db,
    llm_reranking=cfg.llm_reranking,
    llm_reranking_sample_size=cfg.llm_reranking_sample_size,
    top_n_retrieval=cfg.top_n_retrieval,
    api_provider=cfg.api_provider,
    answering_model=cfg.answering_model,
    document_language=cfg.document_language,
    doc_router_enabled=cfg.doc_router_enabled,
    candidate_doc_top_k=cfg.candidate_doc_top_k,
    numeric_grounding_enabled=cfg.numeric_grounding_enabled,
)

result = processor.process_question("贵州茅台2024年年报中的法定代表人是谁？", "name")
print(result["final_answer"])
print(result["references"])
```

## API 说明

当前仓库未提供 REST/OpenAPI/Swagger 接口，主要入口是 CLI 和 Python 类。程序化入口如下：

| 入口 | 类型 | 说明 |
| --- | --- | --- |
| `main.py` | CLI | 数据集构建、解析、索引、批量问答 |
| `src.pipeline.Pipeline` | Python API | 串联整套处理流程 |
| `src.questions_processing.QuestionsProcessor` | Python API | 单题 / 批量问题处理 |
| `eval/run_eval.py` | CLI | 评测答案文件 |
| `demo_app/streamlit_app.py` | Web UI | 交互式问答与调试工作台 |

常用 CLI 命令如下：

| 命令 | 主要参数 | 说明 |
| --- | --- | --- |
| `download-models` | 无 | 预下载 Docling 模型 |
| `prepare-pdfcrawl-dataset` | `--pdfcrawl-root`、`--dataset-dir`、`--link-mode` | 将 PDFCrawl 产物整理成 FinaRAG 数据目录 |
| `parse-pdfs` | `--parallel`、`--chunk-size`、`--max-workers`、`--cuda-devices`、`--config-path` | 解析 PDF 到 `debug_data/01_parsed_reports*` |
| `serialize-tables` | `--max-workers`、`--config-path` | 为解析结果补充 `serialized` 表格块 |
| `process-reports` | `--config` 或 `--config-path` | 合并、导出 Markdown、切块并构建索引 |
| `process-questions` | `--config` 或 `--config-path` | 读取 `questions.json` 并生成答案文件 |

输出答案文件中常见字段包括：

- `question_text`
- `kind`
- `value`
- `references`
- `citations`
- `confidence`
- `confidence_reason`
- `validation_flags`
- `route_info`
- 调试包中的 `answer_details`、`query_plan`、`retrieval_results`、`table_grounding_result`

## 开发指南

### 本地开发方式

```bash
git clone https://github.com/AdamsLiuG/FinaRAG.git
cd FinaRAG

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 常用开发命令

```bash
# 查看 CLI 帮助
.venv/bin/python main.py --help

# 运行测试
python -m pytest -q

# 运行单个评测
.venv/bin/python eval/run_eval.py --help

# 启动 Demo
.venv/bin/streamlit run demo_app/streamlit_app.py
```

### 测试方式

当前测试主要覆盖：

- Parent/Child 切块与检索输出
- Metadata routing 和文档候选路由
- Query rewrite 的年份/币种/证券代码抽取
- Numeric table grounding
- 向量索引参数、检索配置和多 GPU embedding 分片
- Docling fallback 和 CUDA device 分配
- PDFCrawl 数据集适配

### 代码规范

- 待补充：仓库中未提供 `ruff`、`flake8`、`black`、`isort`、`pre-commit` 等统一规范配置
- 当前代码风格以标准 Python 模块化结构和 `pytest`/`unittest` 测试为主

### Commit / Branch 规范

- 待补充：仓库中未提供明确的 commit message 规范和分支策略

### 模块开发建议

- 新增检索后端时，同时修改 `src/ingestion.py`、`src/retrieval.py` 和相关测试
- 新增实验配置时，优先在 `config/` 下添加 YAML，而不是继续扩展硬编码预设
- 如果修改问答输出结构，请同步检查 `eval/metrics.py`、`eval/error_analysis.py` 和 Streamlit 展示逻辑
- 如果修改数据集目录格式，请同步维护 `src/pdfcrawl_dataset.py`、`src/document_manifest.py` 和 Demo 的数据发现逻辑

## 部署说明

### 本地进程部署

本项目当前的“部署”方式主要是本地 Python 进程：

```bash
source .venv/bin/activate
streamlit run demo_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

适用场景：

- 本地开发演示
- 局域网内临时共享 Streamlit 工作台
- 离线批处理问答和评测

### 容器化 / CI / 云部署

- 待补充：仓库未提供 `Dockerfile`
- 待补充：仓库未提供 `docker-compose.yml`
- 待补充：仓库未提供 `nginx`、`k8s`、`helm`、`systemd` 配置
- 待补充：仓库未提供 GitHub Actions / GitLab CI 等 CI/CD 配置

## 常见问题

### 1. 为什么 `parse-pdfs` / `process-reports` 找不到数据文件？

这些命令默认把“当前工作目录”当作数据集根目录。请先 `cd` 到包含 `pdf_reports/`、`document_manifest.csv`、`questions.json` 的目录，再执行 CLI。

### 2. 为什么 Demo 提示检索资产未就绪？

当前数据集和当前配置对应的 `chunked_reports/`、`vector_dbs/`、`bm25_dbs/` 或 `sparse_dbs/` 还没有构建完成。请先运行：

```text
parse-pdfs -> serialize-tables（如启用） -> process-reports
```

### 3. 为什么数字题被返回成 `N/A`？

数字问题会经过币种、年份、期间和 table grounding 校验。如果命中文档与问题过滤条件不一致，或者无法定位到表格单元格，`answer_validation.py` 会主动降级甚至拒答。

### 4. 为什么 `block` 模式报错要求重新运行 `process-reports`？

`block` 模式依赖 `content.parent_chunks` 和子 chunk 的 `parent_chunk_id`。如果你的索引产物是旧格式，必须重新执行 `process-reports` 以生成 Parent/Child 结构。

### 5. 为什么预构建数据库不能直接复用？

`data/test_set/README.md` 和 `data/erc2_set/README.md` 都明确说明：仓库中某些预构建 `databases` 产物与当前本地 embedding 管线不兼容。需要基于当前代码和当前 embedding 配置重新构建。

### 6. 如何把 PDFCrawl 的结果接入当前项目？

使用：

```bash
.venv/bin/python main.py prepare-pdfcrawl-dataset \
  --pdfcrawl-root /path/to/PDFCrawl/output \
  --dataset-dir data/my_dataset
```

该命令会自动生成 `document_manifest.csv`、同步 `pdf_reports/` 并可写出空的 `questions.json`。

## Roadmap

- 待补充：仓库未提供正式的公开 Roadmap 文档
- `data/chinese_benchmark/` 已提供中文金融 benchmark 模板和 gold 答案模板
- `eval/compare_configs.py` 已提供多配置对比评测入口
- `demo_app/streamlit_app.py` 已提供交互式工作台和上传 PDF 工作区

## 贡献指南

欢迎通过 Issue 和 Pull Request 参与改进。

建议贡献流程：

1. 先通过 Issue 描述问题、场景和复现方式
2. Fork 仓库并创建独立分支
3. 安装依赖并补充或更新相关测试
4. 如果修改检索、重排、路由或问答逻辑，建议附上至少一组评测结果或样例输出
5. 提交 PR 时请说明影响范围，包括修改模块、输出格式是否变化、是否需要重建索引，以及是否影响 Demo 或评测脚本

贡献建议：

- 优先保持数据目录结构、输出 JSON 结构和评测脚本兼容
- 涉及配置项新增时，优先补充 YAML 配置和 README 说明
- 涉及路由、检索、table grounding 的变更，建议增加对应单元测试

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
