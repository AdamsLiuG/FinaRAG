# FinaRAG

面向金融研报与年报场景的 RAG 智能问答系统。项目重点不是“把 PDF 塞进向量库”，而是围绕金融文档的真实难点做增量优化：复杂 PDF 解析、表格语义化、结构感知切块、混合检索、可选 rerank、引用溯源、置信度输出，以及多公司对比问答路由。

## 项目背景 / 业务痛点

投研、IR、财务分析等场景下，金融文档问答和普通 PDF QA 有明显差异：

- 研报与年报是多栏排版、表格密集、脚注多、标题层级复杂的长文档
- 关键答案经常依赖表头、单位、币种、脚注和上下文，简单 chunk 容易切坏
- 指标口径严格，`Operating margin`、`Gross margin`、`Total assets` 这类问题不能靠模糊匹配猜
- 多公司对比问题需要先拆解成单公司问答，再回到统一比较逻辑
- 金融问答对“出处”和“低置信度拒答”要求更高，错误回答往往比 `N/A` 更危险

FinaRAG 的目标是把这些问题收敛到一个可复现、可扩展、可继续做实验的金融 RAG 项目里。

## 系统架构

```text
PDF Reports
  -> Docling Parsing
  -> Page / Table Structuring
  -> Report Merging
  -> Structure-Aware Chunking
  -> Vector / BM25 / Sparse Indexing
  -> Retrieval + Query Rewrite + Metadata Filter
  -> Optional Rerank
  -> Structured Answer Generation
  -> References + Citations + Confidence
```

## 核心能力

- 复杂 PDF 解析：使用 Docling 处理多栏、表格、图片和 OCR 场景
- 表格序列化：把表格转换成更适合检索的自然语言信息块
- 结构感知切块：保留标题、页面、section、table 等 metadata
- 多路召回：支持向量检索、BM25、bge-m3 sparse lexical 混合召回
- Query Rewrite：针对财务指标问答做术语扩展、币种与年份抽取
- Metadata Filter：支持 company / currency / year / question kind 等过滤约束
- 多公司对比问答：自动拆分 comparative question，再汇总比较答案
- 引用溯源：返回页码 references、chunk 级 citations 和 confidence

## RAG Pipeline 说明

### 1. PDF 解析

- 入口：`parse-pdfs`
- 模块：[src/pdf_parsing.py](/media/main/lgd/llm/FinaRAG/src/pdf_parsing.py)
- 作用：
  - 解析 PDF 页面结构
  - 提取文本、表格、图片
  - 把 `subset.csv` 中的 `company_name`、`currency`、`major_industry` 注入文档元信息

### 2. 表格处理

- 入口：`serialize-tables`
- 模块：[src/tables_serialization.py](/media/main/lgd/llm/FinaRAG/src/tables_serialization.py)
- 作用：
  - 将 HTML 表格序列化为 context-independent 信息块
  - 保留表格上下文、单位、表头、footnote 等信息

### 3. 合并与结构感知切块

- 模块：
  - [src/parsed_reports_merging.py](/media/main/lgd/llm/FinaRAG/src/parsed_reports_merging.py)
  - [src/text_splitter.py](/media/main/lgd/llm/FinaRAG/src/text_splitter.py)
- 设计：
  - 先把 Docling 输出整理成页面级文本
  - 再按标题和页面结构切成 block
  - 对长 block 再做 token-aware 拆分
- chunk metadata 至少包含：
  - `chunk_id`
  - `chunk_type`
  - `page`
  - `section_title`
  - `table_id`
  - `currency`
  - `report_year`

### 4. 检索与重排

- 模块：
  - [src/ingestion.py](/media/main/lgd/llm/FinaRAG/src/ingestion.py)
  - [src/retrieval.py](/media/main/lgd/llm/FinaRAG/src/retrieval.py)
  - [src/reranking.py](/media/main/lgd/llm/FinaRAG/src/reranking.py)
- 召回方式：
  - Vector retrieval
  - BM25 retrieval
  - bge-m3 sparse lexical retrieval
  - RRF / average fusion
- 增强：
  - Query rewrite
  - Metadata filter
  - Optional rerank

### 5. 生成与后处理

- 模块：
  - [src/questions_processing.py](/media/main/lgd/llm/FinaRAG/src/questions_processing.py)
  - [src/prompts.py](/media/main/lgd/llm/FinaRAG/src/prompts.py)
  - [src/citation_formatter.py](/media/main/lgd/llm/FinaRAG/src/citation_formatter.py)
- 输出结构：
  - `final_answer`
  - `reasoning_summary`
  - `relevant_pages`
  - `references`
  - `citations`
  - `confidence`

## 金融场景优化

### 表格优先

数字类问题优先受益于表格序列化和 `serialized_table` chunk。检索结果会带 `chunk_type`，number question 会对表格类 chunk 获得额外排序 bonus。

### 指标口径严格匹配

Prompt 明确约束：

- 不能把相似指标当成目标指标
- 不能用推导值替代原文值
- 币种不一致时优先拒答
- 证据弱时返回 `N/A`

### Query Rewrite

针对金融术语做轻量扩展，例如：

- `Operating margin` -> `operating profit margin`
- `share buyback` -> `share repurchase`
- `mergers or acquisitions` -> `M&A / acquisition / merger`

### 引用与置信度

系统在生成后会：

- 校验模型返回的页码必须来自检索结果
- 回填 chunk 级 evidence snippet
- 根据检索得分和证据完整度输出 `high / medium / low` confidence

## 项目结构

```text
FinaRAG/
├── main.py
├── config/
│   ├── qwen_base.yaml
│   ├── qwen_rerank.yaml
│   └── qwen_ser_rerank.yaml
├── demo_app/
│   └── streamlit_app.py
├── eval/
│   ├── compare_configs.py
│   ├── metrics.py
│   └── run_eval.py
├── src/
│   ├── api_requests.py
│   ├── citation_formatter.py
│   ├── embedding_backend.py
│   ├── ingestion.py
│   ├── parsed_reports_merging.py
│   ├── pdf_parsing.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── query_rewrite.py
│   ├── questions_processing.py
│   ├── reranking.py
│   ├── retrieval.py
│   ├── retrieval_filters.py
│   ├── tables_serialization.py
│   ├── text_normalization.py
│   └── text_splitter.py
└── tests/
```

## 快速开始

### 1. 安装

```bash
git clone https://github.com/AdamsLiuG/FinaRAG.git
cd FinaRAG
python -m venv venv
source venv/bin/activate
pip install -e . -r requirements.txt
```

### 2. 配置 `.env`

```env
LLM_PROVIDER=qwen
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://your-openai-compatible-endpoint/v1
QWEN_MODEL=Qwen/Qwen2.5-72B-Instruct

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
RERANKING_BACKEND=flag_embedding
```

### 3. 构建 Pipeline

```bash
cd data/test_set

python ../../main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10
python ../../main.py process-reports --config no_ser_tab
python ../../main.py process-questions --config-path ../../config/qwen_base.yaml
```

如果要启用表格序列化与混合召回 + rerank：

```bash
python ../../main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10
python ../../main.py serialize-tables --max-workers 10
python ../../main.py process-reports --config ser_tab
python ../../main.py process-questions --config-path ../../config/qwen_ser_rerank.yaml
```

## 评测闭环

### 指标

当前最小评测框架支持：

- `answer_rate`
- `na_rate`
- `citation_coverage`
- `avg_references_per_answer`
- `confidence_distribution`
- `reference_exact_match`

其中 `reference_exact_match` 默认使用数据目录下的参考答案文件做对齐评估。它更适合做 **配置间对比 / 回归检查**，不等价于严格 benchmark ground truth。

### 运行单配置评测

```bash
python eval/run_eval.py \
  --dataset-dir data/test_set \
  --run-pipeline \
  --config qwen_base \
  --output eval/results/qwen_base.json
```

### 对比多配置

```bash
python eval/compare_configs.py \
  --dataset-dir data/test_set \
  --configs qwen_base,qwen_rerank,qwen_ser_rerank \
  --output eval/results/compare_test_set.json
```

### 结果表模板

| Config | Answer Rate | Citation Coverage | Reference Exact Match |
| --- | --- | --- | --- |
| `qwen_base` | local run | local run | local run |
| `qwen_rerank` | local run | local run | local run |
| `qwen_ser_rerank` | local run | local run | local run |

本仓库不预置虚构指标，建议直接运行上面的脚本生成真实结果后填表。

## Demo

最小交互 Demo 基于 Streamlit：

```bash
streamlit run demo_app/streamlit_app.py
```

Demo 展示内容：

- 输入问题
- config 切换
- 结构化答案
- references
- citations
- confidence
- model/debug metadata

## 难点与解决方案

### 1. README 亮点和代码实现不一致

对比问答分支已统一改为使用 `api_processor`，并补充测试，避免“README 有但代码跑不通”。

### 2. 金融文档的 chunk 不是普通文本 chunk

新增结构感知切块，保留标题、table、page、currency、year 等 metadata，降低财务指标问答时的上下文断裂。

### 3. 不能只返回页码，要返回证据

新增 citation formatter，把检索结果中的 chunk 信息映射成 citation 列表和 evidence snippet。

### 4. 只堆功能不够，必须能做 ablation

新增最小 eval 框架，对不同 config 做 reference-alignment 对比，形成评测闭环。

## 可以直接写进简历的 Bullet

- 设计并实现面向金融研报场景的 RAG 问答系统，基于 Docling 完成复杂 PDF 解析、表格语义化、结构感知切块与混合检索，支持引用溯源与置信度输出。
- 在问答链路中加入 Query Rewrite、metadata filter、Parent Page Retrieval 与可选 rerank，提升数字类与多公司比较问题的检索相关性和答案可解释性。
- 搭建评测闭环，支持对 `base / rerank / table-serialization` 等配置进行 reference-alignment 对比，为 README 和简历输出可量化实验结果。
- 实现多公司 comparative QA 路由，将比较问题拆解为单公司子问题并汇总为统一比较答案，增强项目的业务真实感和面试可追问性。

## 当前状态

- 已完成：comparative QA 修复、结构感知切块、query rewrite、metadata filter、citation/confidence、config YAML、demo 脚手架、最小 eval、单元测试
- 待继续补强：真实跑分结果、更多过滤字段、更多 benchmark case、生产级 API/Web UI
