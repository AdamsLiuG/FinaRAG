# FinaRAG

面向金融研报与年报场景的 RAG 智能问答系统。项目重点不是“把 PDF 塞进向量库”，而是围绕金融文档的真实难点做增量优化：复杂 PDF 解析、中文/双语 OCR、表格语义化、结构感知切块、混合检索、文档候选路由、可选 rerank、table-grounded 数字问答、引用溯源、答案校验、置信度输出，以及多公司对比问答路由。

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
  -> Structure-Aware Parent / Child Chunking
  -> Vector / BM25 / Sparse Indexing
  -> Retrieval + Query Rewrite + Metadata Filter
  -> Optional Rerank
  -> Structured Answer Generation
  -> References + Citations + Confidence
```

## 中文金融版改造

当前版本已经支持把语料从英文年报切换到中文上市公司年报和中文券商研报，核心变化包括：

- `document_manifest` 统一管理文档元数据，替代运行时对 `subset.csv` topic flag 的强依赖
- `document_language / ocr_mode` 配置接入 Pipeline，默认使用 Docling RapidOCR，支持 `zh` 和 `bilingual`
- 中文 query rewrite、中文 BM25 tokenization、ticker/简称/券商名/报告类型路由
- `table_grounding_result` 接入 number QA、citation、confidence 和 validation
- Demo 直接展示候选文档路由、召回证据和命中的表格单元格

建议中文场景优先使用 [config/qwen_zh_finance.yaml](/media/main/lgd/llm/FinaRAG/config/qwen_zh_finance.yaml)。

## 核心能力

- 复杂 PDF 解析：使用 Docling 处理多栏、表格、图片和 OCR 场景
- 表格序列化：把表格转换成更适合检索的自然语言信息块
- 真正 Parent-Child 检索：child 负责召回，命中后回溯结构块 parent 参与融合、重排和生成
- 结构感知切块：保留标题、页面、section、table、parent block 等 metadata
- 多路召回：支持向量检索、BM25、bge-m3 sparse lexical 混合召回
- Query Plan：针对中英文财务指标问答做术语扩展、币种/年份/期间/报告类型抽取
- Metadata Filter：支持 company / currency / year / report type / period / security code / broker / candidate docs 等过滤约束
- Document Catalog Routing：无显式公司名时，根据公司简称、股票代码、券商名、报告标题、年份和报告类型先召回候选文档，再进入文档内检索
- 多公司对比问答：自动拆分 comparative question，再汇总比较答案
- 引用溯源：返回页码 references、chunk/table 级 citations、retrieval debug 和 confidence
- 答案校验：对 currency / year / period / citation coverage / numeric table grounding 做后处理校验，必要时拒答
- 误差分析：基于 debug bundle 输出 routing / retrieval / generation / validation 四类失败归因

## RAG Pipeline 说明

### 1. PDF 解析

- 入口：`parse-pdfs`
- 模块：[src/pdf_parsing.py](/media/main/lgd/llm/FinaRAG/src/pdf_parsing.py)
- 作用：
  - 解析 PDF 页面结构
  - 提取文本、表格、图片
  - 把 `document_manifest.csv/json` 中的 `company_name`、`company_aliases`、`security_code`、`doc_source_type`、`broker_name`、`report_date` 等字段注入文档元信息
  - 中文场景下默认启用 `zh+en` OCR；项目默认使用 Docling RapidOCR，也保留 `docling_easyocr` 作为兼容模式

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
  - 再按标题和页面结构切成 parent block
  - 对每个 parent block 再做 token-aware child 拆分
- 产物结构：
  - `content.parent_chunks`：生成与引用使用的 parent block
  - `content.chunks`：检索索引使用的 child chunk
- chunk metadata 至少包含：
  - `chunk_id`
  - `chunk_type`
  - `node_type`
  - `page`
  - `section_title`
  - `table_id`
  - `currency`
  - `report_year`
  - `parent_block_id`
  - `parent_chunk_id`
  - `report_section`
  - `evidence_type`
  - `has_table_context`

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
  - Query plan
  - Metadata filter
  - Metadata-aware routing
  - Parent retrieval mode: `page` / `block`
  - Optional rerank

### Parent Retrieval Modes

- `parent_document_retrieval=false`
  - 返回 child chunk，适合观察最细粒度召回
- `parent_document_retrieval=true` + `parent_retrieval_mode=page`
  - 兼容旧模式：命中 child 后扩展为整页 page
- `parent_document_retrieval=true` + `parent_retrieval_mode=block`
  - 真正 Parent-Child：命中 child 后回溯结构块 parent，再进入 fusion / rerank / 生成

### 5. 生成与后处理

- 模块：
  - [src/questions_processing.py](/media/main/lgd/llm/FinaRAG/src/questions_processing.py)
  - [src/prompts.py](/media/main/lgd/llm/FinaRAG/src/prompts.py)
  - [src/citation_formatter.py](/media/main/lgd/llm/FinaRAG/src/citation_formatter.py)
  - [src/answer_validation.py](/media/main/lgd/llm/FinaRAG/src/answer_validation.py)
- 输出结构：
  - `final_answer`
  - `reasoning_summary`
  - `relevant_pages`
  - `references`
  - `citations`
  - `confidence`
  - `confidence_reason`
  - `validation_flags`
  - `route_info`

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

针对中英文金融术语做轻量扩展，例如：

- `Operating margin` -> `operating profit margin`
- `share buyback` -> `share repurchase`
- `mergers or acquisitions` -> `M&A / acquisition / merger`
- `营业收入` -> `营收 / 收入`
- `归母净利润` -> `归属于母公司股东的净利润`
- `并购` -> `收购 / 兼并`

### 数字题 Table Grounding

对于 `number` 问题，系统会优先从结构化表格里做 `metric -> row/col header -> cell` 匹配，输出：

- `table_id`
- `page`
- `matched_row_headers`
- `matched_col_headers`
- `raw_value / normalized_value`
- `unit / footnote_refs`

这样可以把“文本检索 + 模型猜数”改造成“表格定位 + 单元格落地 + 数值归一”。

### 引用与置信度

系统在生成后会：

- 校验模型返回的页码必须来自检索结果
- 回填 chunk/table 级 evidence snippet
- 根据检索得分、citation coverage、validation flags 和 table grounding 输出 `high / medium / low` confidence
- 对 currency / year / period / numeric grounding 做一致性检查，必要时强制降级或拒答

### Document Catalog Routing

当问题里没有显式公司名时，系统不会直接报错，而是结合：

- query rewrite 抽取出的 `currency / year / period / report_type / doc_source_type`
- 问题中的 `公司简称 / 股票代码 / 券商名 / 报告标题`
- `document_manifest` 中已有的公司别名、行业、报告标题和文档类型

对候选公司做打分路由，再进入单公司检索和生成流程。

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
│   ├── error_analysis.py
│   ├── metrics.py
│   └── run_eval.py
├── src/
│   ├── answer_validation.py
│   ├── api_requests.py
│   ├── citation_formatter.py
│   ├── embedding_backend.py
│   ├── ingestion.py
│   ├── document_manifest.py
│   ├── parsed_reports_merging.py
│   ├── pdf_parsing.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── query_plan.py
│   ├── query_rewrite.py
│   ├── questions_processing.py
│   ├── reranking.py
│   ├── report_catalog.py
│   ├── retrieval.py
│   ├── retrieval_filters.py
│   ├── table_grounding.py
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
QWEN_MODEL=Qwen3.5-35B-A3B-AWQ-4bit

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
RERANKING_BACKEND=flag_embedding
```

如果你要在向量化阶段使用多卡，可以把 `EMBEDDING_DEVICE` 写成逗号分隔形式，例如 `cuda:0,cuda:1`；`parse-pdfs` 不读取这个变量，它只影响后续 embedding / sparse lexical 阶段。

### 3. 构建 Pipeline

```bash
cd data/test_set

python ../../main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10
python ../../main.py process-reports --config no_ser_tab
python ../../main.py process-questions --config-path ../../config/qwen_base.yaml
```

如果你是从旧版本升级到真正 Parent-Child 检索，必须先重新执行一次 `process-reports`，让 `chunked_reports` 重新生成 `parent_chunks` 和 `parent_chunk_id`。

如果要启用表格序列化与混合召回 + rerank：

```bash
python ../../main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10
python ../../main.py serialize-tables --config-path ../../config/qwen_ser_rerank.yaml --max-workers 10
python ../../main.py process-reports --config ser_tab
python ../../main.py process-questions --config-path ../../config/qwen_ser_rerank.yaml
```

如果你要切中文年报 / 中文研报数据集，建议准备一个 `document_manifest.csv`，最少包含这些列：

```text
doc_id,company_name,company_aliases,security_code,doc_source_type,report_date,fiscal_year,broker_name,major_industry,language
```

如果你的 PDF 已经是通过 `PDFCrawl/output` 按行业抓下来的，也可以直接用仓库内置命令把它整理成 FinaRAG 数据集：

```bash
cd FinaRAG

python main.py prepare-pdfcrawl-dataset \
  --pdfcrawl-root ../PDFCrawl/output \
  --dataset-dir data/chinese_annual_reports_2024 \
  --link-mode symlink
```

这个命令会自动：

- 汇总 `PDFCrawl/output/*/manifest.csv`
- 把 PDF 平铺到 `data/chinese_annual_reports_2024/pdf_reports/`
- 生成 `data/chinese_annual_reports_2024/document_manifest.csv`
- 生成一个空的 `questions.json`，方便你后续补中文问题集

生成完成后，在新数据集目录下直接跑中文配置即可：

```bash
cd data/chinese_annual_reports_2024

python ../../main.py parse-pdfs --config-path ../../config/qwen_zh_finance.yaml --parallel --chunk-size 2 --max-workers 6 --cuda-devices 0,1
python ../../main.py serialize-tables --config-path ../../config/qwen_zh_finance.yaml --max-workers 6
python ../../main.py process-reports --config-path ../../config/qwen_zh_finance.yaml
```

如果 `parse-pdfs` 的 OCR 全压在 `GPU 0`，可以给 `parse-pdfs` 额外传 `--cuda-devices 0,1`；这会把并行解析 worker 轮转绑定到不同 GPU。它和后续向量化阶段使用的 `EMBEDDING_DEVICE` 是两套独立配置。

中文配置建议直接使用：

```bash
python ../../main.py parse-pdfs --config-path ../../config/qwen_zh_finance.yaml --parallel --chunk-size 2 --max-workers 10 --cuda-devices 0,1
python ../../main.py serialize-tables --config-path ../../config/qwen_zh_finance.yaml --max-workers 10
python ../../main.py process-reports --config-path ../../config/qwen_zh_finance.yaml
python ../../main.py process-questions --config-path ../../config/qwen_zh_finance.yaml
```

同样地，启用 `block` 模式前需要重新跑 `process-reports`；旧版 `chunked_reports` 只有 child/page 结构，不能直接用于真正 Parent-Child 回溯。

## 评测闭环

### 指标

当前最小评测框架支持：

- `answer_rate`
- `na_rate`
- `citation_coverage`
- `avg_references_per_answer`
- `confidence_distribution`
- `reference_exact_match`
- `reference_page_hit`
- `citation_page_hit`
- `retrieval_hit_at_k`
- `avg_citation_page_precision`
- `question_type_breakdown`
- `confidence_calibration`

其中 `reference_exact_match` 默认使用数据目录下的参考答案文件做对齐评估。它更适合做 **配置间对比 / 回归检查**，不等价于严格 benchmark ground truth。`run_eval.py` 会自动读取同名 `_debug.json`，因此评测还能覆盖检索命中和误差归因。

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
  --output eval/results/compare_test_set.json \
  --markdown-output eval/results/compare_test_set.md
```

### 误差分析

```bash
python eval/error_analysis.py \
  --answers-file data/test_set/answers_qwen_base.json \
  --debug-file data/test_set/answers_qwen_base_debug.json \
  --reference-answers data/test_set/answers_max_nst_o3m.json
```

`compare_configs.py` 现在可以直接输出 Markdown 表；README 不再保留 `local run` 占位表，避免文档亮点先于证据落地。

## Demo

可演示前端基于 Streamlit，默认适合直接挂在 `data/test_set` 上做现场展示：

```bash
python3 -m streamlit run demo_app/streamlit_app.py
```

仓库自带 `.streamlit/config.toml`，默认关闭 Streamlit file watcher，避免 `torch` / `FlagEmbedding` 依赖在启动时触发 `local_sources_watcher` 兼容问题。

界面能力：

- 输入 query 并选择 `boolean / number / name`
- 侧边栏一键载入样例问题、切换配置文件、查看数据资产状态
- 主界面展示最终回答、相关页码、confidence、validation flags
- 独立展示召回文档列表与 chunk/page 级 evidence preview
- 展示 references、citations、route info、query plan 与原始 debug JSON

推荐演示配置：

- `config/qwen_ser_rerank.yaml`：表格序列化 + 混合召回 + rerank，效果最完整
- `config/qwen_zh_finance.yaml`：中文金融版配置，默认开启文档候选路由和 table grounding
- `config/qwen_base.yaml`：链路更轻，适合先验证基本问答可用性

## 中文 Benchmark 脚手架

仓库内已经预留了中文 benchmark 模板目录 [data/chinese_benchmark/README.md](/media/main/lgd/llm/FinaRAG/data/chinese_benchmark/README.md)，建议按下面结构补齐你自己的数据：

- `document_manifest.template.csv`
- `questions_zh_template.json`
- `answers_zh_gold_template.json`

推荐第一阶段先做 `20` 份年报 + `20` 份研报 + `50-80` 个标注问题，用来跑：

- 英文旧版 baseline
- 中文 baseline
- 中文 hybrid + rerank
- 中文 hybrid + rerank + table-grounded

## 难点与解决方案

### 1. README 亮点和代码实现不一致

对比问答分支已统一改为使用 `api_processor`，并补充测试，避免“README 有但代码跑不通”。

### 2. 金融文档的 chunk 不是普通文本 chunk

新增结构感知切块，保留标题、table、page、currency、year 等 metadata，降低财务指标问答时的上下文断裂。

### 3. 不能只返回页码，要返回证据和校验结果

新增 citation formatter，把检索结果中的 chunk 信息映射成 citation 列表和 evidence snippet；同时增加 answer validation，对 currency / year / numeric grounding 做后处理校验。

### 4. 不能只靠显式公司名做路由

新增 document catalog routing，在问题未提公司名时，结合股票代码、券商名、年份、报告类型和文档标题做候选文档推断。

### 5. 只堆功能不够，必须能做 ablation 和 error analysis

扩展 eval 框架，对不同 config 做 reference-alignment、retrieval hit、citation page hit 对比，并补充 error analysis 脚本做失败归因。

## 可以直接写进简历的 Bullet

- 设计并实现面向金融研报场景的 RAG 问答系统，基于 Docling 完成复杂 PDF 解析、表格语义化、结构感知切块与混合检索，支持引用溯源与置信度输出。
- 在问答链路中加入 Query Plan、document catalog routing、metadata filter、真正 Parent-Child Retrieval、可选 rerank 与 table grounding，提升数字类、事件类和无显式公司名问题的检索相关性与可解释性。
- 搭建评测闭环，支持对 `base / rerank / table-serialization / zh-finance` 等配置进行 reference-alignment、retrieval hit、citation page hit 和 error analysis 对比，为 README 和简历输出可量化实验结果。
- 实现多公司 comparative QA 路由，并加入 answer validation、evidence-grounded citation 与中文 benchmark 脚手架，增强项目的业务真实感、拒答能力和面试可追问性。

## 当前状态

- 已完成：comparative QA 修复、结构感知切块、query plan、document manifest、document catalog routing、table grounding、answer validation、citation/confidence、config YAML、demo、扩展 eval、误差分析脚本、单元测试
- 待继续补强：真实中文 benchmark 跑分结果、多报告时序检索、更强表格 fidelity 校验、生产级 API/Web UI
