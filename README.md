# FinaRAG - 金融研报智能问答系统

面向金融研报场景的 RAG（检索增强生成）系统，支持从大量 PDF 研报中自动检索相关内容并回答自然语言问题。

## 项目特点

- **复杂 PDF 深度解析**：基于 Docling 的版面分析引擎，支持多栏排版、表格识别、图表提取，面对研报类复杂文档不丢信息
- **表格序列化**：将 PDF 中的结构化表格转换为语义化的文本描述，提升财务数据的检索和理解能力
- **向量检索 + 父文档召回**：使用 FAISS 构建向量索引，支持 Parent Document Retrieval（先定位 chunk 再返回所在完整页），保留上下文完整性
- **LLM 重排序**：可选启用 LLM 对检索结果进行二次打分排序，提升上下文相关性
- **结构化输出 + CoT 推理**：生成答案时采用链式推理（Chain-of-Thought），输出包含逐步分析、引用页码、最终答案的结构化 JSON
- **多公司对比问答路由**：自动识别问题涉及多家公司时拆分为子问题分别检索，再汇总对比回答
- **本地部署**：Embedding 使用本地模型（bge-m3），LLM 接入 Qwen 兼容 API，无需依赖 OpenAI

## 系统架构

```
PDF 研报 ──→ Docling 解析 ──→ 文档合并 ──→ 文档切块 ──→ 向量化入库
                  ↓                                         ↓
            表格序列化（可选）                          FAISS 向量索引
                                                          ↓
用户提问 ──→ 公司名提取 ──→ 向量检索 ──→ LLM 重排（可选）──→ CoT 推理生成 ──→ 结构化答案
```

## 技术栈

| 组件 | 选型 |
| --- | --- |
| PDF 解析 | Docling（版面分析 + 表格识别） |
| Embedding | BAAI/bge-m3（本地部署） |
| 向量检索 | FAISS（IndexFlatIP） |
| 重排序 | LLM Reranking（Qwen） |
| 生成模型 | Qwen 系列（通过 OpenAI 兼容 API） |
| 文本切块 | RecursiveCharacterTextSplitter（tiktoken） |

## 快速开始

### 环境安装

```bash
git clone <your-repo-url>
cd FinaRAG
python -m venv venv
source venv/bin/activate
pip install -e . -r requirements.txt
```

### 配置

创建 `.env` 文件：

```env
LLM_PROVIDER=qwen
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://your-openai-compatible-endpoint/v1
QWEN_MODEL=Qwen/Qwen2.5-72B-Instruct

# Embedding 模型：支持 HuggingFace ID 或本地路径
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
```

### 运行 Pipeline

将 PDF 研报放入 `data/test_set/pdf_reports/` 目录，然后执行：

```bash
cd data/test_set

# 1. 解析 PDF
python ../../main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10

# 2. 处理解析结果（合并、切块、构建向量库）
python ../../main.py process-reports --config no_ser_tab

# 3. 执行问答
python ../../main.py process-questions --config qwen_base
```

如需启用表格序列化和 LLM 重排：

```bash
python ../../main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10
python ../../main.py serialize-tables --max-workers 10
python ../../main.py process-reports --config ser_tab
python ../../main.py process-questions --config qwen_ser_rerank
```

### CLI 命令

```bash
python main.py --help
```

| 命令 | 说明 |
| --- | --- |
| `download-models` | 下载 Docling 所需模型 |
| `parse-pdfs` | 解析 PDF 研报（支持并行） |
| `serialize-tables` | 表格序列化处理 |
| `process-reports` | 执行完整的文档处理流水线 |
| `process-questions` | 基于指定配置处理问答 |

### 可用配置

| 配置名 | 说明 |
| --- | --- |
| `qwen_base` | 默认配置：Qwen 生成 + 本地 Embedding + 父文档检索 |
| `qwen_rerank` | 在 base 基础上启用 LLM 重排序 |
| `qwen_ser_rerank` | 在 rerank 基础上启用表格序列化 |

## 数据集

仓库包含两个数据集：

- `data/test_set/`：小规模测试集（5 份年报 + 配套问答）
- `data/erc2_set/`：完整数据集（含所有问答和研报）

每个数据集目录下的 README 有详细说明。

## 项目结构

```
FinaRAG/
├── main.py                         # CLI 入口
├── src/
│   ├── pipeline.py                 # 流水线编排与配置
│   ├── pdf_parsing.py              # PDF 解析（Docling）
│   ├── parsed_reports_merging.py   # 解析结果合并
│   ├── tables_serialization.py     # 表格序列化
│   ├── text_splitter.py            # 文档切块
│   ├── embedding_backend.py        # Embedding 后端
│   ├── ingestion.py                # 向量/BM25 索引构建
│   ├── retrieval.py                # 检索（向量/混合/全文）
│   ├── reranking.py                # 重排序
│   ├── questions_processing.py     # 问答处理
│   ├── prompts.py                  # Prompt 模板
│   ├── api_requests.py             # LLM API 调用
│   └── api_request_parallel_processor.py  # 并行 API 处理
├── data/
│   ├── test_set/                   # 测试数据集
│   └── erc2_set/                   # 完整数据集
├── requirements.txt
└── .env                            # 配置文件（需自行创建）
```

## 许可证

MIT
