# ERC2 数据集

本目录包含完整的金融研报问答数据集。

## 内容
- `questions.json`：问答题目
- `subset.csv`：测试文档元信息
- `subset.json`：JSON 格式的元信息
- `answers_1st_place_o3-mini.json`：示例答案（o3-mini 模型）
- `answers_1st_place_llama_70b.json`：示例答案（Llama 70B 模型）

## 使用方法

1. 下载必要文件至本目录：

   ### 问答必需
   - `databases`（[Google Drive](https://drive.google.com/file/d/1mp-hYhMAit4rdi7RURuIsM33zbXq1nQJ/view?usp=sharing)）
     - 包含运行问答所需的全部预处理数据

   ### 可选文件
   - `pdf_reports`（[Google Drive](https://drive.google.com/file/d/1MvcN_-KpI-9nS4hDFAcPxFU2lRmwMP7M/view?usp=sharing)）
     - 如需从头运行 PDF 解析流水线
   - `debug_data`（[Google Drive](https://drive.google.com/file/d/13RT456tZVTAwPIsy8OndZ1EWASNCdfe3/view?usp=sharing)）
     - 用于调试各流水线阶段或查看中间输出

2. 参照仓库根目录的 README.md 进行配置和运行

> **注意**：预构建的 `databases` 包与本地 Embedding 后端不兼容，需从 `pdf_reports` 重新构建向量数据库。
