请基于现有 PDFCrawl 项目，设计并实现一套用于金融年报 RAG 的库构建与检索系统，要求：

1. 数据源
- 上交所年报 PDF 与正文是主源
- 巨潮公司资料/分红/关联证券是辅助标签源

2. 数据层
- company_master
- annual_report
- report_page
- report_chunk
- company_label_snapshot
- chunk_metadata

3. 库构建
- 年报页级解析
- 章节识别
- chunk 切分
- 标签生成
- embedding_text 生成
- metadata 生成
- embedding 向量生成
- 写入向量库/检索库

4. 检索层
- query parser
- metadata filter
- vector recall
- bm25 recall
- tag recall
- rerank
- 公司级聚合

5. 输出
- 检索结果要包含：
  - 公司名
  - 股票代码
  - 年报年份
  - 页码
  - section_name
  - 命中标签
  - 原文片段
  - 最终分数