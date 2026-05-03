# FinaRAG 简历项目升级评估与增量改造方案

## 1. 总体结论
FinaRAG 当前最大的问题不是“完全没有技术内容”，而是**项目真实能力、README 表达、工程闭环、可验证结果之间没有形成强一致**。仓库里确实有 Docling 解析、表格序列化、混合检索、可选 rerank、结构化输出等能力，但 README 缺少业务痛点、量化结果、评测闭环和关键设计取舍；同时部分关键分支仍有实现缺口，例如多公司对比问答分支直接引用未定义的 `self.openai_processor`，使 README 中“多公司对比问答路由”这一亮点在代码层面站不稳（[questions_processing.py](/media/main/lgd/llm/FinaRAG/src/questions_processing.py#L501)）。因此它现在还不够像一个有竞争力的“大模型应用岗简历项目”，更像“有一定深度的 RAG 工程雏形，但产品化、评测化、表达化不足”。

---

## 2. 按严重程度排序的缺陷清单

### 1. README 亮点与代码落地不完全一致
- 缺陷名称：亮点声明可信度不足
- 具体表现：README 将“多公司对比问答路由”列为项目特点（[README.md](/media/main/lgd/llm/FinaRAG/README.md#L12)），但实现里 `process_comparative_question` 使用了未初始化的 `self.openai_processor`（[questions_processing.py](/media/main/lgd/llm/FinaRAG/src/questions_processing.py#L509) 和 [questions_processing.py](/media/main/lgd/llm/FinaRAG/src/questions_processing.py#L557)），实际会直接报错。
- 为什么削弱竞争力：面试官一旦顺着 README 追问“多公司对比怎么做”，你现场解释会露出破绽，项目可信度大幅下降。
- 可以如何改进：统一改为 `self.api_processor`；比较问答子问题的 schema 不要写死 `"number"`，而应沿用原问题 `schema` 或显式推断；补一个 comparative case 的回归测试和 README 示例。
- 改进优先级：高
- 建议修改文件：`src/questions_processing.py`、新增 `tests/test_comparative_questions.py`、`README.md`

### 2. 没有评测闭环，无法证明“优化有效”
- 缺陷名称：缺少 eval / benchmark 体系
- 具体表现：仓库结构中没有 `eval/`、`tests/`、`benchmark/` 等目录；README 也没有任何准确率、命中率、引用正确率、配置对比结果。MyRAG1 至少明确存在 `final_score.py`、训练数据生成和评估脚本（[MyRAG1/README.md](/media/main/lgd/llm/MyRAG1/README.md#L14)）。
- 为什么削弱竞争力：大模型应用岗最怕“堆模块但说不清效果”；没有指标就无法支撑简历 bullet。
- 可以如何改进：新增最小评测流水线，先做 `retrieval_hit@k`、`answer_exact_match / normalized_match`、`citation_page_hit`、不同配置对比表。
- 改进优先级：高
- 建议修改文件：新增 `eval/run_eval.py`、`eval/metrics.py`、`eval/compare_configs.py`、`README.md`

### 3. 项目定位仍偏“技术拼装”，业务痛点表达弱
- 缺陷名称：业务定位不够锋利
- 具体表现：README 首段只说“从大量 PDF 研报中自动检索相关内容并回答自然语言问题”（[README.md](/media/main/lgd/llm/FinaRAG/README.md#L1)），但没有明确说明具体用户是谁、研报问答难点是什么、为什么普通 RAG 不够。
- 为什么削弱竞争力：招聘者无法快速判断这是“金融场景 AI 应用”还是“通用 PDF QA Demo”。
- 可以如何改进：补“业务背景 / 用户痛点”章节，明确金融研报的难点是表格密集、指标口径严格、币种/单位敏感、来源页要求强、跨公司比较复杂。
- 改进优先级：高
- 建议修改文件：`README.md`

### 4. 检索范围被“先识别公司名再按公司单文档检索”强约束
- 缺陷名称：检索设计偏竞赛/数据集定制，真实场景感不足
- 具体表现：问题处理先从 `subset.csv` 中匹配公司名（[questions_processing.py](/media/main/lgd/llm/FinaRAG/src/questions_processing.py#L252)），随后只在该公司对应的单份报告里检索（[retrieval.py](/media/main/lgd/llm/FinaRAG/src/retrieval.py#L65)）。
- 为什么削弱竞争力：这更像“已知答案范围的封闭数据集问答”，不够像真实投研助手。面试官会问“如果用户只问指标，不提公司怎么办？”“如果一个公司多份年份报告怎么办？”
- 可以如何改进：保留当前单公司模式，同时新增 metadata filter 模式，支持 `company/year/report_type/currency` 筛选；再加一个“全库检索 + metadata reroute”路径。
- 改进优先级：高
- 建议修改文件：`src/questions_processing.py`、`src/retrieval.py`、新增 `src/metadata.py` 或 `src/retrieval_filters.py`

### 5. Chunk 策略过于基础，未体现对金融研报的结构适配
- 缺陷名称：切块策略不够专业
- 具体表现：当前仅按页文本做 `RecursiveCharacterTextSplitter`，默认 `chunk_size=300, overlap=50`（[text_splitter.py](/media/main/lgd/llm/FinaRAG/src/text_splitter.py#L70)），没有标题级、表格级、财务报表级、footnote 绑定等策略。
- 为什么削弱竞争力：金融研报中关键答案经常依赖“表头 + 单位 + 注释 + 上下页连续块”；简单切块很难说明你理解金融文档特性。
- 可以如何改进：做增量升级，不推翻当前结构，只补“结构感知切块”：优先按页面标题、表格块、脚注块组织，再对子块做字符切分；记录 `chunk_type/section_title/page_span/table_id` 元数据。
- 改进优先级：高
- 建议修改文件：`src/text_splitter.py`、`src/parsed_reports_merging.py`

### 6. 没有显式 metadata filter，金融场景特化不足
- 缺陷名称：金融场景检索约束不够
- 具体表现：现有 metadata 只稳定用了 `company_name`（[pdf_parsing.py](/media/main/lgd/llm/FinaRAG/src/pdf_parsing.py#L277)）；虽然 `subset.csv` 包含 `cur`、行业、事件标签等字段，但没有进入检索或回答流程。
- 为什么削弱竞争力：金融问答里币种、年份、报表类型、是否年报/中报、公司主体都很关键，不做 filter 很难讲“场景优化”。
- 可以如何改进：把 `subset.csv` 的 `cur`、`major_industry` 等注入 chunk metadata；先支持最有价值的 4 个过滤条件：`company_name`、`currency`、`year`、`question_kind`。
- 改进优先级：高
- 建议修改文件：`src/pdf_parsing.py`、`src/text_splitter.py`、`src/ingestion.py`、`src/retrieval.py`

### 7. 生成侧虽然有结构化输出，但缺少“引用即证据”的更强约束
- 缺陷名称：引用溯源控制偏弱
- 具体表现：当前会校验页码是否来自检索结果，并在不足时补足页码（[questions_processing.py](/media/main/lgd/llm/FinaRAG/src/questions_processing.py#L114)），但没有“答案片段级证据”“引用块 ID”“高置信度/低置信度判定”。
- 为什么削弱竞争力：金融场景里“答案正确但证据弱”是高风险点。面试官会追问如何控制 hallucination。
- 可以如何改进：回答结构里新增 `citations`，每条引用包含 `page`、`chunk_id`、`evidence_snippet`、`score`；当 top1/top2 分数差过小或回答为推断类时返回 `confidence: low`。
- 改进优先级：高
- 建议修改文件：`src/prompts.py`、`src/questions_processing.py`、`src/retrieval.py`

### 8. README 没有量化结果，简历 bullet 很难写“成果”
- 缺陷名称：缺少可量化成果
- 具体表现：README 没有任何“启用表格序列化后 xxx 提升”“混合检索优于单向量检索”的实验结果。
- 为什么削弱竞争力：简历最强 bullet 通常是“做了什么 + 难点 + 指标提升”；现在只能写“搭了一个系统”，力度不足。
- 可以如何改进：补一个 3xN 对比实验表，至少比较 `base / rerank / ser_rerank` 三组配置。
- 改进优先级：高
- 建议修改文件：新增 `eval/compare_configs.py`、`README.md`

### 9. 项目工程入口清楚，但缺少面向面试展示的 Demo 载体
- 缺陷名称：展示层不足
- 具体表现：目前只有 CLI（[main.py](/media/main/lgd/llm/FinaRAG/main.py)），没有 API / Web Demo / notebook 演示。
- 为什么削弱竞争力：大模型应用岗通常希望看到“可交互应用”，不仅是离线脚本。
- 可以如何改进：最小成本方案是先补 `demo_app/streamlit_app.py` 或 `app.py`，展示问题、检索页、引用页码、答案 JSON。
- 改进优先级：中
- 建议修改文件：新增 `demo_app/streamlit_app.py`、`README.md`

### 10. 配置体系可用，但仍偏硬编码 preset，实验管理不够系统
- 缺陷名称：配置管理粒度不够
- 具体表现：`RunConfig` 和若干 preset 写在 `pipeline.py` 内（[pipeline.py](/media/main/lgd/llm/FinaRAG/src/pipeline.py#L54)），适合少量配置，不适合持续扩展实验。
- 为什么削弱竞争力：当你要展示“我做了多组 ablation”时，不方便复现，也不利于对外说明。
- 可以如何改进：抽出 `config/` 目录，保留现有 dataclass，增加 YAML/JSON 配置装载层。
- 改进优先级：中
- 建议修改文件：`src/pipeline.py`、新增 `config/*.yaml`

### 11. 包装与项目命名不统一
- 缺陷名称：项目形象不稳定
- 具体表现：仓库名是 FinaRAG，但 `setup.py` 里的包名仍是 `erc2`（[setup.py](/media/main/lgd/llm/FinaRAG/setup.py#L3)）。
- 为什么削弱竞争力：这会让面试官感觉项目是比赛仓库改壳，包装还没收尾。
- 可以如何改进：统一 package name、README 标题、CLI 帮助、输出文件命名。
- 改进优先级：中
- 建议修改文件：`setup.py`、`README.md`、`main.py`

### 12. 数据可复现性说明存在明显摩擦
- 缺陷名称：预置数据产物与当前 embedding 管线不兼容
- 具体表现：`data/test_set/README.md` 和 `data/erc2_set/README.md` 都明确写了预构建 `databases` 与当前本地 embedding 管线不兼容，需要重建（[data/test_set/README.md](/media/main/lgd/llm/FinaRAG/data/test_set/README.md#L21)；[data/erc2_set/README.md](/media/main/lgd/llm/FinaRAG/data/erc2_set/README.md#L28)）。
- 为什么削弱竞争力：这会直接影响“开箱即用”和“别人能不能跑起来”，属于简历项目常见扣分项。
- 可以如何改进：给出统一重建脚本、产物版本标识、数据库 manifest 文件。
- 改进优先级：中
- 建议修改文件：新增 `scripts/rebuild_test_set.sh`、`scripts/rebuild_erc2_set.sh`、`README.md`

### 13. Prompt 很重，但缺少面向成本与稳定性的解释
- 缺陷名称：生成成本高且缺少取舍说明
- 具体表现：回答 schema 强制输出长 CoT，要求至少 150 词分析（[prompts.py](/media/main/lgd/llm/FinaRAG/src/prompts.py#L99)、[prompts.py](/media/main/lgd/llm/FinaRAG/src/prompts.py#L148)），这在工程上会显著抬高成本和延迟。
- 为什么削弱竞争力：面试官会问“为什么线上需要这么长的 reasoning？你如何平衡成本？”
- 可以如何改进：区分 debug 模式和 submission 模式；默认只生成 `reasoning_summary`，debug 时再开全量 CoT。
- 改进优先级：中
- 建议修改文件：`src/prompts.py`、`src/questions_processing.py`、`config/*.yaml`

### 14. BM25 实现较朴素，语言与金融术语适配有限
- 缺陷名称：词法检索质量可能偏弱
- 具体表现：BM25 直接 `query.split()` 和 `chunk.split()`（[retrieval.py](/media/main/lgd/llm/FinaRAG/src/retrieval.py#L93)、[ingestion.py](/media/main/lgd/llm/FinaRAG/src/ingestion.py#L16)），没有任何正则归一、单位处理、数字归一、财务缩写展开。
- 为什么削弱竞争力：金融问答对数字、币种、缩写极敏感，朴素分词很难成为亮点。
- 可以如何改进：新增 normalize 层，先做最小改造：大小写、标点、千分位、百分号、币种符号归一。
- 改进优先级：中
- 建议修改文件：`src/ingestion.py`、`src/retrieval.py`、新增 `src/text_normalization.py`

---

## 3. 对照 MyRAG1 的差距分析

| 维度 | MyRAG1 的表现 | FinaRAG 的表现 | 差距本质 | 应该如何补齐 |
| --- | --- | --- | --- | --- |
| 项目定位 | 明确是“用户手册问答助手”，目标用户和返回形式都清楚（[MyRAG1/README.md](/media/main/lgd/llm/MyRAG1/README.md#L3)） | 只写“金融研报智能问答系统”，定位偏泛（[README.md](/media/main/lgd/llm/FinaRAG/README.md#L1)） | 场景名词有了，但产品角色不清 | 写清“投研/金融分析/IR/知识助手”目标用户和典型问题 |
| 业务痛点 | 解释了手册 PDF 脏文本、图片、页码引用等问题 | 没写金融文档的口径、币种、时效、表格难点 | 业务问题没被拆开讲 | 补“为什么通用 PDF QA 不够”的章节 |
| 场景真实感 | 有页码、图片引用、训练与推理闭环 | 有竞赛数据集感，但真实使用路径弱 | 更像 challenge solution | 增加真实用户工作流和 demo 场景 |
| RAG 技术链路表达 | 从解析、清洗、切分、建索引、重排、生成到后处理，链路完整清晰 | 技术点罗列多，但缺少关键取舍与参数说明 | 有组件，缺少叙事 | 在 README 中展开 chunk、检索、重排、引用流程 |
| 金融场景特化 | 无金融特化，但每步都讲清用途 | 有表格序列化与 metric matching prompt，但缺少 metadata / time / currency filter | 特化点不够落地 | 增加币种/年份/来源过滤和相关实验 |
| 工程化程度 | 有 build / infer / data-gen / score 多入口 | 有 CLI 和 pipeline，但缺少 eval、tests、demo | 缺闭环工具链 | 新增 `eval/`、`tests/`、`demo_app/` |
| 评测闭环 | 明确有评估脚本 `final_score.py` | 无显式评估脚本 | 不能证明优化价值 | 构建 retrieval + answer + citation 的三层评测 |
| demo 展示 | README 本身像“讲给面试官听”的结构化 walkthrough | README 更像普通开源项目说明 | 展示方式弱 | 增加架构图、样例输入输出、debug 页面截图 |
| README 表达力 | 强在“解释为什么这么做” | 强在“列 feature”，弱在“证明和拆解” | 说点不说线 | 重写 README 叙事顺序 |
| 简历可写性 | 可以写“微调、混合检索、重排、带引用回答” | 现在只能写“做了一个金融 RAG 系统” | 缺指标和闭环 | 补实验结果和具体优化点 |
| 面试可追问性 | 每个模块都能顺着问下去 | 一问到 comparative / eval / 效果就容易失分 | 证据链断裂 | 补齐实现和实验，准备 FAQ |

---

## 4. 最值得优先做的项目升级路线图

### 第一阶段：低成本高收益改造
目标：尽快让项目更适合写进简历

#### 1. 修复 comparative QA 分支并加示例
- 目的：让 README 声称的亮点真正可用
- 为什么值得做：代码缺陷直接影响可信度
- 预期提升：减少“README 说了但跑不通”的风险
- 实现成本：低
- 简历价值提升点：可写“支持跨公司对比问答路由”

#### 2. 补一个最小 eval 脚本
- 目的：验证 `qwen_base / qwen_rerank / qwen_ser_rerank`
- 为什么值得做：最快把“做了功能”变成“做出了提升”
- 预期提升：生成量化结果表，可直接进 README 和简历
- 实现成本：低到中
- 简历价值提升点：可写“通过 ablation 验证混合检索/表格序列化/重排收益”

#### 3. 重写 README 前三屏
- 目的：把项目从“工具仓库”改成“面试材料”
- 为什么值得做：简历项目首先靠 README 建立第一印象
- 预期提升：定位更清晰、亮点更聚焦
- 实现成本：低
- 简历价值提升点：更容易提炼 2 到 4 条 bullet

#### 4. 增加引用证据结构
- 目的：把页码引用从“有页码”升级成“可解释证据”
- 为什么值得做：金融场景非常吃证据链
- 预期提升：增强幻觉控制和展示效果
- 实现成本：低到中
- 简历价值提升点：可写“支持答案溯源与置信度控制”

#### 5. 补一个最小 Demo
- 目的：提升项目完成度与展示性
- 为什么值得做：比纯 CLI 更像真实 AI 应用
- 预期提升：面试现场更容易演示
- 实现成本：低
- 简历价值提升点：可写“实现可交互投研问答 Demo”

### 第二阶段：中等成本的能力增强
目标：让项目更像成熟的 RAG 应用

#### 1. 增加 metadata filter
- 目的：支持公司、币种、年份、问题类型筛选
- 为什么值得做：金融问答高度依赖这些约束
- 预期提升：检索精度和场景真实性提升
- 实现成本：中
- 简历价值提升点：可写“针对金融场景设计元数据过滤检索链路”

#### 2. 结构感知切块
- 目的：改进表格、标题、脚注连续信息的保留
- 为什么值得做：直接影响财务指标问答正确率
- 预期提升：retrieval hit 和 citation 质量提高
- 实现成本：中
- 简历价值提升点：可写“面向复杂研报结构优化 chunk 策略”

#### 3. query rewrite / metric normalization
- 目的：把自然语言问题映射到财务表达
- 为什么值得做：金融术语同义、多口径问题很常见
- 预期提升：检索召回更稳定
- 实现成本：中
- 简历价值提升点：可写“加入问题标准化与术语改写模块”

#### 4. reranker 抽象统一
- 目的：统一 LLM rerank 和本地 FlagEmbedding rerank 的入口
- 为什么值得做：后续可做成本/效果对比
- 预期提升：更易实验和切换
- 实现成本：中
- 简历价值提升点：可写“支持多后端重排策略和加权融合”

#### 5. 结果缓存与日志标准化
- 目的：降低调试成本和推理成本
- 为什么值得做：工程化成熟度明显提升
- 预期提升：更可复现、更易排障
- 实现成本：中
- 简历价值提升点：可写“完善缓存、日志和中间产物管理”

### 第三阶段：高价值亮点增强
目标：让项目具备更强区分度和面试亮点

#### 1. 配置化实验平台
- 目的：把不同 pipeline 变成真正可对比的实验
- 为什么值得做：利于做 ablation 和 benchmark
- 预期提升：项目更像研究型工程项目
- 实现成本：中到高
- 简历价值提升点：可写“搭建 RAG 配置实验框架”

#### 2. 引入时间维度与多报告版本检索
- 目的：支持同一公司多年份报告、时效性过滤
- 为什么值得做：这是金融场景区别于普通 PDF QA 的关键
- 预期提升：场景真实性大幅提升
- 实现成本：高
- 简历价值提升点：可写“支持多年度研报时序问答”

#### 3. 自动误差分析报告
- 目的：把失败 case 自动归因到解析 / 检索 / 生成阶段
- 为什么值得做：面试可追问性极强
- 预期提升：你能主动讲“我怎么定位问题”
- 实现成本：中到高
- 简历价值提升点：可写“构建 RAG 误差分析闭环”

---

## 5. 代码改造建议

### 5.1 建议新增的模块/文件

#### `eval/`
- 为什么需要它：没有评测就没有“优化证明”
- 解决什么问题：回答效果、检索质量、引用质量不可量化
- 建议包含：
  - `eval/run_eval.py`
  - `eval/metrics.py`
  - `eval/compare_configs.py`
  - `eval/report_template.md`

#### `config/`
- 为什么需要它：当前 preset 都写在 `pipeline.py`，扩展困难
- 解决什么问题：实验配置不可复用，不利于 README 复现
- 建议包含：
  - `config/qwen_base.yaml`
  - `config/qwen_rerank.yaml`
  - `config/qwen_ser_rerank.yaml`

#### `demo_app/`
- 为什么需要它：CLI 展示感不够
- 解决什么问题：无法直观看到引用页与检索证据
- 建议包含：
  - `demo_app/streamlit_app.py`
  - `demo_app/components.py`

#### `tests/`
- 为什么需要它：现在没有关键流程回归保护
- 解决什么问题：comparative QA、引用页处理、metadata filter 容易改坏
- 建议包含：
  - `tests/test_comparative_questions.py`
  - `tests/test_reference_validation.py`
  - `tests/test_retrieval_fusion.py`

#### `src/retrieval_filters.py`
- 为什么需要它：metadata 逻辑不要继续塞进 `questions_processing.py`
- 解决什么问题：检索约束与业务字段耦合混乱
- 建议包含：
  - `build_filters(question, subset_row)`
  - `apply_filters(candidates, filters)`

#### `src/query_rewrite.py`
- 为什么需要它：金融问答术语表达不稳定
- 解决什么问题：自然语言问题与文档表达不一致
- 建议包含：
  - `rewrite_question()`
  - `expand_financial_terms()`
  - `normalize_currency_and_units()`

#### `src/citation_formatter.py`
- 为什么需要它：引用目前只有页码和 pdf_sha1
- 解决什么问题：证据展示不够强
- 建议包含：
  - `build_citations()`
  - `extract_evidence_snippets()`
  - `compute_confidence()`

### 5.2 建议重构的现有模块

#### `src/questions_processing.py`
- 重构目标：把“问题理解 / 检索编排 / 生成 / 后处理”拆开
- 推荐边界：
  - `QuestionRouter`
  - `AnswerPipeline`
  - `ReferencePostProcessor`
- 重构后的调用链路：
  - `process_question`
  - `route_question`
  - `build_query_plan`
  - `retrieve_candidates`
  - `rerank_candidates`
  - `generate_answer`
  - `validate_references`
  - `format_submission_answer`

#### `src/retrieval.py`
- 重构目标：统一单路检索、多路召回、metadata filter、parent page 展开
- 推荐边界：
  - `VectorRetriever`
  - `LexicalRetriever`
  - `HybridRetriever`
  - `RetrievalResult`
  - `RetrievalContextBuilder`
- 重构后的调用链路：
  - `retrieve(query, company, filters)`
  - `collect_backend_results`
  - `merge_results`
  - `expand_parent_context`
  - `return_top_k`

#### `src/text_splitter.py`
- 重构目标：从“字符切块器”升级成“结构感知 chunker”
- 推荐边界：
  - `PageChunker`
  - `TableChunker`
  - `ChunkMetadataBuilder`
- 重构后的调用链路：
  - `prepare_structural_blocks`
  - `split_long_blocks`
  - `attach_metadata`
  - `save_chunks`

### 5.3 建议补充的关键能力

#### 1. metadata 过滤
- 为什么对金融研报 RAG 重要：币种、年份、公司主体、报告类型决定答案合法性
- 代码层面怎么接入：在 `subset.csv` 元信息和 chunk 元信息中加入 `currency/year/company_name/report_type`
- 插入位置：`pdf_parsing.py` 写 metainfo，`text_splitter.py` 落到 chunk，`retrieval.py` 在召回后或召回前过滤

#### 2. 多路召回
- 为什么重要：财务指标既有术语精确匹配，也有语义表达变形
- 代码层面怎么接入：保留现有 vector + BM25 + sparse 结构，增加统一候选对象
- 插入位置：`HybridRetriever.retrieve_candidates_by_company_name`

#### 3. rerank
- 为什么重要：研报中相近表述很多，初召回噪声较高
- 代码层面怎么接入：沿用现有 `LLMReranker / FlagEmbeddingReranker`，但抽象为统一接口
- 插入位置：`HybridRetriever.retrieve_by_company_name`

#### 4. query rewrite
- 为什么重要：问题中的“Operating margin”可能在文档里写作“operating profit margin”
- 代码层面怎么接入：新增 `rewrite_question()`，输出 `original_query` 和 `expanded_queries`
- 插入位置：在 `process_question()` 进入检索前

#### 5. 引用溯源
- 为什么重要：金融问答必须给证据
- 代码层面怎么接入：检索结果中保留 `chunk_id/page/source/retrieval_sources`; 生成后根据引用页回填 snippet
- 插入位置：`get_answer_for_company()` 后处理阶段

#### 6. 结果置信度控制
- 为什么重要：金融指标误答比拒答更危险
- 代码层面怎么接入：基于 top-k score gap、引用数量、是否跨币种冲突给出 `confidence`
- 插入位置：`questions_processing.py` 最终格式化前

#### 7. 金融研报时间维度筛选
- 为什么重要：同公司多年份报告时必须先对齐时间
- 代码层面怎么接入：新增 year metadata，并允许 query parser 抽取年份
- 插入位置：`query_rewrite.py` + `retrieval_filters.py`

#### 8. 研报来源字段过滤
- 为什么重要：年报、季报、ESG 报告的指标覆盖不同
- 代码层面怎么接入：`subset.csv` 增加或映射 `report_type`
- 插入位置：metadata 建库阶段

#### 9. chunk 实验
- 为什么重要：这是最容易讲出“为什么我这样设计”的地方
- 代码层面怎么接入：增加 chunk mode 参数，支持 `page_only / structure_aware / structure_plus_table`
- 插入位置：`TextSplitter` 和 `config/*.yaml`

#### 10. benchmark / eval pipeline
- 为什么重要：能把优化讲成工程结果
- 代码层面怎么接入：读取 questions + gold / pseudo-gold answer，跑多配置并输出 markdown 表格
- 插入位置：新增 `eval/`

### 5.4 示例代码 / 伪代码 / 补丁建议

#### A. comparative QA 修复伪代码
文件建议：`src/questions_processing.py`
```python
def process_comparative_question(self, question: str, companies: list[str], schema: str) -> dict:
    rephrased_questions = self.api_processor.get_rephrased_questions(
        original_question=question,
        companies=companies,
    )

    individual_answers = {}
    aggregated_references = []

    def process_company(company: str):
        sub_question = rephrased_questions[company]
        return company, self.get_answer_for_company(
            company_name=company,
            question=sub_question,
            schema=schema,
        )

    with ThreadPoolExecutor(max_workers=min(len(companies), self.parallel_requests or 4)) as ex:
        for company, answer in ex.map(process_company, companies):
            individual_answers[company] = answer
            aggregated_references.extend(answer.get("references", []))

    comparative_answer = self.api_processor.get_answer_from_rag_context(
        question=question,
        rag_context=json.dumps(individual_answers, ensure_ascii=False),
        schema="comparative",
        model=self.answering_model,
    )
    comparative_answer["references"] = dedupe_refs(aggregated_references)
    return comparative_answer
```

#### B. metadata filter 接口设计
文件建议：`src/retrieval_filters.py`
```python
from dataclasses import dataclass

@dataclass
class RetrievalFilters:
    company_name: str | None = None
    currency: str | None = None
    year: int | None = None
    question_kind: str | None = None

def apply_filters(results: list[dict], filters: RetrievalFilters) -> list[dict]:
    filtered = results
    if filters.company_name:
        filtered = [r for r in filtered if r["metadata"].get("company_name") == filters.company_name]
    if filters.currency:
        filtered = [r for r in filtered if r["metadata"].get("currency") == filters.currency]
    if filters.year:
        filtered = [r for r in filtered if r["metadata"].get("year") == filters.year]
    return filtered
```

#### C. 检索流程重构伪代码
文件建议：`src/questions_processing.py`
```python
def answer_question(question_text: str, schema: str):
    routing = question_router.parse(question_text)
    rewritten_queries = query_rewriter.expand(question_text, schema=schema)

    candidates = []
    for q in rewritten_queries:
        candidates.extend(
            hybrid_retriever.retrieve_candidates(
                query=q,
                filters=routing.filters,
                top_k=top_k_candidates,
            )
        )

    merged = merge_and_dedupe(candidates)
    reranked = reranker.rerank(query=question_text, documents=merged) if use_rerank else merged[:top_k]
    context = context_builder.build(reranked)
    answer = generator.answer(question_text, context, schema=schema)
    return citation_formatter.attach(answer, reranked)
```

#### D. 引用返回格式设计
文件建议：`src/prompts.py`、`src/citation_formatter.py`
```json
{
  "final_answer": 18500342000,
  "reasoning_summary": "...",
  "relevant_pages": [78],
  "citations": [
    {
      "page": 78,
      "chunk_id": 12,
      "source": "194000c9109c6fa628f1fed33b44ae4c2b8365f4",
      "evidence_snippet": "Total assets ... 18,500,342",
      "score": 0.91
    }
  ],
  "confidence": "high"
}
```

#### E. eval 脚本框架
文件建议：`eval/run_eval.py`
```python
def main():
    configs = ["qwen_base", "qwen_rerank", "qwen_ser_rerank"]
    rows = []
    for cfg in configs:
        result = run_pipeline_eval(cfg)
        rows.append({
            "config": cfg,
            "answer_rate": result.answer_rate,
            "na_rate": result.na_rate,
            "retrieval_hit_at_5": result.hit_at_5,
            "citation_hit": result.citation_hit,
        })
    save_markdown_table(rows, "eval/results.md")
```

#### F. config 结构示例
文件建议：`config/qwen_ser_rerank.yaml`
```yaml
name: qwen_ser_rerank
retrieval:
  use_vector: true
  use_bm25: true
  use_sparse: false
  top_k: 6
  candidate_k: 16
  parent_document_retrieval: true
rerank:
  enabled: true
  backend: flag_embedding
generation:
  provider: qwen
  model: Qwen/Qwen2.5-72B-Instruct
  debug_reasoning: false
features:
  use_serialized_tables: true
  use_metadata_filters: true
```

#### G. README 新增章节模板
```md
## 业务背景与痛点
## 系统架构与数据流
## 金融场景特化设计
## RAG Pipeline 设计
## 配置对比与评测结果
## Demo 展示
## 项目亮点与踩坑总结
## 可直接写入简历的 Bullet
```

---

## 6. README 升级建议

### 1. 项目背景 / 业务痛点
- 为什么必须写：这是简历项目的价值起点
- 面试官能判断什么：你是不是理解场景，而不是只会拼框架
- 最好展示哪些内容：投研人员查询难点、表格密集、单位/币种/口径约束、跨公司对比需求

### 2. 系统架构图
- 为什么必须写：快速建立工程感
- 面试官能判断什么：你是否清楚模块边界
- 最好展示哪些内容：解析、切块、建库、召回、重排、生成、引用后处理

### 3. 数据流程
- 为什么必须写：说明从 PDF 到答案的中间产物
- 面试官能判断什么：你是否真的理解 pipeline
- 最好展示哪些内容：`parsed_reports -> merged_reports -> chunked_reports -> vector_dbs`

### 4. RAG 流程说明
- 为什么必须写：不能只列 feature
- 面试官能判断什么：你是否知道每步为什么存在
- 最好展示哪些内容：chunk 策略、混合检索、rerank、parent page、citation 校验

### 5. 金融场景优化
- 为什么必须写：这是区别于通用 PDF QA 的核心
- 面试官能判断什么：你是否真的做了场景特化
- 最好展示哪些内容：表格序列化、strict metric matching、币种/单位处理、时间维度计划

### 6. 评测方法与结果
- 为什么必须写：没有指标就没有说服力
- 面试官能判断什么：你是否具备实验意识
- 最好展示哪些内容：不同配置的 answer rate / hit@k / citation hit / case study

### 7. demo 展示
- 为什么必须写：增强“应用感”
- 面试官能判断什么：你是否把系统做到了可交互展示
- 最好展示哪些内容：输入问题、检索页、最终答案、引用页

### 8. 项目亮点
- 为什么必须写：帮助招聘者扫读
- 面试官能判断什么：你的亮点是否聚焦
- 最好展示哪些内容：3 到 5 条高价值特性，不要堆功能名词

### 9. 难点与解决方案
- 为什么必须写：这是面试高频追问入口
- 面试官能判断什么：你是否亲自解决过问题
- 最好展示哪些内容：复杂 PDF、表格语义化、长上下文、引用校验、并发/模型加载问题

### 10. 简历可写 bullet 提炼
- 为什么必须写：直接服务简历产出
- 面试官能判断什么：你是否知道项目价值点
- 最好展示哪些内容：2 到 4 条“动作 + 技术 + 结果”式 bullet

---

## 7. 面试官最容易质疑的问题

### 1. 你说支持多公司对比问答，具体怎么实现的？
- 为什么会问：这是 README 里最像亮点的 feature
- 目前证据不足：实现分支有未定义对象，可信度不足
- 应该怎么补足证据：修复代码、补 comparative demo 和测试

### 2. 你怎么证明表格序列化真的有用？
- 为什么会问：这是金融场景最像特化能力的点
- 目前证据不足：README 没有对比实验
- 应该怎么补足证据：做 `base vs ser_tab` 的 retrieval / answer 对比

### 3. 你为什么选择这种 chunk 策略？
- 为什么会问：RAG 项目最常见追问
- 目前证据不足：只有 `RecursiveCharacterTextSplitter`，没有结构策略说明
- 应该怎么补足证据：补结构感知 chunk 方案和实验

### 4. 你如何控制幻觉和错误引用？
- 为什么会问：金融问答是高风险场景
- 目前证据不足：目前只有页码校验，没有 chunk 级证据和置信度
- 应该怎么补足证据：补 citation schema、evidence snippet、置信度规则

### 5. 你如何评价 rerank 的收益？
- 为什么会问：README 把 rerank 写成核心能力
- 目前证据不足：没有任何指标
- 应该怎么补足证据：输出 rerank 前后命中率和最终答对率变化

### 6. 这个系统和普通 PDF QA 的区别是什么？
- 为什么会问：招聘者需要确认项目深度
- 目前证据不足：README 还没把金融特殊性讲透
- 应该怎么补足证据：补金融场景难点和对应设计

### 7. 如果一个问题不写公司名怎么办？
- 为什么会问：当前流程明显依赖公司识别
- 目前证据不足：实现上必须先抽公司名
- 应该怎么补足证据：增加全库检索或 company candidate generation 模式

### 8. 这个项目是工程项目还是比赛项目改造？
- 为什么会问：`setup.py` 仍是 `erc2`，数据目录也有 challenge 痕迹
- 目前证据不足：命名和包装没收尾
- 应该怎么补足证据：统一命名、重写 README、强调你做的改造和场景化部分

### 9. 你有做过失败样例分析吗？
- 为什么会问：真正做过 RAG 的人通常会讲错误类型
- 目前证据不足：没有 eval / error analysis 产物
- 应该怎么补足证据：输出错误样例表和阶段归因

---

## 8. 最终结论：它现在更像哪一类项目

**C. 有一定深度但包装不足的项目**

理由：
- 不是 A 或 B，因为仓库里确实有真实的 PDF 解析、表格序列化、向量/BM25/sparse 检索、rerank、结构化输出、并发处理和 CLI 流水线。
- 还不是 D，因为最关键的“工程闭环 + 指标证明 + README 叙事 + feature 可信度”没有做好，尤其是 comparative QA 分支存在代码级缺口，评测体系缺失，README 无法支撑强有力简历 bullet。
- 这类项目最适合做“增量强化”，不需要推倒重来，补齐评测、README、demo、metadata filter、citation 和少量重构后，竞争力会显著上升。

---

## 9. 可执行清单

- [高优先级] 修复多公司对比问答分支
- [高优先级] 要改哪些文件：`src/questions_processing.py`、`src/api_requests.py`、新增 `tests/test_comparative_questions.py`
- [高优先级] 做完后简历上能怎么写：实现跨公司对比问答路由，支持将比较问题拆解为子问题并汇总引用证据

- [高优先级] 新增最小评测框架，比较 `qwen_base / qwen_rerank / qwen_ser_rerank`
- [高优先级] 要改哪些文件：新增 `eval/run_eval.py`、`eval/metrics.py`、`eval/compare_configs.py`
- [高优先级] 做完后简历上能怎么写：搭建 RAG 评测闭环，量化验证混合检索、重排和表格序列化的收益

- [高优先级] 重写 README 前三屏，补业务痛点、架构图、核心链路
- [高优先级] 要改哪些文件：`README.md`
- [高优先级] 做完后简历上能怎么写：将金融研报 RAG 从技术 Demo 包装为面向投研场景的 AI 应用项目

- [高优先级] 在输出结果中新增 citation 结构和 confidence 字段
- [高优先级] 要改哪些文件：`src/prompts.py`、`src/questions_processing.py`、新增 `src/citation_formatter.py`
- [高优先级] 做完后简历上能怎么写：实现答案页码溯源、证据片段返回与低置信度控制，降低金融问答幻觉风险

- [高优先级] 将 `subset.csv` 的币种等元信息接入检索过滤
- [高优先级] 要改哪些文件：`src/pdf_parsing.py`、`src/text_splitter.py`、`src/retrieval.py`、新增 `src/retrieval_filters.py`
- [高优先级] 做完后简历上能怎么写：设计面向金融问答的 metadata filter，支持公司/币种等约束检索

- [中优先级] 升级 chunk 策略为结构感知切块
- [中优先级] 要改哪些文件：`src/text_splitter.py`、`src/parsed_reports_merging.py`
- [中优先级] 做完后简历上能怎么写：针对复杂年报页面、表格和脚注设计结构感知 chunk 策略，提升检索上下文完整性

- [中优先级] 抽离 YAML 配置层，支持实验复现
- [中优先级] 要改哪些文件：`src/pipeline.py`、新增 `config/*.yaml`
- [中优先级] 做完后简历上能怎么写：构建可配置化 RAG 实验框架，支持多检索/重排策略快速切换

- [中优先级] 增加 Demo 页面展示检索证据与最终答案
- [中优先级] 要改哪些文件：新增 `demo_app/streamlit_app.py`、`README.md`
- [中优先级] 做完后简历上能怎么写：实现可交互金融研报问答 Demo，支持答案、引用页和中间检索结果可视化

- [中优先级] 统一项目命名与打包信息
- [中优先级] 要改哪些文件：`setup.py`、`README.md`、`main.py`
- [中优先级] 做完后简历上能怎么写：完成项目工程化整理与规范化交付，提升仓库专业度与可复现性

- [中优先级] 增加 query rewrite / financial term normalization
- [中优先级] 要改哪些文件：新增 `src/query_rewrite.py`、`src/text_normalization.py`、接入 `src/questions_processing.py`
- [中优先级] 做完后简历上能怎么写：加入金融术语标准化与查询改写，提升指标类问题召回稳定性

## 假设与默认选择
- 默认以“面向大模型应用岗暑期实习的简历项目”作为目标，而不是竞赛刷榜方案。
- 默认优先做增量改造，不推翻现有 CLI、pipeline、Docling、FAISS、Qwen 接口。
- 默认把“提升简历价值”放在“追求最强 SOTA”之前，所以优先级偏向 eval、README、demo、citation、metadata filter，而不是直接上复杂训练方案。
- 默认允许保留现有 challenge 数据集结构，但需要弱化“比赛仓库”痕迹，强化“金融 AI 应用”叙事。
