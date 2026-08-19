# FinaWiki：FinaRAG 的结构化金融知识层

> 分支：`feature/llm-wiki-metric-atoms`
> FinaWiki = FinaRAG 的 LLM-Wiki 实现：把散落在代码常量与语料中的金融领域知识，
> 收敛为**结构化、可引用、可校验**的知识资产，供检索/生成/校验各环节精确调用。

## 1. 为什么需要知识层（从 FinaRAG 自身病灶推导）

FinaRAG 是纯检索驱动架构：领域知识要么硬编码在代码常量里，要么隐含在语料中，从未被当作**资产**管理。四个具体病灶：

| # | 病灶 | 代码位置 | 后果 |
|---|---|---|---|
| B1 | 术语知识住在代码里 | `_METRIC_PATTERNS`（11 指标 dict）、`_TERM_EXPANSIONS` | 新增指标须改源码；口径知识无处存放 |
| B2 | 口径歧义无处声明 | 全流程缺失 | 归母/扣非、合并/母公司混用；同一问题不同表述答案不一致 |
| B3 | 取证规则单指标过拟合 | `table_grounding.py` 全部规则以 `_is_revenue_question()` 门控 | 精调的行列过滤规则**只服务营业收入**，其余指标一条用不上 |
| B4 | 分析方法硬编码为字符串 | `_IMPLICIT_INTENT_PATTERNS`（5 组意图→证据词） | 方法论不可引用、不可校验、无判断依据 |

**选择判据**（决定什么知识值得进 FinaWiki，四条缺一不可）：
1. **错不起**：口径错=答案错；
2. **反复用**：多环节高频消费；
3. **相对稳**：不随文档集变化（区别于报告事实）；
4. **RAG 管不好**：短术语+口径歧义是向量检索的天然弱项。

据此**明确排除**：chunk 库（报告特定事实，量大易变，检索友好）、report_catalog 元数据（是数据不是知识）、编译式知识发布（个人项目语料规模不支撑，规模决定架构）。

## 2. 知识分层：对齐 RAG 流水线阶段

分层依据是**流水线的哪个环节需要它**——每层都能回答“谁消费、什么时候消费”：

| 流水线阶段 | 需要什么知识 | FinaWiki 层 | ID 前缀 | 状态 |
|---|---|---|---|---|
| ① 查询理解 | 指标标准名/别名/口径定义 | **指标原子** MetricAtom | `MET-` | ✅ Stage A |
| ② 取证规划 | 分析问题→必需指标+判断依据 | **分析剧本** AnalysisPlaybook | `PLB-` | ✅ Stage B |
| ③ 证据定位 | 指标在哪张表、行列特征、单位约束 | **取证规则** GroundingRule（MET 子结构） | `MET-*.grounding` | Stage C |
| ④ 答案校验 | 易混口径、常见误区 | **风险标注**（MET 字段：`confusable_with`/`pitfalls`） | `MET-*.risks` | ✅ Stage A |

两条结构性约束（知识层的完整性保证）：
- **剧本必须引用原子**：`PLB.required_metrics` 只能填合法 `MET-` ID，加载时校验——分析方法不得脱离指标事实基础（防“无据可依的方法论”）；
- **原子互斥性**：任一表述（standard_name/alias）不得同时属于两个原子；`confusable_with` 双向引用有效——防口径混并。

## 3. Stage A：指标原子层（已完成，commit 5638af1 / d985e26）

**知识资产**：`config/metric_atoms.yaml`，12 原子。

```yaml
- atom_id: MET-002
  standard_name: 归母净利润
  aliases: [归母净利润, 母公司股东净利润, 归属于母公司股东的净利润, 归母]
  definition: 净利润扣除少数股东损益后归属母公司股东的部分
  caliber_notes: 与净利润区分（后者含少数股东损益）；与扣非归母净利润区分
  unit: 元
  statement_location: 合并利润表
  confusable_with: [MET-001]
  pitfalls: 部分研报"净利润"默认指归母口径，需核对报告口径声明
  evidence_targets: [归母净利润, 归属于母公司股东的净利润, 合并利润表]
```

**加载层** `src/metric_atoms.py`：最长匹配索引（子串包含关系下长词优先，匹配正确性不依赖配置顺序）+ 加载时完整性校验 + 惰性单例。

**四个消费点**：
1. 查询理解：`_extract_metric` 原子库驱动；
2. 检索扩展：`_expand_financial_terms` 仅用同口径 aliases 生成扩展查询（**口径内扩展**）；
3. 生成注入：命中原子时把 definition/caliber_notes/易混提示注入 system prompt（未命中零 token 成本）；
4. 答案校验：`metric_evidence_mismatch` 依据原子 `evidence_targets`。

**顺带修复的三处缺陷**（资产化过程中发现，均有回归测试固化）：
- **口径放宽扩展**：`"营业收入": ["营收", "收入"]` 中裸“收入”会命中其他业务收入/投资收益 → 扩展词域收敛为原子 aliases；
- **口径混并**：研发投入 aliases 含“研发费用”（投入=费用化+资本化）→ 拆为两个独立原子；
- **顺序敏感匹配**：扣非归母/归母的命中靠 dict 插入顺序保证 → 最长匹配替代。

**评测** `eval/metric_consistency/`：38 变体（26 数值 + 12 定义），Level1 四项指标 **100%**（意图命中率/原子挂载率/零口径污染/定义路由命中率）。

**已实现的知识消费增强**：
- 定义类问题分层路由（`definition_qa` → 原子直接回答，不走检索，fail-closed 回落 RAG）；
- 取证报表先验（`statement_location` 匹配的表格 chunk 加分）；
- 原子扩容链路（`metric_atom_mining.py` 规则校验器 + CLI：口径放宽词拒绝/口径重叠检测/字段完整性）。

## 4. Stage B：分析剧本层（已完成）

**解决 B4**：`_IMPLICIT_INTENT_PATTERNS` 把“评估流动性压力要看哪些指标”硬编码成裸字符串列表——方法不可引用、无判断依据、拿不到口径注入。

**知识资产**：`config/analysis_playbooks.yaml`（5 剧本）

```yaml
- playbook_id: PLB-002
  intent: liquidity_pressure
  question_forms: [现金压力, 流动性压力, 偿债压力, 短期偿债, 资金压力]
  required_metrics: [MET-003, MET-005, MET-006, MET-011]   # 校验引用合法性
  judgment: 货币资金/短期借款 < 1 提示压力；经营现金流连续为负加重；资产负债率高于行业中位加重
  evidence_keywords: [偿债能力]        # 非指标类补充检索词
  pitfalls: 货币资金含受限资金时会高估偿债能力
```

**实现要点**：
1. `src/analysis_playbooks.py`：`AnalysisPlaybook` + `PlaybookStore`（加载校验 `required_metrics` 全部可解析，否则启动即失败）；
2. `query_rewrite._extract_implicit_intent` 改剧本驱动；`evidence_targets` 由 `required_metrics` 展开为各原子的 evidence_targets 并集（**知识引用替代字符串复制**）；
3. `QueryPlan` 新增 `playbook` 字段（playbook_id/judgment/涉及原子），隐含意图问答由此获得**口径注入能力**（当前仅单指标查询有）；
4. 生成注入：把 `judgment` 作为判断依据提示（“请依据以下判断标准作答，缺失指标须说明”），把各原子 caliber_notes 一并注入；
5. 校验升级：`implicit_intent_evidence_mismatch` 按 `required_metrics` 逐项检查覆盖度，缺项在答案中显式声明（而非静默给结论）。

**新增测试**：剧本加载校验（含非法 MET 引用应报错）、意图→剧本命中、evidence_targets 展开正确性、QueryPlan.playbook 挂载、口径注入含 judgment。

**评测扩展**：变体集加入 5 类意图的表述变体（每类 3 种问法），指标为**意图命中率**与**required_metrics 覆盖率**。

**实测结果**：42/42 相关测试全绿（新增 10：剧本加载/引用校验/双粒度展开/覆盖度/注入/挂载）；
现有隐含意图测试**未改任何断言**即通过（行为兼容性验收线）；变体集 38→53，六项指标均 **100%**：

| 指标 | 结果 |
|---|---|
| 意图命中率（26 数值变体） | 100% |
| 原子挂载率 | 100% |
| 口径污染违规 | 0 |
| 定义类路由命中率（12 变体） | 100% |
| **剧本意图命中率（15 变体）** | **100%** |
| **剧本必需指标挂载率** | **100%** |

## 5. Stage C：取证规则原子化 + 拒答门禁（1.5 天）

**解决 B3**：`table_grounding.py` 的 `_revenue_row_score`/`_table_scope_score`/`_reject_revenue_unit` 全部以 `_is_revenue_question()` 为前置门控——精调的规则集只服务一个指标，是 benchmark 驱动的特调而非可扩展结构。

**知识资产**：MetricAtom 增加 `grounding` 子结构

```yaml
  grounding:
    preferred_statements: [合并利润表, 主要财务指标]
    good_row_headers: [营业收入, 一、营业收入]
    bad_row_headers: [占营业收入比例, 研发投入占营业收入]
    bad_col_headers: [同比, 变动, 占比, 分部收入, 关联方]
    segment_guard: true      # 分部/地区/产品列须问题明确提及才允许
    unit_kind: currency      # currency|percent，用于拒绝错误单位单元格
```

**实现要点**：
1. `_revenue_row_score` 泛化为 `_metric_row_score(atom, row_text, col_text)`：营收现有规则作为 `MET-004.grounding` 的第一个实例迁移进去，其余指标按同结构补齐；
2. `_reject_revenue_unit` 泛化为 `unit_kind` 驱动的单位守卫；
3. 保留兜底：无 `grounding` 配置的原子走通用 token 重叠打分（行为不回退）。

**拒答门禁（fail-closed）**：`answer_validation` 当前对口径不符只**降 confidence**；升级为——数值类答案的证据完全不含原子 `evidence_targets`、且无结构化 grounding 支撑时，**强制 `final_answer = N/A`** 并标注原因。宁可拒答，不可给错口径数字。

**验收**：number 类问题准确率不回退（迁移正确性）；非营收指标的 grounding 命中率提升；新增拒答门禁的误拒率可控（人工抽检 20 例）。

## 6. 命名与边界

- **命名自解释**：`MET-xxx`（Metric）、`PLB-xxx`（Playbook）、`MET-*.grounding`——ID 在金融 RAG 语境下直接可读，不依赖任何外部编码体系；
- **不做的三件事及理由**：编译式知识发布（规模不支撑）、chunk 库知识化（RAG 的地盘）、操作步骤层（当前非 Agent 架构，无消费方；架构演进后再评估）；
- **知识准确性**：冷启动原子全部取自会计准则标准定义；扩容走“LLM 提炼→规则质检→人工校准→入库”，质检与提炼使用不同模型（**生产者不自批**）。

## 7. 里程碑

| 阶段 | 交付 | 状态 |
|---|---|---|
| Stage A | 指标原子层 + 四消费点 + 三缺陷修复 + 一致性评测 100% | ✅ 已完成 |
| Stage B | 分析剧本层 + 剧本→原子引用约束 + 意图口径注入 + 覆盖度校验 | ✅ 已完成 |
| Stage C | 取证规则原子化（治过拟合）+ fail-closed 拒答门禁 | 下一步 |
| 待环境 | Level2 端到端口径一致率/答案一致率；原子扩容 12→50+ | 需 API key + 数据集 |
