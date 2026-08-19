# FinaRAG × LLM-Wiki：金融指标口径原子库改造方案

> 分支：`feature/llm-wiki-metric-atoms`（Phase 1/2/3 均已实施，见文末实施结果）
> 方法论来源：XX公司实习（四K全书知识工程化改造）→ 淘天（工业级知识治理）→ 本方案（方法论迁移到个人项目）
> 核心判断：**LLM-Wiki 不替代 RAG，只接管 RAG 管不好的那部分知识**——判据：错不起（口径唯一）、反复用（高频）、相对稳（不随文档集变）、RAG 管不好（语义近似命中不一致版本）。

## 0. 背景与已实锤的问题

现状：金融指标领域知识散落在 `query_rewrite.py` 的三个代码结构里——`_METRIC_PATTERNS`（11 个指标 dict）、`_TERM_EXPANSIONS`（中英术语替换表）、SFT 白名单（6 个指标）。新增加指标要改源码；且代码审查已发现两处口径缺陷：

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| D1 | 口径放宽扩展：`"营业收入": ["营收", "收入"]` 中的 **"收入"** | `_TERM_EXPANSIONS` L110 | 问营业收入生成"XX公司收入"扩展查询，可能命中其他业务收入/投资收益 chunk，答案口径被污染 |
| D2 | 口径混并：研发投入的 aliases 含 **"研发费用"** | `_METRIC_PATTERNS` L69 | 研发费用（费用化）问题命中研发投入（含资本化）指标，evidence_targets 错位 |
| D3 | 顺序敏感匹配：dict 插入顺序保证"扣非归母净利润"先于"归母净利润"命中（两者是子串包含关系） | `_extract_metric` L534-539 | 重排 dict 即引入静默 bug，脆弱设计 |

这三个缺陷就是本改造的"P0 故事"（对标 XX公司评测发现的兜底条款缺陷——每个改造都应有非做不可的实锤）。

## 1. 目标与非目标

**目标**：
1. 指标知识资产化：从代码内 dict 迁移到独立 YAML 原子库（K1 事实层 + K4 对抗字段）；
2. 修复 D1/D2/D3；
3. 一库四用：意图解析、检索扩展、生成注入、答案校验；
4. 建立"口径一致性"评测闭环，拿到改造前后量化对比。

**非目标**：
- 不整体 LLM-Wiki 化 chunk 库（报告特定事实是 RAG 的地盘，量大、随文档变、检索友好）；
- 不做编译式 Generation（个人项目语料规模不支持，规模化才合理——规模决定架构）；
- 不做 K2 方法层原子（FinaRAG 是 QA 场景，方法注入场景少；XX公司的分析报告场景才需要）。

## 2. 总体架构

```
config/metric_atoms.yaml（知识资产：12 原子起步）
        │ load
        ▼
src/metric_atoms.py::MetricAtomStore（最长匹配索引 + 口径内扩展词）
        │
        ├─① 意图解析   query_rewrite._extract_metric      （Phase 1）
        ├─② 检索扩展   query_rewrite._expand_financial_terms（Phase 1）
        ├─③ 生成注入   questions_processing / prompts      （Phase 2）
        ├─④ 答案校验   answer_validation                   （Phase 2）
        └─⑤ 评测层     eval/run_metric_consistency.py     （Phase 2）
```

原子结构（已定稿，`config/metric_atoms.yaml` 已建）：

| 字段 | 层 | 用途 |
|---|---|---|
| atom_id / standard_name / version | K0 | 索引与版本管理 |
| aliases | K1 | 匹配与口径内扩展（**禁止口径放宽词**） |
| definition / unit / statement_location | K1 | 生成注入（Phase 2） |
| caliber_notes / confusable_with / pitfalls | K4 | 口径消歧提示（Phase 2/3） |
| evidence_targets | K1 | 意图解析与校验依据（Phase 1/2） |

---

## Phase 1：知识资产化与口径修复（检索侧）

**目标**：原子库上线，替换 `_METRIC_PATTERNS`；修复 D1/D2/D3；现有行为完全兼容。

**交付物（文件级）**：

1. `config/metric_atoms.yaml` ✅ 已完成——12 原子（原 11 个 + D2 修复拆分出的"研发费用"独立原子 K1-MET-012；"研发投入"aliases 移除"研发费用"；"经营活动现金流量净额"aliases 补充"经营活动现金流净额"变体保持扩展行为兼容）。

2. `src/metric_atoms.py`（新建，~120 行）：
   - `MetricAtom`（frozen dataclass）：字段同 YAML；`match_terms` 属性 = [standard_name, *aliases] 去重；
   - `MetricAtomStore`：
     - `load(path)`：YAML → 原子列表；启动时校验（atom_id 唯一、standard_name/aliases/definition 非空、confusable_with 引用的 atom_id 存在）；
     - **最长匹配索引**：所有 match_terms 按 `len(normalize_text(term))` 降序排列，线性扫描取首个命中——解决 D3，匹配正确性不再依赖配置顺序；
     - `find_metric(normalized_question) -> Optional[MetricAtom]`；
     - `find_metric_with_term(normalized_question) -> Optional[(atom, matched_term)]`（供扩展用）；
     - `get(atom_id)` / `all_atoms()` / `__len__`；
   - `get_default_store()`：模块级惰性单例 + `reset_default_store()`（测试隔离用）。

3. `src/query_rewrite.py` 改造（4 处，方法级）：
   - **删除** `_METRIC_PATTERNS`（L35-80）；
   - `_extract_metric`：改为 `get_default_store().find_metric()`，返回签名不变 `(metric_name, evidence_targets)`——`_parse_intent` 及其下游零改动；
   - **删除** `_TERM_EXPANSIONS` 中被原子覆盖的中文指标条目（营业收入/归母净利润/扣非归母净利润/毛利率/经营活动现金流净额），保留英文条目与非指标术语（净利率/期间费用率/分红/回购/并购/定增——Phase 3 蒸馏时再收编）；
   - `_expand_financial_terms`：头部新增原子驱动块——`find_metric_with_term` 命中后，在原文中定位实际出现的口径内表述（standard_name 或某 alias），替换为其余 aliases 生成扩展查询；**D1 修复**：扩展词域从硬编码表变为原子 aliases（营业收入只扩展出"营收"/"主营业务收入"，永不出"收入"）。

4. `tests/test_metric_atoms.py`（新建，用例级）：
   - `test_load_default_atoms`：≥12 原子、atom_id 唯一、必填字段非空；
   - `test_longest_match_wins`（D3 回归）："扣非归母净利润是多少" 命中扣非原子而非归母原子；
   - `test_alias_matching`："归母"、"归属于母公司股东的净利润"、"经营活动现金流净额"（无"量"字变体）各自命中正确原子；
   - `test_caliber_separation`（D2 回归）："研发费用" 命中 K1-MET-012、"研发投入" 命中 K1-MET-009，互不误触；
   - `test_confusable_references_valid`：confusable_with 引用的 atom_id 均存在；
   - `test_no_caliber_pollution_in_aliases`：任何原子的 aliases 不含"收入"这类口径放宽词（语义断言：alias 要么含完整指标限定词，要么在标准词表内）。

5. `tests/test_query_rewrite.py`（回归 + 新增）：
   - 现有 7 个测试**不改断言**全通过（行为兼容的验收线——已逐一核对：L19 归母扩展、L88 营收 metric、L94 主营业务收入 evidence 均可由原子满足）；
   - 新增 `test_no_caliber_pollution_expansion`（D1 回归）：`rewrite("SH600000在2024年年报里的营业收入是多少？")` 的 search_queries 中**不存在**"…收入是多少"（裸"收入"替换产物），且存在"主营业务收入"扩展；
   - 新增 `test_metric_atom_intent_from_store`：营收 top_k 问题的 `plan.metric == "营业收入"`（原子驱动意图解析）。

**关键设计决策**：
- 最长匹配而非最大别名优先排序：O(terms) 线性扫描，12 原子 ~40 词规模无性能问题，语义正确性优先；
- 行为兼容策略：凡是现有测试断言覆盖的行为，原子库必须原样复现（aliases 补充变体），差异点（D1/D2 修复）单独写回归测试固化。

**验收标准**：`pytest tests/test_metric_atoms.py tests/test_query_rewrite.py tests/test_answer_validation.py -v` 全绿；D1/D2/D3 各有对应回归测试。

**工作量**：1 天。**提交**：单个 commit（feat: metric atom store + 口径修复）。

---

## Phase 2：生成侧口径注入与校验升级（拿到量化数据）

**目标**：原子知识进入生成与校验环节；建立口径一致性评测，产出改造前后对比数据（面试的核心弹药）。

**交付物**：

1. `src/query_plan.py`：`QueryPlan` 新增 `metric_atom: Optional[Dict]`（携带 atom_id/definition/caliber_notes/confusable_names——用 dict 而非对象引用，保持 QueryPlan 可序列化惯例）；`query_rewrite.rewrite` 在 `_parse_intent` 命中 metric 时填充。

2. `src/questions_processing.py` + `src/prompts.py`：生成阶段口径注入——
   - 注入内容（刻意克制，只注口径冲突相关）：`definition`（一句话）+ `caliber_notes`（易混口径提示）+ confusable 原子的 standard_name；
   - 注入形式：prompt 模板新增可选段"指标口径说明"（如"【口径提示】归母净利润=净利润-少数股东损益；注意与扣非归母净利润区分，若证据为扣非口径需在答案中说明"）；
   - 未命中原子的查询：不注入，token 成本为零（按需注入原则）。

3. `src/answer_validation.py`：`metric_evidence_mismatch` 检查的指标-证据对照改由原子库 `evidence_targets` 驱动（消除最后一个硬编码知识点）。

4. `eval/metric_consistency/`（新建目录）：
   - `variants.jsonl`：口径一致性测试集——每个原子用 aliases 自动生成变体查询对（如"宁德时代2024归母净利润"/"宁德时代2024归属于母公司股东的净利润"），约 12 原子 × 2-3 变体 ≈ 30-36 条；
   - `run_metric_consistency.py`：跑变体查询对，计算三个指标——
     - **意图命中率**：变体查询的 QueryPlan.metric 是否命中同一原子；
     - **口径一致率**：答案 citation 指向的 chunk 是否含该原子 evidence_targets（口径正确的证据）；
     - **答案一致率**：同原子变体对的数值答案是否一致；
   - 输出改造前后对比表（main 分支跑 baseline vs 本分支跑 treatment）。

**验收标准**：
- 现有 benchmark（answer rate / citation coverage）不回退；
- 口径一致率、答案一致率拿到前后对比数字（哪怕基线已经不错，"设计评测→量化验证"本身就是评测能力展示）。

**工作量**：1 天。**提交**：`feat: caliber injection + metric consistency eval`。

---

## Phase 3：知识生产规模化与消费深化（演进，按需做）

> 定位：Phase 1/2 是"闭环最小可行"，Phase 3 是深挖方向——每个子项独立可停，优先级按面试价值排序。

**3a. 蒸馏扩原子（知识生产规模化）** ⭐ 优先级最高
- `scripts/mine_metric_atoms.py`：从年报语料/会计准则文本用强模型生成原子草稿（结构化 JSON：候选指标/定义/口径/别名），规则校验（别名冲突检测、与现有原子口径重叠检测）→ 人工校准 → 合入 YAML；
- 目标：12 → 50+ 原子；与 SFT 优化方案 S1（指标挖掘 6→50+）**产物同源**——蒸馏一次，训练数据和在线知识库双受益；
- 方法论价值：完整复刻"多源提炼→规则质检→人工校准→入库"生产链路（XX公司链路的个人项目版）。

**3b. Table Grounding 知识驱动**
- `statement_location` 先验进入 `TableGrounder` 评分：chunk 的 evidence_type/来源报表与原子 statement_location 匹配时加分（如归母净利润问题优先命中合并利润表的表格 chunk）；
- 从硬编码权重（行首×2.2）走向"启发式+知识先验"混合。

**3c. 分层路由：定义类问题直达原子**
- `_parse_intent` 新增 `definition_qa` 任务类型（触发词："是什么/什么意思/和XX的区别/口径"）；
- 路由到原子直接回答（definition + caliber_notes + confusable 区分 + 原子 ID 引用），**不走检索**——"结构化优先、语义兜底"分层路由的最小实现；
- fail-closed：未命中原子回落 general_rag。

**3d. 消歧提示（K4 消费深化）**
- 查询同时命中多个 confusable 原子（如"扣非后的净利润"歧义）时，prompt 注入消歧分支："若证据为扣非口径回答扣非归母净利润，否则回答归母净利润并标注口径"。

**工作量**：3a 约 1.5 天，3b/3c/3d 各 0.5 天。**验收**：3a 原子数 50+ 且校验通过；3c 定义类问题回答含原子 ID 引用且不触发检索。

---

## 风险与对策

| 风险 | 对策 |
|---|---|
| 行为兼容性破坏（下游依赖 _METRIC_PATTERNS） | grep 确认引用面（仅 _extract_metric）；接口签名不变；现有 7 测试全绿为验收线 |
| prompt 注入推高 token 成本/干扰生成 | 只注入口径冲突字段；未命中原子的查询零成本；benchmark 不回退为验收线 |
| 口径一致性评测需要可跑的 pipeline 数据集 | 变体集设计为现有数据集公司×年份的组合；跑不动时退化为"意图命中率+扩展查询检查"两级评测（不依赖端到端） |
| 原子库知识准确性 | 冷启动原子全部来自会计准则标准定义；confusable_with 交叉校验；人工校准入库前把关 |

## 里程碑

| 节点 | 内容 | 面试叙事增量 |
|---|---|---|
| M1（Phase 1）✅ | 原子库 + D1/D2/D3 修复 + 回归测试 | "迁移顺手修了三个隐藏口径缺陷"——对标兑底条款故事 |
| M2（Phase 2）✅ | 口径注入 + 分层路由 + Level1 评测 100% | "设计评测→量化验证"闭环 + 数字 |
| M3（Phase 3）✅ | 蒸馏链路 + grounding 先验 + definition 路由 | 知识生产规模化链路就绪，三段经历矩阵中间格坐实 |

## 实施结果（2026-08-19）

**Phase 1（commit 5638af1）**：12 原子入库、MetricAtomStore 最长匹配、D1/D2/D3 修复，11 个新增回归测试。

**Phase 2**：
- `QueryPlan.metric_atom` 原子挂载（definition/caliber_notes/confusable_names）；
- 生成侧口径注入：`get_answer_from_rag_context(metric_caliber_hint=...)`，命中原子才注入（未命中零 token 成本）；
- 校验侧：`metric_evidence_mismatch` 已随 Phase 1 由原子库驱动（读 QueryPlan.evidence_targets）；
- 评测：`eval/metric_consistency/`（gen_variants 38 变体 + Level1 评测）——

| 指标 | 结果 |
|---|---|
| 意图命中率（26 数值变体） | **100%** |
| 原子挂载率 | **100%** |
| 口径污染违规 | **0**（旧实现会产生"…收入是多少"污染查询） |
| 定义类分层路由命中率（12 变体） | **100%** |

**Phase 3**：
- 3a 蒸馏链路：`src/metric_atom_mining.py`（规则校验器：口径放宽词拒绝/重叠检测/字段校验）+ `scripts/mine_metric_atoms.py` CLI（需 API key）；
- 3b grounding 先验：`_metric_statement_prior_score`（statement_location 强命中+2.5/弱命中+1.5）；
- 3c 分层路由：definition_qa → metric_atom_store，原子直接回答（含 fail-closed 回落）；
- 3d 消歧提示：融入口径注入文本（confusable 分支指令）。

**测试**：32/32 相关测试全绿；全量 268 测试中 6 个失败为历史遗留环境依赖（多 GPU/vLLM/量化/数据集，stash 验证与本次改动无关）。

**待办（需环境）**：Level2 端到端口径一致率/答案一致率（API key + 数据集）；蒸馏扩原子 12→50+（跑 mine_metric_atoms.py + 人工校准）。
