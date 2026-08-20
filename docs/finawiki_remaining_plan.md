# FinaWiki 剩余阶段实施计划（Stage C / D / E / F）

> 本文是 [`finawiki_design.md`](./finawiki_design.md) 的**实施细化**：design 讲分层依据与"为什么"，本文讲"改哪个文件、哪个方法、加哪些测试、凭什么判定通过"。
> 设计依据冲突时以 design 为准。阶段完成后在本文对应章节标注实测结果，不删除原计划内容（便于对照计划与实际的偏差）。

## 0. 前置声明：验收环境约束

当前环境**无 API key、无完整数据集**，因此：

- **可验收**：离线单测、Level1 评测（意图/挂载/污染/路由/剧本）、行为等价断言；
- **不可验收**：number 类问题端到端准确率、Level2 口径一致率/答案一致率。

这直接影响 Stage C 的价值判定——见 3.4。**任何不可验收的收益都不得表述为已验证**，只能表述为结构性改进。

## 1. 阶段总览与顺序依据

| 顺序 | 阶段 | 内容 | 风险 | 状态 |
|---|---|---|---|---|
| 1 | **Stage D** | flag 命名空间契约（修两个已存在缺陷） | 零（纯增量） | ✅ 完成 `ed06dca` |
| 2 | **Stage C** | 取证规则原子化 + fail-closed 门禁 | **高**（改打分主路径） | ✅ 完成 `a41b035` `9dcd4a6` |
| 3 | **Stage E** | 原子对抗质检（LLM 多维打分） | 低 | ✅ 完成 `94587aa` |
| 4 | **Stage F** | 原子版本化 | 低 | ✅ 完成 `ddad281` |

**顺序依据**：D 独立且零风险，先做立即交付；C 风险最高、需最完整注意力，放第二位；E/F 属治理类且都要覆盖 C 新增的 `grounding` 字段，必须在 C 之后。

---

## 2. Stage D：flag 命名空间契约（0.5 天）

### 2.1 病灶：开工前核实，找到两个**已存在**缺陷

> 修正记录（两轮）：①早期判断为“三处各自硬编码词表”——不成立，打标侧无词表（读元数据布尔字段）；
> ②随后判断为“契约漂移风险”——低估了，核实后发现是两个**已经发生**的缺陷，不是风险。

**缺陷 D-1：命名空间污染（必然发生）**

- `src/pdfcrawl_dataset.py:116` 向 metainfo 写入 `"has_pdfcrawl_metadata": "true"`（内部技术字段）；
- `src/text_splitter.py:148 _report_topic_flags` 无条件收集所有 `has_`/`mentions_` 前缀且值为 `"true"` 的键；
- → **凡走 pdfcrawl 数据集的报告，其全部 chunk 的 `topic_flags` 都多一个假 flag `has_pdfcrawl_metadata`**。

影响：不致检索错误（查询侧词表无此 key，不会被要求匹配），但污染知识字段，使基于 topic_flags 的统计/打分失真。

**缺陷 D-2：能力缺口（已存在）**

数据集（`data/*/subset.csv`，多个数据集取并集稳定）共 **17 个** topic flag 字段，而 `query_rewrite._TOPIC_PATTERNS` 只覆盖 **6 个**。
缺失的 11 个（裁员/高管薪酬/研发数据/新品发布/资本开支/财务绩效指标/资本结构变动/新风险因素/战略重组/供应链中断/ESG）
——数据集已打标，但**查询侧永远无法通过 topic_flag 路由命中**。这不是漂移风险，是现存能力缺口。

**第三个（仍属风险）**：词表 key 拼写错了也不报错，静默失败。

### 2.2 改动

**新增** `config/topic_flag_contract.yaml`：命名空间单一来源

```yaml
version: "1.0"
# 数据集合法 topic flag 全集（取自 data/*/subset.csv 表头并集，17 项）
flags:
  - {name: mentions_recent_mergers_and_acquisitions, query_side: true}
  - {name: has_layoffs, query_side: false}   # 未覆盖：待补词表
  # ...
# 内部技术字段：前缀碰巧但**不得**当作 topic flag（修 D-1）
reserved_non_flags: [has_pdfcrawl_metadata, has_table_context, has_chart_context, has_literal_evidence]
```

**新增** `src/topic_flag_contract.py`：

| 方法 | 签名 | 职责 |
|---|---|---|
| `load_flag_contract` | `(path=None) -> FlagContract` | 加载（flags / query_side_flags / reserved） |
| `validate_topic_patterns` | `(patterns) -> List[str]` | 词表 key 不在契约内 → 违规（拼写错即报） |
| `assert_topic_patterns_valid` | `(patterns) -> None` | 违规则 raise（import 期调用） |
| `filter_report_topic_flags` | `(report_meta) -> List[str]` | **修 D-1**：只收契约内 flag，reserved 字段排除 |
| `coverage_report` | `(patterns) -> Dict` | **量化 D-2**：覆盖率 + 缺口清单 |

**改动** `src/query_rewrite.py`：模块尾调用 `assert_topic_patterns_valid(_TOPIC_PATTERNS)`（import 期暴露）。

**改动** `src/text_splitter.py::_report_topic_flags`：改为委派 `filter_report_topic_flags`（修 D-1）。

> D-2 的词表补全不在本阶段做（需为 11 类逐个设计中英文问法，属知识扩容）；
> 本阶段只做到**把缺口从不可见变为可度量**（`coverage_report` + 测试输出）。

### 2.3 测试（`tests/test_topic_flag_contract.py`，7 例）

| 用例 | 断言 |
|---|---|
| `test_contract_covers_all_topic_patterns` | `_TOPIC_PATTERNS` 全部 key 在契约内（回归锁） |
| `test_contract_matches_dataset_columns` | 契约 flags 与 `data/*/subset.csv` 表头一致（数据集变了即失败） |
| `test_reserved_field_not_treated_as_flag` | **D-1 回归**：`{"has_pdfcrawl_metadata": "true"}` 不得进入 topic_flags |
| `test_legit_flag_still_collected` | 契约内 flag 且值 true 仍正常收集 |
| `test_unknown_flag_rejected` | 词表含未登记 key → 违规 |
| `test_assert_raises_on_violation` | 非法输入抛 `ValueError` |
| `test_coverage_report` | **D-2 量化**：覆盖 6/17，缺口清单含 `has_layoffs` |

### 2.4 验收

- 7/7 新增测试通过；全量测试无新增失败；
- 故障注入：临时给 `_TOPIC_PATTERNS` 加未登记 key，`import src.query_rewrite` 必须失败；
- D-1 修复证据：包含内部字段的 report_meta 输入下，topic_flags 不再含 `has_pdfcrawl_metadata`。

### 2.5 实测结果（已完成）

**12/12 通过**（比计划 7 例更全：补了契约自相矛盾、空 flags、TextSplitter 端到端、单例重置）；
全量 290 测试（+12），失败数维持历史遗留的 6 个环境依赖，无新增失败。

| 验收项 | 结果 |
|---|---|
| D-1 修复 | `{"has_pdfcrawl_metadata": "true"}` 不再进入 topic_flags（单元 + 端到端双重验证） |
| D-2 量化 | 覆盖 6/17 = **35.3%**，缺口 11 项已列入 `coverage_report().missing` |
| 契约与数据集一致 | 契约 flags == `data/*/subset.csv` 表头并集（数据集变则测试失败） |
| 故障注入 | 未登记 key → `ValueError: has_forgotten_flag: not declared in ...` |
| 声明与实现一致 | `query_side: true` 的 6 项均有词表（`declared_query_side_without_patterns` 为空） |

### 2.5 实测结果（已完成）

**12/12 通过**（比计划 7 例更全：补了契约自相矛盾、空 flags、text_splitter 端到端、单例重置）；
全量 290 测试（+12），失败数维持历史遗留的 6 个环境依赖，无新增失败。

| 验收项 | 结果 |
|---|---|
| D-1 修复 | `{"has_pdfcrawl_metadata": "true"}` 不再进入 topic_flags（单元 + TextSplitter 端到端双重验证） |
| D-2 量化 | 覆盖 6/17 = **35.3%**，缺口 11 项已列入 `coverage_report().missing` |
| 契约与数据集一致 | 契约 flags == `data/*/subset.csv` 表头并集（数据集变则测试失败） |
| 故障注入 | 未登记 key → `ValueError: has_forgotten_flag: not declared in ...` |
| 声明与实现一致 | `query_side: true` 的 6 项均有词表（`declared_query_side_without_patterns` 为空） |

---

## 3. Stage C：取证规则原子化 + fail-closed 门禁（1.5 天）

### 3.1 病灶精确定位

`src/table_grounding.py` 的四个函数全部以 `_is_revenue_question()`（L121，判据仅 `"营业收入" in q or "营收" in q`）为前置门控，非营收问题直接返回 0/False：

| 函数 | 行 | 门控行为 |
|---|---|---|
| `_revenue_row_score` | L222-268 | 非营收 → `return 0.0`（15 条规则全部跳过） |
| `_table_scope_score` | L157 | 非营收 → `return 0.0` |
| `_reject_revenue_unit` | L148 | 非营收 → `return False` |
| 页面/值级后处理 | L511、L516、L597 | 非营收 → 跳过 |

**关键发现：`_revenue_row_score` 的 15 条规则里有 9 条与指标无关**——它们是通用取证守卫，却被囚禁在营收专用函数中，其余 11 个指标一条都用不上：

| 原规则 | 性质 | 迁移归属 |
|---|---|---|
| 行列不含指标词 → `None` | 指标专属 | `atom.grounding.match_terms` |
| `_BAD_REVENUE_ROW_TERMS` → `None` | 指标专属 | `atom.grounding.bad_row_headers` |
| `_BAD_REVENUE_COL_TERMS` → `None` | **混合** | 同比/变动/比例/占比/百分比/期末余额/期初 → **通用**；分部收入/关联方/数量合计 → 指标专属 |
| "增减" 且无年份列 → `None` | **通用** | 通用守卫：变动列 |
| `_BAD_SEGMENT_COL_TERMS` 且无"合计" → `None` | **通用** | 通用守卫：分部列（`atom.grounding.segment_guard` 开关） |
| "季度" 列但问题无"季度" → `None` | **通用** | 通用守卫：期间粒度不匹配 |
| 问题无 `%` 但列有 `%`/百分点 → `None` | **通用** | 通用守卫：由 `atom.unit` 驱动（货币类拒 %） |
| "母公司"/"本行" 不匹配 → `None` | **通用** | 通用守卫：主体范围 |
| 分部/地区/产品/关联方/客户 不匹配 → `None` | **通用** | 通用守卫：口径维度 |
| 精确行首 → `8.0` | 指标专属 | `atom.grounding.exact_row_headers` |
| 次优行首 → `7.5` | 指标专属 | `atom.grounding.secondary_row_headers` |
| 列含"本集团"/"合计" → `+2.0` | **通用** | 通用守卫：合并口径加成 |
| `_LOW_PRIORITY_REVENUE_TERMS` → `1.0` | 指标专属 | `atom.grounding.low_priority_row_headers` |
| `_CONSOLIDATED_REVENUE_TERMS` → `4.0` | 指标专属 | `atom.grounding.weak_row_headers` |
| 兜底 → `1.5` | **通用** | 通用默认分 |

即 **9 通用 + 6 专属**。原方案表述"精调规则其他指标用不上"不够准确，准确形态是：**通用守卫与指标专属词表耦合在同一函数内，被同一个门控一并关闭**。

### 3.2 知识结构：`MetricAtom.grounding` 子结构

`config/metric_atoms.yaml` 中每个原子可选新增（缺省时仅通用守卫生效）：

```yaml
  - atom_id: MET-004
    standard_name: 营业收入
    # ... 现有字段
    grounding:
      match_terms: [营业收入, 营业总收入, 营收]
      exact_row_headers: [营业收入, 一、营业收入, -、营业收入]
      secondary_row_headers: [营业总收入, 一、营业总收入, -、营业总收入]
      weak_row_headers: [营业收入, 营业总收入, 一、营业总收入, -、营业总收入]
      bad_row_headers: [占营业收入比例, 营业收入比例, 销售费用占, 研发投入占营业收入]
      bad_col_headers: [分部收入, 关联方, 数量合计]
      low_priority_row_headers: [扣除与主营业务无关, 不具备商业实质]
      preferred_statements:
        - {terms: [主要会计数据, 主要财务指标], score: 5.0}
        - {terms: [公司简介和主要财务指标], score: 2.0}
        - {terms: [合并利润表, 利润表], score: 2.5}
      penalized_statements:
        - {terms: [关联方, 分部信息, 按产品, 按地区, 母公司], score: -3.0}
      segment_guard: true
```

### 3.3 代码改动

**`src/metric_atoms.py`**

| 改动 | 内容 |
|---|---|
| 新增 `GroundingRule` dataclass | 上述 9 字段，全部有默认值（空列表/False） |
| `MetricAtom` 新增字段 | `grounding: Optional[GroundingRule] = None` |
| `MetricAtomStore.load` | 解析 `grounding` 子结构；`preferred_statements`/`penalized_statements` 解析为 `List[Tuple[List[str], float]]` |
| 新增 `MetricAtomStore.grounding_for` | `(question: str) -> Optional[Tuple[MetricAtom, GroundingRule]]`：命中且该原子有 grounding 时返回 |

**`src/table_grounding.py`**（核心改造，保留旧函数直至等价性验证通过）

| 新增函数 | 签名 | 职责 |
|---|---|---|
| `_universal_row_guard` | `(question, row_text, col_text, unit_kind) -> Optional[bool]` | 9 条通用守卫，返回 `None`=否决 / `True`=通过。**对所有指标生效，无门控** |
| `_consolidated_bonus` | `(col_text) -> float` | 列含"本集团"/"合计" → +2.0（通用） |
| `_atom_row_score` | `(question, row_text, col_text, raw_value) -> Optional[float]` | 原子驱动打分：先跑 `_universal_row_guard`，再按 grounding 词表分级给分；无 grounding 原子 → 仅通用守卫 + 默认分 1.5 |
| `_atom_statement_score` | `(question, table_context) -> float` | 替代 `_table_scope_score`，读 `preferred_statements`/`penalized_statements`；与已有 `_metric_statement_prior_score` 合并，避免双重加分 |
| `_reject_unit_by_atom` | `(question, unit_hint) -> bool` | 替代 `_reject_revenue_unit`：按 `atom.unit` 判定（货币类拒 %），无原子命中时回退旧行为 |

**调用点替换**（`ground_number_query`，L500-520）：`_reject_revenue_unit`→`_reject_unit_by_atom`；`_revenue_row_score`→`_atom_row_score`；`_table_scope_score`+`_metric_statement_prior_score`→`_atom_statement_score`；L511/L516/L597 的 `_is_revenue_question` 门控改为「命中任一带 grounding 的原子」。

**删除**：`_is_revenue_question`、`_revenue_row_score`、`_table_scope_score`、`_reject_revenue_unit` 及 5 个 `_*REVENUE*` 词表常量（内容已迁入 yaml）。删除时机 = 等价性测试全绿之后。

### 3.4 行为等价验收（本阶段最关键）

**验收线**：应为「number 类问题准确率不回退」，但**无环境跑不了**。替代方案：

**`tests/test_table_grounding_equivalence.py`**——固化旧实现为**黄金基线**：

1. 用 `git show HEAD:src/table_grounding.py` 抽取改造前的 `_revenue_row_score`/`_table_scope_score`/`_reject_revenue_unit`，落为测试内的 `_legacy_*` 参照实现（**不 import 生产代码**，避免被后续改动带走）；
2. 构造 **≥40 组** (question, row_text, col_text, unit_hint) 输入矩阵，覆盖 3.1 表格全部 15 条规则分支 + 边界（年份列/季度列/百分号/母公司/分部/本集团合计/低优先词/兜底）；
3. 断言 `_atom_row_score(...) == _legacy_revenue_row_score(...)`（`None` 与数值均需一致），报表打分同理。

**必须诚实声明的局限**：等价性只证明「营收路径未改坏」。其余 11 个指标获得的通用守卫属**结构性改进**，其准确率收益**未经端到端验证**——叙述时须表述为"具备了可扩展结构"而非"提升了准确率"。

### 3.5 fail-closed 口径门禁

**病灶**：`src/answer_validation.py` 现有口径不符仅 `_downgrade_confidence`，答案照常返回。

**改动**：新增 `_metric_caliber_conflict(query_plan, retrieval_results) -> Optional[str]`——命中原子且证据中**只出现 `confusable_with` 指向的其他口径标准名、不出现本原子任一 `evidence_targets`** 时，返回冲突口径名。

命中冲突时（仅当 `schema == "number"` 且 `intent_confidence >= 0.7`）：`final_answer` 强制置 `"N/A"`，`confidence="low"`，追加 flag `metric_caliber_conflict_refused`。

**fail-closed 边界**（防误拒）：仅 number 类、仅高置信度意图、仅证据中确实出现易混口径名时触发；三条件任一不满足则退回原降级逻辑。

### 3.6 测试清单

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `tests/test_table_grounding_equivalence.py` | ≥40 断言 | 营收路径行为等价（黄金基线对照） |
| `tests/test_table_grounding_atoms.py` | 8 | 通用守卫对**非营收**指标生效（货币资金/研发投入拒同比列、拒母公司列、拒季度列）、grounding 缺省仅通用守卫、报表偏好加分、单位守卫按 `atom.unit` 判定、分部守卫开关 |
| `tests/test_metric_atoms.py` | +3 | `grounding` 解析、`grounding_for` 命中/未命中、statements 元组结构 |
| `tests/test_answer_validation.py` | +3 | 口径冲突强制 N/A、非 number 类不拒答、低置信度不拒答 |

### 3.7 风险与回退

| 风险 | 缓解 |
|---|---|
| 通用守卫放开后误否决其他指标的合法行 | 守卫全部是"问题未提及则否决"的**对称条件**（问题提到季度则允许季度列），不引入新的单边否决 |
| 报表打分双重加分（新旧两处叠加） | 合并为单一入口 `_atom_statement_score`，旧 `_metric_statement_prior_score` 一并移除 |
| 等价性测试自身写错 | 基线实现从 git 历史抽取，非手写复述 |
| fail-closed 误拒真答案 | 三条件收紧 + 3 个负向测试 |
| 回退 | 本阶段独立 commit；`git revert` 即可回到 Stage B 状态 |

---

## 4. Stage E：原子对抗质检（1 天）

### 4.1 目标

现有 `src/metric_atom_mining.py::validate_atom_draft` 只做规则质检（口径放宽词黑名单/口径重叠/字段完整）——**规则查不出语义错误**（如定义写错准则条款、caliber_notes 与 definition 自相矛盾）。补 LLM 多维打分层。

### 4.2 改动

**`src/metric_atom_review.py`**（新文件，与 mining 分离——**生产者不自批**）

| 方法 | 签名 | 职责 |
|---|---|---|
| `REVIEW_SYSTEM_PROMPT` | - | 四维评分（定义准确性/口径完整性/别名覆盖/易混淆标注），0-2 三分制，要求逐维给依据 |
| `review_atom_with_llm` | `(api_processor, atom_payload, model) -> Dict` | 返回 `{scores: {dim: int}, issues: [str], verdict: accept/revise/reject}` |
| `aggregate_review` | `(reviews: List[Dict]) -> Dict` | 汇总：任一维 0 分 → `reject`；总分 < 6/8 → `revise` |
| `render_review_report` | `(results) -> str` | Markdown 报告（供人工校准） |

**`scripts/review_metric_atoms.py`**：CLI 薄壳，`--model` 必须与 mining 模型不同，相同则拒绝执行并提示原因。

### 4.3 测试（`tests/test_metric_atom_review.py`，6 例）

打分聚合（任一维 0 → reject）、总分阈值、`issues` 透传、报告渲染、同模型拒绝执行、LLM 返回非法 JSON 时的 fail-closed（判 `revise` 而非 `accept`）。**全部用假 api_processor，不发真实请求。**

---

## 5. Stage F：原子版本化（0.5 天）

### 5.1 改动

**`src/metric_atoms.py`**

| 方法 | 职责 |
|---|---|
| `MetricAtom.content_hash` | 对语义字段（含 `grounding`）稳定序列化后取 sha256 前 12 位；**排除** `version` 自身，避免自指 |
| `MetricAtomStore.snapshot` | `{atom_id: content_hash}` 全库快照 |
| `MetricAtomStore.diff_snapshot` | `(old) -> {added, removed, changed}` |

**`scripts/atom_changelog.py`**：对比 `config/atom_snapshot.json` 与当前库，输出变更清单并可 `--update` 刷新快照。

**`config/atom_snapshot.json`**：纳入版本控制——知识变更在 code review 中可见（yaml 改动必然伴随快照 diff，改了不刷新则测试失败）。

### 5.2 测试（`tests/test_metric_atom_versioning.py`，5 例）

哈希稳定性（同内容同哈希、字段顺序无关）、语义变更导致哈希变化、`version` 字段变化不影响哈希、diff 三类正确、**快照与当前库一致性回归锁**（yaml 改了忘刷新快照即失败）。

---

## 6. 统一验收清单

| 项 | 判定方式 | 环境 |
|---|---|---|
| 各阶段单测全绿 | `./venv/bin/python -m unittest` | ✅ 本地 |
| 全量无新增失败 | 与 `git stash` 基线对照（已知 6 个历史环境依赖失败） | ✅ 本地 |
| Level1 评测不回退 | 六项指标维持 100% | ✅ 本地 |
| 营收取证行为等价 | ≥40 组黄金基线断言 | ✅ 本地 |
| flag 契约故障注入 | 未登记 key 导致 import 失败 | ✅ 本地 |
| number 类准确率不回退 | 端到端跑分 | ❌ **需 API key + 数据集** |
| Level2 口径/答案一致率 | 端到端跑分 | ❌ **需 API key + 数据集** |

最后两项在环境就绪前**保持空缺**，不以任何替代指标冒充。

---

## 7. 实测结果汇总（四阶段完成）

| 阶段 | 交付 | 验收证据 |
|---|---|---|
| Stage D | flag 契约 + 白名单过滤 + 覆盖率报告 | 12 测试；D-1 内部字段不再污染 topic_flags（单元 + 端到端双验）；D-2 覆盖率量化为 6/17 = 35.3%；故障注入未登记 key 触发 ValueError |
| Stage C | 9 条通用守卫提升 + 6 条专属词表下沉 + 删 108 行硬编码 + fail-closed 门禁 | 13 + 4 测试；黄金基线（git 动态加载旧实现）264 组行打分 / 54 组报表 / 42 组单位判定全部同分；拒答 1 正向 + 3 负向边界 |
| Stage E | LLM 四维打分 + 生产者不自批 + fail-closed 裁决 | 16 测试（全假 processor）；CLI 同模型退出码 2 |
| Stage F | 语义内容哈希 + 快照回归锁 | 12 测试；故障注入改 `MET-004.definition` 精确报出 `changed: ['MET-004']` |

**测试总量**：303 → 333（新增 53 个测试）；失败数维持历史遗留的 6 个环境依赖项，无新增。
**Level1 评测**：六项指标维持 100%。

**仍然空缺（不以替代指标冒充）**：

| 项 | 阻塞原因 |
|---|---|
| number 类问题端到端准确率 | 需 API key + 数据集 |
| Level2 口径一致率 / 答案一致率 | 需 API key + 数据集 |
| 原子扩容 12 → 50+ | 需 API key（mine + review 两模型） |
| topic flag 词表补全 11 类 | 属知识扩容，需逐类设计中英文问法 |

**过程中暴露并已建防线的两类质量问题**：
1. 形近字错写（「润」→「潦」，同一会话复现 3 次）使规则静默失效——防线为"知识与旧常量逐字一致断言" + 黄金基线从 git 加载而非手抄 + 码点转义书写负向断言；
2. 磁盘写入报失败但延迟落盘，重试导致函数/YAML 键重复（Python 与 YAML 均不报错）——处理流程为报失败后先计数验证再决定重试，去重用带 assert 保护的脚本。

---

## 8. Stage G：治理机制对齐 LLM-Wiki（2026-08-20 立项）

> 背景纠偏：本项目的知识层技术锚点是 KnowledgeBrain `_llm-wiki/` 工程的四条架构决策
> （编译而非检索 / fail-closed 拒答语义 / 三层 Registry 状态机 / Builder 永不自批）
> 及其 Claim 粒度证据链（provenance 刻度、有效期、Gap 一等公民），不是"实习方法论泛化"。
> Stage E/F 已自发同构决策三/四的雏形；Stage G 补齐证据链与决策一/二的落地形态。

| 序 | 项 | 对齐特点 | 设计要点 |
|---|---|---|---|
| G-1 | provenance 刻度进原子 schema | Claim 证据链、缺口不包装 | 每原子标注各字段来源 extracted（从生产代码迁移，等价测试可证）/ inferred（手写知识，未语义验证）；缺省按 inferred（fail-closed）。存量：aliases=extracted（源自 query_rewrite 旧正则），定义类字段=inferred，MET-004.grounding=extracted（源自 _revenue_row_score） |
| G-2 | 有效期 verified_at/expires_at | 知识带有效期 | 过期原子 payload 带 expired 标志，answer_validation 加 metric_atom_expired flag + 降置信度（温和语义：不拒答——口径可能仍正确，完全 Gap 语义适用于独立 Serving 而非 RAG 内知识层）；validity_audit() 把未声明有效期暴露为显式缺口 |
| G-3 | 快照升级为冻结 Generation | 决策一+三 | atom_snapshot.json 从 hash 映射升级为完整编译产物（原子规范化序列化 + per-atom hash + buildId + 源 yaml sha256）；MetricAtomStore.load() 运行时只读冻结版并校验源同步，yaml 变更未重编译即启动失败（fail-closed）；load_source() 保留源码直读供构建工具 |
| G-4 | 条目填充（后续） | — | grounding 扩容 11 原子、flag 词表补全、剧本收编分红词表——每个新条目必须带 provenance |

验收：provenance 非法值加载期拒绝；过期边界（今天/明天）单测；audit 显式列缺口；
yaml 改动未重编译时 load() 报错指路；Generation 加载与源码加载行为等价（hash 全等）。

### 8.1 实测结果（G-1/G-2/G-3 完成）

| 项 | 交付 | 验收证据 |
|---|---|---|
| G-1 | provenance 刻度（per-field extracted/inferred，缺省 inferred） | 12 原子如实标注；非法值/坏日期加载期 ValueError；快照回归锁首次真实运转（yaml 加字段即报 12 个变更） |
| G-2 | verified_at/expires_at + validity_audit + 温和过期门禁 | 12 原子全部未声明有效期（显式缺口而非伪装永不过期）；过期 → metric_atom_expired flag + 降一级（不拒答，与口径门禁严格语义刻意区分） |
| G-3 | 冻结 Generation（编译产物）+ 运行时只读编译产物 | 等价性：Generation 加载 vs 源码加载 12 原子 hash 全等（含 grounding tuple 还原）；源失步演练：改 yaml 不重编译 → RuntimeError 拦截；Generation 缺失 → 报错指路 compile_atoms.py；格式代差（旧快照无 schema_version）触发重编译 |

开发流程变更：改 `config/metric_atoms.yaml` 后必须运行 `python scripts/compile_atoms.py --update`，
否则 `MetricAtomStore.load()` 启动即失败——知识变更因此必须在 code review 中显式出现。
atom_changelog.py 已删除（功能并入 compile_atoms.py）。测试 349（+16 于 Stage F 基线），
失败维持 6 个历史环境依赖；Level1 六项维持 100%。
