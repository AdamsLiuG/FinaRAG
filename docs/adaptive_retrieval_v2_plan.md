# Agentic 自适应检索 v2 加厚方案（①闭环 ②多轮收敛 ③策略多样化 ④trace 反哺）

> 立项：2026-08-21。前置：v1 已交付（commit c71664f：预检模式、280 行、12 测试）。
> 本方案把单环扩成五环骨架：感知(多源) → 决策(策略) → 行动(多策略) → 重评估(收敛)
> → 记忆(trace 反哺知识层)。全部离线可实施可验收；LLM 决策器接口预留（卡 API key）。
> 代码事实均经本轮 grep 核实（行号为当前 HEAD）。

## 0. 五环框架与 v1 现状对照

```
v1：  感知(3规则) ──────────────→ 行动(同库补检) → 停(预算1)
v2：  感知(预检3源+校验后3源) → 决策(策略选择) → 行动(同库/父粒度) → 重评估(收敛判定)
      → 记忆(trace → 别名扩充建议 → 人工确认入库)
```

| 环节 | v1 | v2 | 交付项 |
|---|---|---|---|
| 感知 | 生成前预检 3 规则 | + 生成后校验 flags 3 源 | ① |
| 决策 | 无（单一动作） | gap 类型 × 轮次 → 策略 | ③ |
| 行动 | 同库同粒度补检 | + 父粒度回溯 | ③ |
| 重评估 | 单轮硬停 | 缺口闭合/无新增证据/预算，三条件收敛 | ①② |
| 记忆 | trace 落盘 | trace → 知识资产进化建议 | ④ |

**不做**（防堆功能稀释主线）：多 agent 协作、自由文本反思链、工具注册框架、
metadata_reroute 跨文档路由（仅设计预留）。厚度的来源是知识层参与更深。

---

## 1. ① 校验后反思闭环（try-then-closed）

### 1.1 现状与代码证据

- 主生成段：questions_processing.py L3209-3261——`get_answer_from_rag_context`
  → 组装（references/citations/confidence/route_info/retrieval_debug）→
  `validate_answer`（L3261）→ `return validated_answer.answer`（L3262）。
- 闭环触发源（answer_validation.py 实测产生点）：
  - `metric_caliber_conflict_refused`（L269）：口径门禁拒答——证据只有易混口径
  - `metric_evidence_mismatch`（L262）：指标证据与问题不匹配
  - `implicit_intent_evidence_incomplete`（L254）：剧本必需指标证据不足
- v1 预检挂点在 L3110（检索后/生成前）；校验后无任何补检路径。

### 1.2 详细设计

**重入机制：闭包方案（不做大抽取）**。get_answer_for_company 内把
L3209-3261 的「生成→组装→校验」段包进局部函数：

```python
def _generate_and_validate(current_results: List[Dict]) -> "ValidatedAnswer":
    # 闭包捕获 question/schema/company_name/rewrite_result/route_info/
    # route_mode/selected_report/reranking_debug/hyde_debug 等局部变量，
    # 参数只有 current_results——零参数穿透，不动方法签名
    ...
    validated_answer = validate_answer(answer_dict, current_results, rewrite_result)
    return validated_answer
```

主流程改为：

```python
validated = _generate_and_validate(retrieval_results)
# try-then-closed：拒答/证据缺口 flags 触发补检 → 重新生成 → 二次校验
if adaptive_retrieval_enabled() and mode != "full_context":
    post_gap = detect_post_validate_gap(
        validated.validation_flags, rewrite_result, retrieval_results)
    if post_gap is not None and adaptive_rounds_used < budget:
        supplement = self._collect_supplement(retriever, company_name,
                                              rewrite_result, post_gap)
        if supplement:
            retrieval_results = merge_supplement_results(retrieval_results, supplement)
            retrieval_results = retriever.rerank_candidates(...)  # 统一重排
            validated = _generate_and_validate(retrieval_results)  # 重入
            adaptive_rounds_used += 1
            trace_outcome = "supplement_succeeded" if ... else "refused_maintained"
```

**新增（src/adaptive_retrieval.py）**：

```python
POST_VALIDATE_TRIGGERS = {
    "metric_caliber_conflict_refused": TRIGGER_EVIDENCE_GAP,   # 定向补本口径
    "metric_evidence_mismatch": TRIGGER_EVIDENCE_GAP,
    "implicit_intent_evidence_incomplete": TRIGGER_PLAYBOOK_INCOMPLETE,
}

def detect_post_validate_gap(validation_flags: List[str], query_plan,
                             retrieval_results: List[Dict],
                             atom_store=None) -> Optional[RetrievalGap]:
    """校验后触发：flags 中首个可补救触发源映射到 gap。
    与预检共用 _atom_supplement_queries / playbook 分支的构造逻辑；
    caliber_conflict 的补检方向 = 本口径 evidence_targets（排除易混标准名，
    防强化错口径——补的是本口径证据，不是对照取证）。"""
```

**try-then-closed 语义**（测试锁定）：
- try 分支：补检后二次校验 `metric_caliber_conflict_refused` 消失 → 采用二次答案
- then-closed 分支：补检后仍只有易混口径证据 → 二次校验维持拒答（Stage C
  fail-closed 语义不变，闭环只是给了它一次补救机会）
- 预算：与预检共享 `adaptive_rounds_used` 计数，总轮次 ≤ ADAPTIVE_MAX_ROUNDS

### 1.3 测试用例表（新增 8 例，挂 tests/test_adaptive_retrieval.py）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| P1 | caliber 拒答触发 | flags=[metric_caliber_conflict_refused] + MET-001 plan | gap.trigger=evidence_gap，queries 含扣非 targets、不含归母 |
| P2 | evidence_mismatch 触发 | flags=[metric_evidence_mismatch] | 同上机制 |
| P3 | playbook 不全触发 | flags=[implicit_intent_evidence_incomplete] | playbook_incomplete |
| P4 | 无关 flags 不触发 | flags=[numeric_grounding_weak] | None |
| P5 | try 分支（重入级） | mock 生成：首轮拒答→补检回扣非证据→二次生成作答 | 最终非 N/A，outcome=supplement_succeeded |
| P6 | then-closed 分支 | mock 补检仍只回归母 | 最终维持 N/A，outcome=refused_maintained |
| P7 | 预算共享 | 预检已用 1 轮 + 校验后再触发 | 闭环不再执行（计数不超限） |
| P8 | 默认关闭零影响 | enabled=False + 构造触发场景 | 生成只调一次（mock 计数=1） |

### 1.4 风险与回退

- 闭包重入使 L3209-3261 缩进一层——纯机械改动，行为等价由现有全量测试守护
- 二次生成的 token 成本：预算硬限保证 ≤1 次额外生成；budget_exceeded 标记
  （D3）对二次生成同样生效
- 回退：ADAPTIVE_RETRIEVAL_ENABLED=0 即回到 v1 行为；单 commit revert

---

## 2. ② 收敛驱动的多轮

### 2.1 设计

预检循环从「单轮硬停」改为「三条件收敛」：

```python
def _adaptive_loop(...) -> Tuple[List[Dict], Dict]:
    rounds = 0
    while rounds < adaptive_max_rounds():
        gap = detect_gap(query_plan, retrieval_results, top_score)
        if gap is None:
            break  # 收敛条件1：缺口闭合（剧本 required_metrics 全覆盖 = 知识说够了）
        supplement = self._collect_supplement(...)
        if not supplement:
            break  # 收敛条件2：无新增证据——边际收益为零
        merged = merge_supplement_results(retrieval_results, supplement)
        if len(merged) == len(retrieval_results):
            break  # 收敛条件2的显式形态（合并后未增）
        retrieval_results = rerank(merged)
        rounds += 1
    # 循环自然退出 = 收敛条件3：预算耗尽
```

- `ADAPTIVE_MAX_ROUNDS` 默认 1 → **2**（预检 ≤1 + 闭环 ≤1 共享池）；
  语义从「轮数」变「轮数上限」——停止的第一依据是收敛而非预算
- **停止条件知识化**（叙事核心）：收敛条件 1 的判定就是 detect_gap 的
  playbook 分支——required_metrics 全覆盖即收敛，不是"跑 N 轮碰运气"
- 环形风险防护：每轮 gap 的 supplement_queries 若与上轮完全相同则 break
  （同查询补检是浪费——查询级幂等检查，新增 `_queries_unchanged` 判定）

### 2.2 测试用例表（4 例）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| C1 | 缺口闭合即停 | 第一轮补检覆盖 targets | 只 1 轮，detect_gap 第二次返回 None |
| C2 | 无新增证据即停 | mock 补检返回全重复结果 | 1 轮后停（len 未增） |
| C3 | 同查询幂等停 | 第二轮 gap queries == 第一轮 | break，不发起第三次检索 |
| C4 | 预算上限 | 永远新证据场景 | 恰好 max_rounds 轮 |

---

## 3. ③ 行动策略多样化

### 3.1 策略空间（实施范围收敛后）

| 策略 | 触发条件 | 实现 | 状态 |
|---|---|---|---|
| `dense_recall`（v1 现状） | 首轮各触发 | 同库同粒度补检 | 已有 |
| `parent_recall` | 第 2 轮起 / low_confidence 且首轮无新增 | parent_retrieval_mode 显式传 "page"（切父粒度拿页级上下文） | 本轮实施 |
| `adversarial_caliber` | caliber_conflict 触发 | 定向补本口径 evidence_targets（与 dense 同机制，方向语义不同） | 本轮实施（合并入①触发源） |
| `metadata_reroute` | 换年份/文档场景 | ReportCatalog 路由 | **仅设计预留不实施** |

`metadata_reroute` 设计要点（归档供演进）：补检前判定 gold 年份/报告类型是否
在候选池——不在则 ReportCatalog 重路由后整轮重检。涉及路由层改造，收益依赖
多文档题占比，暂缓。

### 3.2 详细设计

**RetrievalGap 扩展**（adaptive_retrieval.py）：

```python
@dataclass(frozen=True)
class RetrievalGap:
    trigger: str
    supplement_queries: List[str]
    rationale: str
    strategy: str = "dense_recall"   # 新增：本轮补检策略
```

**策略选择**（`_choose_strategy(gap, round_index, prior_evidence_gain: bool) -> str`）：

```
round 0：dense_recall（任何触发）
round ≥1：上轮有新增证据 → 维持 dense_recall（有效就继续）
          上轮无新增 → parent_recall（粒度切换是低置信的第二手段）
```

**parent_recall 实现**：`_collect_supplement` 的检索调用处按策略覆盖粒度：

```python
retrieve_kwargs = dict(..., parent_retrieval_mode=(
    "page" if strategy == "parent_recall" else self.parent_retrieval_mode
))
```

现有设施确认：`retrieve_candidates_by_company_name` 的
`parent_retrieval_mode` 参数全程透传（questions_processing L982/L992），
策略层只需按次覆盖，不改传递链。

**merge 去重键升级**（关键细节，实测确认）：parent 结果（page 粒度）与
child 结果（chunk 粒度）同页共存——v1 的 `(pdf_sha1, page)` 去重会把
child 证据误判为 parent 重复。升级为：

```python
def _merge_key(item: Dict) -> Tuple:
    scope = item.get("result_scope") or (item.get("metadata") or {}).get("node_type") or "child"
    return (str(item.get("pdf_sha1") or ""), item.get("page"), scope)
```

（retrieval.py L130 的 `result_scope` 判定逻辑复用其语义。）

### 3.3 测试用例表（5 例）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| S1 | 首轮策略 | 任意触发 | strategy=dense_recall |
| S2 | 粒度切换 | 第 2 轮且上轮无新增 | strategy=parent_recall，补检索调用收到 parent_retrieval_mode="page" |
| S3 | 有效不切换 | 第 2 轮但上轮有新增 | 维持 dense_recall |
| S4 | merge 键 scope 维度 | child(p1) + parent(p1) | 两条都保留 |
| S5 | 同页同 scope 去重 | 两条 child(p1) | 保留一条 |

---

## 4. ④ trace 反哺知识层

### 4.1 设计

自适应检索的运行经验成为知识资产进化的依据——**系统改进自己的知识层**，
且守住 Builder 永不自批（建议必须人工确认入库）。

**trace 字段扩展**（v2 闭环补齐后自然齐备）：

```json
{"timestamp": "...", "question": "...", "trigger": "evidence_gap",
 "strategy": "parent_recall", "queries": ["扣除非经常性损益后的净利润"],
 "new_hits": 3, "rounds": 2, "outcome": "supplement_succeeded",
 "atom_id": "MET-001", "gap_closed": true}
```

**新增 `scripts/analyze_adaptive_trace.py`**：

```python
def load_trace(path: Path) -> List[Dict]                       # JSONL 读取 + 脏行跳过
def per_atom_stats(entries: List[Dict]) -> List[Dict]:
    # 每原子：触发次数 / 触发源分布 / 查询级命中率（query → new_hits>0 比例）
    # / 缺口闭合率（gap_closed 比例）/ outcome 分布

def alias_expansion_suggestions(per_atom: List[Dict], store) -> List[Dict]:
    # 建议规则（只出建议不自动改库）：
    # 1. 某查询词补检命中率 > 80% 且缺口常闭 → 建议该词加入该原子 aliases
    #    （检索侧常靠它才命中 = 查询改写侧应预展开）
    # 2. 某原子 evidence_target 触发补检后仍零命中 → 建议复核该 target 表述
    #    （可能文档从不这么写）
    # 3. 某原子从未触发补检 → 无操作（知识已足够，不是缺口）

def main():   # CLI: --trace logs/adaptive_trace.jsonl --output reports/adaptive_analysis.md
```

产出 Markdown 报告：统计矩阵 + 建议清单（每条建议附 evidence 计数与依据）。
**入库流程**：人工确认建议 → 编辑 yaml → `compile_atoms.py --update`
→ 既有等价测试与门禁全量守护（与 mine→review→人工→compile 同构，
Builder 不自批原则不变）。

### 4.2 测试用例表（5 例，tests/test_adaptive_trace_analysis.py）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| A1 | 每原子统计 | 3 条 trace（2 触发 1 闭合） | 命中率/闭合率计算正确 |
| A2 | 高命中别名建议 | query "扣除非经常性损益后的净利润" 命中率 100% | 建议清单含该词 → MET-001 aliases |
| A3 | 零命中 target 复核建议 | 某 target 补检后 new_hits 全 0 | 建议复核该 target |
| A4 | 脏行防御 | trace 文件混入非法 JSON 行 | 跳过不炸，报告注明 skipped |
| A5 | 空文件 | 空迹文件 | 空报告不炸 |

---

## 5. 实施顺序与统一验收

```
commit 1  ① 闭包重入 + detect_post_validate_gap + try-then-closed（最大件）
commit 2  ② 多轮收敛循环 + 查询幂等停（依赖①的循环骨架）
commit 3  ③ 策略选择 + parent_recall + merge 键 scope 升级（独立于②可并行）
commit 4  ④ trace 字段补齐（随①③自然落）+ analyze 脚本 + 报告
commit 5  docs 回填实测结果
```

| 项 | 离线验收 | 卡 key |
|---|---|---|
| ① 闭环 | P1-P8（try/then-closed 两分支行为锁定 + 预算共享 + 默认关闭零影响） | 拒救率（refused→answered 转化） |
| ② 多轮 | C1-C4（三收敛条件各一 + 预算） | 轮次分布统计 |
| ③ 策略 | S1-S5（策略选择 + 粒度覆盖 + merge 键） | 各策略命中率对比 |
| ④ 反哺 | A1-A5 + 模拟 trace 报告产出 | 真实 trace 积累后的首轮报告 |

**纪律**（同前）：独立 commit；现有断言不改通过；ADAPTIVE_RETRIEVAL_ENABLED
默认关闭保证全量零回归；trace/报告字段名以本方案 §4.1 schema 为准不得漂移；
LLM 决策器（detect_gap 接口喂给模型做策略选择）为卡 key 的 v3 项，接口形态
已由 RetrievalGap 结构化字段预留。

**预期量级**：生产代码 +400~500 行（重入闭包 ~80 / 循环与收敛 ~80 /
策略与 merge ~80 / trace 分析脚本 ~150 / 触发判定 ~60），测试 +22 例，
全量 458 → ~480。
