# Agentic 自适应检索 v2 加厚方案（实现级蓝图）

> 立项：2026-08-21；v2 修订：按「照着能直接写代码」标准落细——闭包完整改造规则、
> 每个新函数全文、主流程插入段全文、diff 清单、边界条件表、测试构造骨架。
> 前置：v1 已交付（c71664f）。行号基于当前 HEAD（questions_processing.py 3800+ 行）。
> 全部离线可实施可验收；LLM 决策器为卡 key 的 v3 项（接口已由 RetrievalGap 预留）。

## 0. 五环框架 / commit 划分 / 状态传递设计

```
v1：  感知(预检3规则) ─────────────→ 行动(同库补检) → 停(预算1)
v2：  感知(预检3源+校验后3源) → 决策(策略选择) → 行动(dense/父粒度)
      → 重评估(三条件收敛) → 记忆(trace→知识进化建议→人工入库)
```

| commit | 内容 | 文件 |
|---|---|---|
| 1 | ① 闭包重入 + detect_post_validate_gap + try-then-closed | questions_processing.py / adaptive_retrieval.py / test |
| 2 | ② 多轮收敛循环（_run_adaptive_supplement 重构 + _collect_supplement 抽取） | 同上 |
| 3 | ③ 策略选择 + parent_recall + merge 键 scope 升级 | adaptive_retrieval.py / 同上 |
| 4 | ④ trace 字段补齐 + analyze_adaptive_trace.py + 报告 | scripts/ + test |
| 5 | docs 回填实测 | docs/ |

**状态传递设计（关键决策）**：adaptive 轮次预算不进 processor 实例属性
（D1 教训：实例属性在多线程共享 processor 下串状态）——全部以**局部变量 +
返回值**传递：

```python
# get_answer_for_company 内（局部作用域，线程安全）
adaptive_state = {"rounds_used": 0, "traces": []}   # 预检与闭环共享的预算池
```

`_run_adaptive_supplement(..., state=adaptive_state)` 与闭环段共同读写该 dict
（dict 按引用传递，同一请求局部）。

---

## 1. ① 校验后反思闭环（try-then-closed）

### 1.1 现状精确边界（闭包要包的代码）

**闭包范围 = L3124-3262 整段**（不是只有生成段！）：

```
L3124  grounding_result = None                     ← 段首
L3125  chart_grounding_result = None
L3126-3163  grounding 计算（table/chart，消费 retrieval_results，
            其中 L3163 会改写 retrieval_results——闭包内用 current_results）
L3165-3196  no_context 早退分支（retrieval_results 为空 → 拒答 return）
L3198-3221  生成分支（table_grounded / chart_grounded / RAG 三路）
L3223-3260  组装（pages 校验/references/citations/confidence/debug 字段）
L3261  validated_answer = validate_answer(answer_dict, retrieval_results, rewrite_result)
L3262  return validated_answer.answer              ← 段尾
```

**为什么从 grounding 开始包**：闭环补检改变 retrieval_results 后，表格取证
（grounding）消费的是新证据——只重跑生成不重跑 grounding，数字题的闭环语义
不完整。`_cached_revenue_grounding` 有缓存，重跑成本低。

### 1.2 闭包改造（机械规则，逐点列出）

**改造动作**：L3124-3262 整体缩进 4 格，包进：

```python
        def _generate_and_validate(current_results: List[Dict]) -> "ValidatedAnswer":
            """生成→组装→校验（闭包捕获请求上下文；current_results 唯一可变输入）。

            闭包重入的语义：补检后以新证据池重走 grounding→生成→校验全链。
            捕获变量（只读）：question/schema/company_name/rewrite_result/
            route_info/route_mode/candidate_pool_size_before_rerank/
            reranking_debug/hyde_debug/candidate_doc_ids/allowed_grounding_doc_ids。
            """
```

**段内 `retrieval_results` → `current_results` 逐点替换表**（闭包内统一改名，
防误读外层变量）：

| 原行 | 上下文 | 替换 |
|---|---|---|
| L3155 | `if not retrieval_results:`（no_context 判定） | `if not current_results:` |
| L3163 | `retrieval_results = self._ensure_chart_grounding_retrieval_result(retrieval_results, ...)` | `current_results = self._ensure_chart_grounding_retrieval_result(current_results, ...)`（闭包内局部再赋值，安全） |
| L3173/3177 | grounding 调用入参 ×2 | `current_results` |
| L3209 | `answer_context_results = retrieval_results[: self.top_n_retrieval]` | `current_results[: ...]` |
| L3226 | `_validate_page_references(pages, retrieval_results)` | `current_results` |
| L3235/3238 | citations 的 `build_citations(retrieval_results, ...)` 及 table_support | `current_results` |
| L3243 | `compute_confidence(answer_dict, retrieval_results)` | `current_results` |
| L3256-3258 | retrieval_pages/retrieval_results/retrieval_report_groups 组装 ×3 | `current_results` |
| L3261 | `validate_answer(answer_dict, retrieval_results, rewrite_result)` | `current_results` |

**段尾 return 变更**：`return validated_answer.answer` → `return validated_answer`
（主流程需要 validation_flags 做闭环判定；最终对外仍 `.answer`）。

### 1.3 主流程改造（完整插入代码，替换 L3108-3121 预检段 + 接闭包）

```python
        # ===== Agentic v2：预检（生成前）+ 闭环（校验后）=====
        # 预算池与 trace 以局部 dict 传递（D1 教训：不进实例属性）
        adaptive_state: Dict[str, Any] = {"rounds_used": 0, "traces": []}

        if adaptive_retrieval_enabled() and retrieval_results and mode != "full_context":
            supplemented = self._run_adaptive_supplement(
                retriever=retriever,
                company_name=company_name,
                question=question,
                rewrite_result=rewrite_result,
                retrieval_results=retrieval_results,
                candidate_doc_ids=candidate_doc_ids,
                state=adaptive_state,
            )
            if supplemented is not None:
                retrieval_results = supplemented

        # ---- 闭包定义（L3124-3262 原段整体移入，见 §1.2 机械规则）----
        def _generate_and_validate(current_results): ...

        # ---- 首轮生成 + 校验后闭环（try-then-closed）----
        validated = _generate_and_validate(retrieval_results)
        if (
            adaptive_retrieval_enabled()
            and mode != "full_context"
            and adaptive_state["rounds_used"] < adaptive_max_rounds()
        ):
            post_gap = detect_post_validate_gap(
                validated.validation_flags, rewrite_result, retrieval_results
            )
            if post_gap is not None:
                supplement = self._collect_supplement(
                    retriever=retriever,
                    company_name=company_name,
                    rewrite_result=rewrite_result,
                    gap=post_gap,
                    candidate_doc_ids=candidate_doc_ids,
                )
                if supplement:
                    retrieval_results = merge_supplement_results(retrieval_results, supplement)
                    retrieval_results = retriever.rerank_candidates(
                        query=question,
                        candidate_results=retrieval_results,
                        documents_batch_size=self.final_reranking_batch_size,
                        top_n=self.retrieval_debug_top_n,
                        task_type=getattr(rewrite_result, "task_type", None),
                    )
                    adaptive_state["rounds_used"] += 1
                    # try：以新证据池重走 grounding→生成→校验
                    validated = _generate_and_validate(retrieval_results)
                    self._record_adaptive_trace(adaptive_state, {
                        "question": question, "company_name": company_name,
                        "trigger": post_gap.trigger,
                        "queries": post_gap.supplement_queries,
                        "new_hits": len(supplement),
                        "rounds": adaptive_state["rounds_used"],
                        "outcome": (
                            "supplement_succeeded"
                            if "metric_caliber_conflict_refused" not in validated.validation_flags
                            else "refused_maintained"
                        ),
                        "atom_id": (getattr(rewrite_result, "metric_atom", None) or {}).get("atom_id"),
                        "gap_closed": detect_post_validate_gap(
                            validated.validation_flags, rewrite_result, retrieval_results
                        ) is None,
                    })
        return validated.answer
```

注意：v1 的 `_append_adaptive_trace`（静态方法直写文件）改造为
`_record_adaptive_trace(state, entry)`——先入 state["traces"]，由
`_run_adaptive_supplement` 与闭环段结束时统一落盘一次（一次请求一条聚合
trace，不是每轮一条；落盘函数复用 v1 的 JSONL 追加逻辑）。

### 1.4 detect_post_validate_gap 完整实现（src/adaptive_retrieval.py 追加）

```python
POST_VALIDATE_TRIGGERS = {
    # 校验 flag → gap trigger（产生点：answer_validation.py L254/L262/L269）
    "metric_caliber_conflict_refused": TRIGGER_EVIDENCE_GAP,      # 口径门禁拒答
    "metric_evidence_mismatch": TRIGGER_EVIDENCE_GAP,
    "implicit_intent_evidence_incomplete": TRIGGER_PLAYBOOK_INCOMPLETE,
}


def detect_post_validate_gap(
    validation_flags: List[str],
    query_plan,
    retrieval_results: List[Dict],
    atom_store: Optional[MetricAtomStore] = None,
) -> Optional[RetrievalGap]:
    """校验后触发：flags 中首个可补救触发源 → RetrievalGap。

    构造逻辑复用预检：evidence_gap 走 _atom_supplement_queries（排除易混
    口径标准名——补的是本口径证据，caliber_conflict 场景即「定向补本口径」）；
    playbook_incomplete 走 payload.required_metrics 逐指标判定。
    与预检的差异：触发源是校验层判定（语义更准），构造逻辑完全同源。
    """
    if atom_store is None:
        from src.metric_atoms import get_default_store
        atom_store = get_default_store()

    for flag in validation_flags or []:
        trigger = POST_VALIDATE_TRIGGERS.get(flag)
        if trigger is None:
            continue
        metric_atom_payload = getattr(query_plan, "metric_atom", None) or {}
        if trigger == TRIGGER_EVIDENCE_GAP:
            atom = atom_store.get(metric_atom_payload.get("atom_id") or "")
            if atom is None:
                continue  # 无原子命中：flags 不可由补检补救
            queries = _atom_supplement_queries(atom, ADAPTIVE_MAX_SUPPLEMENT_QUERIES)
            confusable = set(metric_atom_payload.get("confusable_names") or [])
            queries = [q for q in queries if q not in confusable]
            if not queries:
                continue
            return RetrievalGap(
                trigger=trigger,
                supplement_queries=queries,
                rationale=f"校验 flag={flag}；本口径证据定向补检",
            )
        # TRIGGER_PLAYBOOK_INCOMPLETE
        playbook_payload = getattr(query_plan, "playbook", None) or {}
        corpus_text = _result_texts(retrieval_results)
        missing: List[str] = []
        for metric_atom_id in list(playbook_payload.get("required_metrics") or []):
            metric = atom_store.get(metric_atom_id)
            if metric is None:
                continue
            targets = [metric.standard_name] + metric.evidence_targets
            if len(_targets_covered(targets, corpus_text)) == len(targets):
                missing.append(metric.standard_name)
        if missing:
            return RetrievalGap(
                trigger=trigger,
                supplement_queries=missing[:ADAPTIVE_MAX_SUPPLEMENT_QUERIES + 1],
                rationale=f"校验 flag={flag}；剧本必需指标缺证据：{missing}",
            )
    return None
```

### 1.5 边界条件表（①）

| 边界 | 处理 |
|---|---|
| validation_flags 为空/None | 循环不进 → None |
| flags 有触发源但原子未命中（无 atom_id） | `continue` 跳过该 flag——flags 不可补救时不硬造 gap |
| 预算已耗尽（预检用满） | 主流程条件 `rounds_used < max` 先判——闭环不执行 |
| 补检无新证据（supplement 空） | 不重入、不耗预算——闭环只在有新证据时 try |
| 二次生成抛异常 | 异常向上传播（fail-closed：生成失败不做三次尝试） |
| mode == "full_context" | 闭环跳过（与预检同条件） |
| 闭包内 chart grounding 改写 current_results | 闭包局部再赋值，不影响外层（语义正确：该改写本来就只影响本轮生成） |

---

## 2. ② 收敛驱动的多轮（_run_adaptive_supplement 重构）

### 2.1 重构前置：_collect_supplement 抽取（①③ 都要用）

从 v1 的 `_run_adaptive_supplement` 中抽出「单轮补检收集」为独立方法（②的
循环体、①的闭环段共用）：

```python
    def _collect_supplement(
        self,
        retriever,
        company_name: str,
        rewrite_result,
        gap,                                   # RetrievalGap
        candidate_doc_ids: Optional[List[str]] = None,
        parent_mode_override: Optional[str] = None,   # ③ 策略层按次覆盖粒度
    ) -> List[Dict]:
        """单轮补检收集：gap.supplement_queries 逐条走纯召回（hybrid 无 rerank）。

        parent_mode_override：非 None 时覆盖 self.parent_retrieval_mode
        （③ parent_recall 策略传 "page"；默认 None 走 processor 配置）。
        """
        supplement: List[Dict] = []
        for supplement_query in gap.supplement_queries:
            results = self._run_retrieval(
                retriever, "hybrid", company_name, supplement_query,
                rewrite_result.filters, candidate_doc_ids,
                task_type=getattr(rewrite_result, "task_type", None),
            )
            # 粒度覆盖需要绕过 _run_retrieval 的 self.parent_retrieval_mode——
            # 直接调 retriever（与 _retrieve_multi_query_candidate_runs 同型）
            if parent_mode_override is not None:
                results = retriever.retrieve_candidates_by_company_name(
                    company_name=company_name,
                    query=supplement_query,
                    top_n=max(self.llm_reranking_sample_size, self.retrieval_debug_top_n),
                    parent_retrieval_mode=parent_mode_override,
                    filters=rewrite_result.filters,
                    candidate_doc_ids=candidate_doc_ids,
                )
            supplement.extend(results)
        return supplement
```

注：`parent_mode_override` 分支不走 `_run_retrieval` 的缓存语义差异——
`_run_retrieval` 的 hybrid 分支同样直调 `retrieve_candidates_by_company_name`
（L978 附近），此处为覆盖粒度直调同型接口，行为一致。

### 2.2 _run_adaptive_supplement 完整新版（v1 → v2 对照）

```python
    def _run_adaptive_supplement(
        self,
        retriever,
        company_name: str,
        question: str,
        rewrite_result,
        retrieval_results: List[Dict],
        candidate_doc_ids: Optional[List[str]] = None,
        state: Optional[Dict] = None,          # 新增：预算池（§0 设计）
    ) -> Optional[List[Dict]]:
        """预检多轮循环：detect_gap → 补检 → merge → rerank，直到收敛。

        收敛三条件（任一满足即停）：
        1. 缺口闭合：detect_gap 返回 None（剧本 required_metrics 全覆盖——
           停止条件知识化，不是跑 N 轮碰运气）
        2. 无新增证据：merge 后池长不变（边际收益为零）
        3. 预算耗尽：rounds 达 adaptive_max_rounds()
        附加防护：查询幂等停——本轮 gap.queries 与上轮完全相同 → break
        （同查询重复补检是浪费）。
        返回最终 retrieval_results（含补检合并）；未触发返回 None。
        """
        from src.adaptive_retrieval import (
            adaptive_max_rounds,
            detect_gap,
            merge_supplement_results,
        )

        state = state if state is not None else {"rounds_used": 0, "traces": []}

        top_score: Optional[float] = None
        for item in retrieval_results:
            score = item.get("combined_score")
            if score is not None:
                numeric = float(score)
                top_score = numeric if top_score is None else max(top_score, numeric)

        gap = detect_gap(rewrite_result, retrieval_results, top_score=top_score)
        if gap is None:
            return None

        current_results = list(retrieval_results)
        prior_queries: List[str] = []
        rounds_this_call = 0

        while gap is not None:
            if state["rounds_used"] >= adaptive_max_rounds():
                break                                   # 收敛条件 3：预算
            if gap.supplement_queries == prior_queries:
                break                                   # 查询幂等停
            prior_queries = list(gap.supplement_queries)

            strategy = _choose_strategy(                # ③ 接入点
                round_index=state["rounds_used"],
                prior_evidence_gain=state.get("last_evidence_gain", True),
            )
            parent_override = "page" if strategy == "parent_recall" else None
            supplement = self._collect_supplement(
                retriever, company_name, rewrite_result, gap,
                candidate_doc_ids=candidate_doc_ids,
                parent_mode_override=parent_override,
            )

            merged = merge_supplement_results(current_results, supplement)
            state["last_evidence_gain"] = len(merged) > len(current_results)
            if not state["last_evidence_gain"]:
                break                                   # 收敛条件 2：无新增
            current_results = merged

            current_results = retriever.rerank_candidates(
                query=question,
                candidate_results=current_results,
                documents_batch_size=self.final_reranking_batch_size,
                top_n=self.retrieval_debug_top_n,
                task_type=getattr(rewrite_result, "task_type", None),
            )
            state["rounds_used"] += 1
            rounds_this_call += 1
            state["traces"].append({
                "question": question, "company_name": company_name,
                "trigger": gap.trigger, "strategy": strategy,
                "queries": list(gap.supplement_queries),
                "new_hits": len(supplement), "rounds": state["rounds_used"],
                "atom_id": (getattr(rewrite_result, "metric_atom", None) or {}).get("atom_id"),
            })

            gap = detect_gap(rewrite_result, current_results)   # 重判：条件 1

        self._record_adaptive_trace(state, state["traces"].pop()) if state["traces"] else None
        # ↑ 末轮 trace 已含聚合信息；空保护见边界表
        return current_results if rounds_this_call else None
```

**v1 → v2 行为对照**：

| 方面 | v1 | v2 |
|---|---|---|
| 循环 | `while ... gap = None`（强制单轮） | 真循环 + 三条件收敛 |
| 返回 | `{"adaptive_retrieval": trace, "retrieval_results": ...}` | 直接返回 `List[Dict]`（简化：调用方只关心结果池；trace 走 state） |
| 预算 | 方法内局部计数 | state 共享池（预检+闭环总预算） |
| trace | 每次调用直写文件 | state 聚合，主流程统一落盘 |
| rerank | 合并后一次 | 每轮一次（池在变） |

### 2.3 边界条件表（②）

| 边界 | 处理 |
|---|---|
| detect_gap 首判 None | 返回 None（零成本路径，调用方不替换结果池） |
| 循环 0 轮（预算已被闭环/前次耗尽） | 返回 None |
| 补检结果与现有池全重复 | merge 长度不变 → break（条件 2） |
| rerank 抛异常 | 向上传播（fail-closed；预检失败不做降级重试） |
| state 为 None（旧调用方兼容） | 方法内建临时 state——v2 内部调用必传 |

### 2.4 测试构造骨架（C1-C4 共用 fixture）

```python
def _make_processor_with_fake_retriever(supplement_pages_by_round):
    """fake retriever：rerank_candidates 原样排序返回；retrieve 按 round 计数
    返回不同页（控制'有无新增证据'）。processor 用 object.__new__ 裸构造。"""
```

| # | 用例 | 构造要点 | 期望 |
|---|---|---|---|
| C1 | 缺口闭合即停 | round1 补检回含 targets 文本的结果 | rounds_this_call==1；detect_gap 二次为 None；返回池含新页 |
| C2 | 无新增即停 | supplement 全与 base 同 (sha,page,scope) | 1 轮后停；返回池 len 不变 |
| C3 | 查询幂等停 | gap.queries 两轮相同（mock detect_gap 恒返同 gap） | 第 2 轮 break，检索调用数==首轮 queries 数 |
| C4 | 预算上限 | max_rounds=2 + 永远新证据 | 恰 2 轮；rounds_used==2 |

---

## 3. ③ 行动策略多样化

### 3.1 RetrievalGap 扩展与 _choose_strategy 完整实现

```python
# adaptive_retrieval.py
STRATEGY_DENSE_RECALL = "dense_recall"
STRATEGY_PARENT_RECALL = "parent_recall"


@dataclass(frozen=True)
class RetrievalGap:
    trigger: str
    supplement_queries: List[str] = field(default_factory=list)
    rationale: str = ""
    strategy: str = STRATEGY_DENSE_RECALL      # 新增：由 _choose_strategy 填充


def _choose_strategy(round_index: int, prior_evidence_gain: bool) -> str:
    """策略选择（规则版决策器；v3 换 LLM 时接口不变）。

    round 0：dense_recall（任何触发）。
    round ≥1：上轮有新增证据 → 维持 dense（有效就继续）；
              上轮无新增 → parent_recall（粒度切换是低置信的第二手段——
              child chunk 证据不全时回溯 page 级上下文）。
    """
    if round_index == 0 or prior_evidence_gain:
        return STRATEGY_DENSE_RECALL
    return STRATEGY_PARENT_RECALL
```

策略选择放 adaptive_retrieval.py（纯函数可测），_run_adaptive_supplement
调用（§2.2 已埋接入点）。

### 3.2 merge 键升级（diff）

```python
# adaptive_retrieval.py — merge_supplement_results 改动
# v1: key = (str(item.get("pdf_sha1") or ""), item.get("page"))
# v2:
def _merge_key(item: Dict) -> Tuple[str, object, str]:
    """去重键含粒度维度：child/parent 同页共存（parent 页级证据与 child
    chunk 证据不是重复信息）。scope 判定复用 retrieval.py L130 语义。"""
    scope = (
        item.get("result_scope")
        or (item.get("metadata") or {}).get("node_type")
        or "child"
    )
    return (str(item.get("pdf_sha1") or ""), item.get("page"), str(scope))
```

函数体内两处 `key = (...)` 换 `_merge_key(item)`。

### 3.3 metadata_reroute 设计预留（不实施）

触发条件（未来）：gold 年份/报告类型不在候选池 doc_ids → ReportCatalog
重路由 → 整轮重检。涉及路由层状态回滚，收益依赖多文档题占比，本轮不做。
实施时挂点：`_collect_supplement` 前置判定 `route_info["selected_report"]`
的时间范围 vs `rewrite_result.filters`。

### 3.4 测试用例（S1-S5）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| S1 | 首轮策略 | _choose_strategy(0, True/False) | dense_recall |
| S2 | 粒度切换 | _choose_strategy(1, prior_evidence_gain=False) | parent_recall |
| S3 | 有效不切换 | _choose_strategy(1, prior_evidence_gain=True) | dense_recall |
| S4 | merge 键 scope | base=[child(p1)] supplement=[parent(p1), child(p1)] | 2 条（child 新被去重、parent 保留） |
| S5 | 端到端粒度覆盖 | round1 无新增 → round2 | fake retriever 收到 parent_retrieval_mode="page" |
| S6 | RetrievalGap 默认 | 不传 strategy | "dense_recall"（向后兼容） |

---

## 4. ④ trace 反哺知识层

### 4.1 trace 聚合落盘（_record_adaptive_trace + 主流程收尾）

```python
    # questions_processing.py — 新增（替代 v1 的 _append_adaptive_trace 调用点）
    @staticmethod
    def _record_adaptive_trace(state: Dict, entry: Optional[Dict] = None) -> None:
        """trace 入 state（entry 非 None）或统一落盘（entry 为 None）。"""
        if entry is not None:
            state.setdefault("traces", []).append(entry)
            return
        for trace in state.get("traces", []):
            _append_adaptive_trace_line(trace)     # v1 的 JSONL 追加逻辑改名复用
```

主流程收尾（`return validated.answer` 之前）：

```python
        self._record_adaptive_trace(adaptive_state)   # 落盘一次（无 trace 则空操作）
```

### 4.2 分析脚本（scripts/analyze_adaptive_trace.py 完整核心）

```python
"""④ trace 反哺：自适应检索运行经验 → 知识资产进化建议。

只出建议不自动改库（Builder 永不自批）：建议经人工确认 → 编辑 yaml →
compile_atoms.py --update → 既有等价测试与门禁守护。
三条规则：
R1 高命中词扩 aliases：query 命中率 > 80% 且 gap 常闭 → 检索侧常靠它命中，
   查询改写侧应预展开（aliases 命中即免补检一轮）。
R2 零命中 target 复核：某 evidence_target 触发补检后 new_hits 全 0 →
   文档可能从不这么写，表述需复核。
R3 无触发无操作：原子从未触发补检 → 知识已足够，不是缺口。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

MIN_HIT_RATE_FOR_ALIAS = 0.8


def load_trace(path: Path) -> tuple[List[Dict], int]:
    """JSONL 读取；脏行跳过并计数（缺口显式）。"""
    entries, skipped = [], 0
    if not path.exists():
        return entries, skipped
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    return entries, skipped


def _query_stats(entries: List[Dict]) -> Dict[str, Dict]:
    """query → {hits, total}：new_hits>0 记一次命中。"""
    stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
    for entry in entries:
        queries = entry.get("queries") or []
        hit = (entry.get("new_hits") or 0) > 0
        for query in queries:
            stats[query]["total"] += 1
            if hit:
                stats[query]["hits"] += 1
    return stats


def per_atom_stats(entries: List[Dict]) -> List[Dict]:
    rows = []
    by_atom: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        by_atom[entry.get("atom_id") or "-"].append(entry)
    for atom_id, atom_entries in sorted(by_atom.items()):
        n = len(atom_entries)
        rows.append({
            "atom_id": atom_id,
            "triggers": n,
            "trigger_dist": dict(__import__("collections").Counter(
                e.get("trigger") for e in atom_entries)),
            "strategy_dist": dict(__import__("collections").Counter(
                e.get("strategy") for e in atom_entries)),
            "rounds_total": sum(e.get("rounds") or 0 for e in atom_entries),
            "gap_closed_rate": round(
                sum(1 for e in atom_entries if e.get("gap_closed")) / n, 4),
            "outcome_dist": dict(__import__("collections").Counter(
                e.get("outcome") for e in atom_entries)),
        })
    return rows


def alias_expansion_suggestions(entries: List[Dict], store) -> List[Dict]:
    """R1/R2（R3 无操作）。建议含 evidence 计数，供人工裁决。"""
    suggestions = []
    query_stats = _query_stats(entries)
    atom_ids = {e.get("atom_id") for e in entries if e.get("atom_id")}
    for atom_id in sorted(atom_ids):
        atom = store.get(atom_id)
        if atom is None:
            continue
        for query, stat in sorted(query_stats.items()):
            if stat["total"] < 3:
                continue                      # 样本不足不出建议
            hit_rate = stat["hits"] / stat["total"]
            if hit_rate >= MIN_HIT_RATE_FOR_ALIAS and query not in atom.aliases:
                suggestions.append({
                    "rule": "R1_expand_alias",
                    "atom_id": atom_id,
                    "term": query,
                    "evidence": f"补检命中率 {hit_rate:.0%}（{stat['hits']}/{stat['total']}）",
                })
            if hit_rate == 0.0 and query in atom.evidence_targets:
                suggestions.append({
                    "rule": "R2_review_target",
                    "atom_id": atom_id,
                    "term": query,
                    "evidence": f"补检零命中（0/{stat['total']}）——文档可能无此表述",
                })
    return suggestions


def render_report(rows: List[Dict], suggestions: List[Dict], skipped: int) -> str:
    """Markdown 报告（模板见 §4.3）。"""
    lines = ["# Adaptive Trace 分析报告", "",
             f"- trace 条数：{sum(r['triggers'] for r in rows)}",
             f"- 脏行跳过：{skipped}", "", "## 每原子统计", "",
             "| atom | 触发 | 收敛率 | outcome 分布 |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['atom_id']} | {row['triggers']} | "
                     f"{row['gap_closed_rate']} | {row['outcome_dist']} |")
    lines += ["", "## 建议（人工确认后入库）", ""]
    if not suggestions:
        lines.append("（无——样本不足或知识已覆盖）")
    for suggestion in suggestions:
        lines.append(f"- **{suggestion['rule']}** {suggestion['atom_id']}: "
                     f"`{suggestion['term']}` —— {suggestion['evidence']}")
    lines += ["", "> 入库流程：人工确认 → 编辑 config/metric_atoms.yaml → "
              "python scripts/compile_atoms.py --update（等价测试与门禁守护）"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="adaptive trace → 知识进化建议")
    parser.add_argument("--trace", type=Path, default=Path("logs/adaptive_trace.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/adaptive_analysis.md"))
    args = parser.parse_args()
    entries, skipped = load_trace(args.trace)
    rows = per_atom_stats(entries)
    from src.metric_atoms import get_default_store
    suggestions = alias_expansion_suggestions(entries, get_default_store())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(rows, suggestions, skipped), encoding="utf-8")
    print(f"报告已写入 {args.output}（{len(entries)} 条 trace，{len(suggestions)} 条建议）")


if __name__ == "__main__":
    main()
```

### 4.3 测试用例（tests/test_adaptive_trace_analysis.py）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| A1 | 每原子统计 | 3 条 trace（atom MET-001 ×2，gap_closed 1） | triggers=2、gap_closed_rate=0.5、outcome_dist 正确 |
| A2 | R1 建议触发 | query "扣除非经常性损益后的净利润" 3/3 命中，不在 aliases | 建议 R1_expand_alias MET-001 |
| A3 | R2 建议触发 | 某 target 0/3 命中 | 建议 R2_review_target |
| A4 | 样本不足 | total=2 的高命中 query | 不出建议（<3） |
| A5 | 已在 aliases | 高命中 query 已是 aliases 成员 | 不出 R1（无需扩） |
| A6 | 脏行防御 | 混入 1 行非法 JSON | skipped=1，其余正常 |
| A7 | 空文件/缺失 | 不存在的路径 | 空报告不炸 |
| A8 | R3 无操作 | 原子零触发 | 该原子不出现在任何建议 |

---

## 5. 统一验收与实施检查单

```bash
# 每个 commit 后：
./venv/bin/python -m unittest tests.test_adaptive_retrieval tests.test_adaptive_trace_analysis -v
./venv/bin/python -m unittest discover -s tests -p "test_*.py"   # 默认关闭 → 零回归
# ④ 完成后端到端冒烟：
echo '{"question":"国网信通扣非归母净利润是多少","schema":"number"}' | \
  ADAPTIVE_RETRIEVAL_ENABLED=1 ./venv/bin/python -c "..."        # 手动冒烟（无索引则标注）
./venv/bin/python scripts/analyze_adaptive_trace.py --trace logs/adaptive_trace.jsonl
```

| 项 | 离线验收 | 端到端（卡 key） |
|---|---|---|
| ① 闭环 | P1-P8：三触发源映射 / 无关 flag 不触发 / try 分支（二次生成被调 + refused 消失）/ then-closed（维持 N/A）/ 预算共享（预检耗尽则闭环不执行）/ 默认关闭生成仅一次 | 拒救率（refused→answered） |
| ② 多轮 | C1-C4：三收敛条件 + 幂等 + 预算 | 轮次分布 |
| ③ 策略 | S1-S6：选择逻辑 / merge 键 / 粒度覆盖端到端 | 各策略命中率 |
| ④ 反哺 | A1-A8 + 模拟 trace 报告产出 | 真实 trace 首轮报告 |

**纪律**：独立 commit；现有断言不改通过；ADAPTIVE_RETRIEVAL_ENABLED 默认
关闭保证全量零回归（458 → ~480）；trace 字段以 §4.1/§2.2 schema 为准；闭包
改造的机械正确性由全量测试守护（缩进/变量替换错误会在 458 个测试里当场炸）。

**风险与回退**：最大风险是闭包缩进改造的机械错误——缓解：改造后立即跑
test_questions_processing 相关全量；回退：ADAPTIVE_RETRIEVAL_ENABLED=0 回
v1，单 commit revert 回 v1 代码。
