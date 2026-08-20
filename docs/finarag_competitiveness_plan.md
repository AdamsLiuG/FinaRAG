# FinaRAG 竞争力提升实施计划（详细版 v2）

> 立项：2026-08-21。v2 修订：按 FinaWiki 方案标准扩至函数签名/测试用例编号级；
> 修正 v1 两处错误假设（debug payload 实存错误堆栈非过程数据；F3 数据源改定为
> finance_eval_benchmark_v2 的 gold）。所有行号基于当前分支实测。

## 0. 总览：顺序、批次、工时、commit 划分

```
第一批（可并行验收，~1.5 天）
  A1 融合量纲修复 + A4 题型加权     commit×2（修复 / 加权联动）
  F3 错误归因 v1                    commit×1
第二批（~2 天）
  A2 并行检索三层                   commit×2（索引常驻 / 四路+多查询并行）
  Agentic 自适应检索                commit×1
第三批（~1.5 天）
  D1 并发审计补齐 + D3 服务化       commit×2
```

依赖：A1 先于 A2（融合层稳定后再动检索结构）；F3 基线矩阵在 A1 修复**前**跑一次
（存档），修复后跑第二次形成对比；Agentic 依赖 A1 的 rerank 路径；D3 依赖 A2。

---

## 1. A1：融合量纲修复 + A4 题型加权

### 1.1 现状与代码证据

**rerank 层三个类**（`src/reranking.py`）：

| 类 | 行号 | relevance_score 来源 | 融合位置 |
|---|---|---|---|
| `FlagEmbeddingReranker` | L250 | `_compute_scores` → L140 `torch.sigmoid` → 0-1 | L346 |
| `VLLMApiReranker` | L362 | `_normalize_remote_scores`（L44：全 0-1 保留，否则 min-max） | L466 |
| `LLMReranker` | L482 | LLM 打分（经解析） | L529 `_build_scored_document` |

三处融合式相同：`combined = llm_weight * relevance + vector_weight * distance`，
`distance` 量纲随检索层漂移（`retrieval.py` L744 默认 `HYBRID_RETRIEVAL_FUSION=rrf`）：

| fusion | distance 实际 | 0.7/0.3 名义权重的实际效果 |
|---|---|---|
| rrf（默认） | rrf_score=Σ1/(60+rank)，量级 <0.1 | 向量路实际 ~2% |
| average | 四路原始分直平（L885-889），BM25 无上界 0-30+ | BM25 淹没其余信号 |

**LLMReranker 的结构特殊性**：`_rerank_single_document`（L536）逐文档调用
`_build_scored_document`——单文档视角无法做池内归一化，需改为「先收集全部
raw 分，再批量融合」。

### 1.2 详细设计

**新增（reranking.py 模块级）**：

```python
def _minmax_normalize(values: list[float]) -> list[float]:
    """池内 min-max；max==min（常数池）时全 0.5，空列表返回空。"""

def _fuse_scores(rel_scores: list[float], dist_scores: list[float],
                 llm_weight: float) -> list[float]:
    """两路各自池内归一到 0-1 后加权。语义：combined_score 变为池内相对强度，
    名义权重 = 实际权重。长度不一致取 min 截断并记 warning。"""
```

**三个调用点改造**：

1. `FlagEmbeddingReranker.rerank_documents`：L342-349 循环改为先收集
   `[(doc, score)]`，循环后 `fused = _fuse_scores(scores, distances, llm_weight)`
   一次性赋值。
2. `VLLMApiReranker.rerank_documents`：同上（L460-468）。
3. `LLMReranker.rerank_documents`：保留 `_rerank_single_document` 只做打分与
   raw 记录（`doc["_raw_relevance"]`），主循环结束后新增批量融合步骤，再删除
   临时字段；`_build_scored_document` 退化为「赋 relevance + 融合延后」。

**检索层 average 模式修复**（`retrieval.py`）：

```python
def _normalize_backend_scores(results: list[dict], score_field: str) -> None:
    """单路 top_n 池内 min-max 归一，写回 result[score_field]。在 L872 聚合循环
    之前，对四路各自调用一次。rrf 路径不经过此函数（rank-based 天然免量纲）。"""
```

average 聚合（L885-889）改为消费归一化后的分数。

**A4 题型加权（`config/retrieval_fusion_weights.yaml` 全文）**：

```yaml
# 题型 → rerank 融合的 llm_weight（知识层驱动：task_type 由 QueryPlan 路由产出）
version: "1.0"
task_type_weights:
  definition_qa:        {llm_weight: 0.85}  # 定义类：语义信号主导
  single_metric_lookup: {llm_weight: 0.50}  # 数字类：精确词（sparse/bm25）补强
  metric_aggregation:   {llm_weight: 0.50}
  comparison:           {llm_weight: 0.55}
  implicit_intent_qa:   {llm_weight: 0.60}  # 剧本类：混合证据
  default:              {llm_weight: 0.70}
```

加载器（reranking.py）：`resolve_llm_weight(task_type: Optional[str], default: float) -> float`，
yaml 缺失/坏值回退 default 并 warning；权重越界 (0,1) 加载期 ValueError。
`rerank_documents` 增可选参数 `task_type: Optional[str] = None`；
`questions_processing` 调用点传 `query_plan.task_type`（两个调用处：
hybrid_rerank 主路径与 multi-query 路径）。

### 1.3 测试用例表（`tests/test_score_fusion.py`，12 例）

| # | 用例 | 输入 | 期望 | 防护意图 |
|---|---|---|---|---|
| T1 | rrf 量纲 | rel=[0.9,0.1], dist=[0.03,0.02], w=0.7 | 排序=doc0 优先 | 修复前向量差异被抹平 |
| T2 | BM25 量纲 | rel=[0.5,0.9], dist=[25.0,24.9], w=0.3 | doc1 优先 | 修复前 BM25 主导 |
| T3 | 常数池 | rel=[0.5,0.5], dist=[0.1,0.1] | 全 0.5*w+0.5*(1-w) | 除零保护 |
| T4 | 单元素池 | rel=[0.7], dist=[999] | [0.5*w+0.5*(1-w)] | 退化定义 |
| T5 | 权重边界 | w=1.0 / w=0.0 | 纯一路 | 端点语义 |
| T6 | LLMReranker 批量融合 | mock 3 文档不同 rel | combined 为归一后融合（非逐个） | 结构改造正确性 |
| T7 | average 模式归一 | vector=[0.9,0.1], bm25=[30,1] | 两路同权参与平均 | 检索层修复 |
| T8 | rrf 模式不变 | 现有 rrf 聚合输入 | 输出与旧逻辑全等 | 不误伤正确路径 |
| T9 | 题型权重加载 | yaml 各题型 | resolve 返回对应值 | A4 |
| T10 | 未注册题型 | task_type="novel" | 回落 default 0.70 | 容错 |
| T11 | 越界权重 | llm_weight: 1.5 | ValueError | 加载期校验 |
| T12 | yaml 缺失 | 文件不存在 | 回退调用方 default + warning | 部署容错 |

### 1.4 验收命令与风险

```bash
./venv/bin/python -m unittest tests.test_score_fusion tests.test_reranking -v   # 新12 + 现有回归
./venv/bin/python -m unittest discover -s tests -p "test_*.py"                  # 全量（失败维持6个历史项）
cd eval && ../venv/bin/python retrieval_rank_eval.py \
  --answers-file <跑分产物> --reference-answers data/finance_eval_benchmark_v2/core120/answers_gold.json \
  --recall-k 10 --precision-k 3 --output reports/rank_eval_a1.json
```

风险与回退：combined_score 语义变化（绝对分→池内相对分）影响下游若按绝对阈值
过滤的逻辑——grep 确认下游消费仅排序（`sort(key=combined_score)`），无绝对阈值
消费点；若发现消费点，加 `raw_distance` 字段保留原始值。回退：单 commit revert。

---

## 2. F3：错误归因自动化 v1（基于现有 gold + references）

### 2.1 v1 修正后的数据基础（v1 方案假设有误，实测修正）

- `data/finance_eval_benchmark_v2/core120/`：120 题，gold 含 `gold_pages` /
  `gold_chunk_ids` / `doc_ids` / `should_refuse` / `capability` / `metric_name`。
  gold[0] 即法代题（question_id=zh-top10-01-01, gold_pages=[7]）。
- 预测侧：跑分产物的每题 `references`（含 pdf_sha1/page_index）+ `value`。
- **debug payload（answers_*_debug.json）实测不可用**：answer_details 存的是
  逐题错误堆栈（questions_processing._process_single_question 的 traceback），
  非过程数据。→ v1 归因不依赖它；过程级归因留给 D3 trace（v2 升级路径）。

### 2.2 v1 归因树（贴现有数据可判定的五分支）

```
逐题（pred × gold 对齐 by question_id）短路判定：
V1 should_refuse=true 且 pred 未拒答（value != "N/A"）      → refusal_policy（该拒不拒）
V2 should_refuse=true 且拒答                                  → correct_refusal（正确，非错误）
V3 pred.value == "N/A" 且非 refuse 题                          → over_refusal（不该拒而拒）
V4 pred.references 为空但 value 非 N/A                        → ungrounded_generation（无证据作答）
V5 references 不含 gold_pages（同 doc 比对 page_index）       → evidence_mislocalization
   └─ v2 细分前置条件：D3 trace 落地后拆 retrieval_miss / rerank_sink
V6 references 含 gold 页但 value != gold_value                → generation_error（证据对答案错）
V7 value == gold_value 但引用页错误                            → citation_error
其余                                                          → correct
```

输出（`attribution_matrix`）：
```json
{"stage_counts": {"evidence_mislocalization": 18, "generation_error": 9, ...},
 "stage_rates": {...}, "n": 120, "unknown": 0,
 "by_capability": {"metric_numeric": {"evidence_mislocalization": 10, ...}, ...}}
```

### 2.3 文件与函数（新增 `eval/stage_attribution.py`）

```python
STAGES = ["refusal_policy", "over_refusal", "ungrounded_generation",
          "evidence_mislocalization", "generation_error", "citation_error"]

def attribute_case(pred: dict, gold: dict) -> str          # 返回 stage 或 correct_refusal/correct
def _references_cover_gold_pages(pred: dict, gold: dict) -> bool
def attribution_matrix(preds: list[dict], golds: list[dict]) -> dict
def main()  # CLI: --pred answers.json --gold answers_gold.json --output reports/...
```

值比对复用 `eval/error_analysis.py` 的 `_normalize_answer`（数值容差/多值集合语义
已处理，不重造）。断点测试数据缺失（gold_pages 空）→ 该题 stage=unknown 显式计数。

### 2.4 测试用例表（`tests/test_stage_attribution.py`，8 例）

| # | 用例 | 构造 | 期望 stage |
|---|---|---|---|
| T1 | 该拒不拒 | should_refuse=true, value="123" | refusal_policy |
| T2 | 正确拒答 | should_refuse=true, value="N/A" | correct_refusal |
| T3 | 过度拒答 | refuse=false, value="N/A" | over_refusal |
| T4 | 无证据作答 | value="123", references=[] | ungrounded_generation |
| T5 | 证据错位 | references 页=[3], gold 页=[7] | evidence_mislocalization |
| T6 | 证据对答案错 | references 页=[7], value≠gold | generation_error |
| T7 | 答案对引用错 | value==gold, references 页=[3] | citation_error |
| T8 | gold 缺页 | gold_pages=[] | unknown（计数不归错） |

### 2.5 验收

```bash
./venv/bin/python -m unittest tests.test_stage_attribution -v
# 对存量跑分（任一 answers_*.json 若与 core120 对齐）产出基线矩阵；不齐则标注
# 「待 A1 修复后统一跑分产出」，基线矩阵挂 NULL 不伪造
```

---

## 3. A2：并行检索 + 索引常驻（含 A3 设计）

### 3.1 现状与代码证据

| 串行点 | 位置 | 证据 |
|---|---|---|
| 四路顺序调用 | `retrieve_candidates_by_company_name`（retrieval.py L899+） | L920/L930/L939/L948 四个 if 顺序执行 |
| BM25 反序列化 | L388 `pickle.load(f)` | 每次检索整文件加载（数十 MB） |
| sparse 反序列化 | L438 `pickle.load(f)` | 同上 |
| 多查询串行 | `questions_processing._retrieve_multi_query_candidate_runs` | for 循环逐查询跑四路 |

### 3.2 详细设计（三层，独立 commit）

**层 1（commit 1）：索引常驻缓存**

```python
# retrieval.py 模块级
@lru_cache(maxsize=None)  # 外层自管容量，见下
def _load_pickle_cached(path_str: str, mtime_ns: int):
    """mtime 进键：索引重建（mtime 变）自动失效；lru_cache 负责驱逐。"""
    with open(path_str, "rb") as f:
        return pickle.load(f)

def _load_index(path: Path):
    stat = path.stat()
    obj = _load_pickle_cached(str(path), stat.st_mtime_ns)
    _INDEX_CACHE_SLOTS.record(str(path))   # 有序 LRU 名册，超预算驱逐
    return obj
```

容量控制：`INDEX_CACHE_MAX_SLOTS`（env，默认 16）。自建名册的原因：
`lru_cache` 按调用键驱逐而非「索引个数」，直接用会把同一索引的旧 mtime 键
挤掉而非最旧索引。名册记 path → 最近使用序，超限调用
`_load_pickle_cached.cache_clear` 级联或逐键失效（记录键集合）。
内存预算估算进 docstring（单索引 ~数十 MB × 16 slot ≈ 0.5-1GB 上限）。

**层 2（commit 2a）：四路并行**

```python
def retrieve_candidates_by_company_name(...):
    backends = [("vector", self.vector_retriever, scope_ok_vector),
                ("bm25", self.bm25_retriever, backend_scope == "all"),
                ("sparse", self.sparse_retriever, backend_scope == "all"),
                ("tag", self.tag_retriever, backend_scope == "all")]
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, retriever, enabled in backends:
            if retriever is None or not enabled:
                continue
            futures[name] = pool.submit(self._safe_backend_retrieve,
                                        retriever, name, **retrieve_kwargs)
    # _safe_backend_retrieve：try/except → 失败返回 [] 并 logger.warning（含异常类型）；
    # 单路失败不拖垮整体（现状：一路抛错整个检索失败）
    vector_results = futures.get("vector", ...).result() 等  # 按 backend 名取回
    # 聚合逻辑（_merge_...）不变——输入顺序与串行版一致 → 输出确定一致
```

**层 3（commit 2b）：多查询并行**

`_retrieve_multi_query_candidate_runs` 外层 for 改
`ThreadPoolExecutor(max_workers=min(4, len(search_queries)))`；内层（层 2）也用
线程池 → 最坏 4×4=16 线程，设全局上限 `RETRIEVAL_MAX_THREADS`（env，默认 8）：
层 2 的 max_workers 取 `min(4, 上限 // 并行查询数)`。

**A3（全局索引 + 元数据预过滤）——仅设计归档**

现状：`target_report["vector_db"]`（retrieval.py L650 附近）按公司独立库，
`candidate_doc_ids` 循环多库。10 万文档演进路线：
1. 全局单 FAISS 库，doc_id 进 IDMap；`RetrievalFilters` → 预过滤 IDSelector
  （公司/年份/报告类型在向量检索前收敛候选集，免多库循环）；
2. BM25/sparse 并入同样的 doc 级预过滤（tokenizer 层倒排索引带 doc 前缀）；
3. 索引版本管理：Generation 同款 hash 语义（索引 buildId + 源 PDF 清单 hash）。
不在本轮实施；作为「200 → 10 万份」面试题的架构演进答案。

### 3.3 测试用例表（`tests/test_parallel_retrieval.py`，8 例）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| T1 | 并行==串行 | 同输入跑两版（保留串行实现为测试对照） | 输出列表全等（含顺序） |
| T2 | 单路异常降级 | mock bm25 抛 RuntimeError | 其余三路结果正常返回 + warning |
| T3 | backend_scope=vector_only | 只挂 vector | 仅 vector 结果 |
| T4 | 缓存命中 | 同 path 两次 _load_index | 第二次不读盘（mock stat/pickle 计数） |
| T5 | mtime 失效 | 同 path 改 mtime | 重新加载 |
| T6 | 容量驱逐 | 装满 maxsize+1 个索引 | 最旧的被驱逐（名册断言） |
| T7 | 多查询并行一致 | 3 条查询 | 与串行版逐查询结果全等 |
| T8 | 线程上限 | 上限 8、3 查询 | 层 2 workers=min(4, 8//3)=2 |

### 3.4 验收

```bash
./venv/bin/python -m unittest tests.test_parallel_retrieval -v
# 延迟基准（现成脚本扩展 per-stage 计时后）：
./venv/bin/python eval/benchmark_ttft.py --queries <采样集> --output eval/reports/latency_a2.json
# 修复前后 P50/P95 对比进报告；注意：本机无 GPU 时 embedding 路延迟失真，标注环境
```

---

## 4. Agentic 自适应检索（知识层驱动，try-then-closed）

### 4.1 触发-构造-预算三表（核心设计）

**触发判定**（`detect_gap`，按优先级短路）：

| 优先级 | trigger | 判定输入 | 语义 |
|---|---|---|---|
| 1 | `caliber_conflict` | validation_flags 含 `metric_caliber_conflict_refused`（Stage C 门禁已拒答） | 证据只有易混口径——补检本口径 |
| 2 | `metric_evidence_mismatch` | flags 含 `metric_evidence_mismatch` 且 query_plan.metric_atom 非空 | 原子证据目标未命中 |
| 3 | `playbook_incomplete` | flags 含 `implicit_intent_evidence_incomplete` 且有 playbook | 剧本必需指标缺证据 |
| 4 | `low_confidence` | retrieval_results 为空 或 top combined_score < `ADAPTIVE_MIN_TOP_SCORE`（env，默认 0.15） | 检索置信度不足 |

**补检查询构造规则**（`_build_supplement_queries`，纯函数）：

| trigger | 查询来源（FinaWiki 知识资产） | 查询数上限 |
|---|---|---|
| caliber_conflict | 本原子 evidence_targets（**不含**易混原子 standard_name——避免强化错口径） | 3 |
| metric_evidence_mismatch | 本原子 aliases ∪ evidence_targets 去重 | 3 |
| playbook_incomplete | 未覆盖指标的 standard_name（剧本 required_metrics 逐个） | 4 |
| low_confidence | 本原子 aliases；无原子时用原查询去停用词重写 | 2 |

所有补充查询拼公司名（保持 company_name 过滤一致性）。

**预算与停止**：`adaptive_retrieval.enabled`（env，默认 **False**——回归零影响）；
`ADAPTIVE_MAX_ROUNDS`（默认 1，防循环）；补检结果 `merge_supplement_results`
（按 chunk_id/sha1+page 去重合并，新结果带 `retrieval_sources += ["adaptive"]` 标记）
→ 重跑一次 rerank（复用现有 reranker 实例，不重建）→ 重新生成 → 再校验。
二次仍触发同 trigger → 不再补检，按原 fail-closed 语义收尾（try-then-closed 的
closed 分支）。

### 4.2 模块与挂点

新增 `src/adaptive_retrieval.py`（纯函数，无 IO，控制流全离线可测）：

```python
@dataclass(frozen=True)
class RetrievalGap:
    trigger: str
    supplement_queries: List[str]
    rationale: str

def detect_gap(query_plan, validation_flags: List[str],
               retrieval_results: List[dict],
               top_score: Optional[float]) -> Optional[RetrievalGap]
def _build_supplement_queries(trigger: str, query_plan, playbook) -> List[str]
def merge_supplement_results(base: List[dict], supplement: List[dict]) -> List[dict]
```

挂点（`questions_processing.get_answer_for_company`）：validate_answer 调用之后、
返回组装之前——插入「补检决策点」：
```
validated = validate_answer(...)
if enabled and rounds_used < MAX_ROUNDS:
    gap = detect_gap(query_plan, validated.validation_flags, retrieval_results, top_score)
    if gap:
        supplement = self._run_retrieval(retriever, mode, company_name, 每条 gap 查询, filters)
        merged = merge_supplement_results(retrieval_results, supplement)
        → rerank → 重新生成 → 再 validate（rounds_used += 1）
        trace: {"question_id", "trigger", "queries", "new_hits": len(supplement), "outcome"}
```

trace 落 `logs/adaptive_trace.jsonl`（追加式；D3 的 trace 体系前身）。

### 4.3 测试用例表（`tests/test_adaptive_retrieval.py`，9 例）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| T1 | 口径冲突触发 | flags=[metric_caliber_conflict_refused] + MET-001 plan | trigger=caliber_conflict，queries 含"扣除非经常性损益后的净利润"且**不含**"归母净利润" |
| T2 | 证据不匹配触发 | flags=[metric_evidence_mismatch] | queries ⊆ 原子 aliases∪targets，≤3 条 |
| T3 | 剧本缺指标触发 | flags=[implicit_intent_evidence_incomplete] + PLB-002 plan（缺 MET-005 证据） | queries 含"货币资金" |
| T4 | 低置信触发 | top_score=0.05 | trigger=low_confidence |
| T5 | 不触发（零成本） | flags 无相关 + top_score=0.8 | detect_gap 返回 None（无任何检索副作用） |
| T6 | try 分支 | mock：首轮只回归母，补检回扣非 | 最终作答，validation_flags 无 refused |
| T7 | then-closed 分支 | mock：补检仍只回归母 | 维持 N/A 拒答（Stage C 语义不变） |
| T8 | 预算上限 | 永远触发场景 | 只补检 1 轮（计数断言） |
| T9 | 默认关闭 | enabled=False | 全流程无补检调用（mock 计数=0） |

### 4.4 验收

```bash
./venv/bin/python -m unittest tests.test_adaptive_retrieval -v
./venv/bin/python -m unittest discover -s tests -p "test_*.py"   # 默认关闭 → 全量零变化
```
端到端收益（卡 API key）：F3 矩阵中 evidence_mislocalization / over_refusal
占比在开关前后对比。

---

## 5. D1+D3：并发审计 + 服务化与可观测

### 5.1 D1 现状实测（比 v1 方案预估的好，仍有缺口）

| 状态 | 证据 | 评估 |
|---|---|---|
| `self._retriever_cache = threading.local()` | L186 | 线程隔离 ✓ |
| `self._document_cache` 无锁读 | L340 `if doc_id in self._document_cache`（锁外） | CPython GIL 下无害，但 check-then-act 在写锁内、读锁外——**统一到锁内**（规范修复） |
| `self._revenue_grounding_cache` + Lock | L188-189 | ✓ 已有锁 |
| `reranker.last_debug` | reranking.py 各类实例属性 | **请求间共享残留**——多线程下后写覆盖先写；服务化下改为方法级返回值传递 |

D1 commit 内容：读路径入锁；`last_debug` 改为 `rerank_documents(..., collect_debug=True)` 返回值携带（调用点同步改）；全量回归。

### 5.2 D3 服务层（`src/service/app.py` + `src/service/tracing.py`）

```python
# app.py 骨架
app = FastAPI(title="FinaRAG QA Service")

@app.post("/qa")
def qa(req: QARequest) -> QAResponse:           # question, schema, company_name
    # QARequest/QAResponse pydantic 模型；内部调 QuestionsProcessor（单例，
    # 索引/模型常驻——依赖 A2 层 1 缓存与 D1 无状态化）

@app.post("/qa/stream")
def qa_stream(req: QARequest) -> EventSourceResponse:
    # SSE 事件序列（阶段名固定顺序，前端可渲染进度）：
    # {"stage": "route",    "ms": 12,  "route_mode": "explicit_company", ...}
    # {"stage": "retrieve", "ms": 340, "n_candidates": 28, "sources": ["vector","bm25"]}
    # {"stage": "rerank",   "ms": 850, "n_reranked": 10}
    # {"stage": "ground",   "ms": 45,  "grounding_hits": 2}
    # {"stage": "generate", "ms": 2100, "tokens": {"prompt": 3800, "completion": 120}}
    # {"stage": "validate", "ms": 8,   "flags": [], "adaptive_rounds": 0}
    # {"stage": "done",     "final_answer": ..., "confidence": "high"}

@app.get("/healthz")
def healthz():   # {"status", "knowledge_generation": {"valid": true, "build_id": ...},
                 #  "index_cache": {"slots": 12, "max": 16}, "adaptive_enabled": false}
```

**tracing.py**：`StageTimer` 上下文管理器（stage 名 → 起止毫秒，自动进 trace dict）；
token 聚合从 `api_requests` 的 usage 回收点接线（get_answer_from_rag_context 返回
后取 `processor.last_usage`——D1 改造后随请求携带）；trace 每题一行 JSONL
（`logs/service_trace.jsonl`），字段与 F3/A3 前述联动。

**预算控制**：`SERVICE_TOKEN_BUDGET`（env，默认 8000）；
降级链（逐级）：跳过 HyDE → rerank 候选 28→10 → 返回 `budget_exceeded`（带
已产生证据的部分回答标记）。

### 5.3 测试用例表（`tests/test_service.py`，7 例；fastapi 用 TestClient + mock LLM）

| # | 用例 | 构造 | 期望 |
|---|---|---|---|
| T1 | /qa 冒烟 | mock get_answer | QAResponse 结构 + trace 字段齐 |
| T2 | /qa/stream 事件序 | SSE 收集 | 7 阶段事件按固定顺序 |
| T3 | /healthz | 正常启动 | 知识层 Generation 校验状态（valid/build_id） |
| T4 | healthz 源失步 | 篡改 yaml 后 load | valid=false（fail-closed 透出） |
| T5 | 并发隔离 | 2 线程各发 1 题 | 各自 trace 的 question 字段不串 |
| T6 | 预算降级链 | mock 超预算 usage | 依次触发跳 HyDE/砍候选/标记 |
| T7 | 404 输入校验 | 非法 schema | 422 + 错误信息 |

### 5.4 验收

```bash
pip install fastapi uvicorn sse-starlette   # 或确认 venv 已有（无则 requirements 补录）
./venv/bin/python -m unittest tests.test_service -v
./venv/bin/python -m uvicorn src.service.app:app --port 8300 &   # 手动冒烟 curl
```

---

## 6. 统一验收清单（v2 终版）

| 项 | commit | 离线验收（全部可做） | 端到端（卡 API key） |
|---|---|---|---|
| A1 修复 | ×1 | T1-T8 + 现有 reranking/retrieval 回归 + Level1 100% | retrieval_rank_eval 前后对比 |
| A4 加权 | ×1 | T9-T12 + 调用点传参断言 | 按题型分桶的 rank_eval 对比 |
| F3 归因 | ×1 | T1-T8 + 基线矩阵（可跑则出，不可跑挂 NULL 不伪造） | 修复前后矩阵对比 |
| A2 层1 | ×1 | T4-T6 | 延迟 P50/P95 |
| A2 层2+3 | ×1 | T1-T3/T7/T8 | 同上 |
| Agentic | ×1 | T1-T9 + 默认关闭全量零变化 | F3 矩阵 evidence_mislocalization/over_refusal 占比 |
| D1+D3 | ×2 | T1-T7 + 全量回归 | 线上 P95、trace 覆盖率 |

**纪律（同 FinaWiki）**：独立 commit；现有断言不改通过（行为等价）；新增配置
带加载期校验 + 缺失容错降级；Agentic 默认关闭；所有 trace/日志字段名进本文档
§5.2/§4.2 的 schema，不得漂移；F3 基线矩阵在 A1 之前采样存档。


---

## 7. 实施结果回填（2026-08-21 实测）

第一批（A1/A4/F3）与第二批（A2/Agentic）已完成，共 6 个 commit：

| commit | 内容 | 实测要点 |
|---|---|---|
| 1a22cbc | A1 融合量纲修复 | 检索层实测已有每路归一（方案病灶二不成立，范围收窄为仅 rerank 层三类）；reranking.py 原有 _minmax_normalize 被发现并复用（v2 方案漏调研）；顺带修复 test_event_atoms 基线 HEAD 漂移炸弹（钉死 2f2bc8e） |
| ae80d0f | A4 题型加权 | yaml 键覆盖 QueryPlan 8 类 task_type 实测值域；穿链 5 处；测试替身签名同步 2 文件 |
| 7c537a1 | F3 阶段归因 v1 | 八分支归因树（比方案五分支多出 routing/parse 异常归类与 citation_error 优先序修正）；基线矩阵挂 NULL 不伪造（存量跑分与 core120 零对齐，记录于 eval/reports/stage_attribution_baseline.json） |
| bf69124 | A2 层1 索引常驻 | _IndexCache 自建（lru_cache 逐键删除缺失 + 驱逐语义不符）；粗粒度锁预留给层 2 |
| 5690fca | A2 层2/3 并行 | 串行回退路径保留旧异常传播语义（HYBRID_RETRIEVAL_PARALLEL=0 完整回退）；并行==串行输出全等双测试 |
| c71664f | Agentic v1 | 挂点收窄为检索后预检（生成段重入留 v2 与 D3 配套，如实声明）；覆盖判定 OR 语义修正（evidence_targets 混有位置词，AND 语义几乎永触发）；try 语义以模块级测试锁定 |

**最终状态**：全量 446 tests（较实施前 405 +41），失败维持 6 个历史环境依赖项（multi_gpu×2 / core200 / quantization / colbert_model env 污染 / ragas），零回归。

**磁盘故障记录（本轮 3 次）**：SearchReplace 报 save failed 两次实写一次真失败；终端 grep 输出一次陈旧假象（显示旧内容实际已替换）。防线：python 直改 + assert 锚点计数 + 字节级回读校验，未发生重复写入。

**剩余**：D1+D3（第三批，服务化）未开工。Agentic v2（校验后 try-then-closed 闭环）与 D3 trace 配套。端到端收益验收（Level2/number 准确率/归因矩阵首跑）仍卡 API key。
