# FinaRAG 竞争力提升实施计划（五项）

> 立项：2026-08-21。范围：FinaWiki 知识层之外的系统能力——检索质量、性能、归因、
> Agentic 检索、服务化。E1（ListWise reranker）经决策暂不做（需 GPU 训练环境）。
> 所有代码事实均经本轮 grep 核实（行号为当前分支）。

## 0. 实施顺序与依赖

```
A1 融合量纲修复（bug，最小改动，独立）
 → F3 错误归因（验收基础设施，为后续各项提供前后对比度量）
 → A2 并行检索+索引常驻（性能；在融合层稳定后动检索结构）
 → Agentic 自适应检索（功能增量；补检收益由 F3 度量）
 → D3 服务化可观测（包装层；依赖 A2 的常驻索引与 D1 并发修复）
```

批次建议：A1+F3（第一批）→ A2+Agentic（第二批）→ D3（第三批）。

---

## 1. A1：融合量纲修复（含 A4 题型加权联动）

### 1.1 精确诊断（比原方案更严重）

**病灶一（rerank 层，`reranking.py` L346/L466/L529）**：
`combined_score = llm_weight * relevance_score + vector_weight * distance`。
`relevance_score` 经 `_normalize_remote_scores`（L44，0-1 检查否则 min-max）或本地
sigmoid 已是 0-1；但 `distance` 的量纲**随检索层 fusion_method 漂移**：

| fusion_method（默认 rrf，L744） | distance 实际量纲 | 0.7/0.3 名义权重的实际效果 |
|---|---|---|
| rrf | rrf_score ≈ 1/(60+rank)，量级 < 0.1 | **向量路实际权重 ~2%**（0.3×0.03≈0.01 vs 0.7×0.9=0.63） |
| average | 四路原始分平均（BM25 无上界，常见 0-30+） | 向量/LLM 信号被 BM25 **淹没** |

即：名义权重与实际权重严重脱钩，且表现随配置漂移——bug 的完整形态。

**病灶二（检索层 average 模式，`retrieval.py` L885-889）**：
`average_score = (vector_score + bm25_score + sparse_score + tag_score) / active_backends`
——四路原始分数直接平均。vector 是余弦 0-1，BM25 无上界。rrf 模式（基于排名）
天然免量纲，但 average 模式存在同样问题。

### 1.2 方案

**修复 1（rerank 层归一化）**：`reranking.py` 新增模块级函数
`_fuse_scores(rel_scores, distances, llm_weight) -> list[float]`——候选池内对两路
各自 min-max 归一到 0-1 后加权（空池/常数池退化保护：全同值返回 0.5 基准）。
三处调用点（L346/L466/L529）替换为该函数。combined_score 语义变为
「池内相对强度」，排序不变性仅在归一化单调意义下成立（这正是我们要的）。

**修复 2（检索层 average 模式）**：`retrieval.py` 的 average 聚合改为
每路先 min-max 归一再平均（各路 top_n 池内）。rrf 路径不动。

**A4 联动（题型加权，知识层驱动）**：新增 `config/retrieval_fusion_weights.yaml`：
```yaml
# 题型 × (llm_weight, bm25_boost)——知识层驱动检索融合
task_type_weights:
  definition_qa:      {llm_weight: 0.85}   # 定义类：语义信号为主
  single_metric_lookup: {llm_weight: 0.5}  # 数字类：精确词信号补强（sparse/bm25）
  implicit_intent_qa: {llm_weight: 0.6}
  default:            {llm_weight: 0.7}
```
`rerank_documents` 增加可选参数 `task_type`；`questions_processing` 调用点从
`query_plan.task_type` 传入。加载期校验权重范围 (0,1)。

### 1.3 测试与验收

- 单测（新建 `tests/test_score_fusion.py`）：
  - rrf 量纲用例：rel=[0.9,0.1]、distance=[0.03,0.02] → 归一化后 top1 保持 rel 高者（修复前 distance 差异被抹平）
  - BM25 量纲用例：rel=[0.5,0.9]、distance=[25.0,24.9] → rel=0.9 者胜（修复前 BM25 主导）
  - 常数池/空池退化保护
  - 题型权重：definition_qa 传 0.85 生效；未注册题型回落 default
- 回归：现有 reranking/retrieval 测试全部不改断言通过；Level1 六项 100% 不回退
- 量化验收：`eval/retrieval_rank_eval.py`（现成脚本）修复前后各跑一次，
  hit@k / NDCG@k 对比进入评测报告——**归因矩阵（F3）记录排序错误占比变化**

---

## 2. F3：错误归因自动化（归因到 pipeline 阶段）

### 2.1 现状

`eval/error_analysis.py`（`_classify_case` L26 + `summarize_error_analysis` L51）：
归因维度是题型/对错，未映射 pipeline 阶段。数据源已具备：
`data/test_set/answers_*_debug.json` 含 `{questions, answer_details, statistics}`，
answer_details 是逐题过程数据（检索命中、路由、校验 flags）。

### 2.2 方案

新增 `eval/stage_attribution.py`（独立模块，不侵入现有 error_analysis）：

```python
STAGE_ORDER = ["pdf_parsing", "text_splitter", "report_catalog",
               "retrieval", "rerank", "generation", "validation"]

def attribute_case(pred, ref, detail) -> dict:
    # 短路判定，归因到第一个失败阶段：
    # 1. gold 页内容在解析产物中缺失/错乱        → pdf_parsing
    # 2. gold 证据被切断（跨 chunk 断裂标记）      → text_splitter
    # 3. 候选文档池不含 gold doc                  → report_catalog
    # 4. gold 页不在 retrieval_results            → retrieval
    # 5. gold 页在候选池但 rank > rerank 截断位    → rerank
    # 6. 证据正确（gold 页被引用）但答案错         → generation
    # 7. 答案与 gold 一致但被判 N/A               → validation（误杀）
    # gold 页缺失时对应判定跳过并标 stage=unknown（不算错，缺口显式）

def attribution_matrix(cases) -> dict:  # stage × error_rate 矩阵 + 样本数
```

判定输入：pred（answers_*.json）+ ref（数据集 gold：页码/证据字段）+
detail（debug payload）。CLI 入口 `python -m eval.stage_attribution --pred ... --debug ...`。

### 2.3 测试与验收

- 单测：每阶段构造一个典型失败样本（mock detail），断言归因到正确阶段；
  误杀分支（答案对但 N/A）；gold 缺失 → unknown
- 验收：对存量 answers_qwen_base_01 跑一次真实归因矩阵，输出进
  `eval/reports/stage_attribution_baseline.json`（A1/A2/Agentic 的**前后对比基线**）

---

## 3. A2：并行检索 + 索引常驻缓存（含 A3 设计）

### 3.1 现状（四个串行点，代码事实）

1. `retrieve_candidates_by_company_name`（retrieval.py L899+）四路顺序调用
2. BM25/sparse 每次检索 `pickle.load`（L388/L438），单索引数十 MB 反复磁盘反序列化
3. `_retrieve_multi_query_candidate_runs` 多条改写查询 for 循环串行
4. 同一查询在 dense/sparse 两路各编码一次

### 3.2 方案（三个层次，按收益/风险排序）

**层 1（索引常驻，零风险先做）**：BM25/sparse 加载处包一层
`functools.lru_cache(maxsize=可配)`（按 doc_id 键；文档重建索引时按 mtime 失效）。
注意内存预算：数十 MB × 常用集，maxsize 默认 16、env 可调；单测覆盖缓存命中与驱逐。

**层 2（四路并行）**：`retrieve_candidates_by_company_name` 内四路检索用
`ThreadPoolExecutor` 并行（IO 密集，GIL 无碍）；**每路独立 try/except 降级**
（一路失败返回空并记 trace，不拖垮整体）；收集后按原顺序聚合，保证与串行版
**结果一致性**（单测断言并行==串行输出，含顺序）。

**层 3（多查询并行）**：`_retrieve_multi_query_candidate_runs` 的查询级并行
（每查询内部已是并行四路，两层 executor 注意线程数上限：外层 4 × 内层 4 = 16
设为可配上限，默认 8）。

查询编码共享（dense/sparse 合并 encode）改动面大，列为后续可选，本轮不做。

### 3.3 A3（全局索引 + 元数据预过滤）——本轮只出设计不实施

现状每公司独立向量库（`target_report["vector_db"]`），200 文档可接受，
10 万文档演进：全局单库 + `RetrievalFilters` 转 FAISS IDSelector/元数据预过滤 +
索引版本管理。设计文档随本计划归档，面试讲架构演进时用（「200 → 10 万份怎么演进」）。

### 3.4 测试与验收

- 并行==串行一致性单测（确定性排序断言）；单路异常降级测试（mock 抛错）
- lru 缓存命中/失效/驱逐单测
- 延迟基准：`eval/benchmark_ttft.py`（现成）扩展 per-stage 计时，
  修复前后各跑一次进报告（P50/P95）；归因矩阵记录检索阶段延迟变化

---

## 4. Agentic 自适应检索（知识层驱动，try-then-closed）

### 4.1 设计原则

不是「给 RAG 加 agent 循环」，是**知识层告诉系统何时补检、往哪补、何时停止**：
触发条件来自校验层 flags（FinaWiki 产出），补检方向来自原子别名/易混口径/剧本
必需指标（FinaWiki 产出），停止条件来自预算上限与 fail-closed 门禁。

### 4.2 方案

新增 `src/adaptive_retrieval.py`（纯函数模块，控制流全部离线可测）：

```python
@dataclass
class RetrievalGap:
    trigger: str            # metric_evidence_mismatch / caliber_conflict / playbook_incomplete / low_confidence
    supplement_queries: List[str]
    rationale: str          # 进 trace

def detect_gap(query_plan, validation_flags, retrieval_results, top_score) -> Optional[RetrievalGap]:
    # 触发判定（按优先级）：
    # 1. metric_evidence_mismatch → 用命中原子 aliases + evidence_targets 构造 2-3 条补充查询
    # 2. caliber_conflict（证据只有易混口径）→ 按易混原子 standard_name 定向补检——
    #    补检后找到本口径 → 正常作答；仍只有易混口径 → 维持拒答（try-then-closed）
    # 3. implicit_intent_evidence_incomplete → 剧本 required_metrics 未覆盖的指标逐个补检
    # 4. top_score < 阈值（可配）→ 原子别名扩展补检

def merge_supplement_results(base_results, supplement_results) -> list:
    # 合并进候选池 → 重跑一次 rerank（复用现有 reranker，不走第二次完整检索）
```

挂点：`questions_processing.get_answer_for_company` 在 validate_answer 之后加
补检决策点：`detect_gap` 命中且本轮预算（默认 max 1 轮，防循环）未耗尽 →
补检 → 合并 → 重新生成 → 再校验（仍不过则按原 fail-closed 语义收尾）。

trace：每次补检记录 trigger/queries/新命中/最终影响（JSONL，喂 F3 与 D3）。

### 4.3 测试与验收

- 触发判定单测：四种 trigger 各一例 + 不触发（零成本）例
- 补检查询构造单测：原子驱动（MET-001 命中时 queries 含其 evidence_targets）
- try-then-closed 行为：mock 检索（第一次只有归母、补检返回扣非 → 作答；
  补检仍归母 → 拒答），两个断言
- 预算上限：构造永远触发的场景，断言只补检 1 轮
- 回归：无触发场景下现有全量测试零变化（Agentic 路径默认关闭开关下跑）
- 端到端收益：F3 归因矩阵中 retrieval 类错误占比变化（需 API key 跑真实评测）

---

## 5. D3：服务化与可观测

### 5.1 前置：D1 并发正确性修复

`QuestionsProcessor` 实例级共享状态（last_debug、缓存 dict 等）在多线程下串题。
修复：请求路径状态全部改局部传递；模型/索引单例只读共享（`_build_retriever`
的重复构建问题由 A2 层 1 缓解）。单测：两线程交错处理不同问题，断言各自
debug payload 不串。

### 5.2 方案

**服务层**（新增 `src/service/app.py`，FastAPI）：
- `POST /qa`：同步 JSON（question, schema, company_name）
- `POST /qa/stream`：SSE 依次推送五阶段事件 route→retrieve→rerank→ground→generate
  （每事件含耗时与关键中间产物）
- `GET /healthz`：索引就绪/知识层 Generation 校验状态
- 启动时预热：索引常驻（A2）、知识层 load()（fail-closed 校验在启动期暴露）

**可观测**：
- `src/service/tracing.py`：阶段计时上下文管理器 + token usage 聚合
  （api_requests 已回收 usage 数据，接上即可）
- per-question trace（JSONL 落盘）：route_path、各路检索命中、rerank 前后顺序、
  grounding 命中、校验 flags、Agentic 补检记录——同时是 F3 归因与线上排障数据源

**预算控制**：单题 token/成本上限（可配），超限降级链：跳过 HyDE → 砍 rerank
候选数 → 拒答并返回 budget_exceeded 标记。

### 5.3 测试与验收

- `TestClient` 冒烟（mock LLM）：/qa 返回结构、/qa/stream 事件序列与顺序、
  /healthz 的知识层校验状态
- trace 完整性：一次请求产出完整五阶段 trace
- 并发正确性：D1 场景回归（并发 N 请求各自 trace 隔离）
- 预算降级：mock 超限 token，断言降级链逐级触发

---

## 6. 统一验收清单

| 项 | 离线验收（本轮全部可做） | 端到端验收（卡 API key） |
|---|---|---|
| A1 | 量纲单测 + 并行一致性 + Level1 不回退 + retrieval_rank_eval 前后对比 | number 准确率 |
| F3 | 七阶段归因单测 + 存量基线矩阵 | — |
| A2 | 并行==串行 + 缓存单测 + benchmark_ttft 延迟对比 | — |
| Agentic | 触发/构造/预算/try-then-closed 单测 + 默认关闭零回归 | 归因矩阵 retrieval 错误占比下降 |
| D3 | TestClient 冒烟 + trace 完整性 + 并发隔离 + 预算降级 | 线上延迟 P95 |

通用纪律：每项独立 commit；行为等价优先（现有断言不改通过）；新增配置全部
带加载期校验；知识类配置（fusion_weights）进 yaml 不进代码。
