# Generator SFT V2 清洗规则与重切分方案

本文档给出一版可直接落地的 `generator_sft` 数据清洗与重切分方案，面向当前这批用于 Qwen3.5 9B SFT 的年报问答数据。

适用范围：

- 当前 `training/generator_sft/llamafactory_data` 中的 571 条样本
- 后续从 `teacher_answers_raw.jsonl` 重新构建的新版本样本

目标：

- 去掉会直接伤害监督信号的硬错误
- 把主训练集切成真正的文档级 holdout
- 把单文档抽取任务和多文档/比较任务拆开处理
- 为后续 `v2` 数据版本提供统一的命名和流程

## 1. 当前版本的核心问题

基于当前 `llamafactory_data` 的检查结果，主要风险如下：

- 总样本数为 571，规模可用于原型训练，但不足以支撑稳定泛化。
- `number` 类样本共有 181 条，但存在大量 `step_by_step_analysis` 写对、`final_answer` 数字写错的情况。
- `dev/test` 不是干净的文档级 holdout。
- `names` 与两公司比较题会把多个 `doc_id` 混在一起，导致主 split 发生文档泄漏。
- 问题模板非常集中，规范化后只有 25 种问法，其中 `number` 实际只有 1 种问法。
- 拒答逻辑基本一致，但“禁止计算”和“允许单位换算/求和/反推”存在冲突。
- 检索上下文里标题页、章节页、空片段偏多，会稀释有效证据。
- `reasoning_summary` 中存在明显模板痕迹，例如 `grounding`、`validation` 等英文残留。

## 2. V2 数据的总体策略

推荐把数据先拆成两个池子再处理：

### 2.1 Core 单文档池

满足以下条件的样本进入主训练/验证/测试：

- `len(doc_ids) == 1`
- 问题不是跨公司比较题
- `schema` 属于 `name`、`number`、`boolean`

这是主 SFT 集，后续 train/dev/test 只从这里切。

### 2.2 Aux 多文档池

满足以下任一条件的样本进入辅助池：

- `len(doc_ids) > 1`
- `schema == "names"`
- 问题中出现比较模式，例如 `谁的营业收入更高`

这部分当前总量很小，建议不要混入主 `dev/test`。

推荐做法：

- 当前版本全部放入 `train_aux_multidoc`
- 单独做一个 `benchmark_multidoc.jsonl`
- 等该类样本至少扩充到 50 条以上，再单独做多文档评测 split

## 3. V2 清洗规则

清洗输出只分四类：

- `keep`: 直接保留
- `auto_fix`: 可自动修复后保留
- `manual_review`: 放人工复核队列
- `reject`: 直接丢弃

### 3.1 结构层硬规则

这些规则一旦不满足，直接 `reject`。

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `STRUCT-001` | `messages` 不是 3 轮，或角色顺序不是 `system -> user -> assistant` | reject |
| `STRUCT-002` | assistant 不是合法 JSON | reject |
| `STRUCT-003` | assistant 缺少 `step_by_step_analysis`、`reasoning_summary`、`relevant_pages`、`final_answer` 任一字段 | reject |
| `STRUCT-004` | `relevant_pages` 不是整数列表 | reject |
| `STRUCT-005` | `final_answer` 为空、空字符串、空列表或 `null` | reject |
| `STRUCT-006` | `relevant_pages` 不是 `retrieval_pages` 的子集 | reject |
| `STRUCT-007` | 单文档题缺失 `company_name` 或 `doc_ids` | reject |

### 3.2 检索上下文规则

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `CTX-001` | 无检索结果，或 `rag_context` 为空 | reject |
| `CTX-002` | 所有 chunk 都是标题页、章节页、封面页或极短片段 | reject |
| `CTX-003` | 标题/空片段占比过高，但仍有 1 条以上有效证据 chunk | auto_fix，裁掉低信息 chunk |
| `CTX-004` | 完全重复的标题 chunk 或重复页面说明 | auto_fix，去重 |

建议把“低信息 chunk”定义为以下任一情况：

- 去空白后长度小于 40
- 只包含公司名、`2024 年年度报告`、章节标题
- 没有任何实体、指标、金额、布尔触发词

建议阈值：

- 若低信息 chunk 比例大于 50%，进入 `manual_review`
- 若大于 80% 且没有有效证据 chunk，直接 `reject`

### 3.3 `name` 类规则

适用于问“法定代表人是谁”这类单值名称抽取。

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `NAME-001` | 问题问的是 `法定代表人`，但上下文没有显式出现 `法定代表人` | 必须返回 `N/A`；否则 reject |
| `NAME-002` | 用 `公司负责人`、`董事长`、`董事会秘书` 等近义但非等价字段替代 `法定代表人` | reject |
| `NAME-003` | `final_answer` 为非字符串且不是 `N/A` | reject |
| `NAME-004` | 题目实际是两公司比较题，但 `schema` 仍写成 `name` | manual_review，改任务类型 |

补充说明：

- 当前两公司比较题实际上不应该留在 `name` 主池中。
- 推荐把这类样本标成 `task_type = comparative`，并移到 Aux 池。

### 3.4 `number` 类规则

这是当前版本最需要重点修的部分。

#### 3.4.1 强一致性规则

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `NUM-001` | `final_answer != "N/A"` 但没有 `table_grounding_result.normalized_value` | reject |
| `NUM-002` | `final_answer` 与 `normalized_value` 不一致 | auto_fix 为 `normalized_value`；若冲突严重则 manual_review |
| `NUM-003` | 推理文本已经给出明确最终数值，但 `final_answer` 与之不一致 | reject |
| `NUM-004` | 推理文本明确写“应返回 N/A/无法确定”，但 `final_answer` 仍为数字 | reject |
| `NUM-005` | 问题要求单位为 `元`，但证据里没有明确单位，也没有可验证的标准化数值 | reject |

#### 3.4.2 允许与禁止的计算

V2 建议统一规则：

- 允许：直接单位换算
- 禁止：复杂推导

允许的仅限以下一跳换算：

- `千元 -> 元`
- `万元 -> 元`
- `百万元 -> 元`
- `亿元 -> 元`

不允许的情况：

- 用占比反推总额
- 汇总季度收入得到全年收入
- 由主营业务收入推断营业收入
- 用“营业总收入”替代“营业收入”
- 用多个表格的间接信息做二次推导

因此新增规则：

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `NUM-006` | 需要反推、求和、补全、跨概念映射才能得到数字 | 必须返回 `N/A`；否则 reject |
| `NUM-007` | 只做单位换算即可得到答案 | keep 或 auto_fix |
| `NUM-008` | 原始证据是 `主营业务收入`，问题问的是 `营业收入` | 必须返回 `N/A` |
| `NUM-009` | 原始证据是 `营业总收入`，问题问的是 `营业收入`，且无明确“其中：营业收入” | 必须返回 `N/A` |

#### 3.4.3 数值比对阈值

建议用以下容差：

- 浮点绝对误差 `<= 1e-6`
- 或相对误差 `<= 1e-6`

如果是整数化后的货币值：

- 标准化后必须与 `normalized_value` 完全一致

### 3.5 `boolean` 类规则

当前这类数据几乎只有 `true` 和 `N/A`，缺少真实 `false`。

V2 规则建议如下：

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `BOOL-001` | `final_answer` 不是 `true`、`false` 或 `N/A` | reject |
| `BOOL-002` | 回答 `true`，但上下文没有显式提到目标事实 | reject |
| `BOOL-003` | 回答 `false`，但只是“当前检索片段没看到”，没有直接否定证据 | 改为 `N/A` |
| `BOOL-004` | 有显式否定表述，例如“不进行现金分红”“不派发现金红利” | 可以保留 `false` |

建议在下一轮构造中，给 `boolean` 单独补一批真实 `false` 样本，不要只依赖检索缺失造成的 `N/A`。

### 3.6 `names` 与比较题规则

当前这两类题的样本量太少，而且天然跨文档。

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `MULTI-001` | `schema == "names"` | 移入 Aux 池 |
| `MULTI-002` | 问题中含 `谁的...更高`、`哪家公司` 等跨公司比较模式 | 移入 Aux 池 |
| `MULTI-003` | `len(doc_ids) > 1` | 移入 Aux 池 |

当前版本不建议把这类样本混入主 `dev/test`。

### 3.7 文本与风格规则

这部分不一定要 `reject`，但建议统一改写。

| 规则 ID | 条件 | 动作 |
| --- | --- | --- |
| `TEXT-001` | `reasoning_summary` 中出现 `grounding`、`validation` 等英文模板词 | auto_fix，改成自然中文 |
| `TEXT-002` | system prompt 内存在空 schema 代码块 | auto_fix，清理空块 |
| `TEXT-003` | `reasoning_summary` 没有总结依据，只是模板句 | auto_fix |
| `TEXT-004` | `relevant_pages` 含无意义标题页，且不构成证据 | auto_fix，删掉噪声页 |

## 4. V2 清洗流程

推荐按以下顺序执行：

1. 结构校验
2. 检索上下文去噪
3. 任务类型归一化
4. `number` 类强校验与自动修复
5. `name/boolean` 规则校验
6. 样本分池：Core vs Aux
7. 去重
8. 文本改写与输出

### 4.1 去重建议

去重分三层：

- 问题去重：完全相同问题只保留 1 条
- 上下文去重：完全相同上下文只保留 1 条
- 语义去重：同一 `doc_id` 上同一任务只保留最佳版本

同一 `doc_id` 上如果同时存在多条“营业收入是多少元”的候选，优先级如下：

1. `normalized_value` 存在且和 `final_answer` 一致
2. `relevant_pages` 最短且最精确
3. 上下文含直接证据页，不是标题页
4. `reasoning_summary` 为自然中文

## 5. V2 重切分方案

## 5.1 主原则

主 split 只对 Core 单文档池做文档级切分。

切分单元：

- 首选 `doc_id`
- 同时记录 `(company_name, report_year)` 作为二次校验键

不要再把多文档题与单文档题一起哈希切分。

### 5.2 推荐切分比例

对 Core 单文档池使用：

- train: 0.80
- dev: 0.10
- test: 0.10

以当前 182 个单文档 `doc_id` 为基数，目标文档数约为：

- train: 146
- dev: 18
- test: 18

样本数会随清洗结果波动，但大致可期待：

- train: 420 到 450
- dev: 50 到 60
- test: 50 到 60

### 5.3 Aux 池处理方式

对多文档/比较题不做主 split，建议单独输出：

- `processed/train_aux_multidoc.chat.v2.jsonl`
- `processed/benchmark_multidoc.chat.v2.jsonl`

推荐当前版本先这样做：

- `names` 全部进入 `train_aux_multidoc`
- 两公司比较题全部进入 `benchmark_multidoc`

原因：

- 总量只有 20 条
- 题型和主任务分布完全不同
- 直接混入主 `dev/test` 会制造虚高或虚低的评测波动

### 5.4 推荐的 split salt 与分组字段

对 Core 单文档池建议使用：

```yaml
split_salt: finarag_generator_v2_doc_holdout
group_fields:
  - doc_ids
```

前提是输入文件已经只包含单文档样本，即每条记录的 `doc_ids` 长度都为 1。

### 5.5 切分后的强校验

split 完成后必须检查：

- `train/dev/test` 之间 `doc_id` 交集为 0
- `train/dev/test` 之间 `(company_name, report_year)` 交集为 0
- `dev/test` 都至少覆盖 `name`、`number`、`boolean`
- `dev/test` 的 `number` 样本不能再出现数值错标
- `dev/test` 中低信息 chunk 比例不能高于 train 太多

## 6. 推荐的 V2 目录产物

推荐输出以下文件：

```text
training/generator_sft/
├── processed/
│   ├── teacher_answers_filtered.v2.jsonl
│   ├── all.chat.v2.jsonl
│   ├── core_single_doc.chat.v2.jsonl
│   ├── train.chat.v2.jsonl
│   ├── dev.chat.v2.jsonl
│   ├── test.chat.v2.jsonl
│   ├── train_aux_multidoc.chat.v2.jsonl
│   └── benchmark_multidoc.chat.v2.jsonl
├── manifests/
│   ├── filter_stats.v2.json
│   ├── rejected_samples.v2.jsonl
│   ├── split_stats.v2.json
│   └── multidoc_manifest.v2.json
└── llamafactory_data/
    ├── finarag_generator_v2_train.json
    ├── finarag_generator_v2_dev.json
    ├── finarag_generator_v2_test.json
    └── dataset_info.json
```

## 7. 用现有脚本落地时的最小改动建议

当前已有：

- `scripts/filter_sft_samples.py`
- `scripts/convert_to_chat_sft.py`
- `scripts/split_train_dev_test.py`

最小可行落地方案如下：

1. 在 `filter_sft_samples.py` 增加 `NUM-002`、`NUM-003`、`NUM-004`、`NUM-006`、`NUM-008`、`NUM-009`
2. 增加任务分池步骤，把多文档/比较题单独导出
3. 只对 `core_single_doc.chat.v2.jsonl` 执行 `split_train_dev_test.py`
4. 使用新的 `split_salt`

如果只允许非常小的代码改动，优先顺序如下：

1. 先修 `number` 标签一致性
2. 再把多文档题从主 split 移出
3. 最后再做文本去模板化

## 8. 下一轮数据构造的目标配额

为了让 SFT 更稳，建议下一轮不是只“清洗旧数据”，还要补样本。

推荐最小目标：

- `name`: 300 条以上
- `number`: 300 条以上
- `boolean`: 300 条以上
- `names/multidoc/comparative`: 80 条以上

并控制分布：

- `name` 中 `N/A` 比例控制在 35% 到 50%
- `number` 中 `N/A` 比例控制在 10% 到 20%
- `boolean` 中显式 `false` 至少占 20%

## 9. 一句话结论

V2 的关键不是“继续往里堆样本”，而是先做三件事：

1. 把 `number` 的硬错标签清掉
2. 把多文档/比较题从主 split 里拆出去
3. 把主 `train/dev/test` 改成真正的单文档级 holdout

这三件事完成之后，这套数据才适合作为 FinaRAG 生成模型的主 SFT 集。
