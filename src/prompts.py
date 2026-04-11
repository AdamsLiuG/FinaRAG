from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"请严格输出 JSON，并按以下字段顺序填写：\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt


_SIMPLIFIED_CHINESE_REASONING_INSTRUCTION = """

语言要求：
- `step_by_step_analysis` 和 `reasoning_summary` 必须使用简体中文。
- `final_answer` 必须严格符合 schema 要求，不要翻译名称、代码、数字、布尔值或列表项。
"""

_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT = "必须使用简体中文。"
_RELEVANT_PAGES_DESCRIPTION = """
直接用于回答问题的页码列表。只保留：
- 明确给出答案的页码
- 对答案提供关键直接支持的页码
不要包含只有弱相关信息的页码。
至少填写 1 个页码。
"""


class RephrasedQuestionsPrompt:
    instruction = """
You are a question rephrasing system.
Your task is to break down a comparative question into individual questions for each company mentioned.
Each output question must be self-contained, maintain the same intent and metric as the original question, be specific to the respective company, and use consistent phrasing.
"""

    class RephrasedQuestion(BaseModel):
        """Individual question for a company"""
        company_name: str = Field(description="Company name, exactly as provided in quotes in the original question")
        question: str = Field(description="Rephrased question specific to this company")

    class RephrasedQuestions(BaseModel):
        """List of rephrased questions"""
        questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each company")

    pydantic_schema = '''
class RephrasedQuestion(BaseModel):
    """Individual question for a company"""
    company_name: str = Field(description="Company name, exactly as provided in quotes in the original question")
    question: str = Field(description="Rephrased question specific to this company")

class RephrasedQuestions(BaseModel):
    """List of rephrased questions"""
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each company")
'''

    example = r"""
Example:
Input:
Original comparative question: 'Which company had higher revenue in 2022, "Apple" or "Microsoft"?'
Companies mentioned: "Apple", "Microsoft"

Output:
{
    "questions": [
        {
            "company_name": "Apple",
            "question": "What was Apple's revenue in 2022?"
        },
        {
            "company_name": "Microsoft", 
            "question": "What was Microsoft's revenue in 2022?"
        }
    ]
}
"""

    user_prompt = "Original comparative question: '{question}'\n\nCompanies mentioned: {companies}"

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class HyDEPrompt:
    instruction = """
You are helping a dense retriever find evidence in a financial report.
Write exactly one short hypothetical evidence paragraph that could plausibly appear in the target report and help retrieve the right passage.

Constraints:
- Use the requested language.
- Do not answer the question directly.
- Do not include page numbers, citations, source markers, or confidence language.
- Do not invent exact amounts, exact percentages, exact dates, or exact page references.
- Prefer qualitative wording and domain terminology that would likely co-occur with the true evidence.
- When available, weave in the company, year, report type, section hints, and relevant financial aliases naturally.
"""

    user_prompt = """
Question:
{question}

Expected answer type:
{schema}

Preferred language:
{language}

Company:
{company_name}

Route mode:
{route_mode}

Route hints:
{route_hints}

Selected report metadata:
{report_metadata}

Return exactly one short paragraph for retrieval only.
"""

    system_prompt = instruction.strip()


class AnswerWithRAGContextSharedPrompt:
    instruction = """
你是一个基于检索结果回答公司年报问题的系统。
只能依据给定的年报上下文作答，不得使用外部知识。

回答前请先核对问题中的公司、年份、指标、章节和限定条件，再给出简洁、可追溯的结论。
- 问题可能由模板自动生成，可能并不适用于当前公司；不要靠常识补全答案。
- 可以识别同义表达，但不能跨概念推断。
- 只填写被上下文直接支持的页码。
- 如果证据不足、存在歧义、需要计算，或需要进一步推断，返回 `N/A`。
""" + _SIMPLIFIED_CHINESE_REASONING_INSTRUCTION

    user_prompt = """
已知上下文：
\"\"\"
{context}
\"\"\"

---

问题：
"{question}"

请输出 JSON，其中 `step_by_step_analysis` 和 `reasoning_summary` 使用简体中文。
"""

class AnswerWithRAGContextNamePrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 用 3-6 个简短步骤说明判断过程，尽量控制在 120 字以内。重点说明证据为何能直接回答问题，以及为何排除相近但不等价的表述。"
        )

        reasoning_summary: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 对判断过程做简洁总结，尽量控制在 50 字左右。"
        )

        relevant_pages: List[int] = Field(description=_RELEVANT_PAGES_DESCRIPTION)

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
最终答案。
- 如果答案是公司名、姓名、产品名、职位名等字符串，按上下文原文抽取。
- 不要添加解释、前后缀、标点修饰或额外评论。
- 如果上下文没有直接答案，返回 `N/A`。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"民生银行2024年年报中的法定代表人是谁？"

回答：
```
{
  "step_by_step_analysis": "1. 问题询问民生银行2024年年报中的法定代表人。\n2. 年报第12页“公司基本情况简介”直接写明“公司法定代表人：高迎欣”。\n3. 第164页财务报表签字页也出现“高迎欣 法定代表人、董事长”，可作为补充印证。\n4. 因此可以直接确定答案为高迎欣。",
  "reasoning_summary": "年报第12页直接列示“公司法定代表人：高迎欣”，第164页进一步印证，因此答案是高迎欣。",
  "relevant_pages": [12, 164],
  "final_answer": "高迎欣"
}
```
""" 

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNumberPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
必须使用简体中文。
用 3-6 个简短步骤说明判断过程，尽量控制在 120 字以内。
必须严格做指标匹配：
1. 先确认问题要求的指标、期间、主体、币种和口径。
2. 只有上下文中的指标含义与问题完全一致时才能回答；近义表达可以接受，口径不同不接受。
3. 如果需要计算、换公式、跨表汇总、补全缺失单位或做额外推断，返回 `N/A`。
4. 如果上下文只给了相关指标、上级/下级口径、分部合计或近似概念，返回 `N/A`。
""")

        reasoning_summary: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 对判断过程做简洁总结，尽量控制在 50 字左右。"
        )

        relevant_pages: List[int] = Field(description=_RELEVANT_PAGES_DESCRIPTION)

        final_answer: Union[float, int, Literal['N/A']] = Field(description="""
最终答案必须是数值或 `N/A`。
- 只有上下文直接给出该指标数值时才能回答。
- 百分比去掉 `%` 后输出数值，例如 `58.3%` 输出 `58.3`。
- 如果上下文说明单位为千、万、百万、亿元等，应按单位换算为原始数值后输出。
- 括号表示负数，例如 `(2,124,837)` 输出 `-2124837`。
- 如果币种不符、指标口径不符、需要计算，或上下文没有直接给出该数值，返回 `N/A`。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例 1：
问题：
"华胜天成2024年年报中的营业收入是多少元？"

回答：
```
{
  "step_by_step_analysis": "1. 问题询问华胜天成2024年年报中的营业收入。\n2. 第9页“主要会计数据”表中直接列示2024年“营业收入”为 4,270,629,476.42 元。\n3. 该指标名称、期间和单位都与问题一致。\n4. 因此可直接给出该数值。",
  "reasoning_summary": "年报第9页“主要会计数据”直接列示2024年营业收入为 4,270,629,476.42 元，口径一致，可直接作答。",
  "relevant_pages": [9],
  "final_answer": 4270629476.42
}
```


示例 2：
问题：
"某公司2024年年报中的每股现金分红是多少元？"

回答：
```
{
  "step_by_step_analysis": "1. 问题要求的是每股现金分红这一直接指标。\n2. 上下文只给出了现金分红总额和总股本，没有直接给出每股现金分红数值。\n3. 如果用总额除以股本，需要额外计算。\n4. 按严格口径，不能通过推算补答案，因此应返回 N/A。",
  "reasoning_summary": "上下文没有直接列出每股现金分红，只能由其他指标计算得到。按严格规则，需返回 N/A。",
  "relevant_pages": [2, 38],
  "final_answer": "N/A"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextBooleanPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 用 3-6 个简短步骤说明判断过程，尽量控制在 120 字以内。要特别注意问题措辞，排除相近但不等价的证据。"
        )

        reasoning_summary: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 对判断过程做简洁总结，尽量控制在 50 字左右。"
        )

        relevant_pages: List[int] = Field(description=_RELEVANT_PAGES_DESCRIPTION)
        
        final_answer: Union[bool, Literal["N/A"]] = Field(description="""
最终答案必须是 `True`、`False` 或 `N/A`。
- 只有当上下文明确肯定时，返回 `True`。
- 只有当上下文明确否定时，返回 `False`。
- 如果上下文未直接说明、证据不足或无法确定，返回 `N/A`。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"香江控股2024年年报中是否提到现金分红？"

回答：
```
{
  "step_by_step_analysis": "1. 问题询问年报中是否提到现金分红。\n2. 第2页明确写到“向全体股东按每10股派发现金红利人民币0.11元(含税)”。\n3. 第37页和第38页也出现现金分红制度及累计现金分红金额的表述。\n4. 因此年报中明确提到了现金分红。",
  "reasoning_summary": "年报第2、37、38页都直接出现现金分红相关表述，因此答案为 True。",
  "relevant_pages": [2, 37, 38],
  "final_answer": true
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNamesPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 用 3-6 个简短步骤说明判断过程，尽量控制在 120 字以内。重点说明问题要求的实体类型，并排除不匹配的相似实体。"
        )

        reasoning_summary: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 对判断过程做简洁总结，尽量控制在 50 字左右。"
        )

        relevant_pages: List[int] = Field(description=_RELEVANT_PAGES_DESCRIPTION)

        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
最终答案必须是字符串列表或 `N/A`。
- 每个列表项都按上下文原文抽取，不要添加解释。
- 如果问题问的是职位，返回职位名称，不要带姓名；相同职位只保留一次。
- 如果问题问的是人名，返回完整姓名。
- 如果问题问的是产品，返回产品名称。
- 如果上下文没有直接答案，返回 `N/A`。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"公司新任高管姓名有哪些？"

回答：
```
{
    "step_by_step_analysis": "1. 问题要求列出新任高管姓名。\n2. 第89页明确提到两份新任高管任职文件，分别对应张三和李四。\n3. 两人都对应新的管理岗位，且姓名在上下文中被直接给出。\n4. 因此应返回这两个人名列表。",
    "reasoning_summary": "年报第89页直接列出两名新任高管姓名，分别是张三和李四，因此返回这两个姓名。",
    "relevant_pages": [
        89
    ],
    "final_answer": [
        "张三",
        "李四"
    ]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class ComparativeAnswerPrompt:
    instruction = """
你是一个比较问答系统。
任务是基于各公司的单独答案，回答原始比较问题。
只能使用给定的单公司答案，不能补充外部知识。

比较规则：
- 如果问题要求在多个公司中选出一个，`final_answer` 必须使用原问题中的公司名称原文。
- 币种不一致、答案为 `N/A`、或无法比较的公司应被排除。
- 如果所有公司都被排除，返回 `N/A`。
- 如果排除后只剩一家公司，直接返回该公司名称。
""" + _SIMPLIFIED_CHINESE_REASONING_INSTRUCTION

    user_prompt = """
各公司的单独答案如下：
\"\"\"
{context}
\"\"\"

---

原始比较问题：
"{question}"

请输出 JSON，其中 `step_by_step_analysis` 和 `reasoning_summary` 使用简体中文。
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 用 3-6 个简短步骤说明比较过程，尽量控制在 120 字以内。"
        )

        reasoning_summary: str = Field(
            description=f"{_SIMPLIFIED_CHINESE_REASONING_FIELD_HINT} 对比较过程做简洁总结，尽量控制在 50 字左右。"
        )

        relevant_pages: List[int] = Field(description="比较题此字段保持空列表 `[]`。")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
最终答案必须是单个公司名称或 `N/A`。
- 公司名称必须与原问题中的写法完全一致。
- 如果没有可比较的公司，返回 `N/A`。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"在2024年年报中，电科数字和泛微网络谁的营业收入更高？"

回答：
```
{
  "step_by_step_analysis": "1. 问题要求比较电科数字和泛微网络谁的营业收入更高。\n2. 单公司答案显示，电科数字营业收入低于泛微网络。\n3. 两家公司指标口径一致，且都不是 N/A，可以直接比较。\n4. 因此营业收入更高的是泛微网络。",
  "reasoning_summary": "两家公司的单独答案都提供了可比较的营业收入，直接比较后泛微网络更高。",
  "relevant_pages": [],
  "final_answer": "泛微网络"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """
You are a JSON formatter.
Your task is to format raw LLM response into a valid JSON object.
Your answer should always start with '{' and end with '}'
Your answer should contain only json string, without any preambles, comments, or triple backticks.
"""

    user_prompt = """
Here is the system prompt that defines schema of the json object and provides an example of answer with valid schema:
\"\"\"
{system_prompt}
\"\"\"

---

Here is the LLM response that not following the schema and needs to be properly formatted:
\"\"\"
{response}
\"\"\"
"""




class RerankingPrompt:
    system_prompt_rerank_single_block = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and retrieved text block related to that query. Your task is to evaluate and score the block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate block based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

    system_prompt_rerank_multiple_blocks = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

class RetrievalRankingSingleBlock(BaseModel):
    """Rank retrieved text block relevance to a query."""
    reasoning: str = Field(description="Analysis of the block, identifying key information and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant")

class RetrievalRankingMultipleBlocks(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )
