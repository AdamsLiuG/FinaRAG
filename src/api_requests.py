import os
import json
import ipaddress
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Any, get_args, get_origin
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_fixed


def _env_flag(*names: str, default: bool = False) -> bool:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value is None:
            continue
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _unwrap_singleton_json_container(parsed):
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]
    return parsed


def _field_union_members(field_annotation) -> tuple:
    origin = get_origin(field_annotation)
    if origin is Union:
        return get_args(field_annotation)
    return (field_annotation,)


def _field_allows_literal_na(field_annotation) -> bool:
    for member in _field_union_members(field_annotation):
        if get_origin(member) is None and member == str:
            continue
        if get_origin(member) is not None:
            member_origin = get_origin(member)
            if member_origin is list:
                continue
        literal_values = get_args(member)
        if "N/A" in literal_values:
            return True
    return False


def _field_base_kind(field_annotation) -> str:
    for member in _field_union_members(field_annotation):
        member_origin = get_origin(member)
        if member_origin is list:
            list_args = get_args(member)
            if list_args and list_args[0] is str:
                return "list_str"
        if member is bool:
            return "bool"
        if member in {int, float}:
            return "number"
        if member is str:
            return "str"
    return "str"


def _parse_jsonish_fragment(fragment: str) -> Any | None:
    try:
        return json.loads(fragment)
    except Exception:
        try:
            return json.loads(repair_json(fragment))
        except Exception:
            return None


def _extract_value_by_key(text: str, key: str) -> Any | None:
    pattern = re.compile(
        rf'["\']?{re.escape(key)}["\']?\s*:\s*(\[[^\]]*\]|"(?:\\.|[^"])*"|\'(?:\\.|[^\'])*\'|true|false|null|-?\d+(?:\.\d+)?)',
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    for match in reversed(matches):
        parsed = _parse_jsonish_fragment(match.group(1))
        if parsed is not None:
            return parsed
    return None


def _extract_pages_from_text(text: str) -> list[int]:
    pages: list[int] = []
    relevant_pages_value = _extract_value_by_key(text, "relevant_pages")
    if isinstance(relevant_pages_value, list):
        for item in relevant_pages_value:
            try:
                pages.append(int(item))
            except Exception:
                continue

    for pattern in (
        re.compile(r"\bpages\s+((?:\d+\s*,\s*)+\d+)\b", re.IGNORECASE),
        re.compile(r"第\s*(\d+)\s*页"),
        re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE),
    ):
        for match in pattern.finditer(text):
            for page_text in re.findall(r"\d+", match.group(1)):
                pages.append(int(page_text))

    seen = set()
    ordered_pages = []
    for page in pages:
        if page <= 0 or page in seen:
            continue
        seen.add(page)
        ordered_pages.append(page)
    return ordered_pages


def _has_explicit_na_signal(text: str) -> bool:
    for key in ("final_answer", "answer"):
        extracted = _extract_value_by_key(text, key)
        if isinstance(extracted, str) and extracted.strip().upper() == "N/A":
            return True

    terminal_window = text.strip()[-1200:]
    patterns = [
        r'answer should be\s*["`]?N/A["`]?',
        r'final answer(?:\s+should\s+be|\s+is|:)\s*["`]?N/A["`]?',
        r'return(?:ed|s)?\s*["`]?N/A["`]?',
        r'返回\s*["`]?N/A["`]?',
        r'应返回\s*["`]?N/A["`]?',
        r'故返回\s*["`]?N/A["`]?',
        r'因此返回\s*["`]?N/A["`]?',
        r'最终(?:答案|结论)(?:应为|为|是)?\s*["`]?N/A["`]?',
        r'答案(?:应为|为|是)\s*["`]?N/A["`]?',
    ]
    return any(re.search(pattern, terminal_window, re.IGNORECASE) for pattern in patterns)


def _extract_list_string_answer(text: str) -> list[str] | None:
    for key in ("final_answer", "answer"):
        extracted = _extract_value_by_key(text, key)
        if isinstance(extracted, list) and extracted and all(isinstance(item, str) for item in extracted):
            return [item.strip() for item in extracted if str(item).strip()]

    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
        if len(line) < 2:
            continue
        if not re.search(r"[\u4e00-\u9fff]{2,}", line):
            continue
        if any(token in line.lower() for token in ("question", "answer", "reasoning", "schema", "page", "context")):
            continue
        if re.match(r"^(问题|检索|上下文|根据|给定|返回|因此|故|说明|无法|未发现)", line):
            continue
        entity_match = re.match(
            r'^([\u4e00-\u9fffA-Za-z0-9（）()·\-/]{2,}?)(?:\s+(?:mentioned|mentions|提到|指出|列出)\b|[，,:：]|$)',
            line,
            re.IGNORECASE,
        )
        entity = (
            entity_match.group(1).strip(" \"'“”")
            if entity_match
            else re.split(r"[，,:：]", line, maxsplit=1)[0].strip(" \"'“”")
        )
        if len(entity) < 2:
            continue
        if entity not in candidates:
            candidates.append(entity)
    return candidates or None


def _extract_string_answer(text: str) -> str | None:
    for key in ("final_answer", "answer"):
        extracted = _extract_value_by_key(text, key)
        if isinstance(extracted, str) and extracted.strip():
            return extracted.strip()

    patterns = [
        r'answer should be\s*["“]?([^"\n”]+)["”]?',
        r'final answer(?:\s+should\s+be|\s+is|:)\s*["“]?([^"\n”]+)["”]?',
        r'答案(?:应为|为|是)\s*["“]?([^"\n”]+)["”]?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    return None


def _extract_bool_answer(text: str) -> bool | None:
    for key in ("final_answer", "answer"):
        extracted = _extract_value_by_key(text, key)
        if isinstance(extracted, bool):
            return extracted

    for pattern, value in (
        (r'answer should be\s*true\b', True),
        (r'final answer(?:\s+should\s+be|\s+is|:)\s*true\b', True),
        (r'答案(?:应为|为|是)\s*true\b', True),
        (r'answer should be\s*false\b', False),
        (r'final answer(?:\s+should\s+be|\s+is|:)\s*false\b', False),
        (r'答案(?:应为|为|是)\s*false\b', False),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def _extract_number_answer(text: str) -> int | float | None:
    for key in ("final_answer", "answer"):
        extracted = _extract_value_by_key(text, key)
        if isinstance(extracted, (int, float)) and not isinstance(extracted, bool):
            return extracted

    patterns = [
        r'answer should be\s*(-?\d+(?:\.\d+)?)',
        r'final answer(?:\s+should\s+be|\s+is|:)\s*(-?\d+(?:\.\d+)?)',
        r'答案(?:应为|为|是)\s*(-?\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        raw_number = match.group(1)
        if "." in raw_number:
            return float(raw_number)
        return int(raw_number)
    return None


def _synthesize_reasoning_fields(final_answer: Any, relevant_pages: list[int]) -> tuple[str, str]:
    pages_text = ""
    if relevant_pages:
        pages_text = "第" + "、".join(str(page) for page in relevant_pages[:5]) + "页"

    if final_answer == "N/A":
        if pages_text:
            step = (
                f"1. 问题询问目标信息。\n"
                f"2. 检索{pages_text}等上下文，未见与问题完全一致的直接表述。\n"
                f"3. 现有信息与目标概念不完全等价，不能据此推断。\n"
                f"4. 因此返回 N/A。"
            )
            summary = f"检索{pages_text}等上下文，未见与问题完全一致的直接证据，按规则返回 N/A。"
        else:
            step = (
                "1. 问题询问目标信息。\n"
                "2. 给定上下文未见与问题完全一致的直接表述。\n"
                "3. 现有信息不足以支持确定答案。\n"
                "4. 因此返回 N/A。"
            )
            summary = "给定上下文缺少与问题完全一致的直接证据，按规则返回 N/A。"
        return step, summary

    if isinstance(final_answer, list):
        if pages_text:
            step = (
                f"1. 问题要求列出符合条件的实体。\n"
                f"2. 检索{pages_text}等上下文，识别到与问题直接对应的实体表述。\n"
                f"3. 去重后按原文返回实体列表。"
            )
            summary = f"{pages_text}等上下文出现与问题直接对应的实体表述，可据此返回名单。"
        else:
            step = (
                "1. 问题要求列出符合条件的实体。\n"
                "2. 给定上下文中出现了与问题直接对应的实体表述。\n"
                "3. 去重后按原文返回实体列表。"
            )
            summary = "给定上下文出现了与问题直接对应的实体表述，可据此返回名单。"
        return step, summary

    if pages_text:
        step = (
            f"1. 问题询问目标信息。\n"
            f"2. 检索{pages_text}等上下文，提取到与问题直接对应的表述。\n"
            f"3. 相关证据与问题要求一致，因此可确定答案。"
        )
        summary = f"{pages_text}等上下文出现与问题直接对应的表述，可据此确定答案。"
    else:
        step = (
            "1. 问题询问目标信息。\n"
            "2. 给定上下文中出现了与问题直接对应的表述。\n"
            "3. 相关证据与问题要求一致，因此可确定答案。"
        )
        summary = "给定上下文出现了与问题直接对应的表述，可据此确定答案。"
    return step, summary


def _construct_structured_payload_from_text(text: str, response_format: Type[BaseModel]) -> Dict | None:
    if not isinstance(text, str) or not text.strip():
        return None

    model_fields = response_format.model_fields
    final_answer_field = model_fields.get("final_answer")
    relevant_pages_field = model_fields.get("relevant_pages")
    if final_answer_field is None:
        return None

    final_answer_annotation = final_answer_field.annotation
    field_kind = _field_base_kind(final_answer_annotation)
    allows_na = _field_allows_literal_na(final_answer_annotation)

    extracted_answer = None
    if field_kind == "list_str":
        extracted_answer = _extract_list_string_answer(text)
    elif field_kind == "bool":
        extracted_answer = _extract_bool_answer(text)
    elif field_kind == "number":
        extracted_answer = _extract_number_answer(text)
    else:
        extracted_answer = _extract_string_answer(text)

    if extracted_answer is None and allows_na and _has_explicit_na_signal(text):
        extracted_answer = "N/A"

    if extracted_answer is None:
        return None

    relevant_pages = _extract_pages_from_text(text) if relevant_pages_field is not None else []
    if extracted_answer == "N/A":
        relevant_pages = []

    step_by_step_analysis, reasoning_summary = _synthesize_reasoning_fields(extracted_answer, relevant_pages)
    candidate_payload: Dict[str, Any] = {}
    if "step_by_step_analysis" in model_fields:
        candidate_payload["step_by_step_analysis"] = step_by_step_analysis
    if "reasoning_summary" in model_fields:
        candidate_payload["reasoning_summary"] = reasoning_summary
    if "relevant_pages" in model_fields:
        candidate_payload["relevant_pages"] = relevant_pages
    candidate_payload["final_answer"] = extracted_answer

    try:
        validated = response_format.model_validate(candidate_payload)
        return validated.model_dump()
    except Exception:
        return None


def _validate_structured_payload(parsed, response_format: Type[BaseModel]) -> Dict:
    normalized = _unwrap_singleton_json_container(parsed)
    if normalized is None:
        raise ValueError("Model returned JSON null.")

    try:
        validated = response_format.model_validate(normalized)
        return validated.model_dump()
    except Exception as original_error:
        if isinstance(normalized, list):
            dict_candidates = [item for item in normalized if isinstance(item, dict)]
            for candidate in reversed(dict_candidates):
                try:
                    validated = response_format.model_validate(candidate)
                    return validated.model_dump()
                except Exception:
                    continue
        raise original_error



class BaseCompatibleProcessor:
    def __init__(self, provider: str = "qwen"):
        load_dotenv()
        self.provider = provider.lower()
        default_wire_api = "responses" if self.provider == "sub2api" else "chat_completions"
        provider_env_prefix = self.provider.upper()
        self.api_key = self._get_env_value(f"{provider_env_prefix}_API_KEY", "LLM_API_KEY")
        self.base_url = self._get_env_value(f"{provider_env_prefix}_BASE_URL", "LLM_BASE_URL")
        self.default_model = self._get_env_value(
            f"{provider_env_prefix}_MODEL",
            "LLM_MODEL",
            default="Qwen3.5-35B-A3B-AWQ-4bit"
        )
        self.wire_api = self._normalize_wire_api(
            self._get_env_value(
                f"{provider_env_prefix}_WIRE_API",
                "LLM_WIRE_API",
                default=default_wire_api,
            )
        )
        self.reasoning_effort = self._get_env_value(
            f"{provider_env_prefix}_REASONING_EFFORT",
            "LLM_REASONING_EFFORT",
        )
        self.disable_response_storage = _env_flag(
            f"{provider_env_prefix}_DISABLE_RESPONSE_STORAGE",
            "LLM_DISABLE_RESPONSE_STORAGE",
            default=False,
        )
        max_tokens = self._get_env_value(
            f"{provider_env_prefix}_MAX_TOKENS",
            "LLM_MAX_TOKENS"
        )
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.use_stream = _env_flag(
            f"{provider_env_prefix}_STREAM",
            "LLM_STREAM",
            default=self.provider == "qwen"
        )
        self.enable_thinking = _env_flag(
            f"{provider_env_prefix}_ENABLE_THINKING",
            "LLM_ENABLE_THINKING",
            default=False if self.provider == "qwen" else True
        )

    @staticmethod
    def _get_env_value(*names: str, default: str = None) -> Optional[str]:
        for name in names:
            if not name:
                continue
            value = os.getenv(name)
            if value:
                return value
        return default

    @staticmethod
    def _normalize_wire_api(value: Optional[str]) -> str:
        normalized = (value or "chat_completions").strip().lower()
        normalized = normalized.replace(".", "_").replace("/", "_")
        alias_map = {
            "chat": "chat_completions",
            "chat_completions": "chat_completions",
            "chatcompletions": "chat_completions",
            "responses": "responses",
        }
        return alias_map.get(normalized, normalized)

    def _get_request_url(self) -> str:
        if not self.base_url:
            raise ValueError(
                f"Missing API base URL for provider '{self.provider}'. "
                f"Set {self.provider.upper()}_BASE_URL or LLM_BASE_URL."
            )

        normalized_base = self.base_url.rstrip("/")
        if self.wire_api == "responses":
            if normalized_base.endswith("/chat/completions"):
                normalized_base = normalized_base[: -len("/chat/completions")]
            if normalized_base.endswith("/responses"):
                return normalized_base
            return f"{normalized_base}/responses"

        if normalized_base.endswith("/responses"):
            normalized_base = normalized_base[: -len("/responses")]
        if normalized_base.endswith("/chat/completions"):
            return normalized_base
        return f"{normalized_base}/chat/completions"

    def _is_dashscope_compatible(self) -> bool:
        return bool(self.base_url and "dashscope.aliyuncs.com" in self.base_url)

    def _should_bypass_env_proxy(self) -> bool:
        if not self.base_url:
            return False

        hostname = urlparse(self.base_url).hostname
        if not hostname:
            return False

        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return True

        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return hostname.endswith(".local")

    def _parse_structured_response(
        self,
        response_text: str,
        response_format: Type[BaseModel],
        model: str,
        original_system_prompt: str | None = None,
    ) -> Dict:
        def _try_parse(raw: str) -> Dict:
            """Repair JSON, decode, and validate; raises on any failure including null."""
            repaired = repair_json(raw)
            parsed = json.loads(repaired)
            try:
                return _validate_structured_payload(parsed, response_format)
            except Exception as exc:
                if parsed is None:
                    raise ValueError(f"Model returned JSON null: {raw!r}") from exc
                raise

        try:
            return _try_parse(response_text)
        except Exception:
            reparsed_response = self._reparse_response(
                response_text,
                response_format,
                model,
                original_system_prompt=original_system_prompt,
            )
            try:
                return _try_parse(reparsed_response)
            except Exception:
                heuristic_payload = _construct_structured_payload_from_text(reparsed_response, response_format)
                if heuristic_payload is None:
                    heuristic_payload = _construct_structured_payload_from_text(response_text, response_format)
                if heuristic_payload is not None:
                    return heuristic_payload
                raise ValueError(
                    f"Structured response parsing failed after reparsing attempt. "
                    f"Raw model output was: {reparsed_response!r}"
                )

    @staticmethod
    def _augment_system_prompt_with_schema(system_content: str, response_format: Type[BaseModel]) -> str:
        if "strictly follow this schema" in system_content.lower():
            return system_content

        schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False, indent=2)
        schema_instruction = (
            "\n\n---\n\n"
            "Return valid JSON only. Return a data object instance, not the JSON schema itself.\n"
            "Do not include schema metadata such as `$defs`, `properties`, `required`, `title`, or `type` unless those are explicit fields in the target model.\n"
            "The JSON data object must strictly follow this schema:\n"
            f"```json\n{schema}\n```"
        )
        return f"{system_content}{schema_instruction}"

    def _reparse_response(
        self,
        response: str,
        response_format: Type[BaseModel],
        model: str,
        original_system_prompt: str | None = None,
    ) -> Union[str, Dict]:
        schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False, indent=2)
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=original_system_prompt or "",
            target_schema=schema,
            response=response,
        )
        reparsed = self.send_message(
            model=model,
            temperature=0,
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        if isinstance(reparsed, str):
            return reparsed
        return json.dumps(reparsed, ensure_ascii=False)

    def _build_chat_completions_payload(
        self,
        *,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        seed,
        system_content: str,
        human_content: str,
    ) -> Dict:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
        }
        if seed is not None:
            payload["seed"] = seed
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if self.use_stream:
            payload["stream"] = True
        if self.provider == "qwen":
            if self._is_dashscope_compatible():
                payload["enable_thinking"] = self.enable_thinking
            elif not self.enable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _build_responses_payload(
        self,
        *,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        seed,
        system_content: str,
        human_content: str,
    ) -> Dict:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
        }
        if seed is not None:
            payload["seed"] = seed
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_output_tokens"] = int(max_tokens)
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.disable_response_storage:
            payload["store"] = False
        return payload

    def _build_payload(
        self,
        *,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        seed,
        system_content: str,
        human_content: str,
    ) -> Dict:
        if self.wire_api == "responses":
            return self._build_responses_payload(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                system_content=system_content,
                human_content=human_content,
            )
        return self._build_chat_completions_payload(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
        )

    @staticmethod
    def _extract_responses_content(response_json: Dict) -> str:
        output_text = response_json.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        chunks: List[str] = []
        for item in response_json.get("output") or []:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "message":
                content_items = item.get("content") or []
                for part in content_items:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            chunks.append(text)
                            continue
                        content = part.get("content")
                        if isinstance(content, str) and content:
                            chunks.append(content)
                    elif part:
                        chunks.append(str(part))
                continue

            text = item.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
                continue

            content = item.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)

        if chunks:
            return "".join(chunks)
        raise ValueError(f"Provider returned no text output. Response: {json.dumps(response_json)[:1000]}")

    @staticmethod
    def _extract_text_from_message_payload(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text:
                return text
            content = value.get("content")
            if isinstance(content, str) and content:
                return content
            return ""
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                        continue
                    content = item.get("content")
                    if isinstance(content, str) and content:
                        parts.append(content)
                        continue
                    output = item.get("output")
                    if output is not None:
                        nested = BaseCompatibleProcessor._extract_text_from_message_payload(output)
                        if nested:
                            parts.append(nested)
                elif item is not None:
                    parts.append(str(item))
            return "".join(parts)
        return str(value)

    def _extract_chat_completions_content(self, completion: Dict) -> str:
        choices = completion.get("choices") or []
        if not choices:
            raise ValueError(f"Provider returned no choices. Response: {json.dumps(completion)[:1000]}")

        for choice in choices:
            message_payload = choice.get("message") or {}

            for candidate in (
                message_payload.get("content"),
                choice.get("text"),
                choice.get("content"),
                message_payload.get("reasoning"),
                message_payload.get("reasoning_content"),
            ):
                content = self._extract_text_from_message_payload(candidate)
                if content:
                    return content

        return ""

    def _post_request(self, request_kwargs: Dict) -> requests.Response:
        if self._should_bypass_env_proxy():
            session = requests.Session()
            session.trust_env = False
            try:
                response = session.post(**request_kwargs)
            except Exception:
                session.close()
                raise
            setattr(response, "_finarag_session", session)
            return response
        return requests.post(**request_kwargs)

    @staticmethod
    def _close_response(response: requests.Response) -> None:
        session = getattr(response, "_finarag_session", None)
        try:
            response.close()
        finally:
            if session is not None:
                session.close()

    def _retry_chat_completions_as_stream(
        self,
        *,
        request_kwargs: Dict,
        model: str,
        fallback_usage: Optional[Dict] = None,
    ) -> str:
        stream_request_kwargs = dict(request_kwargs)
        stream_payload = dict(stream_request_kwargs["json"])
        stream_payload["stream"] = True
        stream_request_kwargs["json"] = stream_payload
        stream_request_kwargs["stream"] = True

        response = self._post_request(stream_request_kwargs)
        try:
            if not response.ok:
                raise requests.HTTPError(
                    f"{response.status_code} Client Error for url: {response.url}\n{response.text[:1000]}",
                    response=response
                )

            content = self._read_streaming_content(response, model)
            if fallback_usage and getattr(self, "response_data", None):
                self.response_data["input_tokens"] = (
                    self.response_data.get("input_tokens") or fallback_usage.get("prompt_tokens")
                )
                self.response_data["output_tokens"] = (
                    self.response_data.get("output_tokens") or fallback_usage.get("completion_tokens")
                )
            return content
        finally:
            self._close_response(response)

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None
    ):
        if model is None:
            model = self.default_model
        original_system_content = system_content
        if is_structured and response_format is not None:
            system_content = self._augment_system_prompt_with_schema(system_content, response_format)

        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        payload = self._build_payload(
            model=model,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
        )

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request_kwargs = {
            "url": self._get_request_url(),
            "headers": headers,
            "json": payload,
            "timeout": 300,
            "stream": bool(payload.get("stream")) if self.wire_api == "chat_completions" else False,
        }

        response = self._post_request(request_kwargs)
        try:
            if not response.ok:
                raise requests.HTTPError(
                    f"{response.status_code} Client Error for url: {response.url}\n{response.text[:1000]}",
                    response=response
                )

            if self.wire_api == "chat_completions" and payload.get("stream"):
                content = self._read_streaming_content(response, model)
            else:
                completion = response.json()
                if self.wire_api == "responses":
                    content = self._extract_responses_content(completion)
                    usage = completion.get("usage", {})
                    self.response_data = {
                        "model": completion.get("model", model),
                        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                    }
                else:
                    usage = completion.get("usage", {})
                    self.response_data = {
                        "model": completion.get("model", model),
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                    }
                    content = self._extract_chat_completions_content(completion)
                    if not content and not payload.get("stream"):
                        content = self._retry_chat_completions_as_stream(
                            request_kwargs=request_kwargs,
                            model=model,
                            fallback_usage=usage,
                        )
        finally:
            self._close_response(response)

        print(self.response_data)
        if is_structured and response_format is not None:
            return self._parse_structured_response(
                content,
                response_format,
                model,
                original_system_prompt=original_system_content,
            )

        return content

    def _read_streaming_content(self, response: requests.Response, model: str) -> str:
        collected_content = []
        collected_reasoning = []
        response_model = model
        usage = {}

        for data in self._iter_sse_event_data(response):
            if data == "[DONE]":
                break

            chunk = self._parse_streaming_chunk(data)
            response_model = chunk.get("model", response_model)
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            collected_content.append(self._stringify_stream_delta_field(delta.get("content")))
            collected_reasoning.append(
                self._stringify_stream_delta_field(
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
            )

        final_content = "".join(collected_content).strip() or "".join(collected_reasoning).strip()
        if not final_content:
            raise ValueError("Streaming response completed without any content.")

        self.response_data = {
            "model": response_model,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }
        return final_content

    @staticmethod
    def _parse_streaming_chunk(data: str) -> Dict:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            repaired = repair_json(data)
            return json.loads(repaired)

    @staticmethod
    def _iter_sse_event_data(response: requests.Response):
        data_lines: List[str] = []

        for raw_line in response.iter_lines(decode_unicode=False):
            if raw_line is None:
                continue

            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
            else:
                line = raw_line.rstrip("\r")
            stripped = line.strip()
            if not stripped:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue

            if stripped.startswith(":"):
                continue

            if stripped.startswith("data:"):
                data_lines.append(stripped[5:].lstrip())
                continue

            if data_lines and not any(stripped.startswith(prefix) for prefix in ("event:", "id:", "retry:")):
                # Some compatible proxies pretty-print one JSON event across multiple lines.
                data_lines.append(stripped)

        if data_lines:
            yield "\n".join(data_lines)

    @staticmethod
    def _stringify_stream_delta_field(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                return text
            content = value.get("content")
            if isinstance(content, str):
                return content
            return ""
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                    content = item.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                elif item is not None:
                    parts.append(str(item))
            return "".join(parts)
        return str(value)

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    content = _validate_structured_payload(parsed_dict, response_format)
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = _validate_structured_payload(reparsed_dict, response_format)
                            print("Reparsing successful!")
                            return validated_data
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            target_schema="",
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            return _validate_structured_payload(parsed_dict, response_format)
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            if isinstance(reparsed, dict):
                try:
                    return _validate_structured_payload(reparsed, response_format)
                except Exception:
                    pass

            if isinstance(reparsed, str):
                try:
                    repaired_json = repair_json(reparsed)
                    reparsed_dict = json.loads(repaired_json)
                    return _validate_structured_payload(reparsed_dict, response_format)
                except Exception:
                    heuristic_payload = _construct_structured_payload_from_text(reparsed, response_format)
                    if heuristic_payload is None:
                        heuristic_payload = _construct_structured_payload_from_text(response_text, response_format)
                    if heuristic_payload is not None:
                        return heuristic_payload

            raise ValueError(
                f"Structured response parsing failed after reparsing attempt. "
                f"Raw model output was: {reparsed!r}"
            )

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False, indent=2)
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt="",
            target_schema=schema,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = _validate_structured_payload(reparsed_dict, response_format)
                    print("Reparsing successful!")
                    return validated_data
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: str = "qwen"):
        self.provider = provider.lower()
        if self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        else:
            self.processor = BaseCompatibleProcessor(provider=self.provider)

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model, temperature: float = 0):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        if not isinstance(rag_context, str):
            rag_context = json.dumps(rag_context, ensure_ascii=False, indent=2)
        
        answer_dict = self.processor.send_message(
            model=model,
            temperature=temperature,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = self.provider in {"ibm", "gemini", "qwen"}
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema in {"text", "long_text"}:
            system_prompt = (prompts.AnswerWithRAGContextTextPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextTextPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextTextPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextTextPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str], model: Optional[str] = None) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        use_schema_prompt = self.provider in {"ibm", "gemini", "qwen"}
        system_prompt = (
            prompts.RephrasedQuestionsPrompt.system_prompt_with_schema
            if use_schema_prompt else prompts.RephrasedQuestionsPrompt.system_prompt
        )
        answer_dict = self.processor.send_message(
            model=model or self.processor.default_model,
            system_content=system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict
