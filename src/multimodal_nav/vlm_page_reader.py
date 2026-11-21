"""Qwen2.5-VL multimodal page reader and online visual grounding verifier."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from json_repair import repair_json

from src.storage.base import StorageError, StorageSecurityError, validate_safe_path
from src.text_normalization import parse_numeric_value

_ALLOWED_IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


@dataclass
class VLMExtractionResult:
    doc_id: str
    page: int
    title: str
    unit: Optional[str]
    headers: List[str]
    rows: List[List[str]]
    records: List[Dict[str, Any]] = field(default_factory=list)
    table_markdown: str = ""
    confidence: float = 0.0
    raw_text: str = ""


@dataclass
class VLMVerificationResult:
    verified: bool
    answer_value: Any = None
    unit: Optional[str] = None
    confidence: float = 0.0
    evidence_quote: str = ""
    page: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "answer_value": self.answer_value,
            "unit": self.unit,
            "confidence": self.confidence,
            "evidence_quote": self.evidence_quote,
            "page": self.page,
            "reason": self.reason,
        }


def _rows_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    if not headers and not rows:
        return ""
    width = max([len(headers)] + [len(r) for r in rows] + [1])
    hdr = (headers + [""] * width)[:width]
    lines = [
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join("---" for _ in hdr) + " |",
    ]
    for r in rows:
        padded = ([str(c) for c in r] + [""] * width)[:width]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


class VLMPageReader:
    """OpenAI-compatible multimodal VLM client for complex charts/tables & grounding verification."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        http_transport: Optional[httpx.BaseTransport] = None,
    ):
        raw_url = (
            base_url
            if base_url is not None
            else (os.getenv("VLM_BASE_URL") or os.getenv("QWEN_BASE_URL") or os.getenv("LLM_BASE_URL") or "")
        ).strip().rstrip("/")
        if raw_url:
            parsed = urlparse(raw_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise StorageSecurityError("Invalid VLM base_url scheme or host")
        self.base_url = raw_url
        resolved_key = (
            api_key
            if api_key is not None
            else (os.getenv("VLM_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("LLM_API_KEY") or "")
        )
        self.model = (model or os.getenv("VLM_MODEL") or "Qwen2.5-VL-7B-Instruct-AWQ").strip()
        headers = {"Content-Type": "application/json"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        self._client = httpx.Client(headers=headers, timeout=timeout, transport=http_transport)

    def _chat_endpoint(self) -> str:
        if not self.base_url:
            raise StorageError("VLM base_url is not configured")
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _encode_image_data_url(
        image_path: Path | str,
        *,
        allowed_base_dir: Optional[Path | str] = None,
        max_bytes: int = _MAX_IMAGE_BYTES,
    ) -> str:
        safe_path = validate_safe_path(image_path, allowed_base_dir)
        if not safe_path.exists() or not safe_path.is_file():
            raise FileNotFoundError(f"Image file not found: {safe_path}")
        suffix = safe_path.suffix.lower()
        mime = _ALLOWED_IMAGE_SUFFIXES.get(suffix)
        if mime is None:
            raise StorageSecurityError(f"Unsupported image extension: {suffix}")
        size = safe_path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise StorageSecurityError(f"Image size out of safe bounds: {size} bytes")
        b64 = base64.b64encode(safe_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _call_vlm_json(self, system_prompt: str, user_text: str, data_url: str) -> tuple[Dict[str, Any], str]:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        resp = self._client.post(self._chat_endpoint(), json=payload)
        if resp.status_code != 200:
            raise StorageError(f"VLM request failed (HTTP {resp.status_code})")
        body = resp.json()
        raw_content = str((((body.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")
        decoded = json.loads(repair_json(raw_content))
        if not isinstance(decoded, dict):
            raise StorageError("VLM response did not decode to a JSON object")
        return decoded, raw_content

    def extract_structured_table_from_image(
        self,
        image_path: Path | str,
        *,
        doc_id: str = "",
        page: int = 0,
        context_hint: str = "",
        allowed_base_dir: Optional[Path | str] = None,
    ) -> VLMExtractionResult:
        data_url = self._encode_image_data_url(image_path, allowed_base_dir=allowed_base_dir)
        sys_prompt = (
            "你是金融文档视觉解析专家。从年报页面截图中精确抽取表格或图表中的数值。"
            "严格输出 JSON：{\"title\": \"...\", \"unit\": \"亿元|百万元|万元|元|%\", "
            "\"headers\": [...], \"rows\": [[...]], \"confidence\": 0.95}"
        )
        usr_prompt = f"页码：{page}\n补充线索：{context_hint or '无'}\n请提取页面核心表格或图表数据并严格以 JSON 返回。"
        parsed, raw_text = self._call_vlm_json(sys_prompt, usr_prompt, data_url)
        headers = [str(h) for h in (parsed.get("headers") or [])]
        rows = [[str(c) for c in row] for row in (parsed.get("rows") or []) if isinstance(row, list)]
        unit = str(parsed.get("unit") or "").strip() or None
        md = _rows_to_markdown(headers, rows)

        records: List[Dict[str, Any]] = []
        for row in rows:
            if not row:
                continue
            series_name = str(row[0]).strip()
            for col_idx in range(1, len(row)):
                x_label = headers[col_idx] if col_idx < len(headers) else ""
                raw_val = str(row[col_idx]).strip()
                norm_val = parse_numeric_value(raw_val, unit_hint=unit)
                if norm_val is None:
                    continue
                records.append(
                    {
                        "page": int(page),
                        "series_name": series_name,
                        "x_label": x_label,
                        "raw_value": raw_val,
                        "normalized_value": norm_val,
                        "unit": unit,
                        "confidence": float(parsed.get("confidence") or 0.85),
                        "table_markdown": md,
                    }
                )

        return VLMExtractionResult(
            doc_id=doc_id,
            page=int(page),
            title=str(parsed.get("title") or ""),
            unit=unit,
            headers=headers,
            rows=rows,
            records=records,
            table_markdown=md,
            confidence=float(parsed.get("confidence") or 0.0),
            raw_text=raw_text,
        )

    def verify_grounding_with_vlm(
        self,
        image_path: Path | str,
        *,
        question: str,
        candidate_answer: Any = None,
        page: int = 0,
        allowed_base_dir: Optional[Path | str] = None,
    ) -> VLMVerificationResult:
        try:
            data_url = self._encode_image_data_url(image_path, allowed_base_dir=allowed_base_dir)
            sys_prompt = (
                "你是金融年报视觉核验专家。对照原始页面截图核对候选答案是否准确、单位与期间是否正确。"
                "严格输出 JSON：{\"verified\": true/false, \"answer_value\": ..., \"unit\": \"...\", "
                "\"confidence\": 0.95, \"evidence_quote\": \"...\", \"reason\": \"...\"}"
            )
            usr_prompt = (
                f"问题：{question}\n"
                f"待核验候选答案：{candidate_answer}\n"
                f"目标页码：{page}\n"
                "请直接读取截图给出视觉核验 JSON。"
            )
            parsed, _ = self._call_vlm_json(sys_prompt, usr_prompt, data_url)
            return VLMVerificationResult(
                verified=bool(parsed.get("verified")),
                answer_value=parsed.get("answer_value"),
                unit=str(parsed.get("unit") or "").strip() or None,
                confidence=float(parsed.get("confidence") or 0.0),
                evidence_quote=str(parsed.get("evidence_quote") or ""),
                page=int(page),
                reason=str(parsed.get("reason") or ""),
            )
        except Exception as exc:
            return VLMVerificationResult(
                verified=False,
                page=int(page),
                reason=f"fail_closed:{type(exc).__name__}",
            )

    def close(self) -> None:
        self._client.close()
