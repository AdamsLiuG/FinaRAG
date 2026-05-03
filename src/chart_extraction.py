from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.text_normalization import parse_numeric_value


_UNIT_PRIORITY = (
    "人民币百万元",
    "百万元",
    "人民币千万元",
    "千万元",
    "人民币万元",
    "万元人民币",
    "万元",
    "亿元",
    "亿股",
    "万股",
    "元",
    "%",
    "百分点",
)


def generate_chart_id(document_id: str, page: int, picture_id: int | str) -> str:
    """Build a stable chart id from document, page, and Docling picture id."""
    return f"{document_id}_p{int(page)}_pic{picture_id}"


def _extract_unit_hint(text: str) -> Optional[str]:
    normalized = text or ""
    for unit in _UNIT_PRIORITY:
        if unit in normalized:
            return unit
    return None


def _is_separator_row(cells: List[str]) -> bool:
    if not cells:
        return True
    return all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in cells)


def _markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _stable_numeric_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    nearest_integer = round(numeric)
    if math.isclose(numeric, nearest_integer, rel_tol=0.0, abs_tol=1e-4):
        return float(nearest_integer)
    return round(numeric, 6)


class ChartTableParser:
    """Parse DePlot linearized table output into markdown and numeric chart records."""

    def parse(
        self,
        raw_output: str,
        *,
        chart_id: str,
        page: int,
        picture_id: int | str,
        context_text: str = "",
    ) -> Dict[str, Any]:
        rows = self._parse_rows(raw_output)
        table_markdown = _markdown_table(rows)
        unit_hint = _extract_unit_hint(f"{context_text}\n{raw_output}")
        confidence = self._estimate_confidence(rows, raw_output, context_text, unit_hint)
        records = self._build_records(
            rows,
            chart_id=chart_id,
            page=page,
            picture_id=picture_id,
            context_text=context_text,
            unit_hint=unit_hint,
            confidence=confidence,
        )
        return {
            "table_markdown": table_markdown,
            "records": records,
            "unit_hint": unit_hint,
            "confidence": confidence,
            "parse_status": "ok" if rows else "empty",
        }

    def _parse_rows(self, raw_output: str) -> List[List[str]]:
        text = self._normalize_output(raw_output)
        rows: List[List[str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            cells = self._split_line(stripped)
            if _is_separator_row(cells):
                continue
            if len(cells) >= 2:
                rows.append(cells)
        if len(rows) >= 2:
            return rows
        return []

    @staticmethod
    def _normalize_output(raw_output: str) -> str:
        text = str(raw_output or "")
        replacements = {
            "<0x0A>": "\n",
            "<0x0a>": "\n",
            "<newline>": "\n",
            "\\n": "\n",
            "\r\n": "\n",
            "\r": "\n",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _split_line(line: str) -> List[str]:
        if "|" in line:
            return [cell.strip() for cell in line.strip("|").split("|")]
        if "\t" in line:
            return [cell.strip() for cell in line.split("\t")]
        if "," in line and line.count(",") >= 1:
            return [cell.strip() for cell in line.split(",")]
        return [cell.strip() for cell in re.split(r"\s{2,}", line) if cell.strip()]

    @staticmethod
    def _estimate_confidence(
        rows: List[List[str]],
        raw_output: str,
        context_text: str,
        unit_hint: Optional[str],
    ) -> float:
        if not rows:
            return 0.0
        score = 0.35
        if len(rows) >= 2 and len(rows[0]) >= 2:
            score += 0.2
        numeric_cells = sum(
            1
            for row in rows[1:]
            for cell in row[1:]
            if parse_numeric_value(cell, unit_hint=unit_hint) is not None
        )
        if numeric_cells:
            score += 0.2
        if unit_hint:
            score += 0.15
        if re.search(r"20\d{2}", "\n".join(" ".join(row) for row in rows)):
            score += 0.05
        if context_text.strip():
            score += 0.05
        if raw_output.strip():
            score += 0.05
        return round(min(score, 0.95), 4)

    @staticmethod
    def _build_records(
        rows: List[List[str]],
        *,
        chart_id: str,
        page: int,
        picture_id: int | str,
        context_text: str,
        unit_hint: Optional[str],
        confidence: float,
    ) -> List[Dict[str, Any]]:
        if len(rows) < 2:
            return []
        header = rows[0]
        records: List[Dict[str, Any]] = []
        for row_index, row in enumerate(rows[1:], start=1):
            if not row:
                continue
            x_label = row[0]
            for col_index, raw_value in enumerate(row[1:], start=1):
                series_name = header[col_index] if col_index < len(header) else f"series_{col_index}"
                normalized_value = parse_numeric_value(raw_value, unit_hint=unit_hint)
                if normalized_value is None:
                    continue
                normalized_value = _stable_numeric_value(normalized_value)
                records.append(
                    {
                        "chart_id": chart_id,
                        "page": page,
                        "picture_id": picture_id,
                        "series_name": series_name,
                        "x_label": x_label,
                        "raw_value": raw_value,
                        "normalized_value": normalized_value,
                        "unit": unit_hint,
                        "context_text": context_text,
                        "confidence": confidence,
                        "row_idx": row_index,
                        "col_idx": col_index,
                    }
                )
        return records


@dataclass
class ChartExtractionConfig:
    backend: str = "deplot"
    model: str = "google/deplot"
    device: str = "cpu"
    batch_size: int = 1
    max_new_tokens: int = 512
    overwrite: bool = False
    image_dpi: int = 200
    crop_padding_px: int = 12
    min_picture_area_ratio: float = 0.01
    context_window_blocks: int = 3


class ChartImageCropper:
    """Render PDF pages and crop Docling picture bboxes for chart extraction."""

    def __init__(
        self,
        *,
        image_dpi: int = 200,
        padding_px: int = 12,
        min_picture_area_ratio: float = 0.01,
    ):
        self.image_dpi = int(image_dpi)
        self.padding_px = int(padding_px)
        self.min_picture_area_ratio = float(min_picture_area_ratio)

    def crop_picture(
        self,
        *,
        pdf_path: Path,
        picture: Dict[str, Any],
        chart_id: str,
        image_output_dir: Path,
        overlay_output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        try:
            import pypdfium2 as pdfium  # type: ignore
            from PIL import ImageDraw  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps
            raise RuntimeError("Chart image cropping requires pypdfium2 and Pillow.") from exc

        page_number = int(picture.get("page"))
        bbox = picture.get("bbox") or []
        if len(bbox) != 4:
            raise ValueError(f"Picture {picture.get('picture_id')} has invalid bbox: {bbox}")

        image_output_dir.mkdir(parents=True, exist_ok=True)
        if overlay_output_dir is not None:
            overlay_output_dir.mkdir(parents=True, exist_ok=True)

        document = pdfium.PdfDocument(str(pdf_path))
        try:
            page = document[page_number - 1]
            scale = self.image_dpi / 72.0
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            page_width, page_height = page.get_size()
            crop_box = self._bbox_to_pixel_box(
                bbox,
                page_width=page_width,
                page_height=page_height,
                image_width=image.width,
                image_height=image.height,
                scale=scale,
                padding_px=self.padding_px,
            )
            crop_area = max(0, crop_box[2] - crop_box[0]) * max(0, crop_box[3] - crop_box[1])
            page_area = max(1, image.width * image.height)
            area_ratio = crop_area / page_area
            if area_ratio < self.min_picture_area_ratio:
                raise ValueError(
                    f"Picture area ratio {area_ratio:.4f} below threshold {self.min_picture_area_ratio:.4f}"
                )

            cropped = image.crop(crop_box)
            image_path = image_output_dir / f"{chart_id}.png"
            cropped.save(image_path)

            page_image_path = image_output_dir / f"{chart_id}_page.png"
            image.save(page_image_path)

            overlay_path = None
            if overlay_output_dir is not None:
                overlay = image.copy()
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(crop_box, outline="red", width=4)
                overlay_path = overlay_output_dir / f"{chart_id}_overlay.png"
                overlay.save(overlay_path)

            return {
                "image_path": str(image_path),
                "page_image_path": str(page_image_path),
                "overlay_path": str(overlay_path) if overlay_path else None,
                "crop_box_px": list(crop_box),
                "area_ratio": round(area_ratio, 6),
            }
        finally:
            document.close()

    @staticmethod
    def _bbox_to_pixel_box(
        bbox: Iterable[float],
        *,
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int,
        scale: float,
        padding_px: int,
    ) -> Tuple[int, int, int, int]:
        left, top, right, bottom = [float(value) for value in bbox]
        x0 = int(math.floor(min(left, right) * scale)) - padding_px
        x1 = int(math.ceil(max(left, right) * scale)) + padding_px

        if 0 <= top <= page_height and 0 <= bottom <= page_height and top <= bottom:
            y0 = int(math.floor(top * scale)) - padding_px
            y1 = int(math.ceil(bottom * scale)) + padding_px
        else:
            y0 = int(math.floor((page_height - max(top, bottom)) * scale)) - padding_px
            y1 = int(math.ceil((page_height - min(top, bottom)) * scale)) + padding_px

        x0 = max(0, min(image_width - 1, x0))
        x1 = max(x0 + 1, min(image_width, x1))
        y0 = max(0, min(image_height - 1, y0))
        y1 = max(y0 + 1, min(image_height, y1))
        return x0, y0, x1, y1


class DePlotExtractor:
    """Thin lazy-loading wrapper around google/deplot."""

    def __init__(
        self,
        *,
        model_name: str = "google/deplot",
        device: str = "cpu",
        max_new_tokens: int = 512,
        prompt: str = "Generate underlying data table of the figure below:",
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.prompt = prompt
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._processor is not None and self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, Pix2StructForConditionalGeneration  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on runtime deps
            raise RuntimeError("DePlot extraction requires torch and transformers.") from exc

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Pix2StructForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def extract(self, image_path: Path) -> str:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DePlot extraction requires Pillow.") from exc

        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=self.prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""


class ChartResultWriter:
    """Persist chart extraction results back into parsed report JSON files."""

    def write_results(
        self,
        report_path: Path,
        chart_results: List[Dict[str, Any]],
        *,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        existing = payload.get("charts") or []
        if overwrite:
            by_id = {str(chart.get("chart_id")): chart for chart in existing if chart.get("chart_id")}
            for chart in chart_results:
                by_id[str(chart.get("chart_id"))] = chart
            payload["charts"] = list(by_id.values())
        else:
            existing_ids = {str(chart.get("chart_id")) for chart in existing if chart.get("chart_id")}
            payload["charts"] = existing + [
                chart for chart in chart_results if str(chart.get("chart_id")) not in existing_ids
            ]
        Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


class ChartExtractionRunner:
    """Orchestrate crop -> DePlot -> parser -> parsed report update for one report."""

    def __init__(
        self,
        *,
        config: ChartExtractionConfig,
        image_dir: Path,
        overlay_dir: Path,
        extractor: Optional[DePlotExtractor] = None,
        parser: Optional[ChartTableParser] = None,
        cropper: Optional[ChartImageCropper] = None,
    ):
        self.config = config
        self.image_dir = image_dir
        self.overlay_dir = overlay_dir
        self.extractor = extractor or DePlotExtractor(
            model_name=config.model,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
        )
        self.parser = parser or ChartTableParser()
        self.cropper = cropper or ChartImageCropper(
            image_dpi=config.image_dpi,
            padding_px=config.crop_padding_px,
            min_picture_area_ratio=config.min_picture_area_ratio,
        )

    def process_report(self, *, report_path: Path, pdf_path: Path) -> List[Dict[str, Any]]:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        doc_id = str((report.get("metainfo") or {}).get("sha1_name") or Path(report_path).stem)
        existing_ok_ids = {
            str(chart.get("chart_id"))
            for chart in report.get("charts") or []
            if chart.get("status") == "ok"
        }
        results: List[Dict[str, Any]] = []
        for picture in report.get("pictures") or []:
            page = picture.get("page")
            picture_id = picture.get("picture_id")
            if page is None or picture_id is None:
                continue
            chart_id = generate_chart_id(doc_id, int(page), picture_id)
            if not self.config.overwrite and chart_id in existing_ok_ids:
                continue
            results.append(self._process_picture(report, pdf_path, picture, chart_id))
        return results

    def _process_picture(
        self,
        report: Dict[str, Any],
        pdf_path: Path,
        picture: Dict[str, Any],
        chart_id: str,
    ) -> Dict[str, Any]:
        base = {
            "chart_id": chart_id,
            "picture_id": picture.get("picture_id"),
            "page": picture.get("page"),
            "bbox": picture.get("bbox"),
            "backend": self.config.backend,
            "model": self.config.model,
            "status": "error",
        }
        try:
            crop = self.cropper.crop_picture(
                pdf_path=pdf_path,
                picture=picture,
                chart_id=chart_id,
                image_output_dir=self.image_dir,
                overlay_output_dir=self.overlay_dir,
            )
            raw_output = self.extractor.extract(Path(crop["image_path"]))
            context_text = self._picture_context(report, picture)
            parsed = self.parser.parse(
                raw_output,
                chart_id=chart_id,
                page=int(picture.get("page")),
                picture_id=picture.get("picture_id"),
                context_text=context_text,
            )
            return {
                **base,
                **crop,
                "raw_output": raw_output,
                "table_markdown": parsed["table_markdown"],
                "records": parsed["records"],
                "unit_hint": parsed.get("unit_hint"),
                "confidence": parsed.get("confidence"),
                "context_text": context_text,
                "status": "ok" if parsed["records"] or parsed["table_markdown"] else "empty",
            }
        except Exception as exc:
            return {**base, "error": str(exc)}

    def _picture_context(self, report: Dict[str, Any], picture: Dict[str, Any]) -> str:
        page_number = picture.get("page")
        picture_id = picture.get("picture_id")
        blocks = []
        for page in report.get("content") or []:
            if page.get("page") != page_number:
                continue
            content = page.get("content") or []
            picture_index = next(
                (
                    index
                    for index, block in enumerate(content)
                    if block.get("type") == "picture" and str(block.get("picture_id")) == str(picture_id)
                ),
                None,
            )
            if picture_index is None:
                break
            start = max(0, picture_index - self.config.context_window_blocks)
            end = min(len(content), picture_index + self.config.context_window_blocks + 1)
            for block in content[start:end]:
                if block.get("type") == "picture":
                    continue
                text = str(block.get("text") or "").strip()
                if text:
                    blocks.append(text)
            break
        for child in picture.get("children") or []:
            text = str(child.get("text") or "").strip()
            if text and text not in blocks:
                blocks.append(text)
        return "\n".join(blocks)
