"""Selective visual complexity scoring, 150 DPI PDF rendering, and ColQwen2.5 MaxSim visual index."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from src.storage.base import StorageError, StorageSecurityError, validate_safe_path

_CHART_CUE_RE = re.compile(r"(?:柱状图|折线图|走势图|分布图|饼图|架构图|流程图|示意图|如图所示|参见下图|见下图)")


@dataclass(frozen=True)
class PageVisualScore:
    doc_id: str
    page: int
    score: float
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    table_count: int = 0
    has_chart: bool = False
    low_text_density: bool = False


def score_page_visual_complexity(
    page_dict: Dict[str, Any],
    page_tables: Sequence[Dict[str, Any]],
    page_charts: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    doc_id: str = "",
) -> PageVisualScore:
    """Compute visual complexity score in [0.0, 1.0] to selectively route ~15%-25% complex pages."""
    page_num = int(page_dict.get("page") or 0)
    text = str(page_dict.get("text") or "").strip()
    reasons: List[str] = []
    score = 0.0

    has_chart = bool(page_charts) or bool(_CHART_CUE_RE.search(text))
    if has_chart:
        score += 0.40
        reasons.append("chart_or_figure")

    table_count = len(page_tables or [])
    if table_count >= 2:
        score += 0.25
        reasons.append("multi_table_page")
    elif table_count == 1:
        score += 0.15
        reasons.append("has_table")

    has_large_table = False
    has_continuation = False
    for tbl in page_tables or []:
        nrows = int(tbl.get("nrows") or 0)
        ncols = int(tbl.get("ncols") or 0)
        md = str(tbl.get("markdown") or "")
        if nrows == 0 and md:
            nrows = sum(1 for ln in md.splitlines() if "|" in ln)
        if ncols == 0 and md:
            first_bar = next((ln for ln in md.splitlines() if "|" in ln), "")
            ncols = max(0, first_bar.count("|") - 1)
        if nrows >= 12 or ncols >= 5 or len(md) >= 1200:
            has_large_table = True
        if tbl.get("continuation_of") or tbl.get("merge_state") in {"merged", "continued"}:
            has_continuation = True

    if has_large_table:
        score += 0.30
        reasons.append("complex_or_dense_table")
    if has_continuation:
        score += 0.25
        reasons.append("continued_table")

    compact_len = len(re.sub(r"\s+", "", text))
    low_density = 0 < compact_len < 160
    if low_density and (has_chart or table_count > 0 or page_num > 3):
        score += 0.30
        reasons.append("low_ocr_text_density")

    clamped = round(min(1.0, max(0.0, score)), 4)
    return PageVisualScore(
        doc_id=doc_id,
        page=page_num,
        score=clamped,
        reasons=tuple(reasons),
        table_count=table_count,
        has_chart=has_chart,
        low_text_density=low_density,
    )


def select_visual_pages(
    report_dict: Dict[str, Any],
    *,
    min_ratio: float = 0.0,
    max_ratio: float = 0.25,
    min_score: float = 0.35,
) -> List[PageVisualScore]:
    """Select only the top visually complex pages capped at `max_ratio` (default 25%)."""
    metainfo = report_dict.get("metainfo") or {}
    content = report_dict.get("content") or {}
    doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or "")
    pages = [p for p in (content.get("pages") or []) if isinstance(p, dict) and p.get("page") is not None]
    if not pages:
        return []

    tables_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for tbl in content.get("structured_tables") or []:
        if isinstance(tbl, dict) and tbl.get("page") is not None:
            tables_by_page.setdefault(int(tbl["page"]), []).append(tbl)

    charts_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for ch in content.get("charts") or []:
        if isinstance(ch, dict) and ch.get("page") is not None:
            charts_by_page.setdefault(int(ch["page"]), []).append(ch)

    scored = [
        score_page_visual_complexity(
            p,
            tables_by_page.get(int(p["page"]), []),
            charts_by_page.get(int(p["page"]), []),
            doc_id=doc_id,
        )
        for p in pages
    ]

    total = len(pages)
    max_keep = max(1, int(round(total * max_ratio)))
    min_keep = max(0, min(max_keep, int(round(total * min_ratio))))

    qualified = [s for s in scored if s.score >= float(min_score)]
    qualified.sort(key=lambda s: (s.score, s.table_count, -s.page), reverse=True)
    selected = qualified[:max_keep]

    if len(selected) < min_keep:
        already = {s.page for s in selected}
        remainder = sorted(
            (s for s in scored if s.page not in already and s.score > 0),
            key=lambda s: (s.score, s.table_count, -s.page),
            reverse=True,
        )
        selected.extend(remainder[: min_keep - len(selected)])

    selected.sort(key=lambda s: s.page)
    return selected


def render_pdf_pages(
    pdf_path: Path | str,
    pages: Sequence[int],
    output_dir: Path | str,
    *,
    dpi: int = 150,
    allowed_pdf_dir: Optional[Path | str] = None,
    allowed_output_dir: Optional[Path | str] = None,
) -> Dict[int, Path]:
    """Render requested 1-indexed PDF pages at `dpi` PNGs using pure-Python pypdfium2."""
    safe_pdf = validate_safe_path(pdf_path, allowed_pdf_dir)
    if not safe_pdf.exists() or not safe_pdf.is_file():
        raise FileNotFoundError(f"PDF file not found: {safe_pdf}")
    safe_out = validate_safe_path(output_dir, allowed_output_dir)
    safe_out.mkdir(parents=True, exist_ok=True)

    scale = max(72, int(dpi)) / 72.0
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", safe_pdf.stem) or "report"
    rendered: Dict[int, Path] = {}

    pdf = pdfium.PdfDocument(str(safe_pdf))
    try:
        num_pages = len(pdf)
        for pno in sorted(set(int(p) for p in pages)):
            if pno < 1 or pno > num_pages:
                continue
            page = pdf[pno - 1]
            try:
                pil_img = page.render(scale=scale).to_pil()
                out_path = safe_out / f"{stem}_p{pno:04d}.png"
                pil_img.save(out_path, format="PNG")
                rendered[pno] = out_path
            finally:
                page.close()
    finally:
        pdf.close()

    return rendered


def _pillow_patch_multivector(image_path: Path, *, dim: int = 64, grid: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """Deterministic zero-GPU spatial-patch visual descriptor matching ColBERT matrix interface."""
    rows, cols = grid
    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize((cols * 32, rows * 32))
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    patches: List[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            patch = arr[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32, :]
            gray = patch.mean(axis=2)
            # Horizontal/vertical edge energy + block brightness histogram + channel means
            gx = np.abs(np.diff(gray, axis=1)).mean(axis=1)
            gy = np.abs(np.diff(gray, axis=0)).mean(axis=0)
            feat = np.concatenate(
                [
                    patch.mean(axis=(0, 1)),
                    patch.std(axis=(0, 1)),
                    gx[: min(24, len(gx))],
                    gy[: min(24, len(gy))],
                ]
            )
            vec = np.zeros(dim, dtype=np.float32)
            take = min(dim, len(feat))
            vec[:take] = feat[:take]
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec /= norm
            patches.append(vec)
    return np.vstack(patches).astype(np.float32)


def _l2_norm_2d(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (m / norms).astype(np.float32, copy=False)


class VisualColBERTIndex:
    """Selective visual multi-vector page index (ColQwen2.5 / patch MaxSim) + Qdrant exporter."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        allowed_base_dir: Optional[Path | str] = None,
        image_encoder: Optional[Callable[[Path], np.ndarray]] = None,
        patch_dim: int = 64,
    ):
        self.db_path = validate_safe_path(db_path, allowed_base_dir)
        self.patch_dim = int(patch_dim)
        self._image_encoder = image_encoder or (lambda p: _pillow_patch_multivector(p, dim=self.patch_dim))
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def _get_ro_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not self.db_path.exists():
            raise StorageError("Visual ColBERT SQLite database does not exist")
        try:
            conn = sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
        except sqlite3.Error as exc:
            raise StorageError("Failed to open Visual ColBERT SQLite index") from exc
        self._local.conn = conn
        with self._lock:
            self._conns.append(conn)
        return conn

    def index_rendered_pages(
        self,
        entries: Sequence[Dict[str, Any]],
        *,
        allowed_image_dir: Optional[Path | str] = None,
    ) -> int:
        """Index rendered page PNGs + multi-vectors cleanly via atomic `.tmp` WAL SQLite."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows: Dict[Tuple[str, int], Tuple[Any, ...]] = {}
        if self.db_path.exists():
            rconn = sqlite3.connect(str(self.db_path))
            rconn.row_factory = sqlite3.Row
            try:
                for r in rconn.execute("SELECT * FROM visual_pages"):
                    existing_rows[(str(r["doc_id"]), int(r["page"]))] = (
                        str(r["doc_id"]),
                        int(r["page"]),
                        str(r["company_name"] or ""),
                        str(r["image_path"]),
                        float(r["complexity_score"] or 0.0),
                        str(r["reasons_json"] or "[]"),
                        int(r["n_patches"]),
                        int(r["dim"]),
                        bytes(r["vec_blob"]),
                    )
            except sqlite3.Error:
                pass
            finally:
                rconn.close()

        for entry in entries:
            doc_id = str(entry["doc_id"])
            page = int(entry["page"])
            img_path = validate_safe_path(entry["image_path"], allowed_image_dir)
            vecs = entry.get("vectors")
            if vecs is None:
                vecs = self._image_encoder(img_path)
            normed = _l2_norm_2d(np.asarray(vecs, dtype=np.float32))
            existing_rows[(doc_id, page)] = (
                doc_id,
                page,
                str(entry.get("company_name") or ""),
                str(img_path),
                float(entry.get("complexity_score") or 0.0),
                json.dumps(list(entry.get("reasons") or []), ensure_ascii=False),
                int(normed.shape[0]),
                int(normed.shape[1]),
                normed.tobytes(),
            )

        self.close()
        tmp_path = self.db_path.with_name(f".{self.db_path.name}.tmp")
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{tmp_path}{suffix}")
            if p.exists():
                p.unlink()

        conn = sqlite3.connect(str(tmp_path))
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS visual_pages (
                    doc_id           TEXT NOT NULL,
                    page             INTEGER NOT NULL,
                    company_name     TEXT,
                    image_path       TEXT NOT NULL,
                    complexity_score REAL,
                    reasons_json     TEXT,
                    n_patches        INTEGER NOT NULL,
                    dim              INTEGER NOT NULL,
                    vec_blob         BLOB NOT NULL,
                    PRIMARY KEY (doc_id, page)
                );
                CREATE INDEX IF NOT EXISTS idx_vis_company ON visual_pages(company_name);
                """
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO visual_pages (
                    doc_id, page, company_name, image_path,
                    complexity_score, reasons_json, n_patches, dim, vec_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                list(existing_rows.values()),
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

        os.replace(tmp_path, self.db_path)
        return len(entries)

    def search(
        self,
        query_vectors: np.ndarray,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        company_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Score visual pages via ColBERT MaxSim (`sum_i max_j q_i * d_j`)."""
        q_mat = _l2_norm_2d(query_vectors)
        clauses: List[str] = []
        params: List[Any] = []
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            clauses.append(f"doc_id IN ({placeholders})")
            params.extend([str(d) for d in doc_ids])
        if company_name:
            clauses.append("company_name = ?")
            params.append(str(company_name))

        sql = "SELECT doc_id, page, company_name, image_path, complexity_score, reasons_json, n_patches, dim, vec_blob FROM visual_pages"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        try:
            rows = self._get_ro_conn().execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("Visual ColBERT query failed") from exc

        scored: List[Dict[str, Any]] = []
        for r in rows:
            dim = int(r["dim"])
            n_patches = int(r["n_patches"])
            if dim != q_mat.shape[1] or n_patches <= 0:
                continue
            doc_mat = np.frombuffer(r["vec_blob"], dtype=np.float32).reshape(n_patches, dim)
            sim = q_mat @ doc_mat.T
            maxsim = float(np.max(sim, axis=1).sum())
            scored.append(
                {
                    "doc_id": str(r["doc_id"]),
                    "page": int(r["page"]),
                    "company_name": str(r["company_name"] or ""),
                    "image_path": str(r["image_path"]),
                    "complexity_score": float(r["complexity_score"] or 0.0),
                    "reasons": json.loads(r["reasons_json"] or "[]"),
                    "score": round(maxsim, 6),
                    "retrieval_sources": ["visual_colbert"],
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(top_k))]

    def export_qdrant_visual_points(self, point_id_start: int = 1_000_000) -> List[Dict[str, Any]]:
        """Export indexed pages into Qdrant/QdrantHybridStore compatible colbert payloads."""
        rows = self._get_ro_conn().execute(
            "SELECT doc_id, page, company_name, n_patches, dim, vec_blob FROM visual_pages ORDER BY doc_id, page"
        ).fetchall()
        points: List[Dict[str, Any]] = []
        pid = int(point_id_start)
        for r in rows:
            mat = np.frombuffer(r["vec_blob"], dtype=np.float32).reshape(int(r["n_patches"]), int(r["dim"])).copy()
            points.append(
                {
                    "point_id": pid,
                    "doc_id": str(r["doc_id"]),
                    "chunk_idx": int(r["page"]),
                    "company_name": str(r["company_name"] or ""),
                    "page": int(r["page"]),
                    "node_type": "visual_page",
                    "text": f"[Visual Page {r['page']}] {r['company_name']}",
                    "colbert_vectors": mat,
                }
            )
            pid += 1
        return points

    def stats(self) -> Dict[str, Any]:
        try:
            row = self._get_ro_conn().execute(
                "SELECT COUNT(*) AS cnt, COUNT(DISTINCT doc_id) AS docs FROM visual_pages"
            ).fetchone()
            return {"db_path": str(self.db_path), "pages_indexed": int(row["cnt"]), "docs_indexed": int(row["docs"])}
        except Exception:
            return {"db_path": str(self.db_path), "pages_indexed": 0, "docs_indexed": 0}

    def close(self) -> None:
        with self._lock:
            for c in self._conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._conns.clear()
        if hasattr(self._local, "conn"):
            delattr(self._local, "conn")
