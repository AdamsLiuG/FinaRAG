"""Migrate per-document chunked JSON, FAISS, and sparse artifacts into SQLite + Unified Hybrid Store."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import faiss
import numpy as np

from src.storage.qdrant_store import QdrantHybridStore
from src.storage.sqlite_doc_store import SqliteDocumentRepository
from src.text_normalization import tokenize_for_bm25

DEFAULT_DATA_ROOT = Path("data/top10_industries_2024_20each/databases_ser_tab")


def _iter_corpus_batches(
    *,
    chunked_reports_dir: Path,
    vector_db_dir: Path,
    sparse_db_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    include_bm25_tokens: bool = True,
) -> Iterator[Tuple[List[Dict[str, Any]], np.ndarray]]:
    files = sorted(chunked_reports_dir.glob("*.json"))
    if limit is not None:
        files = files[: max(0, int(limit))]

    next_pid = 0
    for json_path in files:
        stem = json_path.stem
        with open(json_path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
        if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
            continue
        metainfo = document.get("metainfo") or {}
        chunks = list((document.get("content") or {}).get("chunks") or [])
        if not chunks:
            continue

        doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or stem)
        company_name = str(metainfo.get("company_name") or "")
        industry_l1 = str(metainfo.get("industry_l1") or metainfo.get("major_industry") or "")
        raw_year = metainfo.get("report_year") or metainfo.get("fiscal_year")
        try:
            report_year = int(raw_year) if raw_year is not None else None
        except (TypeError, ValueError):
            report_year = None

        faiss_path = vector_db_dir / f"{stem}.faiss"
        if not faiss_path.exists():
            raise FileNotFoundError(f"Missing FAISS index for {stem}: {faiss_path}")
        idx = faiss.read_index(str(faiss_path))
        if idx.ntotal != len(chunks):
            raise ValueError(f"Vector/chunk count mismatch for {stem}: faiss={idx.ntotal} vs chunks={len(chunks)}")
        vecs = np.asarray(idx.reconstruct_n(0, idx.ntotal), dtype=np.float32)

        lexical_weights: Optional[List[Dict[str, float]]] = None
        if sparse_db_dir is not None:
            sp_path = sparse_db_dir / f"{stem}.pkl"
            if sp_path.exists():
                with open(sp_path, "rb") as sp_fh:
                    sp_obj = pickle.load(sp_fh)
                if isinstance(sp_obj, dict) and isinstance(sp_obj.get("lexical_weights"), list):
                    lexical_weights = sp_obj["lexical_weights"]

        records: List[Dict[str, Any]] = []
        for c_idx, chunk in enumerate(chunks):
            raw_text = str(chunk.get("text") or "")
            search_text = str(chunk.get("search_text") or "")
            combined_bm25_src = f"{search_text}\n{raw_text}" if search_text else raw_text
            rec: Dict[str, Any] = {
                "point_id": next_pid,
                "doc_id": doc_id,
                "chunk_idx": c_idx,
                "company_name": company_name,
                "report_year": report_year,
                "industry_l1": industry_l1,
                "page": int(chunk.get("page") or 0),
                "node_type": str(chunk.get("node_type") or "child"),
                "section_name": str(chunk.get("section_name") or chunk.get("section_title") or ""),
                "statement_location": str(chunk.get("report_section") or chunk.get("statement_location") or ""),
                "topic_flags": list(chunk.get("topic_flags") or []),
                "strategy_tags": list(chunk.get("strategy_tags") or []),
                "text": raw_text,
                "bm25_tokens": tokenize_for_bm25(combined_bm25_src) if include_bm25_tokens else [],
                "bge_sparse": lexical_weights[c_idx] if lexical_weights and c_idx < len(lexical_weights) else None,
            }
            records.append(rec)
            next_pid += 1

        yield records, vecs


def migrate_corpus(
    *,
    chunked_reports_dir: Path | str,
    vector_db_dir: Path | str,
    sparse_db_dir: Optional[Path | str] = None,
    doc_sqlite_path: Path | str,
    unified_store_dir: Path | str,
    mode: str = "embedded",
    qdrant_url: Optional[str] = None,
    collection_name: str = "finarag_chunks_v1",
    limit: Optional[int] = None,
    quantize_sq8: bool = True,
) -> Dict[str, Any]:
    reports_dir = Path(chunked_reports_dir)
    vec_dir = Path(vector_db_dir)
    sp_dir = Path(sparse_db_dir) if sparse_db_dir else None
    doc_db = Path(doc_sqlite_path)
    store_dir = Path(unified_store_dir)

    doc_stats = SqliteDocumentRepository.build_from_chunked_dir(
        reports_dir,
        doc_db,
        limit=limit,
    )

    if mode == "http":
        store = QdrantHybridStore(mode="http", qdrant_url=qdrant_url, collection_name=collection_name)
        store.ensure_collection()
        vectors_uploaded = 0
        for records, vecs in _iter_corpus_batches(
            chunked_reports_dir=reports_dir,
            vector_db_dir=vec_dir,
            sparse_db_dir=sp_dir,
            limit=limit,
        ):
            points = []
            for rec, vec in zip(records, vecs):
                p = dict(rec)
                p["dense_vector"] = vec
                p["bm25_sparse"] = {tok: 1.0 for tok in rec.get("bm25_tokens") or []}
                points.append(p)
            vectors_uploaded += store.upsert_points(points, batch_size=256)
        store_stats = {"mode": "http", "vectors_count": vectors_uploaded, "collection_name": collection_name}
        store.close()
    else:
        store = QdrantHybridStore(mode="embedded", store_dir=store_dir, collection_name=collection_name)
        batches = _iter_corpus_batches(
            chunked_reports_dir=reports_dir,
            vector_db_dir=vec_dir,
            sparse_db_dir=sp_dir,
            limit=limit,
        )
        store_stats = store.build_embedded_from_batches(batches, quantize_sq8=quantize_sq8)
        store.close()

    if int(store_stats["vectors_count"]) != int(doc_stats["chunks_count"]):
        raise RuntimeError(
            f"Migration parity check failed: doc_chunks={doc_stats['chunks_count']} != store_vectors={store_stats['vectors_count']}"
        )

    return {
        "document_store": doc_stats,
        "hybrid_store": store_stats,
        "parity_verified": True,
    }


def upgrade_embedded_faiss_to_ivf_sq8(
    *,
    chunked_reports_dir: Path | str,
    vector_db_dir: Path | str,
    unified_store_dir: Path | str,
) -> Dict[str, Any]:
    """Upgrade existing unified FAISS index to IVF-SQ8 without rebuilding SQLite metadata/FTS5."""
    reports_dir = Path(chunked_reports_dir)
    vec_dir = Path(vector_db_dir)
    store_dir = Path(unified_store_dir)
    meta_db_path = store_dir / "finarag_hybrid_meta.sqlite"
    faiss_path = store_dir / "finarag_unified.faiss"
    if not meta_db_path.exists():
        raise FileNotFoundError(f"Missing unified metadata DB: {meta_db_path}")

    vec_chunks: List[np.ndarray] = []
    for json_path in sorted(reports_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
        if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
            continue
        chunks = list((document.get("content") or {}).get("chunks") or [])
        if not chunks:
            continue
        fp = vec_dir / f"{json_path.stem}.faiss"
        idx = faiss.read_index(str(fp))
        if idx.ntotal != len(chunks):
            raise ValueError(f"Vector/chunk count mismatch for {json_path.stem}")
        vecs = np.asarray(idx.reconstruct_n(0, idx.ntotal), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        vec_chunks.append((vecs / norms).astype(np.float32, copy=False))

    all_vecs = np.vstack(vec_chunks)
    vec_count, dim = int(all_vecs.shape[0]), int(all_vecs.shape[1])

    conn = sqlite3.connect(str(meta_db_path))
    try:
        db_count = int(conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0])
        if db_count != vec_count:
            raise RuntimeError(f"Parity mismatch: chunks_meta={db_count} != vectors={vec_count}")
        nlist = min(512, max(64, int(math.sqrt(vec_count))))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
        )
        sample_n = min(vec_count, 65_536)
        step = max(1, vec_count // sample_n)
        index.train(np.ascontiguousarray(all_vecs[::step][:sample_n], dtype=np.float32))
        index.add(np.ascontiguousarray(all_vecs, dtype=np.float32))
        index.nprobe = min(24, nlist)

        tmp_faiss = faiss_path.with_name(f".{faiss_path.name}.tmp")
        faiss.write_index(index, str(tmp_faiss))
        os.replace(tmp_faiss, faiss_path)

        conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
            ("dense_quantization", "ivf_sq8_int8"),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "store_dir": str(store_dir),
        "vectors_count": vec_count,
        "dense_dim": dim,
        "nlist": nlist,
        "dense_quantization": "ivf_sq8_int8",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate FinaRAG artifacts to SQLite + Unified Hybrid Store.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunked-dir", type=Path, default=None)
    parser.add_argument("--vector-dir", type=Path, default=None)
    parser.add_argument("--sparse-dir", type=Path, default=None)
    parser.add_argument("--doc-sqlite", type=Path, default=None)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("embedded", "http"), default="embedded")
    parser.add_argument("--qdrant-url", type=str, default=None)
    parser.add_argument("--collection", type=str, default="finarag_chunks_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-sq8", action="store_true")
    parser.add_argument("--rebuild-faiss-only", action="store_true")
    args = parser.parse_args(argv)

    root = args.data_root
    if args.rebuild_faiss_only:
        summary = upgrade_embedded_faiss_to_ivf_sq8(
            chunked_reports_dir=args.chunked_dir or (root / "chunked_reports"),
            vector_db_dir=args.vector_dir or (root / "vector_dbs"),
            unified_store_dir=args.store_dir or (root / "unified_store"),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    summary = migrate_corpus(
        chunked_reports_dir=args.chunked_dir or (root / "chunked_reports"),
        vector_db_dir=args.vector_dir or (root / "vector_dbs"),
        sparse_db_dir=args.sparse_dir or (root / "sparse_dbs"),
        doc_sqlite_path=args.doc_sqlite or (root / "finarag_docs.sqlite"),
        unified_store_dir=args.store_dir or (root / "unified_store"),
        mode=args.mode,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        limit=args.limit,
        quantize_sq8=not args.no_sq8,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
