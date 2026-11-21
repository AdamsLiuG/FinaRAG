"""LightRAG dual-level + HippoRAG 2 Personalized PageRank graph retriever and RRF fusion."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from src.knowledge_graph.kg_builder import FinancialKGStore
from src.retrieval_filters import build_result_metadata

_KG_TRIGGER_CUES = (
    "控股股东",
    "实际控制人",
    "实控人",
    "子公司",
    "参股",
    "控股",
    "持股",
    "关联交易",
    "关联方",
    "关联企业",
    "重大担保",
    "违规担保",
    "担保",
    "前五大客户",
    "前五名客户",
    "主要客户",
    "前五大供应商",
    "前五名供应商",
    "主要供应商",
    "供应链",
    "上下游",
    "股权穿透",
    "共同股东",
    "共同投资",
    "产业链",
    "竞争对手",
)

_QUERY_REL_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("控股股东", "实控人", "实际控制人", "股东"), ("CONTROLLING_SHAREHOLDER", "ACTUAL_CONTROLLER", "MAJOR_SHAREHOLDER")),
    (("子公司", "控股", "旗下"), ("HOLDS_SUBSIDIARY",)),
    (("参股", "持股", "股权投资", "联营", "合营"), ("EQUITY_INVESTMENT", "HOLDS_SUBSIDIARY")),
    (("关联交易", "关联方", "授信", "关联企业"), ("RELATED_PARTY_TX",)),
    (("担保",), ("GUARANTEES",)),
    (("客户",), ("TOP_CUSTOMER",)),
    (("供应商", "供应链", "上下游"), ("TOP_SUPPLIER", "SUPPLY_CHAIN_DEPENDENCY")),
)


def should_trigger_kg_retrieval(query_plan: Any, question: str) -> bool:
    """Trigger KG retrieval on equity, related-party, guarantee, supply-chain or multi-hop cues."""
    q_lower = str(question or "").lower()
    if any(cue in q_lower for cue in _KG_TRIGGER_CUES):
        return True
    norm = str(getattr(query_plan, "normalized_query", "") or "").lower()
    return any(cue in norm for cue in _KG_TRIGGER_CUES)


def _infer_target_relations(question: str) -> set[str]:
    q = str(question or "")
    rels: set[str] = set()
    for cues, rtypes in _QUERY_REL_HINTS:
        if any(c in q for c in cues):
            rels.update(rtypes)
    return rels


class LightRAGRetriever:
    """Dual-level entity/relation linker + HippoRAG 2 PPR multi-hop graph retriever."""

    def __init__(self, kg_store: FinancialKGStore):
        self.kg_store = kg_store
        self.graph: nx.MultiDiGraph = kg_store.to_networkx()
        self._alias_index: List[Tuple[str, str]] = self._build_alias_index()

    def _build_alias_index(self) -> List[Tuple[str, str]]:
        pairs: Dict[str, str] = {}
        for node_name, data in self.graph.nodes(data=True):
            if data.get("entity_type") == "anonymized_counterparty":
                continue
            cname = str(node_name)
            if len(cname) >= 2:
                pairs[cname] = cname
            for alias in data.get("aliases") or []:
                al = str(alias or "").strip()
                if len(al) >= 2 and al not in pairs:
                    pairs[al] = cname
        # Longest-first so full corporate names match before substrings
        return sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0]))

    def link_entities(self, question: str, *, company_name: Optional[str] = None, max_seeds: int = 5) -> List[str]:
        seeds: List[str] = []
        seen: set[str] = set()
        if company_name and company_name in self.graph:
            seeds.append(company_name)
            seen.add(company_name)

        q_compact = re.sub(r"\s+", "", str(question or ""))
        for term, canonical in self._alias_index:
            if canonical in seen:
                continue
            if len(term) >= 2 and term.replace(" ", "") in q_compact:
                seeds.append(canonical)
                seen.add(canonical)
                if len(seeds) >= max_seeds:
                    break
        return seeds

    def ppr_expand(
        self,
        seed_entities: Sequence[str],
        *,
        target_relations: Optional[set[str]] = None,
        alpha: float = 0.85,
        top_k: int = 12,
    ) -> List[Tuple[str, float]]:
        """Run HippoRAG 2 style Personalized PageRank over non-anonymized graph structure."""
        valid_seeds = [s for s in seed_entities if s in self.graph]
        if not valid_seeds:
            return []

        undir = nx.Graph()
        for u, v, _, data in self.graph.edges(keys=True, data=True):
            # Exclude anonymized counterfactual nodes from global hub diffusion unless explicitly seeded
            u_type = self.graph.nodes[u].get("entity_type")
            v_type = self.graph.nodes[v].get("entity_type")
            if (u_type == "anonymized_counterparty" and u not in valid_seeds) or (
                v_type == "anonymized_counterparty" and v not in valid_seeds
            ):
                continue
            rel = str(data.get("relation_type") or "")
            w = float(data.get("confidence") or 0.8)
            if target_relations and rel in target_relations:
                w *= 2.0
            if undir.has_edge(u, v):
                undir[u][v]["weight"] += w
            else:
                undir.add_edge(u, v, weight=w)

        for s in valid_seeds:
            if s not in undir:
                undir.add_node(s)

        pers = {node: (1.0 / len(valid_seeds) if node in valid_seeds else 0.0) for node in undir.nodes()}
        if not undir.edges():
            return [(s, 1.0 / len(valid_seeds)) for s in valid_seeds[:top_k]]

        try:
            scores = nx.pagerank(undir, alpha=alpha, personalization=pers, weight="weight", max_iter=200, tol=1e-5)
        except nx.PowerIterationFailedConvergence:
            scores = {}
            for s in valid_seeds:
                lengths = nx.single_source_shortest_path_length(undir, s, cutoff=3)
                for node, dist in lengths.items():
                    scores[node] = scores.get(node, 0.0) + (alpha ** dist) / len(valid_seeds)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    def _incident_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        if node not in self.graph:
            return out
        for u, v, _, d in self.graph.out_edges(node, keys=True, data=True):
            out.append((u, v, dict(d)))
        for u, v, _, d in self.graph.in_edges(node, keys=True, data=True):
            out.append((u, v, dict(d)))
        return out

    def extract_multihop_paths(
        self,
        seed_entities: Sequence[str],
        ppr_scores: Dict[str, float],
        *,
        target_relations: Optional[set[str]] = None,
        max_hops: int = 2,
        max_paths: int = 8,
    ) -> List[Dict[str, Any]]:
        valid_seeds = [s for s in seed_entities if s in self.graph]
        if not valid_seeds:
            return []

        paths: List[Dict[str, Any]] = []
        seen_sigs: set[Tuple[str, ...]] = set()

        def _score_edge(d: Dict[str, Any], other_node: str) -> float:
            base = float(d.get("confidence") or 0.8)
            rel_bonus = 1.5 if (target_relations and d.get("relation_type") in target_relations) else 0.0
            ppr_bonus = float(ppr_scores.get(other_node, 0.0)) * 5.0
            return base + rel_bonus + ppr_bonus

        # Priority 1: paths connecting >= 2 distinct seed entities (up to 2 hops)
        if len(valid_seeds) >= 2:
            for i, s1 in enumerate(valid_seeds):
                for s2 in valid_seeds[i + 1 :]:
                    for u, v, d in self._incident_edges(s1):
                        other = v if u == s1 else u
                        if other == s2:
                            sig = tuple(sorted((u, v)) + [str(d.get("relation_type")), str(d.get("evidence_page"))])
                            if sig not in seen_sigs:
                                seen_sigs.add(sig)
                                paths.append(
                                    {
                                        "hops": 1,
                                        "score": round(_score_edge(d, s2) + 3.0, 4),
                                        "summary": f"{u} -[{d.get('relation_type')}]-> {v} (doc={d.get('evidence_doc_id')}, p{d.get('evidence_page')})",
                                        "edges": [{"head": u, "tail": v, **d}],
                                    }
                                )
                        elif max_hops >= 2 and self.graph.nodes[other].get("entity_type") != "anonymized_counterparty":
                            for u2, v2, d2 in self._incident_edges(other):
                                end_node = v2 if u2 == other else u2
                                if end_node == s2:
                                    sig = ("2hop", s1, other, s2, str(d.get("relation_type")), str(d2.get("relation_type")))
                                    if sig not in seen_sigs:
                                        seen_sigs.add(sig)
                                        sc = _score_edge(d, other) + _score_edge(d2, s2) + 2.5
                                        paths.append(
                                            {
                                                "hops": 2,
                                                "score": round(sc, 4),
                                                "summary": (
                                                    f"{u} -[{d.get('relation_type')}]-> {v} | "
                                                    f"{u2} -[{d2.get('relation_type')}]-> {v2} "
                                                    f"(p{d.get('evidence_page')}/p{d2.get('evidence_page')})"
                                                ),
                                                "edges": [{"head": u, "tail": v, **d}, {"head": u2, "tail": v2, **d2}],
                                            }
                                        )

        # Priority 2: top 1-hop & 2-hop expansions from primary seed(s)
        for seed in valid_seeds:
            hop1_sorted = sorted(
                self._incident_edges(seed),
                key=lambda item: _score_edge(item[2], item[1] if item[0] == seed else item[0]),
                reverse=True,
            )
            for u, v, d in hop1_sorted[: max_paths * 2]:
                other = v if u == seed else u
                sig = (u, str(d.get("relation_type")), v, str(d.get("evidence_doc_id")), str(d.get("evidence_page")))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    paths.append(
                        {
                            "hops": 1,
                            "score": round(_score_edge(d, other), 4),
                            "summary": f"{u} -[{d.get('relation_type')}]-> {v} (doc={d.get('evidence_doc_id')}, p{d.get('evidence_page')})",
                            "edges": [{"head": u, "tail": v, **d}],
                        }
                    )
                if max_hops >= 2 and self.graph.nodes[other].get("entity_type") != "anonymized_counterparty":
                    hop2_sorted = sorted(
                        self._incident_edges(other),
                        key=lambda item: _score_edge(item[2], item[1] if item[0] == other else item[0]),
                        reverse=True,
                    )
                    for u2, v2, d2 in hop2_sorted[:2]:
                        end_node = v2 if u2 == other else u2
                        if end_node == seed:
                            continue
                        sig2 = ("2h", seed, other, end_node, str(d2.get("relation_type")), str(d2.get("evidence_doc_id")))
                        if sig2 in seen_sigs:
                            continue
                        seen_sigs.add(sig2)
                        sc2 = (_score_edge(d, other) + _score_edge(d2, end_node)) * 0.65
                        paths.append(
                            {
                                "hops": 2,
                                "score": round(sc2, 4),
                                "summary": (
                                    f"{u} -[{d.get('relation_type')}]-> {v} | "
                                    f"{u2} -[{d2.get('relation_type')}]-> {v2} "
                                    f"(p{d.get('evidence_page')}/p{d2.get('evidence_page')})"
                                ),
                                "edges": [{"head": u, "tail": v, **d}, {"head": u2, "tail": v2, **d2}],
                            }
                        )

        paths.sort(key=lambda item: (-item["score"], item["hops"], item["summary"]))
        return paths[:max_paths]

    def retrieve(
        self,
        question: str,
        *,
        company_name: Optional[str] = None,
        page_fetcher: Optional[Callable[[str, int], Optional[Dict[str, Any]]]] = None,
        max_hops: int = 2,
        max_paths: int = 8,
        max_pages: int = 4,
    ) -> Dict[str, Any]:
        target_rels = _infer_target_relations(question)
        seeds = self.link_entities(question, company_name=company_name)
        ppr_ranked = self.ppr_expand(seeds, target_relations=target_rels, top_k=15)
        ppr_map = dict(ppr_ranked)
        paths = self.extract_multihop_paths(
            seeds,
            ppr_map,
            target_relations=target_rels,
            max_hops=max_hops,
            max_paths=max_paths,
        )

        context_lines = [f"- {p['summary']}" for p in paths]
        context_summary = "[金融关系图谱路径]\n" + "\n".join(context_lines) if context_lines else ""

        results: List[Dict[str, Any]] = []
        if context_summary:
            first_edge = paths[0]["edges"][0]
            doc_id0 = str(first_edge.get("evidence_doc_id") or "")
            page0 = int(first_edge.get("evidence_page") or 1)
            meta0 = build_result_metadata(
                {"sha1_name": doc_id0, "doc_id": doc_id0, "company_name": company_name or (seeds[0] if seeds else "")},
                {"page": page0, "chunk_type": "kg_path_summary", "node_type": "kg_graph", "section_title": "关系图谱路径"},
            )
            results.append(
                {
                    "distance": round(float(paths[0]["score"]), 4),
                    "combined_score": round(float(paths[0]["score"]), 4),
                    "ranking_score": round(float(paths[0]["score"]), 4),
                    "page": page0,
                    "text": context_summary,
                    "pdf_sha1": doc_id0,
                    "metadata": meta0,
                    "chunk_type": "kg_path_summary",
                    "retrieval_sources": ["kg_lightrag"],
                    "result_scope": "kg_subgraph",
                }
            )

        # Hydrate unique source evidence pages from graph edges
        page_scores: Dict[Tuple[str, int], Tuple[float, str]] = {}
        for p in paths:
            for edge in p.get("edges") or []:
                did = str(edge.get("evidence_doc_id") or "")
                pno = int(edge.get("evidence_page") or 0)
                if did and pno > 0:
                    key = (did, pno)
                    prev_sc = page_scores.get(key, (0.0, ""))[0]
                    if p["score"] > prev_sc:
                        page_scores[key] = (float(p["score"]), str(edge.get("section_title") or ""))

        ranked_evidence_pages = sorted(page_scores.items(), key=lambda kv: (-kv[1][0], kv[0][1]))[:max_pages]
        if page_fetcher is not None:
            for (did, pno), (sc, sec_title) in ranked_evidence_pages:
                pobj = page_fetcher(did, pno)
                if not pobj or not str(pobj.get("text") or "").strip():
                    continue
                meta = build_result_metadata(
                    {"sha1_name": did, "doc_id": did, "company_name": pobj.get("company_name") or company_name or ""},
                    {"page": pno, "chunk_type": "page", "node_type": "page", "section_title": sec_title},
                )
                results.append(
                    {
                        "distance": round(sc, 4),
                        "combined_score": round(sc, 4),
                        "ranking_score": round(sc, 4),
                        "page": pno,
                        "text": str(pobj["text"]),
                        "pdf_sha1": did,
                        "metadata": meta,
                        "chunk_type": "page",
                        "section_title": sec_title,
                        "retrieval_sources": ["kg_lightrag"],
                        "result_scope": "page",
                    }
                )

        return {
            "seed_entities": seeds,
            "ppr_entities": ppr_ranked,
            "paths": paths,
            "evidence_pages": [(did, pno) for (did, pno), _ in ranked_evidence_pages],
            "context_summary": context_summary,
            "results": results,
        }


def fuse_kg_and_hybrid_results(
    hybrid_results: List[Dict[str, Any]],
    kg_results: List[Dict[str, Any]],
    *,
    top_n: int = 10,
    rrf_k: int = 60,
    kg_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """Fuse hybrid retrieval hits with LightRAG+PPR graph evidence using RRF."""
    if not kg_results:
        return list(hybrid_results or [])[:top_n]
    if not hybrid_results:
        return list(kg_results)[:top_n]

    def _key(item: Dict[str, Any]) -> Tuple[str, Any]:
        meta = item.get("metadata") or {}
        sha1 = str(item.get("pdf_sha1") or meta.get("sha1_name") or meta.get("doc_id") or "")
        if item.get("chunk_type") == "kg_path_summary":
            return (sha1, "kg_path_summary")
        return (sha1, item.get("page"))

    fused: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    scores: Dict[Tuple[str, Any], float] = {}

    for rank, item in enumerate(hybrid_results, start=1):
        k = _key(item)
        scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank)
        if k not in fused:
            cp = dict(item)
            cp["retrieval_sources"] = list(item.get("retrieval_sources") or ["hybrid"])
            fused[k] = cp

    for rank, item in enumerate(kg_results, start=1):
        k = _key(item)
        scores[k] = scores.get(k, 0.0) + kg_weight / (rrf_k + rank)
        if k not in fused:
            cp = dict(item)
            cp["retrieval_sources"] = list(item.get("retrieval_sources") or ["kg_lightrag"])
            fused[k] = cp
        else:
            existing_sources = fused[k].setdefault("retrieval_sources", [])
            for src in item.get("retrieval_sources") or ["kg_lightrag"]:
                if src not in existing_sources:
                    existing_sources.append(src)
            if fused[k].get("result_scope") != "page" and len(str(item.get("text") or "")) > len(str(fused[k].get("text") or "")):
                fused[k]["text"] = item["text"]
                fused[k]["result_scope"] = item.get("result_scope") or "page"

    ordered = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)[:top_n]
    output: List[Dict[str, Any]] = []
    for k in ordered:
        entry = fused[k]
        entry["combined_score"] = round(scores[k], 6)
        entry["ranking_score"] = round(scores[k], 6)
        output.append(entry)
    return output
