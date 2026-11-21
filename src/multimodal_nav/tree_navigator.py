"""PageIndex reasoning-based TOC tree navigator and RRF fusion for macro/synthesis queries."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from json_repair import repair_json

from src.multimodal_nav.toc_tree_builder import ReportTocTree
from src.retrieval_filters import build_result_metadata
from src.text_normalization import tokenize_for_bm25

_SYNTHESIS_TASK_TYPES = {
    "summary",
    "generation_summary",
    "generation_risk_synthesis",
    "generation_strategy_synthesis",
    "implicit_intent_qa",
}
_SYNTHESIS_KEYWORDS = (
    "风险",
    "挑战",
    "困难",
    "核心竞争力",
    "竞争力",
    "战略",
    "规划",
    "展望",
    "经营情况",
    "经营分析",
    "主要原因",
    "变化原因",
    "总结",
    "综述",
    "概况",
    "举措",
    "应对",
    "公司治理",
    "治理",
    "供应链",
    "环境",
    "社会",
    "esg",
    "绿色",
    "研发",
)

_SECTION_ONTOLOGY_PRIORS: List[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("风险", "挑战", "不确定", "波动"), ("风险", "可能面对", "管理层讨论")),
    (("战略", "展望", "规划", "数字化", "出海", "竞争力", "经营", "原因"), ("管理层讨论", "核心竞争力", "经营", "业务", "未来发展")),
    (("研发", "技术", "创新", "专利"), ("研发", "核心竞争力", "管理层讨论")),
    (("治理", "董事", "监事", "高管", "薪酬", "内控"), ("公司治理", "董事", "内控")),
    (("环境", "esg", "低碳", "减排", "社会责任", "公益"), ("环境", "社会", "esg")),
    (("分红", "股利", "诉讼", "担保", "关联交易", "重要事项"), ("重要事项", "利润分配", "股利")),
    (("资产", "负债", "收入", "净利润", "现金流", "财务", "毛利率"), ("主要财务", "会计数据", "财务报告")),
]

_TREE_NAV_SYSTEM_PROMPT = (
    "你是年报目录树推理导航器。阅读给定公司年报目录树（包含 node_id、标题、页码区间和摘要），"
    "选出最能直接回答用户问题的 2~3 个章节。\n"
    "严格输出 JSON：{\"selected_nodes\": [{\"node_id\": \"sec-0001\", \"reason\": \"...\"}]}"
)


def should_trigger_tree_navigation(query_plan: Any, question: str) -> bool:
    """Trigger tree navigation on macro, synthesis, risk, strategy, or implicit-intent queries."""
    task_type = str(getattr(query_plan, "task_type", "") or "").lower()
    if task_type in _SYNTHESIS_TASK_TYPES:
        return True
    q_lower = str(question or "").lower()
    return any(tok in q_lower for tok in _SYNTHESIS_KEYWORDS)


class TocTreeNavigator:
    """Navigates hierarchical annual report TOC trees to locate synthesis-relevant page ranges."""

    def select_sections(
        self,
        tree: ReportTocTree,
        question: str,
        *,
        max_sections: int = 3,
        llm_fn: Optional[Callable[[str, str], str]] = None,
    ) -> List[Dict[str, Any]]:
        if not tree.roots:
            return []

        if llm_fn is not None:
            llm_selected = self._select_sections_via_llm(tree, question, max_sections=max_sections, llm_fn=llm_fn)
            if llm_selected:
                return llm_selected

        return self._select_sections_deterministic(tree, question, max_sections=max_sections)

    @staticmethod
    def _select_sections_via_llm(
        tree: ReportTocTree,
        question: str,
        *,
        max_sections: int,
        llm_fn: Callable[[str, str], str],
    ) -> List[Dict[str, Any]]:
        node_map = tree.get_node_map()
        compact_json = tree.to_compact_prompt_json(max_summary_chars=64)
        user_prompt = f"年报目录树：\n{compact_json}\n\n问题：\n{question}\n\n最多选择 {max_sections} 个 node_id。"
        try:
            raw = llm_fn(_TREE_NAV_SYSTEM_PROMPT, user_prompt)
            decoded = json.loads(repair_json(str(raw or "")))
            items = decoded.get("selected_nodes") if isinstance(decoded, dict) else None
            if not isinstance(items, list):
                return []
            selected: List[Dict[str, Any]] = []
            seen = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                nid = str(item.get("node_id") or "").strip()
                if nid not in node_map or nid in seen:
                    continue
                seen.add(nid)
                node, path = node_map[nid]
                selected.append(
                    {
                        "node_id": nid,
                        "level": node.level,
                        "title": node.title,
                        "section_path": path,
                        "start_page": node.start_page,
                        "end_page": node.end_page,
                        "score": round(10.0 - len(selected) * 0.5, 4),
                        "reason": str(item.get("reason") or "LLM目录树推理选中"),
                    }
                )
                if len(selected) >= max_sections:
                    break
            return selected
        except Exception:
            return []

    @staticmethod
    def _select_sections_deterministic(
        tree: ReportTocTree,
        question: str,
        *,
        max_sections: int,
    ) -> List[Dict[str, Any]]:
        q_tokens = set(tokenize_for_bm25(question or ""))
        q_lower = str(question or "").lower()

        active_section_hints: List[str] = []
        for q_cues, sec_cues in _SECTION_ONTOLOGY_PRIORS:
            if any(c in q_lower for c in q_cues):
                active_section_hints.extend(sec_cues)

        scored: List[Dict[str, Any]] = []
        for node, path in tree.iter_nodes():
            title_lower = node.title.lower()
            if any(skip in title_lower for skip in ("释义", "目录", "重要提示", "备查文件")) and "释义" not in q_lower:
                continue
            target_text = f"{path} {node.summary}".lower()
            target_tokens = set(tokenize_for_bm25(target_text))
            overlap = len(q_tokens & target_tokens) if q_tokens else 0
            title_hits = sum(2.0 for tok in q_tokens if len(tok) >= 2 and tok in title_lower)
            prior_hits = sum(2.5 for hint in active_section_hints if hint in title_lower)
            path_prior_hits = sum(1.0 for hint in active_section_hints if hint in path.lower())
            specificity_bonus = 0.6 * (node.level - 1) if (title_hits > 0 or prior_hits > 0) else 0.0
            span_pages = max(1, node.end_page - node.start_page + 1)
            compactness_bonus = 0.4 if 1 <= span_pages <= 25 and (title_hits + prior_hits) > 0 else 0.0

            score = overlap * 1.2 + title_hits + prior_hits + path_prior_hits + specificity_bonus + compactness_bonus
            if score <= 0:
                continue
            scored.append(
                {
                    "node_id": node.node_id,
                    "level": node.level,
                    "title": node.title,
                    "section_path": path,
                    "start_page": node.start_page,
                    "end_page": node.end_page,
                    "score": round(float(score), 4),
                    "reason": "目录标题/摘要与财务章节语义匹配",
                }
            )

        scored.sort(key=lambda item: (item["score"], -item["start_page"]), reverse=True)
        selected: List[Dict[str, Any]] = []
        for item in scored:
            # Avoid selecting a narrow duplicate overlapping child/parent range identically
            is_duplicate = any(
                item["start_page"] == prev["start_page"] and item["end_page"] == prev["end_page"]
                for prev in selected
            )
            if is_duplicate:
                continue
            selected.append(item)
            if len(selected) >= max_sections:
                break
        return selected

    def retrieve_section_pages(
        self,
        tree: ReportTocTree,
        question: str,
        page_fetcher: Callable[[int], Optional[Dict[str, Any]]],
        *,
        doc_metainfo: Dict[str, Any],
        max_sections: int = 2,
        max_pages: int = 4,
        llm_fn: Optional[Callable[[str, str], str]] = None,
    ) -> List[Dict[str, Any]]:
        sections = self.select_sections(tree, question, max_sections=max_sections, llm_fn=llm_fn)
        if not sections:
            return []

        q_tokens = set(tokenize_for_bm25(question or ""))
        candidate_pages: Dict[int, tuple[float, Dict[str, Any], str]] = {}

        for sec_rank, sec in enumerate(sections):
            start_p = int(sec["start_page"])
            end_p = min(int(sec["end_page"]), start_p + 45)
            sec_bonus = float(sec["score"]) * 0.5 + max(0.0, 2.0 - sec_rank * 0.5)
            for pno in range(start_p, end_p + 1):
                if pno in candidate_pages:
                    continue
                page_obj = page_fetcher(pno)
                if not page_obj or not str(page_obj.get("text") or "").strip():
                    continue
                p_text = str(page_obj["text"])
                p_tokens = set(tokenize_for_bm25(p_text[:3000]))
                overlap = len(q_tokens & p_tokens) if q_tokens else 0
                position_bonus = 0.5 if pno == start_p else 0.0
                total_score = sec_bonus + overlap * 1.4 + position_bonus
                candidate_pages[pno] = (total_score, page_obj, sec["section_path"] or sec["title"])

        ranked_pages = sorted(candidate_pages.items(), key=lambda kv: (kv[1][0], -kv[0]), reverse=True)[:max_pages]
        results: List[Dict[str, Any]] = []
        sha1 = str(doc_metainfo.get("sha1_name") or doc_metainfo.get("doc_id") or tree.doc_id)
        base_meta = {**doc_metainfo, "sha1_name": sha1, "doc_id": sha1}

        for pno, (score, page_obj, sec_path) in ranked_pages:
            chunk_stub = {
                "page": pno,
                "chunk_type": "page",
                "node_type": "page",
                "section_name": sec_path.split(" > ")[-1] if sec_path else "",
                "section_title": sec_path.split(" > ")[-1] if sec_path else "",
                "report_section": sec_path,
            }
            meta = build_result_metadata(base_meta, chunk_stub)
            meta["section_path"] = sec_path
            results.append(
                {
                    "distance": round(float(score), 4),
                    "combined_score": round(float(score), 4),
                    "ranking_score": round(float(score), 4),
                    "page": pno,
                    "text": str(page_obj["text"]),
                    "pdf_sha1": sha1,
                    "metadata": meta,
                    "chunk_type": "page",
                    "section_title": meta.get("section_title"),
                    "retrieval_sources": ["toc_tree_nav"],
                    "result_scope": "page",
                }
            )
        return results


def fuse_tree_and_hybrid_results(
    hybrid_results: List[Dict[str, Any]],
    tree_results: List[Dict[str, Any]],
    *,
    top_n: int,
    rrf_k: int = 60,
    tree_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """Fuse traditional hybrid retrieval results and PageIndex TOC tree hits with RRF."""
    if not tree_results:
        return list(hybrid_results or [])[:top_n]
    if not hybrid_results:
        return list(tree_results)[:top_n]

    def _key(item: Dict[str, Any]) -> tuple[str, Any]:
        meta = item.get("metadata") or {}
        sha1 = str(item.get("pdf_sha1") or meta.get("sha1_name") or meta.get("doc_id") or "")
        return (sha1, item.get("page"))

    fused: Dict[tuple[str, Any], Dict[str, Any]] = {}
    scores: Dict[tuple[str, Any], float] = {}

    for rank, item in enumerate(hybrid_results, start=1):
        k = _key(item)
        scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank)
        if k not in fused:
            copy_item = dict(item)
            copy_item["retrieval_sources"] = list(item.get("retrieval_sources") or ["hybrid"])
            fused[k] = copy_item

    for rank, item in enumerate(tree_results, start=1):
        k = _key(item)
        scores[k] = scores.get(k, 0.0) + tree_weight / (rrf_k + rank)
        if k not in fused:
            copy_item = dict(item)
            copy_item["retrieval_sources"] = list(item.get("retrieval_sources") or ["toc_tree_nav"])
            fused[k] = copy_item
        else:
            existing_sources = fused[k].setdefault("retrieval_sources", [])
            for src in item.get("retrieval_sources") or ["toc_tree_nav"]:
                if src not in existing_sources:
                    existing_sources.append(src)
            # Upgrade child text to full page context when TOC tree confirms page importance
            if fused[k].get("result_scope") != "page" and len(str(item.get("text") or "")) > len(str(fused[k].get("text") or "")):
                fused[k]["text"] = item["text"]
                fused[k]["result_scope"] = "page"

    ordered_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)[:top_n]
    output: List[Dict[str, Any]] = []
    for k in ordered_keys:
        entry = fused[k]
        entry["combined_score"] = round(scores[k], 6)
        entry["ranking_score"] = round(scores[k], 6)
        output.append(entry)
    return output
