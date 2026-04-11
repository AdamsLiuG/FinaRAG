from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.metrics import compare_ranked_retrieval, load_answers_bundle


def _default_reference_answers(dataset_dir: Path) -> Path | None:
    candidates = [
        dataset_dir / "answers_max_nst_o3m.json",
        dataset_dir / "answers_1st_place_o3-mini.json",
        dataset_dir / "answers_1st_place_llama_70b.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_debug_bundle(answers_file: Path) -> Path | None:
    debug_candidate = answers_file.with_name(answers_file.stem + "_debug" + answers_file.suffix)
    return debug_candidate if debug_candidate.exists() else None


def evaluate_ranked_retrieval(
    answers_file: Path,
    *,
    reference_answers: Path,
    debug_file: Path,
    recall_k: int = 10,
    precision_k: int = 3,
) -> Dict:
    pred_answers, _ = load_answers_bundle(answers_file)
    ref_answers, _ = load_answers_bundle(reference_answers)
    _, debug_payload = load_answers_bundle(debug_file)

    summary = compare_ranked_retrieval(
        pred_answers,
        ref_answers,
        debug_payload=debug_payload,
        recall_k=recall_k,
        precision_k=precision_k,
    )
    return {
        "answers_file": str(answers_file),
        "reference_answers": str(reference_answers),
        "debug_file": str(debug_file),
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate page-level ranked retrieval metrics such as Recall@10 and Precision@3 from FinaRAG debug bundles."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/test_set"))
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--reference-answers", type=Path, default=None)
    parser.add_argument("--debug-file", type=Path, default=None)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--precision-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reference_answers = args.reference_answers or _default_reference_answers(args.dataset_dir)
    if reference_answers is None or not reference_answers.exists():
        raise ValueError("A reference answers file is required to evaluate ranked retrieval metrics.")

    debug_file = args.debug_file or _default_debug_bundle(args.answers_file)
    if debug_file is None or not debug_file.exists():
        raise ValueError("A debug answers bundle is required to evaluate ranked retrieval metrics.")

    report = evaluate_ranked_retrieval(
        args.answers_file,
        reference_answers=reference_answers,
        debug_file=debug_file,
        recall_k=args.recall_k,
        precision_k=args.precision_k,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
