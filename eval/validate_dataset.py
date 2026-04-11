from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.dataset_schema import (
    export_json_schemas,
    load_gold_answer_set,
    load_manifest,
    load_question_set,
    validate_dataset_alignment,
)


def _validate_prediction_bundle(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "answers" in payload:
        answers = payload["answers"]
    elif isinstance(payload, list):
        answers = payload
    else:
        return {
            "valid": False,
            "errors": ["Prediction bundle must be a list or a dict containing 'answers'."],
            "warnings": [],
        }

    errors = []
    warnings = []
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            errors.append(f"Prediction at index {index} is not a JSON object.")
            continue
        if not answer.get("question_text"):
            errors.append(f"Prediction at index {index} is missing question_text.")
        if "value" not in answer and "final_answer" not in answer:
            warnings.append(f"Prediction at index {index} has neither value nor final_answer.")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(description="Validate finance evaluation datasets and optional prediction bundles.")
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--gold-answers-file", type=Path, default=None)
    parser.add_argument("--manifest-file", type=Path, default=None)
    parser.add_argument("--pred-answers-file", type=Path, default=None)
    parser.add_argument("--dump-schema-dir", type=Path, default=None)
    args = parser.parse_args()

    report: Dict[str, Any] = {}

    if args.questions_file is not None:
        question_set = load_question_set(args.questions_file)
        report["questions"] = {
            "valid": True,
            "question_count": len(question_set.questions),
        }
    else:
        question_set = None

    if args.gold_answers_file is not None:
        answer_set = load_gold_answer_set(args.gold_answers_file)
        report["gold_answers"] = {
            "valid": True,
            "answer_count": len(answer_set.answers),
        }
    else:
        answer_set = None

    if question_set is not None and answer_set is not None:
        report["alignment"] = validate_dataset_alignment(question_set, answer_set)

    if args.manifest_file is not None:
        manifest = load_manifest(args.manifest_file)
        report["manifest"] = {
            "valid": True,
            "dataset_name": manifest.dataset_name,
            "question_count": manifest.question_count,
            "answer_count": manifest.answer_count,
        }

    if args.pred_answers_file is not None:
        report["predictions"] = _validate_prediction_bundle(_load_json(args.pred_answers_file))

    if args.dump_schema_dir is not None:
        args.dump_schema_dir.mkdir(parents=True, exist_ok=True)
        for name, schema in export_json_schemas().items():
            target = args.dump_schema_dir / f"{name}.schema.json"
            with open(target, "w", encoding="utf-8") as file:
                json.dump(schema, file, ensure_ascii=False, indent=2)
        report["schema_dump_dir"] = str(args.dump_schema_dir)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
