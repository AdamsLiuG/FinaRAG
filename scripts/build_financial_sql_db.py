"""Build read-only SQLite financial fact database from FinaRAG chunked reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.structured.fin_table_extractor import FinancialTableExtractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite financial fact lake from chunked reports.")
    parser.add_argument(
        "--chunked-reports-dir",
        type=Path,
        default=REPO_ROOT / "data/top10_industries_2024_20each/databases_ser_tab/chunked_reports",
        help="Directory containing chunked annual report JSON files.",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=REPO_ROOT / "data/top10_industries_2024_20each/databases_ser_tab/finarag_facts.sqlite",
        help="Destination SQLite file path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on reports to ingest.")
    args = parser.parse_args()

    extractor = FinancialTableExtractor()
    stats = extractor.build_sqlite_db(
        chunked_reports_dir=args.chunked_reports_dir,
        db_path=args.output_db,
        limit=args.limit,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
