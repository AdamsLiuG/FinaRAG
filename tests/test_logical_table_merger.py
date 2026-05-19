import unittest

from src.logical_table_merger import LogicalTableMerger


class LogicalTableMergerTests(unittest.TestCase):
    def test_link_tables_confirms_adjacent_continuation_tables(self):
        tables = [
            {
                "table_id": "tbl-head",
                "page": 10,
                "bbox": [40, 120, 560, 780],
                "nrows": 4,
                "ncols": 2,
                "markdown": "主要会计数据 单位：人民币百万元",
                "section_title": "主要会计数据和财务指标",
                "section_name": "主要会计数据和财务指标",
                "report_section": "主要会计数据和财务指标",
                "unit_hint": "人民币百万元",
                "col_headers_by_col": {1: ["2024年"]},
                "row_headers_by_row": {1: ["营业收入"]},
                "cell_records": [
                    {
                        "table_id": "tbl-head",
                        "page": 10,
                        "row_idx": 1,
                        "col_idx": 1,
                        "raw_value": "3000",
                        "matched_row_headers": ["营业成本"],
                        "matched_col_headers": ["2024年"],
                        "unit_hint": "人民币百万元",
                        "period": "2024年报",
                    }
                ],
            },
            {
                "table_id": "tbl-tail",
                "page": 11,
                "bbox": [40, 40, 560, 360],
                "nrows": 3,
                "ncols": 2,
                "markdown": "主要会计数据（续表）",
                "section_title": "主要会计数据和财务指标",
                "section_name": "主要会计数据和财务指标",
                "report_section": "主要会计数据和财务指标",
                "unit_hint": None,
                "col_headers_by_col": {},
                "row_headers_by_row": {1: ["营业收入"]},
                "cell_records": [
                    {
                        "table_id": "tbl-tail",
                        "page": 11,
                        "row_idx": 1,
                        "col_idx": 1,
                        "raw_value": "4000",
                        "matched_row_headers": ["营业收入"],
                        "matched_col_headers": [],
                        "unit_hint": None,
                        "period": None,
                    }
                ],
            },
        ]

        enriched_tables, logical_tables = LogicalTableMerger().link_tables(tables)

        self.assertEqual(len(logical_tables), 1)
        self.assertEqual(enriched_tables[0]["merge_state"], "confirmed")
        self.assertEqual(enriched_tables[1]["merge_state"], "confirmed")
        self.assertEqual(enriched_tables[0]["logical_role"], "head")
        self.assertEqual(enriched_tables[1]["logical_role"], "tail")
        self.assertEqual(enriched_tables[0]["logical_table_id"], enriched_tables[1]["logical_table_id"])
        self.assertEqual(enriched_tables[1]["continuation_of"], "tbl-head")
        self.assertEqual(enriched_tables[0]["page_span"], [10, 11])

    def test_materialize_logical_table_inherits_headers_units_and_provenance(self):
        tables = [
            {
                "table_id": "tbl-head",
                "page": 10,
                "bbox": [40, 120, 560, 780],
                "nrows": 4,
                "ncols": 2,
                "markdown": "主要会计数据 单位：人民币百万元",
                "section_title": "主要会计数据和财务指标",
                "section_name": "主要会计数据和财务指标",
                "report_section": "主要会计数据和财务指标",
                "unit_hint": "人民币百万元",
                "period": "2024年报",
                "col_headers_by_col": {1: ["2024年"]},
                "row_headers_by_row": {1: ["营业成本"]},
                "cell_records": [
                    {
                        "table_id": "tbl-head",
                        "page": 10,
                        "row_idx": 1,
                        "col_idx": 1,
                        "raw_value": "3000",
                        "matched_row_headers": ["营业成本"],
                        "matched_col_headers": ["2024年"],
                        "unit_hint": "人民币百万元",
                        "period": "2024年报",
                    }
                ],
            },
            {
                "table_id": "tbl-tail",
                "page": 11,
                "bbox": [40, 40, 560, 360],
                "nrows": 3,
                "ncols": 2,
                "markdown": "主要会计数据（续表）",
                "section_title": "主要会计数据和财务指标",
                "section_name": "主要会计数据和财务指标",
                "report_section": "主要会计数据和财务指标",
                "unit_hint": None,
                "period": None,
                "col_headers_by_col": {},
                "row_headers_by_row": {1: ["营业收入"]},
                "cell_records": [
                    {
                        "table_id": "tbl-tail",
                        "page": 11,
                        "row_idx": 1,
                        "col_idx": 1,
                        "raw_value": "4000",
                        "matched_row_headers": ["营业收入"],
                        "matched_col_headers": [],
                        "unit_hint": None,
                        "period": None,
                    }
                ],
            },
        ]

        merger = LogicalTableMerger()
        enriched_tables, logical_tables = merger.link_tables(tables)
        materialized = merger.materialize_logical_table(
            enriched_tables,
            logical_tables,
            enriched_tables[0]["logical_table_id"],
        )

        self.assertIsNotNone(materialized)
        self.assertEqual(materialized["logical_table_id"], enriched_tables[0]["logical_table_id"])
        self.assertEqual(materialized["page_span"], [10, 11])
        tail_cell = next(cell for cell in materialized["cell_records"] if cell["source_table_id"] == "tbl-tail")
        self.assertEqual(tail_cell["matched_col_headers"], ["2024年"])
        self.assertEqual(tail_cell["unit_hint"], "人民币百万元")
        self.assertEqual(tail_cell["period"], "2024年报")
        self.assertEqual(tail_cell["source_page"], 11)
        self.assertEqual(tail_cell["source_row_idx"], 1)
        self.assertEqual(tail_cell["source_col_idx"], 1)


if __name__ == "__main__":
    unittest.main()
