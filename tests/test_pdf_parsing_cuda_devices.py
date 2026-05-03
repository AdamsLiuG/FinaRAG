import unittest
from pathlib import Path

from src.pdf_parsing import _assign_chunks_to_devices, _parse_cuda_devices


class PdfParsingCudaDevicesTests(unittest.TestCase):
    def test_parse_cuda_devices_accepts_indices_and_cuda_prefixes(self):
        self.assertEqual(_parse_cuda_devices("0,1"), ["0", "1"])
        self.assertEqual(_parse_cuda_devices("cuda:0, cuda:1"), ["0", "1"])
        self.assertEqual(_parse_cuda_devices(None), [])

    def test_assign_chunks_to_devices_round_robins_chunks(self):
        chunks = [
            [Path("a.pdf")],
            [Path("b.pdf")],
            [Path("c.pdf")],
            [Path("d.pdf")],
        ]

        assignments = _assign_chunks_to_devices(chunks, ["0", "1"])

        self.assertEqual(
            assignments,
            [
                ([Path("a.pdf")], "0"),
                ([Path("b.pdf")], "1"),
                ([Path("c.pdf")], "0"),
                ([Path("d.pdf")], "1"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
