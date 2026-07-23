import unittest
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def _leaderboard_tables():
    lines = README.read_text(encoding="utf-8").splitlines()
    tables = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("| Model | Overall |"):
            table = []
            while index < len(lines) and lines[index].startswith("|"):
                table.append(lines[index])
                index += 1
            tables.append(table)
        index += 1
    return tables


class ReadmeLeaderboardTests(unittest.TestCase):
    def test_validation_and_test_tables_have_antifraud_column(self):
        tables = _leaderboard_tables()
        self.assertEqual(len(tables), 2)
        for table in tables:
            self.assertEqual(
                table[0],
                (
                    "| Model | Overall | img→text | img→markdown | Grounding "
                    "| KIE (JSON) | VQA | Anti-fraud |"
                ),
            )

    def test_overall_remains_mean_of_original_five_categories(self):
        for table in _leaderboard_tables():
            seen_models = set()
            for row in table[2:]:
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                model = cells[0]
                self.assertNotIn(model, seen_models)
                seen_models.add(model)
                overall = float(cells[1])
                components = [float(value) for value in cells[2:7]]
                self.assertLessEqual(
                    abs(overall - sum(components) / 5),
                    0.0011,
                    msg=model,
                )

    def test_tables_have_no_bold_rows_or_failure_footnotes(self):
        for table in _leaderboard_tables():
            joined = "\n".join(table)
            self.assertNotIn("**", joined)
            self.assertNotRegex(joined, r"responses failed")

    def test_tables_have_no_missing_scores(self):
        for table in _leaderboard_tables():
            self.assertNotIn("—", "\n".join(table))
            for row in table[2:]:
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                self.assertEqual(len(cells), 8)
                float(cells[-1])


if __name__ == "__main__":
    unittest.main()
