"""Exact feasible overlaps from the manuscript's marginal contact counts.

No actual pairings, confidence intervals, or p-values are reconstructed.
"""
import argparse
import json
import re
import unittest
from pathlib import Path


def feasible_ranges(n, a, b):
    if not all(isinstance(x, int) for x in (n, a, b)) or not (0 <= a <= n and 0 <= b <= n):
        raise ValueError("Counts must be integers between zero and n")
    lo, hi = max(0, a + b - n), min(a, b)
    return {"both_contact_range": [lo, hi],
            "first_only_range": [a - hi, a - lo],
            "second_only_range": [b - hi, b - lo],
            "discordant_range": [a + b - 2 * hi, a + b - 2 * lo],
            "discordant_step": 2, "net_difference": a - b}


def parse_table(path):
    text = Path(path).read_text()
    if "100 per cell" not in text:
        raise ValueError("Unexpected per-cell population; inspect the table")
    rows = []
    for line in text.splitlines():
        if not line.startswith(("nuPlan &", "nuScenes &")):
            continue
        cells = line.split("&")
        if len(cells) != 10:
            raise ValueError("Unexpected contact table columns")
        counts = [int(re.sub(r"[^0-9]", "", cell)) for cell in cells[2:]]
        for metric, offset in (("Ego CR", 0), ("Initial-target contact", 4)):
            a, b = counts[offset:offset + 2]
            row = {"dataset": cells[0].strip(), "planner": cells[1].strip(),
                   "metric": metric, "n": 100, "CrashDojo": a, "SAFE-SIM": b}
            row.update(feasible_ranges(100, a, b))
            rows.append(row)
    if len(rows) != 20:
        raise ValueError("Expected ten planner/dataset cells for each of two metrics")
    return rows


class RangeTests(unittest.TestCase):
    def test_exhaustive_small_populations(self):
        for n in range(8):
            observed = {}
            for first in range(1 << n):
                for second in range(1 << n):
                    a, b = bin(first).count("1"), bin(second).count("1")
                    c = bin(first & second).count("1")
                    observed.setdefault((a, b), set()).add(c)
            for (a, b), possible in observed.items():
                result = feasible_ranges(n, a, b)
                lo, hi = result["both_contact_range"]
                self.assertEqual(possible, set(range(lo, hi + 1)))
                actual = {a + b - 2 * c for c in possible}
                self.assertEqual(result["discordant_range"], [min(actual), max(actual)])
                self.assertEqual(result["net_difference"], a - b)

    def test_invalid_counts(self):
        for args in ((5, 6, 1), (5, -1, 2), (5, 1, 2.0)):
            with self.assertRaises(ValueError):
                feasible_ranges(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path(__file__).resolve().parents[1] / "figures/tab_per_planner.tex")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RangeTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    result = {"source": "Per-planner marginal contact counts", "rows": parse_table(args.table),
              "interpretation": "Feasible overlap and discordance counts; not observed scene pairings"}
    payload = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
