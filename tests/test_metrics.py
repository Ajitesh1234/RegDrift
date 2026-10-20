import datetime as dt, os, sys, unittest
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import make_metrics as mm

D = dt.date


class Weekly(unittest.TestCase):
    def test_commits_are_bucketed_by_monday_and_old_ones_are_dropped(self):
        today = D(2026, 10, 21)                                   # a Wednesday; its week starts Mon 19 Oct
        dates = [D(2026, 10, 19), D(2026, 10, 21), D(2026, 10, 14), D(2026, 9, 1), D(2020, 1, 1)]
        got = dict(mm.weekly_counts(dates, today, weeks=4))
        self.assertEqual(got[D(2026, 10, 19)], 2)
        self.assertEqual(got[D(2026, 10, 12)], 1)
        self.assertEqual(len(got), 4)
        self.assertEqual(sum(got.values()), 3)                    # 1 Sep and 2020 fall outside four weeks

    def test_weeks_are_oldest_first_and_include_empty_weeks(self):
        got = mm.weekly_counts([], D(2026, 10, 21), weeks=3)
        self.assertEqual([w for w, _ in got], [D(2026, 10, 5), D(2026, 10, 12), D(2026, 10, 19)])
        self.assertTrue(all(c == 0 for _, c in got))


class Svg(unittest.TestCase):
    def test_chart_is_well_formed_svg_with_the_numbers_in_it(self):
        weekly = mm.weekly_counts([D(2026, 10, 20)] * 5 + [D(2026, 10, 13)], D(2026, 10, 21))
        svg = mm.render(weekly, 1234, 126, 4567, D(2026, 10, 21))
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        for text in ("1,234", "126", "4,567", "REPO PULSE"):
            self.assertIn(text, svg)

    def test_an_empty_history_still_draws(self):
        svg = mm.render(mm.weekly_counts([], D(2026, 10, 21)), 0, 0, 0, D(2026, 10, 21))
        ET.fromstring(svg)


if __name__ == "__main__":
    unittest.main()
