import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_judge


class GoldDecisionSet(unittest.TestCase):
    """Hand-labelled (rule, bot setting) cases. Any change that flips a verdict fails here."""

    def test_every_case_matches_its_label(self):
        res = eval_judge.score()
        self.assertGreaterEqual(res["cases"], 30)  # guard against the gold set being emptied
        self.assertEqual([(m["id"], m["expect"], m["got"]) for m in res["misses"]], [])

    def test_yes_no_and_number_are_never_mixed(self):
        res = {r["id"]: r for r in eval_judge.score()["rows"]}
        self.assertEqual(res["mismatch-bool-numeric"]["got"], "at_risk")
        self.assertEqual(res["mismatch-numeric-bool"]["got"], "at_risk")


if __name__ == "__main__":
    unittest.main()
