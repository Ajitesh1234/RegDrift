import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_extract as ev


def ob(summary, value, clause="Clause 1", parameter="x"):
    return {"summary": summary, "required_value": value, "clause": clause, "parameter": parameter,
            "operator": "==", "unit": ""}


class GoldFile(unittest.TestCase):
    def test_every_circular_has_text_required_rules_and_lowercase_keywords(self):
        gold = ev.load_gold()
        base = os.path.dirname(ev.GOLD)
        ids = set()
        for c in gold["circulars"]:
            self.assertNotIn(c["id"], ids)
            ids.add(c["id"])
            self.assertTrue(os.path.exists(os.path.join(base, c["text"])), c["text"])
            self.assertTrue(any(r.get("required") for r in c["rules"]), c["id"])
            for spec in c["rules"] + c.get("must_not", []):
                for group in spec["all"]:
                    for w in group:
                        self.assertEqual(w, w.lower(), f"{c['id']}/{spec['id']}: keywords must be lower case")

    def test_labelled_facts_really_are_in_the_circular_text(self):
        """A label that cannot be found in its circular is a typo in the label, not a miss by the extractor."""
        gold = ev.load_gold()
        base = os.path.dirname(ev.GOLD)
        for c in gold["circulars"]:
            text = ev._text(c, base).lower()
            for r in c["rules"]:
                if r.get("required"):
                    self.assertTrue(any(w in text for g in r["all"] for w in g), f"{c['id']}/{r['id']}")


class Matching(unittest.TestCase):
    def test_all_groups_must_match_and_any_word_in_a_group_will_do(self):
        spec = {"all": [["repo"], ["rate"]]}
        self.assertTrue(ev.matches(spec, ob("The policy repo rate is 5.5", 5.5)))
        self.assertFalse(ev.matches(spec, ob("The repo is 5.5", 5.5)))

    def test_value_is_read_from_text_numbers_and_commas(self):
        spec = {"all": [["limit"]], "value": [5, 5000000]}
        for v in (5, "5", "USD 5 million", 5000000, "5,000,000"):
            self.assertTrue(ev.matches(spec, ob("single limit", v)), v)
        self.assertFalse(ev.matches(spec, ob("single limit", 100)))
        self.assertFalse(ev.matches(spec, ob("single limit", True)))

    def test_parameter_name_counts_as_text(self):
        self.assertTrue(ev.matches({"all": [["rebook"]]}, ob("Do not allow it", False, parameter="allow_rebook_cancelled")))


class Scoring(unittest.TestCase):
    def setUp(self):
        self.case = next(c for c in ev.load_gold()["circulars"] if c["id"] == "RBI/2026-27/291")

    def good(self):
        return [ob("Do not permit users to rebook a cancelled INR derivative contract.", False),
                ob("Threshold for positions without underlying exposure is USD 5 million.", 5),
                ob("Obtain and retain an undertaking from the user that the exposure is not hedged elsewhere.", True),
                ob("Retain the documents for not less than two years.", 2)]

    def test_complete_and_clean_output_scores_full_recall_without_violations(self):
        s = ev.score_circular(self.case, self.good())
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["violations"], [])
        self.assertEqual(s["missed"], [])

    def test_missing_rule_is_reported(self):
        s = ev.score_circular(self.case, self.good()[:2])
        self.assertEqual(sorted(s["missed"]), ["retain_two_years", "undertaking"])
        self.assertAlmostEqual(s["recall"], 0.5)

    def test_the_old_threshold_is_a_violation_not_a_hit(self):
        wrong = self.good()[:1] + [ob("Positions without underlying exposure are limited to USD 100 million.", 100)]
        s = ev.score_circular(self.case, wrong)
        self.assertIn("threshold_5m", s["missed"])
        self.assertEqual([v["id"] for v in s["violations"]], ["old_threshold_100m"])

    def test_commencement_boilerplate_is_a_violation(self):
        s = ev.score_circular(self.case, self.good() + [ob("These Directions shall come into force with immediate effect.", True)])
        self.assertEqual([v["id"] for v in s["violations"]], ["commencement"])

    def test_optional_rules_do_not_change_recall_and_unlabelled_rules_are_listed(self):
        s = ev.score_circular(self.case, self.good() + [ob("AD may continue to allow rollover on maturity.", True),
                                                        ob("Banks must display the rate card.", True)])
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["optional_hit"], ["rollover"])
        self.assertEqual(s["extras"], ["Banks must display the rate card."])

    def test_old_repo_rate_is_caught(self):
        case = next(c for c in ev.load_gold()["circulars"] if c["id"] == "RBI/2026-27/282")
        s = ev.score_circular(case, [ob("Policy repo rate is 5.25 per cent.", 5.25)])
        self.assertEqual([v["id"] for v in s["violations"]], ["old_repo_rate"])
        self.assertIn("repo_rate", s["missed"])


class Repeats(unittest.TestCase):
    def test_hit_rate_shows_which_rule_the_extractor_finds_only_sometimes(self):
        case = next(c for c in ev.load_gold()["circulars"] if c["id"] == "RBI/2026-27/257")
        hit = ev.score_circular(case, [ob("Facility extended to Foreign Portfolio Investors (FPIs).", True)])
        miss = ev.score_circular(case, [])
        s = ev.summarise([{case["id"]: hit}, {case["id"]: miss}, {case["id"]: hit}, {case["id"]: miss}])
        self.assertAlmostEqual(s["recall"], 0.5)
        self.assertAlmostEqual(s["rule_hit_rate"][f"{case['id']} / fpi_extension"], 0.5)


if __name__ == "__main__":
    unittest.main()
