import os, sys, tempfile, unittest
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for k in ("GROQ_API_KEY", "LORA_ADAPTER", "JIRA_URL"):
    os.environ.pop(k, None)
import bench
import regdrift


def row(expected, got, cat="threshold", conf="high"):
    return {"id": "x", "category": cat, "confidence": conf, "clause": "c", "bot": "b", "expected": expected, "got": got, "reason": ""}


class Dataset(unittest.TestCase):
    def test_cases_are_consistent_with_the_bot_pool(self):
        cases, pool = bench.load_cases(), bench.load_pool()
        names = {b["name"] for b in pool}
        self.assertGreaterEqual(len(cases), 100)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        for c in cases:
            self.assertIn(c["bot"], names, c["id"])
            self.assertIn(c["expected"], bench.VERDICTS, c["id"])
            self.assertIn(c["label_confidence"], ("high", "review"), c["id"])
            self.assertIn(c["source"], ("real", "synthetic"), c["id"])
            self.assertGreater(len(c["clause"]), 20, c["id"])

    def test_every_verdict_and_the_hard_categories_are_present(self):
        cases = bench.load_cases()
        self.assertEqual({c["expected"] for c in cases}, set(bench.VERDICTS))
        cats = {c["category"] for c in cases}
        for need in ("boundary", "yes_no", "permissive", "list", "old_new", "unit_trap", "not_applicable", "lookalike",
                     "missing_info", "contradiction", "exception", "ambiguous", "paraphrase"):
            self.assertIn(need, cats)

    def test_pool_contains_the_demo_bots_and_unique_ids(self):
        pool = bench.load_pool()
        self.assertEqual(len({b["id"] for b in pool}), len(pool))
        self.assertTrue({"STR-Filer-Bot", "KYC-Document-Verification-Bot"} <= {b["name"] for b in pool})


class Stats(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = bench.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.490, places=2)
        self.assertAlmostEqual(hi, 0.943, places=2)
        self.assertEqual(bench.wilson(0, 0), (0.0, 1.0))
        lo, hi = bench.wilson(10, 10)
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)

    def test_metrics_for_known_counts(self):
        rows = ([row("violating", "violating")] * 6 + [row("violating", "compliant")] * 2 +
                [row("compliant", "violating")] * 1 + [row("compliant", "compliant")] * 3 + [row("not_applicable", "violating")])
        m = bench.metrics(rows)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (6, 2, 2, 3))
        self.assertAlmostEqual(m["precision"], 0.75)
        self.assertAlmostEqual(m["recall"], 0.75)
        self.assertAlmostEqual(m["f1"], 0.75)
        self.assertAlmostEqual(m["fpr"], 2 / 5)
        self.assertEqual(m["false_high_on_clean"], 2)    # a High on a compliant case and on a not-applicable case

    def test_bootstrap_is_reproducible_and_brackets_the_estimate(self):
        pairs = [("violating", "violating")] * 12 + [("violating", "compliant")] * 3 + [("compliant", "violating")] * 2 + [("compliant", "compliant")] * 10
        a, b = bench.bootstrap_f1(pairs), bench.bootstrap_f1(pairs)
        self.assertEqual(a, b)
        f1 = bench.prf(12, 2, 3)[2]
        self.assertLessEqual(a[0], f1)
        self.assertGreaterEqual(a[1], f1)


class SimpleRules(unittest.TestCase):
    def bot(self, name):
        return next(b for b in bench.load_pool() if b["name"] == name)

    def test_threshold_boundary_and_breach(self):
        bot = self.bot("STR-Filer-Bot")
        self.assertEqual(bench.simple_rules("Suspicious transaction reports shall be filed within 7 days.", bot)["status"], "compliant")
        self.assertEqual(bench.simple_rules("Suspicious transaction reports shall be filed within 5 days.", bot)["status"], "violating")

    def test_spelled_numbers_and_once_in_n_years(self):
        bot = self.bot("KYC-Refresh-Bot")
        got = bench.simple_rules("KYC shall be updated at least once in two years for high risk customers.", bot)
        self.assertEqual(got["status"], "violating")      # 3 years is longer than the 2-year maximum interval

    def test_prohibition_and_permission_and_list(self):
        self.assertEqual(bench.simple_rules("An Authorised Dealer shall not permit users to rebook a cancelled contract.",
                                            self.bot("FX-Derivative-Booking-Bot"))["status"], "violating")
        self.assertEqual(bench.simple_rules("The bank may alternatively obtain the certified copy for Foreign Portfolio Investors (FPIs).",
                                            self.bot("KYC-Document-Verification-Bot"))["status"], "optional_gap")

    def test_unrelated_bot_is_not_applicable(self):
        self.assertEqual(bench.simple_rules("Suspicious transaction reports shall be filed within 5 days.",
                                            self.bot("Card-Limit-Bot"))["status"], "not_applicable")

    def test_it_cannot_read_unit_changes_by_design(self):
        got = bench.simple_rules("An unsecured credit limit on a credit card shall not exceed Rs 3,00,000.", self.bot("Card-Limit-Bot"))
        self.assertEqual(got["status"], "compliant")      # reads 300000 against a lakh setting: the weakness the LLM arms should beat


class Cache(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "c.jsonl")

    def test_a_miss_without_live_returns_none_and_flags_failure(self):
        c = bench.CachedLLM(self.path, live=False)
        self.assertIsNone(c("hello"))
        self.assertTrue(c.failed)

    def test_live_answers_are_stored_and_reused_across_instances(self):
        calls = []
        c = bench.CachedLLM(self.path, live=True)
        c.real = lambda p: calls.append(p) or {"status": "compliant"}
        self.assertEqual(c("q"), {"status": "compliant"})
        self.assertEqual(c("q"), {"status": "compliant"})
        self.assertEqual(len(calls), 1)
        again = bench.CachedLLM(self.path, live=False)
        self.assertEqual(again("q"), {"status": "compliant"})
        self.assertFalse(again.failed)

    def test_a_failed_live_call_is_not_cached(self):
        c = bench.CachedLLM(self.path, live=True)
        c.real = lambda p: None
        self.assertIsNone(c("q"))
        self.assertFalse(os.path.exists(self.path))


class Runner(unittest.TestCase):
    def setUp(self):
        self.orig = regdrift.llm_json
        self.cases = bench.load_cases()[:4]
        self.pool = bench.load_pool()
        self.path = os.path.join(tempfile.mkdtemp(), "c.jsonl")

    def tearDown(self):
        regdrift.llm_json = self.orig

    def test_llm_arm_scores_with_a_stub_and_leaves_regdrift_restored(self):
        llm = bench.CachedLLM(self.path, live=True)
        llm.real = lambda p: {"status": "violating", "reason": "stub"}
        res = bench.run_all(self.cases, self.pool, ["llm", "rules"], llm)
        self.assertEqual([r["got"] for r in res["llm"]], ["violating"] * 4)
        self.assertIs(regdrift.llm_json, self.orig)
        m = bench.summarise(res)["llm"]
        self.assertEqual(m["n"], 4)

    def test_unanswered_cases_are_left_out_not_counted_wrong(self):
        llm = bench.CachedLLM(self.path, live=False)
        res = bench.run_all(self.cases, self.pool, ["llm"], llm)
        m = bench.summarise(res)["llm"]
        self.assertEqual((m["n"], m["unanswered"]), (0, 4))

    def test_malformed_llm_answer_is_unanswered(self):
        llm = bench.CachedLLM(self.path, live=True)
        llm.real = lambda p: {"status": "maybe"}
        res = bench.run_all(self.cases, self.pool, ["llm"], llm)
        self.assertTrue(all(r["got"] is None for r in res["llm"]))


class Reports(unittest.TestCase):
    def results(self):
        cases = bench.load_cases()[:6]
        out = {a: [] for a in ("rules", "full")}
        for i, c in enumerate(cases):
            for a in out:
                got = c["expected"] if (i + (a == "full")) % 3 else "violating"
                out[a].append({"id": c["id"], "category": c["category"], "confidence": c["label_confidence"], "clause": c["clause"],
                               "bot": c["bot"], "expected": c["expected"], "got": got, "reason": "why <b>"})
        return out, cases

    def test_markdown_chart_and_html_render(self):
        res, cases = self.results()
        md = bench.markdown(res, len(cases), "note")
        self.assertIn("Simple rules", md)
        self.assertIn("False-positive rate", md)
        ET.fromstring(bench.chart_svg(res))
        page = bench.cases_html(res, cases)
        self.assertIn("&lt;b&gt;", page)                  # reasoning text is escaped, never injected
        self.assertIn(cases[0]["id"], page)

    def test_chart_with_no_answers_still_renders(self):
        ET.fromstring(bench.chart_svg({"llm": [{"id": "B001", "category": "x", "confidence": "high", "clause": "c", "bot": "b",
                                                 "expected": "compliant", "got": None, "reason": ""}]}))


if __name__ == "__main__":
    unittest.main()
