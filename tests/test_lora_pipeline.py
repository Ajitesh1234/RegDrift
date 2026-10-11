import json, os, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for k in ("GROQ_API_KEY", "LORA_ADAPTER", "JIRA_URL"): os.environ.pop(k, None)
import regdrift, distill, review_candidates, build_lora_data, evaluate, train_lora

OB = {"parameter": "str_filing_days", "operator": "<=", "required_value": 7, "unit": "days", "modality": "mandatory"}


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")


class Seeds(unittest.TestCase):
    def test_every_seed_is_valid_and_visible_to_the_inference_filter(self):
        with open(os.path.join(ROOT, "data", "seed_examples.jsonl")) as fh:
            rows = [json.loads(l) for l in fh]
        self.assertGreaterEqual(len(rows), 20)
        for r in rows:
            [regdrift.canon_obligation(o) for o in r["obligations"]]
            self.assertTrue(len(r["clause_text"]) >= 40 and regdrift.OBLIGATION_HINT.search(r["clause_text"]), r["clause_text"])
        self.assertTrue(any(not r["obligations"] for r in rows), "needs negative examples")
        self.assertTrue({o["operator"] for r in rows for o in r["obligations"]} == {"<=", ">=", "==", "contains"})


class Distill(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        with open(os.path.join(self.d, "c1.txt"), "w") as fh:
            fh.write("RBI/2026-27/1\n1. Suspicious transaction reports shall be filed within 7 days of the conclusion.\n"
                     "2. These directions shall come into force with immediate effect for all entities.\n")
        self.out = os.path.join(self.d, "cand.jsonl")
        self.orig = regdrift.llm_extract_clause

    def tearDown(self):
        regdrift.llm_extract_clause = self.orig

    def test_proposes_then_resumes_without_repeating(self):
        calls = []
        def fake(body):
            calls.append(body)
            return [{"parameter": "str_filing_days", "operator": "<=", "required_value": "7", "unit": "days"}] if "Suspicious" in body else []
        regdrift.llm_extract_clause = fake
        self.assertEqual(distill.propose(self.d, self.out, 0, sleep=lambda s: None), 0)
        rows = distill.load_rows(self.out)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["proposed"], [OB]) ; self.assertEqual(rows[1]["proposed"], [])
        self.assertTrue(all(r["status"] == "pending" and r["final"] is None for r in rows))
        n = len(calls)
        distill.propose(self.d, self.out, 0, sleep=lambda s: None)
        self.assertEqual(len(calls), n)                       # resumable: nothing re-asked

    def test_llm_outage_stops_cleanly_and_keeps_progress(self):
        seq = iter([[], None])
        regdrift.llm_extract_clause = lambda body: next(seq)
        self.assertEqual(distill.propose(self.d, self.out, 0, sleep=lambda s: None), 1)
        self.assertEqual(len(distill.load_rows(self.out)), 1)


class Review(unittest.TestCase):
    def row(self): return {"id": "x", "circular": "c", "clause": "Clause 1", "clause_text": "t", "proposed": [OB], "status": "pending", "final": None}

    def test_actions(self):
        r = review_candidates.apply(self.row(), "a"); self.assertEqual((r["status"], r["final"]), ("accepted", [OB]))
        r = review_candidates.apply(self.row(), "n"); self.assertEqual((r["status"], r["final"]), ("no_rule", []))
        r = review_candidates.apply(self.row(), "e", json.dumps({"parameter": "x", "operator": "<=", "required_value": "5.0"}))
        self.assertEqual(r["final"][0]["required_value"], 5)
        with self.assertRaises(ValueError): review_candidates.apply(self.row(), "e", json.dumps([{"parameter": "x", "operator": "DROP"}]))

    def test_interactive_loop_saves_and_quits(self):
        p = os.path.join(tempfile.mkdtemp(), "c.jsonl")
        rows = [dict(self.row(), id=str(i)) for i in range(3)]
        write_jsonl(p, rows)
        answers = iter(["a", "bogus", "n", "q"])           # accept, invalid key (re-asked), no-rule, quit
        review_candidates.review(p, ask=lambda prompt: next(answers))
        st = [r["status"] for r in distill.load_rows(p)]
        self.assertEqual(st, ["accepted", "no_rule", "pending"])


class Build(unittest.TestCase):
    def make(self, n_circulars):
        d = tempfile.mkdtemp()
        write_jsonl(os.path.join(d, "seed_examples.jsonl"), [{"clause_text": "Seed clause one shall be filed within 7 days of the event.", "obligations": [OB]}])
        rows = []
        for c in range(n_circulars):
            for k in range(3):
                rows.append({"id": f"{c}{k}", "circular": f"c{c}.pdf", "clause": "Clause 1", "clause_text": f"Circular {c} clause {k} shall be done within {k + 1} days of the event.",
                             "proposed": [], "status": "accepted", "final": [dict(OB, required_value=k + 1)]})
        rows.append({"id": "pend", "circular": "c0.pdf", "clause": "Clause 9", "clause_text": "Pending clause that shall never reach training data at all.",
                     "proposed": [OB], "status": "pending", "final": None})
        write_jsonl(os.path.join(d, "candidates.jsonl"), rows)
        return d

    def test_holds_out_whole_circulars_without_leakage(self):
        d = self.make(8)
        r = build_lora_data.build(d)
        self.assertEqual(len(r["test_circulars"]), 2)
        train = distill.load_rows(os.path.join(d, "train.jsonl")); test = distill.load_rows(os.path.join(d, "test.jsonl"))
        self.assertFalse({t["circular"] for t in train} & set(r["test_circulars"]))
        test_prompts = {regdrift.LORA_PROMPT.format(clause=t["clause_text"]) for t in test}
        self.assertFalse({t["prompt"] for t in train} & test_prompts)                 # no test clause appears in training
        self.assertTrue(all("Pending clause" not in t["text"] for t in train))        # unreviewed never trains
        self.assertTrue(any(t["circular"] == "seed" for t in train))                  # seeds are train-only
        self.assertEqual(len(test), 6)

    def test_too_few_circulars_means_no_test_set_not_a_fake_one(self):
        r = build_lora_data.build(self.make(3))
        self.assertEqual((r["test_circulars"], r["test_rows"]), ([], 0))

    def test_training_text_is_prompt_plus_valid_json_completion(self):
        d = self.make(5); build_lora_data.build(d)
        for t in distill.load_rows(os.path.join(d, "train.jsonl")):
            self.assertEqual(t["text"], t["prompt"] + t["completion"])
            self.assertTrue(t["prompt"].endswith("### Output:\n"))
            self.assertIsInstance(json.loads(t["completion"]), list)

    def test_split_is_deterministic(self):
        self.assertEqual(build_lora_data.choose_test_circulars([f"c{i}" for i in range(8)]),
                         build_lora_data.choose_test_circulars([f"c{i}" for i in reversed(range(8))]))


class Scoring(unittest.TestCase):
    def test_metrics(self):
        g = [[OB], [OB], [], []]
        p = [[OB], [dict(OB, parameter="other_name")], [OB], []]     # hit, right rule/wrong name, false alarm, correct negative
        m = evaluate.score(p, g)
        self.assertEqual((m["n"], m["positives"]), (4, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3); self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["exact_match"], 2 / 4)       # clause 1 and clause 4
        self.assertAlmostEqual(m["content_match"], 3 / 4)     # name ignored in clause 2
        self.assertEqual(evaluate.score([[]], [[]])["f1"], 0.0)

    def test_invalid_predictions_count_as_no_prediction(self):
        self.assertEqual(evaluate.score([[{"parameter": "x", "operator": "DROP"}]], [[OB]])["recall"], 0.0)

    def test_baseline_runs_on_rows(self):
        rows = [{"clause_text": "Re-verification of KYC for high-risk customers shall be done every 2 years."}]
        self.assertEqual(evaluate.run_backend("rules", rows)[0][0]["required_value"], 2.0)


class TrainingLabels(unittest.TestCase):
    def test_prompt_masked_completion_trained(self):
        self.assertEqual(train_lora.build_labels([1, 2, 3], [1, 2, 3, 7, 8, 9]), [-100, -100, -100, 7, 8, 9])
        self.assertEqual(train_lora.build_labels([1, 2, 3], [1, 2]), [-100, -100])

    def test_encode_row_appends_eos_and_aligns(self):
        class Tok:
            eos_token_id = 0
            def __call__(self, text, add_special_tokens=False): return {"input_ids": [ord(c) for c in text]}
        e = train_lora.encode_row(Tok(), {"prompt": "ab", "completion": "[]"})
        self.assertEqual(e["input_ids"], [97, 98, 91, 93, 0])
        self.assertEqual(e["labels"], [-100, -100, 91, 93, 0])
        self.assertEqual(len(e["attention_mask"]), len(e["labels"]))


if __name__ == "__main__":
    unittest.main()
