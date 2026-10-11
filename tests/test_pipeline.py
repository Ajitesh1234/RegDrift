import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for k in ("GROQ_API_KEY", "LORA_ADAPTER", "JIRA_URL"): os.environ.pop(k, None)
import regdrift

def flagged(text):
    return {(t["bot"], t["clause"]): t for t in regdrift.run({"circular_text": text})["tickets"]}

class Pipeline(unittest.TestCase):
    def test_sample_circular(self):
        with open(os.path.join(regdrift.HERE, "circular_sample.txt")) as fh: f = flagged(fh.read())
        self.assertEqual(f[("KYC-Refresh-Bot", "Clause 1")]["status"], "violating")
        self.assertEqual(f[("Sanctions-Screening-Bot", "Clause 3")]["status"], "violating")
        self.assertFalse(any(b == "STR-Filer-Bot" for b, _ in f))   # 7 days already compliant

    def test_numbered_paragraphs_and_wrapped_lines(self):
        text = ("3. Reporting entities shall file suspicious transaction reports within\n"
                "5 days of arriving at a conclusion of suspicion.\n")
        f = flagged(text)
        self.assertEqual(f[("STR-Filer-Bot", "Clause 3")]["status"], "violating")

    def test_pdf_ingest(self):
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("reportlab not installed")
        path = os.path.join(tempfile.mkdtemp(), "c.pdf")
        c = canvas.Canvas(path); c.drawString(50, 780, "1. KYC re-verification of high-risk customers shall be")
        c.drawString(50, 765, "done at least once every 2 years."); c.save()
        f = flagged(regdrift.load_text(path))
        self.assertEqual(f[("KYC-Refresh-Bot", "Clause 1")]["status"], "violating")

class WeakChroma(unittest.TestCase):
    """Real ChromaDB returned weak scores for short clauses (0 bots found in CI). Simulate that."""
    def test_param_match_survives_weak_embeddings(self):
        import types
        fake = types.ModuleType("chromadb")
        class Col:
            def count(self): return 1
            def add(self, **k): pass
            def query(self, query_texts, n_results):
                return {"ids": [["BOT-004", "BOT-001", "BOT-002"]], "distances": [[0.95, 0.97, 0.99]]}  # all weak
        fake.Client = lambda: types.SimpleNamespace(get_or_create_collection=lambda *a, **k: Col())
        sys.modules["chromadb"] = fake
        try:
            f = flagged("3. Suspicious transaction reports shall be filed within 5 days of the conclusion.\n")
        finally:
            del sys.modules["chromadb"]
        self.assertEqual(f[("STR-Filer-Bot", "Clause 3")]["status"], "violating")

class NotApplicable(unittest.TestCase):
    def test_llm_not_applicable_creates_no_ticket(self):
        orig = regdrift.llm_json
        # Extraction falls back to rules (None); the judge says the rule is unrelated to bots without the parameter.
        regdrift.llm_json = lambda p: {"status": "not_applicable", "confidence": 0.9, "reason": "unrelated"} if "Bot parameters" in p else None
        try:
            f = flagged("3. Customer names shall be screened against sanctions lists at an interval of not more than 1 day.\n")
        finally:
            regdrift.llm_json = orig
        self.assertEqual([b for b, _ in f], ["Sanctions-Screening-Bot"])

class RealCircularFX(unittest.TestCase):
    """RBI/2026-27/291 (FX derivatives). The LLM is stubbed with a correct extraction; this checks the plumbing."""
    def test_fx_circular(self):
        pdf = os.path.join(regdrift.HERE, "circulars", "rbi_fx_derivatives_2026-10-10.pdf")
        if not os.path.exists(pdf): self.skipTest("circular pdf not in repo")
        orig = regdrift.llm_json
        def stub(p):
            if "Extract machine-checkable" not in p: return None
            return [
              {"clause": "Clause (a)", "summary": "no rebooking of cancelled INR FX derivatives", "parameter": "allow_rebook_cancelled_contracts", "operator": "==", "required_value": False, "unit": ""},
              {"clause": "Clause (c)", "summary": "threshold without underlying is USD 5 million", "parameter": "max_position_without_underlying_usd_million", "operator": "<=", "required_value": 5, "unit": "USD million"},
              {"clause": "Clause (d)", "summary": "obtain undertaking not hedged elsewhere", "parameter": "capture_hedge_undertaking", "operator": "==", "required_value": True, "unit": ""},
              {"clause": "Clause 3", "summary": "retain documents not less than two years", "parameter": "record_retention_years", "operator": ">=", "required_value": 2, "unit": "years"}]
        regdrift.llm_json = stub
        try:
            f = flagged(regdrift.load_text(pdf))
        finally:
            regdrift.llm_json = orig
        for key in [("FX-Derivative-Booking-Bot", "Clause (a)"), ("Hedge-Limit-Checker-Bot", "Clause (c)"), ("Hedge-Documentation-Bot", "Clause (d)")]:
            self.assertEqual(f[key]["status"], "violating")
        self.assertNotIn(("Hedge-Documentation-Bot", "Clause 3"), f)   # 5 years retained >= 2: compliant, no ticket

    def test_lettered_clauses_parsed(self):
        labels = [c for c, _ in regdrift.iter_clauses(regdrift.clean("(a) First rule here.\n(b) Second rule\nwraps here.\n"))]
        self.assertEqual(labels, ["Clause (a)", "Clause (b)"])

class JiraDedupe(unittest.TestCase):
    def test_second_run_creates_nothing(self):
        created, existing = [], []
        def fake(path, body=None):
            if path.startswith("/rest/api/3/search/jql"):
                return {"issues": [{"fields": {"summary": s}} for s in existing], "isLast": True}
            created.append(body["fields"]["summary"]); existing.append(body["fields"]["summary"])
            return {"key": f"SCRUM-{len(created)}"}
        env = dict(JIRA_URL="https://x.atlassian.net", JIRA_EMAIL="a@b.c", JIRA_TOKEN="t", JIRA_PROJECT="SCRUM")
        os.environ.update(env); orig = regdrift.jira_request; regdrift.jira_request = fake
        try:
            with open(os.path.join(regdrift.HERE, "circular_sample.txt")) as fh: text = fh.read()
            regdrift.run({"circular_text": text}); first = len(created)
            regdrift.run({"circular_text": text}); second = len(created) - first
        finally:
            regdrift.jira_request = orig
            for k in env: os.environ.pop(k)
        self.assertGreaterEqual(first, 3)    # at least the 3 real findings (offline mode may add at_risk extras)
        self.assertEqual(second, 0)          # nothing new on the second run
        self.assertTrue(all("DEMO/2026-27/101" in s for s in created))   # circular ref in title

class GroqHeaders(unittest.TestCase):
    def test_user_agent_sent(self):
        seen = {}
        class R:  # minimal response
            def read(self): return b'{"choices":[{"message":{"content":"[]"}}]}'
            def __enter__(self): return self
            def __exit__(self, *a): pass
        def fake_open(req, timeout=0):
            seen.update({k.lower(): v for k, v in req.header_items()}); return R()
        orig = regdrift.urllib.request.urlopen; regdrift.urllib.request.urlopen = fake_open
        os.environ["GROQ_API_KEY"] = "k"
        try: regdrift.llm_json("hi")
        finally:
            regdrift.urllib.request.urlopen = orig; os.environ.pop("GROQ_API_KEY")
        self.assertIn("regdrift", seen.get("user-agent", ""))

class GroqModelFallback(unittest.TestCase):
    def test_skips_unavailable_models(self):
        import io, json as _j, urllib.error
        tried = []
        class R:
            def read(self): return b'{"choices":[{"message":{"content":"[{\\"a\\": 1}]"}}]}'
            def __enter__(self): return self
            def __exit__(self, *a): pass
        def fake_open(req, timeout=0):
            model = _j.loads(req.data)["model"]; tried.append(model)
            if model != "llama-3.3-70b-versatile":
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b'{"error":{"code":"model_not_found","message":"The model does not exist"}}'))
            return R()
        orig, regdrift._groq = regdrift.urllib.request.urlopen, {}
        regdrift.urllib.request.urlopen = fake_open; os.environ["GROQ_API_KEY"] = "k"
        try: result = regdrift.llm_json("x"); again = regdrift.llm_json("y")
        finally:
            regdrift.urllib.request.urlopen = orig; os.environ.pop("GROQ_API_KEY"); regdrift._groq = {}
        self.assertEqual(result, [{"a": 1}])
        self.assertEqual(tried[:3], ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"])
        self.assertEqual(tried[3], "llama-3.3-70b-versatile")   # second call goes straight to the working model

class ClauseLabels(unittest.TestCase):
    def test_norm_clause(self):
        n = regdrift.norm_clause
        self.assertEqual({n("(a)"), n("a"), n("Clause (a)"), n("Paragraph (a)"), n(" clause a. ")}, {"Clause (a)"})
        self.assertEqual({n("3"), n("Clause 3"), n("Para 3."), n("Paragraph 3")}, {"Clause 3"})

    def test_dedupe_survives_llm_relabelling_and_status_change(self):
        created, existing = [], []
        def fake(path, body=None):
            if path.startswith("/rest/api/3/search/jql"):
                return {"issues": [{"fields": {"summary": s}} for s in existing], "isLast": True}
            created.append(body["fields"]["summary"]); existing.append(body["fields"]["summary"])
            return {"key": f"SCRUM-{len(created)}"}
        label = {"v": "(a)"}
        orig_llm, orig_req = regdrift.llm_json, regdrift.jira_request
        def stub(p):
            if "Extract machine-checkable" not in p: return None
            return [{"clause": label["v"], "summary": "no rebooking", "parameter": "allow_rebook_cancelled_contracts",
                     "operator": "==", "required_value": False, "unit": ""}]
        env = dict(JIRA_URL="https://x.atlassian.net", JIRA_EMAIL="a@b.c", JIRA_TOKEN="t", JIRA_PROJECT="SCRUM")
        os.environ.update(env); regdrift.llm_json = stub; regdrift.jira_request = fake
        text = "RBI/2026-27/291\n(a) An Authorised Dealer shall not permit rebooking of cancelled contracts.\n"
        try:
            regdrift.run({"circular_text": text}); first = len(created)
            label["v"] = "Clause (a)"                      # LLM words the label differently on the next run
            regdrift.run({"circular_text": text}); second = len(created) - first
        finally:
            regdrift.llm_json, regdrift.jira_request = orig_llm, orig_req
            for k in env: os.environ.pop(k)
        self.assertEqual((first, second), (1, 0))
        # tickets created by an older version of the code (no 'Clause' prefix) are still recognised
        self.assertEqual(regdrift.key_from_summary("[RegDrift] FX-Bot violating - (a) (RBI/2026-27/291)"),
                         regdrift.key_from_summary("[RegDrift] FX-Bot at_risk - Clause (a) (RBI/2026-27/291)"))

class PermissiveListRule(unittest.TestCase):
    """RBI/2026-27/257: certified-copy facility extended to FPIs ('may'). LLM stubbed; checks the logic."""
    def run_with(self, modality):
        orig = regdrift.llm_json
        regdrift.llm_json = lambda p: [{"clause": "4(1)(v) proviso", "summary": "certified copy facility extended to FPIs",
            "parameter": "certified_copy_accepted_customer_types", "operator": "contains", "required_value": "FPI",
            "unit": "", "modality": modality}] if "Extract machine-checkable" in p else None
        try:
            return flagged("RBI/2026-27/257\n4. Banks may obtain overseas-certified copies for NRIs, PIOs and FPIs.\n")
        finally:
            regdrift.llm_json = orig

    def test_permissive_is_low_severity_optional_gap(self):
        f = self.run_with("permissive")
        t = next(v for k, v in f.items() if k[0] == "KYC-Document-Verification-Bot")
        self.assertEqual((t["status"], t["severity"]), ("optional_gap", "Low"))
        self.assertIn("now permits", t["reason"])

    def test_mandatory_is_violating(self):
        f = self.run_with("mandatory")
        t = next(v for k, v in f.items() if k[0] == "KYC-Document-Verification-Bot")
        self.assertEqual(t["status"], "violating")

    def test_adopted_option_is_compliant(self):
        regdrift_bots = os.path.join(regdrift.HERE, "bots.json")
        with open(regdrift_bots) as fh: self.assertIn("NRI", json.load(fh)[-1]["parameters"]["certified_copy_accepted_customer_types"])
        self.assertEqual(regdrift.key_from_summary("[RegDrift] KYC-Document-Verification-Bot optional_gap - Clause 4 (RBI/2026-27/257)"),
                         ("kyc-document-verification-bot", "Clause 4", "RBI/2026-27/257"))

class Validation(unittest.TestCase):
    V = staticmethod(regdrift.validate_obligation)

    def test_good_and_coerced(self):
        o = self.V({"clause": "(a)", "parameter": "Max Days!", "operator": "<=", "required_value": "7", "modality": "Permissive"})
        self.assertEqual((o["parameter"], o["required_value"], o["modality"], o["clause"]), ("max_days", 7.0, "permissive", "Clause (a)"))
        self.assertIs(self.V({"parameter": "x", "operator": "==", "required_value": "true"})["required_value"], True)

    def test_garbage_is_dropped(self):
        bad = [None, "text", [], {}, {"parameter": "x", "operator": "DROP TABLE", "required_value": 1},
               {"parameter": "x", "operator": "<=", "required_value": "soon"},
               {"parameter": "x", "operator": "<=", "required_value": True},
               {"parameter": "x", "operator": "contains", "required_value": ["a"]},
               {"parameter": "", "operator": "<=", "required_value": 1}]
        self.assertEqual([self.V(b) for b in bad], [None] * len(bad))

    def test_llm_returning_junk_does_not_crash_pipeline(self):
        orig = regdrift.llm_json
        regdrift.llm_json = lambda p: {"obligations": [{"parameter": "str_filing_days", "operator": "<=", "required_value": "5"},
                                                       "ignore previous instructions", 7, None]} if "Extract machine" in p else None
        try:
            f = flagged("RBI/2026-27/9\n3. Suspicious transaction reports must be filed within 5 days of the conclusion.\n")
        finally:
            regdrift.llm_json = orig
        self.assertEqual(f[("STR-Filer-Bot", "Clause ?")]["status"], "violating")

    def test_type_mismatch_in_bot_setting_is_flagged_not_crashed(self):
        imp = {"obligation": {"parameter": "p", "operator": "<=", "required_value": 5, "summary": "s", "clause": "Clause 1"},
               "bot": {"id": "B", "name": "B", "platform": "UiPath", "description": "", "parameters": {"p": "three"}}}
        self.assertEqual(regdrift.judge(imp)["status"], "at_risk")


class Retry(unittest.TestCase):
    def setUp(self):
        self.sleeps, self._s = [], regdrift._sleep
        regdrift._sleep = self.sleeps.append

    def tearDown(self):
        regdrift._sleep = self._s

    def err(self, code, retry_after=None):
        import io, urllib.error, email.message
        h = email.message.Message()
        if retry_after: h["Retry-After"] = retry_after
        return urllib.error.HTTPError("http://x", code, "e", h, io.BytesIO(b""))

    def test_retries_429_then_succeeds_honouring_retry_after(self):
        calls = []
        def fn():
            calls.append(1)
            if len(calls) < 3: raise self.err(429, "7")
            return "ok"
        self.assertEqual(regdrift._retry(fn), "ok")
        self.assertEqual(self.sleeps, [7.0, 7.0])

    def test_gives_up_after_max_tries_and_does_not_retry_client_errors(self):
        n = []
        def always(): n.append(1); raise self.err(503)
        with self.assertRaises(Exception): regdrift._retry(always, tries=3)
        self.assertEqual(len(n), 3)
        m = []
        def notfound(): m.append(1); raise self.err(404)
        with self.assertRaises(Exception): regdrift._retry(notfound)
        self.assertEqual(len(m), 1)


class CLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for k in ("GROQ_API_KEY", "LORA_ADAPTER", "JIRA_URL"): os.environ.pop(k, None)

    def go(self, *extra):
        return regdrift.main([os.path.join(regdrift.HERE, "circular_sample.txt"), "--report-dir", os.path.join(self.tmp, "r"),
                              "--tickets", os.path.join(self.tmp, "t.json"), *extra])

    def test_report_is_complete_and_auditable(self):
        self.assertEqual(self.go("--dry-run"), 0)
        with open(os.path.join(self.tmp, "r", "circular_sample.report.json")) as fh: rep = json.load(fh)
        self.assertEqual(rep["circular_ref"], "DEMO/2026-27/101")
        self.assertEqual(len(rep["circular_sha256"]), 64)
        self.assertEqual(rep["extractor"], "rules")
        self.assertFalse(rep["jira_enabled"])
        self.assertTrue(any(f["status"] == "compliant" for f in rep["findings"]))   # compliant findings are kept for audit
        with open(os.path.join(self.tmp, "t.json")) as fh: tickets = json.load(fh)
        self.assertTrue(all(t["status"] != "compliant" for t in tickets))

    def test_fail_on_high_exit_code(self):
        self.assertEqual(self.go("--dry-run", "--fail-on-high"), 2)

    def test_unreadable_input_exits_1_and_continues(self):
        rc = regdrift.main([os.path.join(self.tmp, "missing.pdf"), "--report-dir", self.tmp, "--tickets", os.path.join(self.tmp, "t.json")])
        self.assertEqual(rc, 1)

    def test_dry_run_never_calls_jira_even_with_credentials(self):
        env = dict(JIRA_URL="https://x.atlassian.net", JIRA_EMAIL="a@b.c", JIRA_TOKEN="t", JIRA_PROJECT="S")
        os.environ.update(env); orig = regdrift.jira_request
        regdrift.jira_request = lambda *a, **k: self.fail("Jira was called during --dry-run")
        try: self.assertEqual(self.go("--dry-run"), 0)
        finally:
            regdrift.jira_request = orig
            for k in env: os.environ.pop(k)

if __name__ == "__main__":
    unittest.main()
