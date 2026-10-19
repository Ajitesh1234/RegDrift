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
        self.assertGreaterEqual(first, 2)    # the 2 real violations (KYC re-verification, sanctions screening)
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


class AliasMatching(unittest.TestCase):
    def test_acronyms_plurals_and_case(self):
        s = regdrift.same_item
        self.assertTrue(s("Foreign Portfolio Investors", "FPI"))
        self.assertTrue(s("fpi", "Foreign Portfolio Investor"))
        self.assertTrue(s("NRIs", "NRI"))
        self.assertFalse(s("NRI", "PIO"))
        self.assertFalse(s("Foreign Portfolio Investors", "FII"))

    def test_judge_contains_uses_aliases(self):
        bot = {"id": "B", "name": "B", "platform": "UiPath", "description": "", "tags": [],
               "parameters": {"accepted_customer_types": ["NRI", "FPI"]}}
        ob = {"clause": "Clause 1", "summary": "s", "parameter": "accepted_customer_types", "operator": "contains",
              "required_value": "Foreign Portfolio Investors", "unit": "", "modality": "permissive"}
        self.assertEqual(regdrift.judge({"obligation": ob, "bot": bot})["status"], "compliant")


class Dashboard(unittest.TestCase):
    def test_render_escapes_and_counts(self):
        import dashboard
        rep = {"source": "x.pdf", "circular_ref": "R/1", "run_at_utc": "t", "tickets": [],
               "findings": [{"status": "violating", "bot": "<b>Bot</b>", "bot_id": "1", "clause": "C1",
                             "reason": "bad & wrong", "ref": "R/1"}]}
        out = dashboard.render([rep])
        self.assertIn("&lt;b&gt;Bot&lt;/b&gt;", out)
        self.assertNotIn("<b>Bot</b>", out)
        self.assertIn("bad &amp; wrong", out)


class LlmVerdictCap(unittest.TestCase):
    def test_llm_only_violation_is_capped_at_risk(self):
        bot = {"id": "B", "name": "B", "platform": "UiPath", "description": "checks limits", "tags": [],
               "parameters": {"other_setting": 1}}
        ob = {"clause": "Clause (d)", "summary": "obtain undertaking", "parameter": "some_unowned_setting",
              "operator": "==", "required_value": True, "unit": "", "modality": "mandatory"}
        orig = regdrift.llm_json
        regdrift.llm_json = lambda p: {"status": "violating", "confidence": 0.99, "reason": "does not collect it"}
        try:
            r = regdrift.judge({"obligation": ob, "bot": bot})
        finally:
            regdrift.llm_json = orig
        self.assertEqual(r["status"], "at_risk")
        self.assertIn("Unconfirmed", r["reason"])
        self.assertLessEqual(r["confidence"], 0.7)


class ChangedLimitGuard(unittest.TestCase):
    TEXT = ("(c) Authorised Dealers may allow positions upto USD 100 million equivalent without underlying. "
            "The threshold shall, henceforth, be USD 5 million equivalent.\n")

    def ob(self, val):
        return {"clause": "Clause (c)", "summary": "s", "parameter": "p", "operator": "<=",
                "required_value": val, "unit": "usd million", "modality": "mandatory"}

    def test_old_value_is_corrected_to_new(self):
        self.assertEqual(regdrift.fix_changed_limit(self.ob(100), self.TEXT)["required_value"], 5)

    def test_summary_notes_the_revised_limit(self):
        out = regdrift.fix_changed_limit(self.ob(100), self.TEXT)
        self.assertIn("Revised limit: at most 5 usd million", out["summary"])

    def test_correct_value_untouched(self):
        self.assertEqual(regdrift.fix_changed_limit(self.ob(5), self.TEXT)["required_value"], 5)

    def test_unrelated_value_untouched(self):
        self.assertEqual(regdrift.fix_changed_limit(self.ob(42), self.TEXT)["required_value"], 42)

    def test_real_circular(self):
        t = regdrift.load_text(os.path.join(regdrift.HERE, "circulars", "rbi_fx_derivatives_2026-10-10.pdf"))
        ob = {**self.ob(100), "clause": "Clause (c)"}
        self.assertEqual(regdrift.fix_changed_limit(ob, t)["required_value"], 5)


class OwnerAndLabels(unittest.TestCase):
    def test_clause_descriptors_are_dropped(self):
        n = regdrift.norm_clause
        self.assertEqual(n("Clause (c) - overall limit"), "Clause (c)")
        self.assertEqual(n("(c) per-facility limit"), "Clause (c)")
        self.assertEqual(n("Clause 5(1)(v)"), "Clause 5(1)(v)")
        self.assertEqual(n("a"), "Clause (a)")

    def test_non_owner_bot_is_not_applicable(self):
        bot = {"id": "X", "name": "X", "platform": "UiPath", "description": "checks positions", "tags": [],
               "parameters": {}}
        ob = {"clause": "Clause (d)", "summary": "undertaking", "parameter": "capture_hedge_undertaking",
              "operator": "==", "required_value": True, "unit": "", "modality": "mandatory"}
        calls = []
        orig = regdrift.llm_json
        regdrift.llm_json = lambda p: calls.append(p)
        try:
            r = regdrift.judge({"obligation": ob, "bot": bot})
        finally:
            regdrift.llm_json = orig
        self.assertEqual(r["status"], "not_applicable")
        self.assertEqual(calls, [])   # no LLM call needed


class AddCirculars(unittest.TestCase):
    def test_clean_name(self):
        import add_circulars
        self.assertEqual(add_circulars.clean_name("RBI Circular (KYC) 12-06-2025.PDF"), "rbi_circular_kyc_12_06_2025.pdf")

    def test_zip_import_and_short_text_rejected(self):
        import add_circulars, zipfile
        tmp = tempfile.mkdtemp()
        z = os.path.join(tmp, "c.zip")
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("Good One.txt", "1. KYC re-verification shall be done every 2 years. " * 20)
            zf.writestr("tiny.txt", "short")
        add_circulars.DEST = os.path.join(tmp, "dest")
        rc = add_circulars.main([z])
        self.assertEqual(sorted(os.listdir(add_circulars.DEST)), ["good_one.txt"])
        self.assertEqual(rc, 2)


class LowMemoryRetriever(unittest.TestCase):
    def test_bow_switch_skips_chroma(self):
        os.environ["REGDRIFT_RETRIEVER"] = "bow"
        called = []
        orig = regdrift.retrieve_chroma
        regdrift.retrieve_chroma = lambda *a, **k: called.append(1)
        try:
            f = flagged("Clause 1: KYC re-verification of high-risk customers shall be done at least once every 2 years.")
        finally:
            regdrift.retrieve_chroma = orig
            os.environ.pop("REGDRIFT_RETRIEVER")
        self.assertEqual(called, [])
        self.assertEqual(f[("KYC-Refresh-Bot", "Clause 1")]["status"], "violating")


class TwinRules(unittest.TestCase):
    def test_identical_rules_are_judged_once(self):
        ob = {"clause": "Clause (c)", "summary": "cap", "parameter": "str_filing_days", "operator": "<=",
              "required_value": 5, "unit": "days", "modality": "mandatory"}
        bot = next(b for b in regdrift.load_bots() if b["name"] == "STR-Filer-Bot")
        out = regdrift.reason({"impacts": [{"obligation": ob, "bot": bot, "score": 1.0},
                                           {"obligation": dict(ob), "bot": bot, "score": 1.0}], "circular_ref": "R/1"})
        self.assertEqual(len(out["findings"]), 1)


class DateRules(unittest.TestCase):
    def test_parse_date_formats_and_rejects_nonsense(self):
        for s in ("31 March 2027", "31st March, 2027", "March 31, 2027", "31/03/2027", "31.3.2027", "2027-03-31"):
            self.assertEqual(regdrift.parse_date(s), "2027-03-31", s)
        for s in ("30/02/2027", "31 Foo 2027", "2027", "soon", None, 5):
            self.assertIsNone(regdrift.parse_date(s), s)

    def test_validation_keeps_dates_and_tags_the_unit(self):
        v = regdrift.validate_obligation({"parameter": "go_live", "operator": "<=", "required_value": "31 March 2027"})
        self.assertEqual((v["required_value"], v["unit"]), ("2027-03-31", "date"))
        self.assertIsNone(regdrift.validate_obligation({"parameter": "x", "operator": "<=", "required_value": "tomorrow-ish"}))

    def test_late_setting_is_violating_and_reason_names_both_dates(self):
        bot = {"id": "B", "name": "B", "platform": "UiPath", "description": "", "tags": [], "parameters": {"go_live": "1 June 2027"}}
        ob = regdrift.validate_obligation({"parameter": "go_live", "operator": "<=", "required_value": "2027-03-31"})
        j = regdrift.judge({"obligation": ob, "bot": bot})
        self.assertEqual(j["status"], "violating")
        self.assertIn("2027-06-01", j["reason"])
        self.assertIn("2027-03-31", j["reason"])


class TimeContext(unittest.TestCase):
    """Is the circular in force today? Withdrawn stamp, dates, expiry and age are read without an LLM."""
    HDR = "RBI/2026-27/10 DOR.XYZ.REC.12/13.03.00/2026-27 October 7, 2026\n"
    BODY = "1. Regulated entities shall file suspicious transaction reports within 3 days of a conclusion of suspicion.\n"
    C = staticmethod(regdrift.circular_context)

    def test_reference_numbers_are_not_dates_and_dated_refs_are_skipped(self):
        c = self.C("RBI/2026-27/138 DOR.SOG(SPE).REC.117/13.03.00/2026-27 June 17, 2026\nSee the Direction dated February 25, 2016.",
                   today="2026-10-18")
        self.assertEqual(c["issued"], "2026-06-17")
        self.assertEqual(self.C("FAQs on the Master Direction dated February 25, 2016 (June 9, 2025)", today="2026-10-18")["issued"],
                         "2025-06-09")

    def test_in_force_with_immediate_effect(self):
        c = self.C(self.HDR + "These Directions shall come into force with immediate effect.", today="2026-10-18")
        self.assertEqual((c["status"], c["warnings"]), ("in_force", []))

    def test_future_effective_date_is_upcoming(self):
        c = self.C(self.HDR + "These Directions shall come into effect from April 1, 2027.", today="2026-10-18")
        self.assertEqual((c["status"], c["effective_dates"]), ("upcoming", ["2027-04-01"]))
        self.assertIn("1 April 2027", c["headline"])

    def test_staggered_dates_are_in_force_with_a_note(self):
        c = self.C(self.HDR + "Para 3 shall come into effect from November 1, 2026 and para 4 from June 15, 2026.", today="2026-10-18")
        self.assertEqual(c["status"], "in_force")
        self.assertTrue(any("2026-11-01" in w for w in c["warnings"]))

    def test_temporary_measure_expires(self):
        t = self.HDR + "The facility is temporarily withdrawn for the period until September 30, 2026."
        self.assertEqual(self.C(t, today="2026-10-18")["status"], "expired")
        self.assertEqual(self.C(t, today="2026-09-01")["status"], "in_force")

    def test_old_circular_warns_but_updated_master_direction_does_not(self):
        old = self.C("RBI/2009-10/69 July 1, 2009\nMaster Circular", today="2026-10-18")
        self.assertEqual(old["status"], "in_force")
        self.assertTrue(old["warnings"] and "17 years" in old["warnings"][0])
        md = self.C("Master Direction February 25, 2016 (Updated as on August 14, 2025)", today="2026-10-18")
        self.assertEqual(md["warnings"], [])

    def test_body_sentence_about_another_circular_is_not_a_withdrawal(self):
        t = self.HDR + "The earlier circular dated March 3, 2020 stands withdrawn. " + self.BODY
        self.assertEqual(self.C(t, today="2026-10-18")["status"], "in_force")

    def test_no_date_is_said_plainly(self):
        self.assertIn("No issue date", self.C("Nothing dated here at all.", today="2026-10-18")["warnings"][0])

    def run_text(self, text, **kw):
        return regdrift.run({"circular_text": text, "today": "2026-10-18", "dry_run": True, **kw})

    def test_withdrawn_stamp_blocks_analysis_unless_overridden(self):
        text = regdrift.STAMP_MARK + "withdrawn]]\n" + self.HDR + self.BODY
        st = self.run_text(text)
        self.assertEqual((st["context"]["status"], st["findings"], st["tickets"], st["extractor"]), ("withdrawn", [], [], "skipped"))
        self.assertNotIn("REGDRIFT-STAMP", st["circular_text"])
        self.assertTrue(self.run_text(text, allow_withdrawn=True)["findings"])

    def test_expired_circular_files_no_tickets_but_upcoming_says_when(self):
        exp = self.run_text(self.HDR + "The facility is temporarily withdrawn for the period until September 30, 2026.\n" + self.BODY)
        self.assertTrue(exp["tickets"] and all(t["jira_key"].startswith("NOT FILED") for t in exp["tickets"]))
        up = self.run_text(self.HDR + "These Directions shall come into effect from April 1, 2027.\n" + self.BODY)
        self.assertTrue(up["findings"] and all(f["timing"] == "upcoming" and f["effective_from"] == "2027-04-01" for f in up["findings"]))
        self.assertEqual(regdrift.build_report(up, "x")["context"]["status"], "upcoming")


class JobDeadline(unittest.TestCase):
    """One time limit covers the whole check; past it nothing is sent to the LLM and the rest is reported, not guessed."""
    def setUp(self):
        self.orig = regdrift.llm_json
        self.orig_sleep = regdrift._sleep
        os.environ["GROQ_API_KEY"] = "x"
        regdrift.start_job_clock(1000)

    def tearDown(self):
        regdrift.llm_json = self.orig
        regdrift._sleep = self.orig_sleep
        os.environ.pop("GROQ_API_KEY", None)
        regdrift.start_job_clock(1e9)

    def test_retry_gives_up_instead_of_sleeping_past_the_deadline(self):
        import urllib.error
        slept = []
        regdrift._sleep = slept.append
        def boom():
            raise urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "50"}, None)
        with self.assertRaises(urllib.error.HTTPError):
            regdrift._retry(boom, deadline=regdrift.time.monotonic() + 5)
        self.assertEqual(slept, [])

    def test_no_request_is_sent_once_out_of_time(self):
        regdrift.start_job_clock(-1)
        self.assertTrue(regdrift.out_of_time())
        called = []
        real = regdrift.urllib.request.urlopen
        regdrift.urllib.request.urlopen = lambda *a, **k: called.append(1)
        try:
            self.assertIsNone(regdrift.llm_json("hi"))
        finally:
            regdrift.urllib.request.urlopen = real
        self.assertEqual(called, [])

    def test_unchecked_pairs_are_counted_not_guessed(self):
        bot = next(b for b in regdrift.load_bots() if b["name"] == "Loan-Disbursal-Bot")
        ob = {"clause": "Clause 1", "summary": "Loans must be disbursed within 5 days.", "parameter": "unmapped_loan",
              "operator": "<=", "required_value": 5, "unit": "days"}
        regdrift.start_job_clock(-1)
        j = regdrift.judge({"obligation": ob, "bot": bot})
        self.assertTrue(j.get("unchecked"))
        self.assertEqual(j["status"], "not_applicable")


class StampDetection(unittest.TestCase):
    """The stamp can be rotated by the page transform (cm) or by the text matrix (Tm). Check both, with each reader alone,
    so one library changing how it reports a transform cannot silently stop withdrawn circulars being caught."""
    def make(self, style):
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("reportlab not installed")
        import math
        path = os.path.join(tempfile.mkdtemp(), "c.pdf")
        c = canvas.Canvas(path)
        c.drawString(50, 780, "RBI/2025-26/36 May 8, 2025")
        c.drawString(50, 760, "The earlier circular stands withdrawn.")  # ordinary sentence: must not count
        if style == "tm":
            t = c.beginText(); k = math.cos(math.radians(45))
            t.setTextTransform(k, k, -k, k, 250, 400); t.textOut("Withdrawn"); c.drawText(t)
        else:
            c.saveState(); c.translate(250, 400)
            if style == "cm":
                c.rotate(45)
            c.drawString(0, 0, "Withdrawn"); c.restoreState()
        c.save()
        return path

    def readers(self):
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            self.skipTest("pdfplumber not installed")
        return (regdrift._slanted_words_pypdf, regdrift._slanted_words_pdfplumber)

    def words(self, reader, path):
        return "".join(reader(path, 3))

    def test_each_reader_finds_both_kinds_of_rotation(self):
        for style in ("cm", "tm"):
            path = self.make(style)
            for reader in self.readers():
                self.assertIn("Withdrawn", self.words(reader, path), f"{reader.__name__} / {style}")
            self.assertEqual(regdrift.pdf_stamps(path), ["withdrawn"], style)
            self.assertTrue(regdrift.load_text(path).startswith(regdrift.STAMP_MARK))

    def test_stamp_repeated_on_every_page_and_twice_on_one(self):
        """Real RBI PDFs carry the stamp on every page; the glyphs of repeats run together."""
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("reportlab not installed")
        self.readers()
        path = os.path.join(tempfile.mkdtemp(), "multi.pdf")
        c = canvas.Canvas(path)
        for page in range(3):
            c.drawString(50, 780, f"RBI/2025-26/36 page {page}")
            for y in ((400, 200) if page == 1 else (400,)):
                c.saveState(); c.translate(250, y); c.rotate(45); c.drawString(0, 0, "Withdrawn"); c.restoreState()
            c.showPage()
        c.save()
        self.assertEqual(regdrift.pdf_stamps(path), ["withdrawn"])
        self.assertTrue(regdrift.load_text(path).startswith(regdrift.STAMP_MARK))

    def test_flat_text_and_body_sentences_are_not_stamps(self):
        path = self.make("flat")
        for reader in self.readers():
            self.assertEqual(self.words(reader, path), "", reader.__name__)
        self.assertEqual(regdrift.pdf_stamps(path), [])
        self.assertFalse(regdrift.load_text(path).startswith(regdrift.STAMP_MARK))

    def test_second_reader_covers_when_the_first_sees_nothing(self):
        path = self.make("cm")
        self.readers()
        orig = regdrift._slanted_words_pypdf
        regdrift._slanted_words_pypdf = lambda p, n: iter(())
        try:
            self.assertEqual(regdrift.pdf_stamps(path), ["withdrawn"])
        finally:
            regdrift._slanted_words_pypdf = orig


class LongCircular(unittest.TestCase):
    """A 29-page circular is far too big for one free-tier LLM request; it must be read in sections."""
    def setUp(self):
        self.orig = regdrift.llm_json
        os.environ["GROQ_API_KEY"] = "x"
        self.prompts = []

    def tearDown(self):
        regdrift.llm_json = self.orig
        os.environ.pop("GROQ_API_KEY", None)

    def long_text(self, n=60):
        head = "RBI/2026-27/474 October 7, 2026\n"
        paras = [f"{i}. Regulated entities shall complete item number {i} " + "and keep records of the same " * 12 for i in range(1, n)]
        return head + "\n".join(paras)

    def test_progress_is_reported_and_a_time_budget_stops_reading(self):
        os.environ["GROQ_API_KEY"] = "x"
        orig_json = regdrift.llm_json
        regdrift.llm_json = lambda p: []
        orig, clock = regdrift.EXTRACT_BUDGET_SECONDS, [0.0]
        real = regdrift.time.monotonic
        try:
            seen = []
            cov = {}
            regdrift.llm_extract_text(self.long_text(), cov, lambda i, n: seen.append((i, n)))
            self.assertEqual(seen[0][0], 1)
            self.assertEqual([i for i, _ in seen], list(range(1, len(seen) + 1)))
            self.assertFalse(cov["stopped"])
            regdrift.EXTRACT_BUDGET_SECONDS = 10
            regdrift.time.monotonic = lambda: clock.__setitem__(0, clock[0] + 6) or clock[0]  # 6 s passes per look at the clock
            cov2 = {}
            regdrift.llm_extract_text(self.long_text(), cov2)
        finally:
            regdrift.time.monotonic = real
            regdrift.EXTRACT_BUDGET_SECONDS = orig
            regdrift.llm_json = orig_json
            os.environ.pop("GROQ_API_KEY", None)
        self.assertTrue(cov2["stopped"])
        self.assertLess(cov2["sent"], cov2["sections"])

    def test_chunks_are_small_whole_and_lose_nothing(self):
        text = self.long_text()
        chunks = regdrift.chunk_text(text)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= regdrift.CHUNK_CHARS for c in chunks))
        self.assertEqual("\n".join(chunks).split(), text.split())  # same words, same order
        one = "word. " * 3000  # a single huge paragraph is still cut up
        self.assertTrue(all(len(c) <= regdrift.CHUNK_CHARS for c in regdrift.chunk_text(one)))

    def test_every_section_is_sent_and_answers_are_merged(self):
        def stub(prompt):
            self.prompts.append(prompt)
            n = len(self.prompts)
            return [{"clause": f"Clause {n}", "summary": "s", "parameter": f"p{n}", "operator": "<=", "required_value": n, "unit": "days"}]
        regdrift.llm_json = stub
        raw = regdrift.llm_extract_text(self.long_text())
        self.assertGreater(len(self.prompts), 3)
        self.assertTrue(all(len(p) < regdrift.CHUNK_CHARS + 2500 for p in self.prompts))  # every request stays small
        self.assertEqual(len(raw), len(self.prompts))

    def test_sections_without_obligation_wording_are_skipped(self):
        regdrift.llm_json = lambda p: self.prompts.append(p) or []
        regdrift.llm_extract_text("ANNEXURE\nName of bank\nAddress\nSignature\n" * 50)
        self.assertEqual(self.prompts, [])

    def test_partial_failure_keeps_what_was_read_and_total_failure_falls_back(self):
        answers = iter([None, [{"clause": "Clause 1", "summary": "s", "parameter": "p", "operator": "<=", "required_value": 3, "unit": "days"}]])
        regdrift.llm_json = lambda p: next(answers, None)
        raw = regdrift.llm_extract_text(self.long_text())
        self.assertEqual([o["parameter"] for o in raw], ["p"])
        regdrift.llm_json = lambda p: None
        self.assertIsNone(regdrift.llm_extract_text(self.long_text()))

    def test_coverage_says_how_much_was_read(self):
        regdrift.llm_json = lambda p: []
        cov = {}
        regdrift.llm_extract_text(self.long_text(n=400), cov)  # far more sections than MAX_CHUNKS
        self.assertGreater(cov["sections"], regdrift.MAX_CHUNKS)
        self.assertEqual((cov["sent"], cov["answered"]), (regdrift.MAX_CHUNKS, regdrift.MAX_CHUNKS))
        st = regdrift.extract({"circular_text": self.long_text(n=400)})
        self.assertGreater(st["coverage"]["sections"], st["coverage"]["answered"])
        os.environ.pop("GROQ_API_KEY", None)  # no key = keyword mode: nothing was "missed", so no coverage claim
        self.assertEqual(regdrift.extract({"circular_text": self.long_text(n=400)})["coverage"], {})


class PercentAndMatchChecks(unittest.TestCase):
    def setUp(self):
        self.orig = regdrift.llm_json

    def tearDown(self):
        regdrift.llm_json = self.orig

    def ob(self, **kw):
        return {"clause": "Clause (a)", "summary": "", "parameter": "x", "operator": "==", "required_value": 1,
                "unit": "", "modality": "mandatory", **kw}

    def test_percent_written_as_a_fraction_is_restored(self):
        o = regdrift.fix_percent(self.ob(summary="rho must be set to 50 %.", required_value=0.5, unit="percent"))
        self.assertEqual(o["required_value"], 50)
        o = regdrift.fix_percent(self.ob(summary="discount rate must be 2.5 per cent", required_value=0.025, unit="percent"))
        self.assertEqual(o["required_value"], 2.5)

    def test_percent_left_alone_when_already_right_or_unrelated(self):
        for o in (self.ob(summary="set to 50 %", required_value=50, unit="percent"),
                  self.ob(summary="multiply by 0.7", required_value=0.7, unit="ratio"),
                  self.ob(summary="must be 50 % of capital", required_value=0.7, unit="percent")):
            self.assertEqual(regdrift.fix_percent(o), o)

    def test_name_only_match_is_unlinked_but_real_match_is_kept(self):
        calls = []
        regdrift.llm_json = lambda p: calls.append(p) or {"concerns_setting": False}
        cva = self.ob(parameter="capture_hedge_undertaking", operator="==", required_value=False,
                      summary="Banks must not recognize CVA hedges.")
        out = regdrift.verify_matches([cva])
        self.assertTrue(out[0]["parameter"].startswith("unmapped_"))
        self.assertEqual(out[0]["summary"], cva["summary"])
        self.assertEqual(len(calls), 1)
        fx = self.ob(parameter="allow_rebook_cancelled_contracts", required_value=False,
                     summary="Authorised Dealers must not permit users to rebook any cancelled contract.")
        calls.clear()
        self.assertEqual(regdrift.verify_matches([fx]), [fx])  # the name is fully covered by the rule: no extra request
        self.assertEqual(calls, [])

    def test_a_failed_or_positive_check_never_drops_a_match(self):
        cva = self.ob(parameter="capture_hedge_undertaking", summary="Banks must not recognize CVA hedges.")
        for answer in (None, {"concerns_setting": True}, "garbage", {}):
            regdrift.llm_json = lambda p, a=answer: a
            self.assertEqual(regdrift.verify_matches([cva]), [cva])

    def test_unknown_parameter_is_never_checked(self):
        regdrift.llm_json = lambda p: self.fail("no request expected")
        o = self.ob(parameter="cva_correlation", summary="rho is 50 %")
        self.assertEqual(regdrift.verify_matches([o]), [o])

    def test_same_pair_is_asked_once(self):
        calls = []
        regdrift.llm_json = lambda p: calls.append(p) or {"concerns_setting": False}
        o = self.ob(parameter="capture_hedge_undertaking", summary="Banks must not recognize CVA hedges.")
        regdrift.verify_matches([o, dict(o), dict(o)])
        self.assertEqual(len(calls), 1)


class SpeculationGuards(unittest.TestCase):
    """The first real long circular (CVA capital rules, no matching bots) produced invented findings."""
    def setUp(self):
        self.orig = regdrift.llm_json

    def tearDown(self):
        regdrift.llm_json = self.orig

    def ob(self, summary, clause="Clause 29", **kw):
        return {"clause": clause, "summary": summary, "parameter": "p", "operator": "==", "required_value": True,
                "unit": "", "modality": "mandatory", **kw}

    def test_notices_about_the_circular_itself_are_not_bot_rules(self):
        meta = [self.ob("Paragraph 85(3) of the RBI Directions shall be repealed on 1 April 2027."),
                self.ob("The Directions shall become effective on or after 1 April 2027.", "Clause A. Short Title and Commencement (2)"),
                self.ob("The Directions shall apply to commercial banks.", "Clause B. Applicability (3)"),
                self.ob("These Directions shall come into force with immediate effect.")]
        real = [self.ob("Authorised Dealers must not permit users to rebook any cancelled contract.", "Clause (a)"),
                self.ob("Suspicious transaction reports shall be filed within 7 days.", "Clause 2"),
                self.ob("Authorised Dealers must obtain and retain an undertaking from the user.", "Clause (d)")]
        self.assertTrue(all(regdrift.is_meta_rule(o) for o in meta))
        self.assertFalse(any(regdrift.is_meta_rule(o) for o in real))

    def test_extract_drops_them(self):
        os.environ["GROQ_API_KEY"] = "x"
        try:
            raw = [{"clause": "Clause 29", "summary": "Paragraph 85(3) shall be repealed on 1 April 2027.", "parameter": "x",
                    "operator": "==", "required_value": "2027-04-01", "unit": "date"},
                   {"clause": "Clause 2", "summary": "STRs shall be filed within 3 days.", "parameter": "str_filing_days",
                    "operator": "<=", "required_value": 3, "unit": "days"}]
            regdrift.llm_json = lambda p: raw if "Extract machine-checkable" in p else {"concerns_setting": True}
            st = regdrift.extract({"circular_text": "9. Banks shall file STRs within 3 days. It shall be repealed.\n" * 3})
            self.assertEqual([o["parameter"] for o in st["obligations"]], ["str_filing_days"])
        finally:
            os.environ.pop("GROQ_API_KEY", None)

    def test_a_bot_must_share_real_words_with_the_rule(self):
        bots = {b["name"]: b for b in regdrift.load_bots()}
        repeal = "Paragraph 85(3) of the RBI Directions shall be repealed on 1 April 2027."
        for b in bots.values():
            self.assertLess(len(regdrift.shared_stems(repeal, b)), regdrift.MIN_SHARED_STEMS, b["name"])
        fx = "For collateralised OTC derivative transactions the effective maturity floor is at least one day."
        self.assertGreaterEqual(len(regdrift.shared_stems(fx, bots["FX-Derivative-Booking-Bot"])), regdrift.MIN_SHARED_STEMS)
        self.assertLess(len(regdrift.shared_stems(fx, bots["KYC-Refresh-Bot"])), regdrift.MIN_SHARED_STEMS)

    def test_retrieval_skips_bots_that_only_look_similar(self):
        os.environ["REGDRIFT_RETRIEVER"] = "bow"
        try:
            ob = self.ob("Reserve Bank directions on customer reporting shall be reviewed by regulated banks.", parameter="unmapped_x")
            got = regdrift.retrieve({"obligations": [ob]})["impacts"]
            for i in got:
                self.assertGreaterEqual(len(regdrift.shared_stems(ob["summary"], i["bot"])), regdrift.MIN_SHARED_STEMS)
        finally:
            os.environ.pop("REGDRIFT_RETRIEVER", None)

    def test_description_only_compliant_is_not_shown_as_compliant(self):
        os.environ["GROQ_API_KEY"] = "x"
        try:
            regdrift.llm_json = lambda p: {"status": "compliant", "confidence": 0.9, "reason": "Looks fine."}
            bot = next(b for b in regdrift.load_bots() if b["name"] == "Loan-Disbursal-Bot")
            j = regdrift.judge({"obligation": self.ob("Loans must be disbursed within 5 days.", parameter="unmapped_loan"), "bot": bot})
            self.assertEqual(j["status"], "not_applicable")
            self.assertIn("cannot be confirmed", j["reason"])
        finally:
            os.environ.pop("GROQ_API_KEY", None)

    def test_the_reasoning_prompt_forbids_guessing_cited_paragraphs(self):
        self.assertIn("never guess", regdrift.PROMPT_REASON)

