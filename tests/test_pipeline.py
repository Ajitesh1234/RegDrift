import os, sys, tempfile, unittest
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

if __name__ == "__main__":
    unittest.main()
