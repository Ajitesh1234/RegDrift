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

if __name__ == "__main__":
    unittest.main()
