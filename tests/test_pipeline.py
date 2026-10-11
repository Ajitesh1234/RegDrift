import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for k in ("GROQ_API_KEY", "LORA_ADAPTER", "JIRA_URL"): os.environ.pop(k, None)
import regdrift

def flagged(text):
    return {(t["bot"], t["clause"]): t for t in regdrift.run({"circular_text": text})["tickets"]}

class Pipeline(unittest.TestCase):
    def test_sample_circular(self):
        f = flagged(open(os.path.join(regdrift.HERE, "circular_sample.txt")).read())
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

if __name__ == "__main__":
    unittest.main()
