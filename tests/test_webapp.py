import json, os, socket, sys, threading, time, unittest, urllib.error, urllib.request, uuid
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for k in ("GROQ_API_KEY", "LORA_ADAPTER", "JIRA_URL", "JIRA_EMAIL", "JIRA_TOKEN", "JIRA_PROJECT", "APP_ACCESS_CODE"):
    os.environ.pop(k, None)
os.environ["RATE_PER_HOUR"] = "1000"
try:
    import uvicorn, multipart  # noqa: F401
    from webapp import app as webapp
except ImportError as e:  # optional dependency: skip instead of failing the whole run
    raise unittest.SkipTest(f"web dependencies not installed ({e}); pip install -r requirements-web.txt")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Server:
    def __enter__(self):
        self.port = free_port()
        self.srv = uvicorn.Server(uvicorn.Config(webapp.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.t = threading.Thread(target=self.srv.run, daemon=True); self.t.start()
        for _ in range(100):
            if self.srv.started: break
            time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self.srv.should_exit = True; self.t.join(5)

    def call(self, path, data=None, headers=None, method=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def post_json(self, path, obj, headers=None):
        return self.call(path, json.dumps(obj).encode(), {"content-type": "application/json", **(headers or {})}, "POST")

    def upload(self, name, content, ctype="application/octet-stream"):
        b = uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode() + content + f"\r\n--{b}--\r\n".encode()
        return self.call("/api/analyze", body, {"content-type": f"multipart/form-data; boundary={b}"}, "POST")

    def wait_job(self, jid, secs=40):
        end = time.time() + secs
        while time.time() < end:
            _, body, _ = self.call(f"/api/jobs/{jid}")
            j = json.loads(body)
            if j["status"] != "running":
                return j
            time.sleep(0.2)
        self.fail("job did not finish")


class WebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Server().__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.s.__exit__()

    def test_static_page_and_security_headers(self):
        code, body, h = self.s.call("/")
        self.assertEqual(code, 200); self.assertIn(b"RegDrift", body)
        h = {k.lower(): v for k, v in h.items()}
        self.assertIn("default-src 'self'", h["content-security-policy"])
        self.assertEqual(h["x-content-type-options"], "nosniff")

    def test_info_endpoints(self):
        st = json.loads(self.s.call("/api/status")[1])
        self.assertFalse(st["ticketing"]); self.assertFalse(st["llm"])
        self.assertTrue(json.loads(self.s.call("/api/samples")[1]))
        self.assertGreaterEqual(len(json.loads(self.s.call("/api/bots")[1])), 4)

    def test_sample_run_finds_drift(self):
        code, body, _ = self.s.post_json("/api/analyze", {"sample": "circular_sample.txt"})
        self.assertEqual(code, 200)
        j = self.s.wait_job(json.loads(body)["job_id"])
        self.assertEqual(j["status"], "done")
        f = {(x["bot"], x["clause"]): x for x in j["result"]["findings"]}
        self.assertEqual(f[("KYC-Refresh-Bot", "Clause 1")]["status"], "violating")
        self.assertEqual(f[("KYC-Refresh-Bot", "Clause 1")]["current"], 3)
        self.assertEqual(f[("STR-Filer-Bot", "Clause 2")]["status"], "compliant")

    def test_text_and_file_upload(self):
        txt = b"1. Suspicious transaction reports shall be filed within 5 days of arriving at a conclusion.\n"
        code, body, _ = self.s.upload("c.txt", txt, "text/plain")
        self.assertEqual(code, 200)
        j = self.s.wait_job(json.loads(body)["job_id"])
        self.assertEqual(j["status"], "done")
        self.assertTrue(any(x["bot"] == "STR-Filer-Bot" and x["status"] == "violating" for x in j["result"]["findings"]))

    def test_bad_inputs_are_rejected_with_plain_messages(self):
        self.assertEqual(self.s.upload("x.zip", b"PK\x03\x04\xff\xfe")[0], 415)
        self.assertEqual(self.s.post_json("/api/analyze", {})[0], 400)
        self.assertEqual(self.s.post_json("/api/analyze", {"sample": "../../etc/passwd"})[0], 404)
        self.assertEqual(self.s.call("/api/jobs/nope")[0], 404)
        code, body, _ = self.s.upload("tiny.txt", b"too short")
        j = self.s.wait_job(json.loads(body)["job_id"])
        self.assertEqual(j["status"], "error"); self.assertIn("readable text", j["error"])

    def test_ticketing_is_locked_without_access_code(self):
        self.assertEqual(self.s.post_json("/api/unlock", {})[0], 403)
        self.assertEqual(self.s.post_json("/api/jobs/x/tickets", {"ids": ["0"]})[0], 403)

    def test_rate_limit(self):
        old = webapp.RATE_PER_HOUR
        webapp.RATE_PER_HOUR = 2
        try:
            webapp.HITS.pop("9.9.9.9", None)
            self.assertIsNone(webapp.over_limit("9.9.9.9")); self.assertIsNone(webapp.over_limit("9.9.9.9"))
            self.assertIn("limit", webapp.over_limit("9.9.9.9"))
        finally:
            webapp.RATE_PER_HOUR = old


class TicketGuard(unittest.TestCase):
    def test_wrong_and_right_code(self):
        os.environ.update(APP_ACCESS_CODE="s3cret", JIRA_URL="https://x.invalid", JIRA_EMAIL="a@b.c", JIRA_TOKEN="t", JIRA_PROJECT="P")
        try:
            with Server() as s:
                self.assertEqual(s.post_json("/api/unlock", {}, {"x-access-code": "nope"})[0], 403)
                self.assertEqual(s.post_json("/api/unlock", {}, {"x-access-code": "s3cret"})[0], 204)
                self.assertTrue(json.loads(s.call("/api/status")[1])["ticketing"])
        finally:
            for k in ("APP_ACCESS_CODE", "JIRA_URL", "JIRA_EMAIL", "JIRA_TOKEN", "JIRA_PROJECT"):
                os.environ.pop(k, None)


class TicketFlow(unittest.TestCase):
    def test_create_duplicate_and_no_double_create(self):
        import regdrift
        env = dict(APP_ACCESS_CODE="code", JIRA_URL="https://x.invalid", JIRA_EMAIL="a@b.c", JIRA_TOKEN="t", JIRA_PROJECT="P")
        os.environ.update(env)
        orig = (regdrift.jira_existing, regdrift.jira_create)
        made = []
        regdrift.jira_existing = lambda: {regdrift.ticket_key("Sanctions-Screening-Bot", "Clause 3", "DEMO/2026-27/101")}
        regdrift.jira_create = lambda f: (made.append(f["bot"]), f"P-{len(made)}")[1]
        try:
            with Server() as s:
                j = s.wait_job(json.loads(s.post_json("/api/analyze", {"sample": "circular_sample.txt"})[1])["job_id"])
                ids = [f["id"] for f in j["result"]["findings"] if f["status"] == "violating"]
                self.assertEqual(len(ids), 2)
                jid = json.loads(s.post_json("/api/analyze", {"sample": "circular_sample.txt"})[1])["job_id"]
                s.wait_job(jid)
                code, body, _ = s.post_json(f"/api/jobs/{jid}/tickets", {"ids": ids}, {"x-access-code": "code"})
                self.assertEqual(code, 200)
                states = sorted(v["state"] for v in json.loads(body)["tickets"].values())
                self.assertEqual(states, ["created", "duplicate"])
                s.post_json(f"/api/jobs/{jid}/tickets", {"ids": ids}, {"x-access-code": "code"})
                self.assertEqual(made, ["KYC-Refresh-Bot"])          # clicking twice never creates twice
                self.assertEqual(s.post_json(f"/api/jobs/{jid}/tickets", {"ids": ids}, {"x-access-code": "bad"})[0], 403)
        finally:
            regdrift.jira_existing, regdrift.jira_create = orig
            for k in env:
                os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
