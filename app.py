"""RegDrift web app: review a circular, see which bots are out of line, decide, file Jira tickets.

Run locally:   python app.py            (then open http://localhost:8000)
Production:    gunicorn -b 0.0.0.0:7860 -w 1 --threads 4 app:app

Environment (all optional):
  GROQ_API_KEY, JIRA_URL, JIRA_EMAIL, JIRA_TOKEN, JIRA_PROJECT   as for the CLI
  REGDRIFT_TOKEN    access code required to create tickets or record feedback
  REGDRIFT_PUBLIC   "1" for a public demo: analysis only, no tickets or feedback unless REGDRIFT_TOKEN is set
  REGDRIFT_RATE     analyses allowed per client per hour (default 30, or 8 when public)
"""
import hmac
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque

from flask import Flask, jsonify, request, send_from_directory

import regdrift

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
CIRCULARS = os.path.join(HERE, "circulars")
FEEDBACK = os.path.join(HERE, "data", "feedback.jsonl")
MAX_UPLOAD = 5 * 1024 * 1024
MAX_CHARS = 120_000
STAGES = ["Reading the circular", "Extracting rules", "Finding affected bots", "Checking bot settings",
          "Preparing findings"]

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD + 64 * 1024

JOBS: dict = {}
JOB_ORDER: deque = deque()
RUN_LOCK = threading.Lock()   # one analysis at a time: the pipeline shares models and API quota
CURRENT = {"job": None}
HITS: dict = defaultdict(deque)


# ---------------------------------------------------------------- configuration helpers
def is_public() -> bool:
    return os.getenv("REGDRIFT_PUBLIC", "").strip() == "1"


def access_code():
    return os.getenv("REGDRIFT_TOKEN") or None


def writes_enabled() -> bool:
    """Creating tickets / recording feedback: always on locally, code-protected when a code is set, off for open demos."""
    return bool(access_code()) or not is_public()


def token_ok() -> bool:
    code = access_code()
    if not code:
        return writes_enabled()
    return hmac.compare_digest(request.headers.get("X-RegDrift-Token", ""), code)


def client_id() -> str:
    if is_public():
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.remote_addr or "local"


def rate_limited() -> bool:
    limit = int(os.getenv("REGDRIFT_RATE", "8" if is_public() else "30"))
    now, q = time.time(), HITS[client_id()]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


def bundled_circulars():
    if not os.path.isdir(CIRCULARS):
        return []
    return sorted(f for f in os.listdir(CIRCULARS)
                  if f.lower().endswith((".pdf", ".txt")) and f != "placeholder.txt")


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; script-src 'self'; frame-ancestors 'none'")
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def err(message, code=400):
    return jsonify({"error": message}), code


# ---------------------------------------------------------------- progress capture
class JobLog(logging.Handler):
    """Copies the pipeline's own log lines into the running job so the page can show live progress."""

    def emit(self, record):
        job = CURRENT["job"]
        if not job:
            return
        msg = record.getMessage().strip()
        job["log"].append(msg)
        m = re.match(r"\[(\d)/5\]", msg)
        if m:
            job["stage"] = min(int(m.group(1)), 5)


_handler = JobLog(level=logging.INFO)
regdrift.log.addHandler(_handler)
regdrift.log.setLevel(logging.INFO)


# ---------------------------------------------------------------- analysis
def drift_of(finding):
    """What the page needs to draw 'setting vs rule' for one finding."""
    ob = finding.get("obligation") or {}
    bots = {b["id"]: b for b in regdrift.load_bots()}
    bot = bots.get(finding.get("bot_id"), {})
    param = ob.get("parameter")
    has = param in (bot.get("parameters") or {})
    return {"parameter": param, "operator": ob.get("operator"), "required": ob.get("required_value"),
            "unit": ob.get("unit") or "", "modality": ob.get("modality"), "has_setting": has,
            "current": (bot.get("parameters") or {}).get(param) if has else None}


def present(state, existing):
    findings = []
    for i, f in enumerate(state.get("findings", [])):
        sev = regdrift.SEVERITY.get(f["status"], "OK")
        key = regdrift.ticket_key(f["bot"], f["clause"], f.get("ref"))
        findings.append({"id": i, "severity": sev, "status": f["status"], "bot": f["bot"], "bot_id": f["bot_id"],
                         "platform": f["platform"], "clause": f["clause"], "clause_text": f["rule"],
                         "reason": f["reason"], "confidence": f["confidence"], "drift": drift_of(f),
                         "ticket_summary": regdrift.ticket_summary(f) if sev != "OK" else None,
                         "in_jira": key in existing})
    order = {"High": 0, "Medium": 1, "Low": 2, "OK": 3}
    findings.sort(key=lambda x: (order[x["severity"]], x["bot"], x["clause"]))
    return {"circular_ref": state.get("circular_ref", ""), "extractor": state.get("extractor", ""),
            "model": regdrift.active_model(), "obligations": len(state.get("obligations", [])),
            "bots_checked": len(regdrift.load_bots()),
            "not_applicable": len(state.get("skipped", [])), "findings": findings}


def worker(job, text, source):
    job["status"] = "queued"
    with RUN_LOCK:
        CURRENT["job"] = job
        job["status"], job["stage"], started = "running", 0, time.time()
        try:
            state = regdrift.run({"circular_text": text, "source": source, "dry_run": True})
            existing = set()
            if regdrift.jira_ready():
                try:
                    existing = regdrift.jira_existing()
                except Exception as e:  # not fatal: only the "already in Jira" hint is lost
                    job["log"].append(f"Could not check Jira for existing tickets: {e}")
            job["result"] = present(state, existing)
            job["raw"] = {f["id"]: state["findings"][f["id"]] for f in job["result"]["findings"]}
            job["seconds"] = round(time.time() - started, 1)
            job["status"] = "done"
        except Exception as e:  # shown to the user, not hidden
            job["status"], job["error"] = "error", f"Analysis failed: {e}"
        finally:
            CURRENT["job"] = None


def start_job(text, source):
    job = {"id": uuid.uuid4().hex[:12], "source": source, "status": "queued", "stage": 0, "log": [],
           "created": time.time()}
    JOBS[job["id"]] = job
    JOB_ORDER.append(job["id"])
    while len(JOB_ORDER) > 40:
        JOBS.pop(JOB_ORDER.popleft(), None)
    threading.Thread(target=worker, args=(job, text, source), daemon=True).start()
    return job


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def assets(name):
    return send_from_directory(STATIC, name)


@app.get("/api/status")
def status():
    return jsonify({"version": regdrift.__version__, "model": regdrift.active_model(),
                    "llm": bool(os.getenv("GROQ_API_KEY")), "jira": regdrift.jira_ready(),
                    "writes": writes_enabled(), "needs_code": bool(access_code()), "public": is_public(),
                    "circulars": bundled_circulars(), "stages": STAGES})


@app.get("/api/bots")
def bots():
    return jsonify(regdrift.load_bots())


@app.post("/api/analyze")
def analyze():
    if rate_limited():
        return err("Too many analyses from this connection. Try again in an hour.", 429)
    upload = request.files.get("file")
    if upload and upload.filename:
        name = os.path.basename(upload.filename)
        ext = os.path.splitext(name)[1].lower()
        data = upload.read(MAX_UPLOAD + 1)
        if ext not in (".pdf", ".txt"):
            return err("Upload a PDF or a plain text file.")
        if len(data) > MAX_UPLOAD:
            return err("That file is over 5 MB. Upload the circular on its own.")
        if ext == ".pdf":
            if not data.startswith(b"%PDF"):
                return err("That file is not a valid PDF.")
            fd, path = tempfile.mkstemp(suffix=".pdf")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                text = regdrift.load_text(path)
            except Exception as e:
                return err(f"Could not read the PDF: {e}")
            finally:
                os.unlink(path)
        else:
            text = data.decode("utf-8", errors="replace")
    else:
        name = (request.get_json(silent=True) or {}).get("circular", "")
        if name not in bundled_circulars():
            return err("Choose one of the listed circulars or upload a file.")
        try:
            text = regdrift.load_text(os.path.join(CIRCULARS, name))
        except Exception as e:
            return err(f"Could not read {name}: {e}")
    text = text.strip()
    if len(text) < 40:
        return err("No readable text found. Scanned PDFs need OCR first; use a text-based PDF.")
    if len(text) > MAX_CHARS:
        return err("This document is too long for one analysis. Upload a single circular.")
    job = start_job(text, name)
    return jsonify({"job": job["id"]}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return err("That analysis is no longer available. Run it again.", 404)
    out = {k: job.get(k) for k in ("id", "source", "status", "stage", "error", "result", "seconds")}
    out["log"] = job["log"][-12:]
    return jsonify(out)


def _selected(job):
    ids = (request.get_json(silent=True) or {}).get("ids", [])
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        return None
    return [i for i in ids if i in job.get("raw", {})]


@app.post("/api/jobs/<job_id>/tickets")
def tickets(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return err("That analysis is no longer available. Run it again.", 404)
    ids = _selected(job)
    if ids is None:
        return err("Send the findings to file as a list of ids.")
    if not ids:
        return err("Select at least one finding.")
    rows, live = [], regdrift.jira_ready() and writes_enabled()
    if live and not token_ok():
        return err("Enter the access code to create tickets.", 403)
    existing = set()
    if live:
        try:
            existing = regdrift.jira_existing()
        except Exception:
            pass
    for i in ids:
        f = job["raw"][i]
        if f["status"] == "compliant":
            continue
        row = {"id": i, "summary": regdrift.ticket_summary(f)}
        if not live:
            row["state"] = "preview"
        elif regdrift.ticket_key(f["bot"], f["clause"], f.get("ref")) in existing:
            row["state"] = "exists"
        else:
            try:
                row.update(state="created", key=regdrift.jira_create(f))
                _feedback(f, "approve")
            except Exception as e:
                row.update(state="failed", error=str(e)[:200])
        rows.append(row)
    return jsonify({"live": live, "tickets": rows})


def _feedback(finding, decision):
    ob = finding.get("obligation")
    key = f"{finding['bot_id']}|{finding['clause']}|{finding['rule']}"
    os.makedirs(os.path.dirname(FEEDBACK), exist_ok=True)
    done = set()
    if os.path.exists(FEEDBACK):
        with open(FEEDBACK, encoding="utf-8") as fh:
            done = {json.loads(l).get("key") for l in fh if l.strip()}
    if key in done:
        return False
    with open(FEEDBACK, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "clause_text": finding["rule"], "obligation": ob, "decision": decision,
                             "corrected": None}) + "\n")
    return True


@app.post("/api/jobs/<job_id>/feedback")
def feedback(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return err("That analysis is no longer available. Run it again.", 404)
    if not writes_enabled():
        return err("Feedback is switched off on this demo.", 403)
    if not token_ok():
        return err("Enter the access code to record feedback.", 403)
    body = request.get_json(silent=True) or {}
    fid, decision = body.get("id"), body.get("decision")
    if decision not in ("approve", "reject") or fid not in job.get("raw", {}):
        return err("Unknown finding or decision.")
    return jsonify({"saved": _feedback(job["raw"][fid], decision)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False, threaded=True)
