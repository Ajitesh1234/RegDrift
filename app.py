"""RegDrift web app: a small Starlette API plus a hand-written front end (webapp/static).

    uvicorn webapp.app:app --port 8000

Safe to host publicly by design:
  * analysis never writes to Jira; ticket creation needs APP_ACCESS_CODE and a configured Jira
  * only an uploaded PDF/TXT, pasted text, or a file from the curated samples list is read (no URL fetching)
  * per-IP and daily limits protect the free Groq quota
"""
import hmac, json, os, re, sys, tempfile, threading, time, uuid
from collections import OrderedDict, defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import regdrift  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import JSONResponse, Response  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

ROOT = regdrift.HERE
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "6")) * 1024 * 1024
RATE_PER_HOUR = int(os.getenv("RATE_PER_HOUR", "10"))
MAX_PER_DAY = int(os.getenv("MAX_PER_DAY", "200"))
MAX_JOBS = 60
STAGES = [("ingest", regdrift.ingest), ("extract", regdrift.extract),
          ("retrieve", regdrift.retrieve), ("reason", regdrift.reason)]

JOBS: "OrderedDict[str, dict]" = OrderedDict()
LOCK = threading.Lock()
SLOTS = threading.BoundedSemaphore(int(os.getenv("MAX_CONCURRENT", "2")))
HITS = defaultdict(deque)
DAY = {"day": "", "n": 0}


def err(msg, code=400, **extra):
    return JSONResponse({"error": msg, **extra}, status_code=code)


def client_ip(request):
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")) or "?"


def over_limit(ip):
    now, today = time.time(), time.strftime("%Y-%m-%d")
    with LOCK:
        if DAY["day"] != today:
            DAY.update(day=today, n=0)
        q = HITS[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= RATE_PER_HOUR:
            return f"You have reached the limit of {RATE_PER_HOUR} checks per hour. Try again later."
        if DAY["n"] >= MAX_PER_DAY:
            return "This demo has reached its daily limit. Try again tomorrow."
        q.append(now)
        DAY["n"] += 1
    return None


# ---------------------------------------------------------------- samples
def samples():
    path = os.path.join(ROOT, "circulars", "samples.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return [s for s in json.load(fh) if os.path.isfile(os.path.join(ROOT, s["file"]))]
    d = os.path.join(ROOT, "circulars")
    names = sorted(f for f in os.listdir(d) if f.lower().endswith((".pdf", ".txt")) and f != "placeholder.txt")
    return [{"file": f"circulars/{n}", "label": os.path.splitext(n)[0].replace("_", " ")} for n in names]


# ---------------------------------------------------------------- pipeline job
def _public_finding(i, f):
    ob = f["obligation"]
    param = ob.get("parameter")
    has = param in f.get("_bot_params", {})
    return {"id": str(i), "bot": f["bot"], "bot_id": f["bot_id"], "platform": f["platform"], "clause": f["clause"],
            "rule": f["rule"], "status": f["status"], "severity": regdrift.SEVERITY.get(f["status"], "OK"),
            "confidence": f["confidence"], "reason": f["reason"], "modality": ob.get("modality"),
            "parameter": param, "operator": ob.get("operator"), "required": ob.get("required_value"),
            "unit": ob.get("unit", ""), "basis": "setting" if has else "description",
            "current": f["_bot_params"].get(param) if has else None}


def _run_job(job, text, name):
    try:
        state = {"circular_text": text, "source": name, "dry_run": True}
        for stage, fn in STAGES:
            job["stage"] = stage
            state = {**state, **fn(state)}
        bots = {b["id"]: b for b in regdrift.load_bots()}
        rows = []
        for f in state["findings"]:
            rows.append({**f, "_bot_params": bots[f["bot_id"]]["parameters"], "ref": state.get("circular_ref", "")})
        order = {"violating": 0, "at_risk": 1, "optional_gap": 2, "compliant": 3}
        rows.sort(key=lambda r: (order.get(r["status"], 9), r["bot"], r["clause"]))
        report = regdrift.build_report(state, name)
        job["findings"] = rows
        job["result"] = {
            "source": name, "circular_ref": state.get("circular_ref", ""), "chars": len(state["circular_text"]),
            "sha256": report["circular_sha256"], "extractor": state.get("extractor", ""),
            "model": report["llm_model"], "run_at": report["run_at_utc"],
            "obligations": state.get("obligations", []),
            "context": state.get("context") or {},
            "skipped_not_applicable": len(state.get("skipped", [])),
            "findings": [_public_finding(i, r) for i, r in enumerate(rows)],
        }
        job["status"] = "done"
    except Exception as e:  # shown to the user, so keep it readable
        job["status"], job["error"] = "error", str(e)[:300]
    finally:
        SLOTS.release()


async def analyze(request):
    ip = client_ip(request)
    ctype = request.headers.get("content-type", "")
    name, text = "", ""
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        try:
            return await _from_form(form, ip)
        finally:
            await form.close()
    else:
        body = await request.json() if "json" in ctype else {}
        if body.get("sample"):
            return await _sample(str(body["sample"]), ip)
        if body.get("text"):
            name, text = "pasted text", str(body["text"])[:200_000]
    if not text.strip():
        return err("Add a circular first: upload a file, pick a sample, or paste text.")
    return _start(text, name, ip)


async def _from_form(form, ip):
    name, text = "", ""
    if True:
        up = form.get("file")
        if up is not None and hasattr(up, "read"):
            data = await up.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                return err(f"That file is larger than {MAX_UPLOAD // (1024 * 1024)} MB. Upload a smaller circular.", 413)
            name = os.path.basename(up.filename or "upload")[:120]
            if data[:5] == b"%PDF-":
                fd, tmp = tempfile.mkstemp(suffix=".pdf")
                try:
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                    text = await run_in_threadpool(regdrift.load_text, tmp)
                except Exception:
                    return err("Could not read that PDF. Check that it is not password protected.", 422)
                finally:
                    os.unlink(tmp)
            else:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    return err("Upload a PDF or a UTF-8 text file.", 415)
        elif form.get("sample"):
            return await _sample(str(form["sample"]), ip)
        elif form.get("text"):
            name, text = "pasted text", str(form["text"])[:200_000]
    if not text.strip():
        return err("Add a circular first: upload a file, pick a sample, or paste text.")
    return _start(text, name, ip)


async def _sample(file, ip):
    match = next((s for s in samples() if s["file"] == file), None)
    if not match:
        return err("That sample is not available.", 404)
    text = await run_in_threadpool(regdrift.load_text, os.path.join(ROOT, match["file"]))
    return _start(text, os.path.basename(match["file"]), ip)


def _start(text, name, ip):
    if not SLOTS.acquire(blocking=False):
        return err("The checker is busy with other requests. Try again in a minute.", 429)
    limit = over_limit(ip)
    if limit:
        SLOTS.release()
        return err(limit, 429)
    jid = uuid.uuid4().hex[:16]
    job = {"id": jid, "status": "running", "stage": "ingest", "created": time.time(), "tickets": {}}
    with LOCK:
        JOBS[jid] = job
        while len(JOBS) > MAX_JOBS:
            JOBS.popitem(last=False)
    threading.Thread(target=_run_job, args=(job, text, name), daemon=True).start()
    return JSONResponse({"job_id": jid})


async def job_status(request):
    job = JOBS.get(request.path_params["jid"])
    if not job:
        return err("That check has expired. Run it again.", 404)
    out = {"status": job["status"], "stage": job["stage"], "tickets": job["tickets"]}
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "error":
        out["error"] = job["error"]
    return JSONResponse(out)


# ---------------------------------------------------------------- tickets (guarded)
def _code_ok(request):
    expected = os.getenv("APP_ACCESS_CODE", "")
    given = request.headers.get("x-access-code", "")
    return bool(expected) and hmac.compare_digest(expected.encode(), given.encode())


def ticketing_enabled():
    return bool(os.getenv("APP_ACCESS_CODE")) and regdrift.jira_ready()


async def unlock(request):
    if not ticketing_enabled():
        return err("Ticketing is turned off on this deployment.", 403)
    return Response(status_code=204) if _code_ok(request) else err("That access code is not right.", 403)


def _create_tickets(job, ids):
    existing = regdrift.jira_existing()
    out = {}
    by_id = {str(i): f for i, f in enumerate(job["findings"])}
    for fid in ids:
        f = by_id.get(str(fid))
        if not f or f["status"] not in regdrift.SEVERITY:
            continue
        if fid in job["tickets"]:
            out[fid] = job["tickets"][fid]
            continue
        if regdrift.ticket_key(f["bot"], f["clause"], f.get("ref")) in existing:
            out[fid] = {"state": "duplicate"}
        else:
            try:
                out[fid] = {"state": "created", "key": regdrift.jira_create(f)}
            except Exception as e:
                out[fid] = {"state": "failed", "error": str(e)[:160]}
        job["tickets"][fid] = out[fid]
    return out


async def create_tickets(request):
    if not ticketing_enabled():
        return err("Ticketing is turned off on this deployment.", 403)
    if not _code_ok(request):
        return err("That access code is not right.", 403)
    job = JOBS.get(request.path_params["jid"])
    if not job or job["status"] != "done":
        return err("That check has expired. Run it again.", 404)
    ids = [str(i) for i in (await request.json()).get("ids", [])][:100]
    return JSONResponse({"tickets": await run_in_threadpool(_create_tickets, job, ids)})


# ---------------------------------------------------------------- read-only info
async def status(request):
    return JSONResponse({"version": regdrift.__version__, "model": regdrift.active_model(),
                         "llm": bool(os.getenv("GROQ_API_KEY")), "ticketing": ticketing_enabled(),
                         "jira_base": (os.getenv("JIRA_URL") or "").rstrip("/") if ticketing_enabled() else "",
                         "rate_per_hour": RATE_PER_HOUR})


async def list_samples(request):
    return JSONResponse(samples())


async def list_bots(request):
    return JSONResponse(regdrift.load_bots())


async def health(request):
    return JSONResponse({"ok": True})


class Headers(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        r = await call_next(request)
        r.headers.update({
            "Content-Security-Policy": "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
                                       "font-src https://fonts.gstatic.com; img-src 'self' data:; "
                                       "frame-ancestors 'none'; base-uri 'none'",
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"})
        if request.url.path.startswith("/api/"):
            r.headers["Cache-Control"] = "no-store"
        return r


app = Starlette(
    routes=[Route("/api/status", status), Route("/api/samples", list_samples), Route("/api/bots", list_bots),
            Route("/api/analyze", analyze, methods=["POST"]), Route("/api/jobs/{jid}", job_status),
            Route("/api/jobs/{jid}/tickets", create_tickets, methods=["POST"]),
            Route("/api/unlock", unlock, methods=["POST"]), Route("/healthz", health),
            Mount("/", StaticFiles(directory=STATIC, html=True))],
    middleware=[Middleware(Headers)])
