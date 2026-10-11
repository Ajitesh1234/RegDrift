"""RegDrift: regulator circular -> affected RPA bots -> Jira tickets.

Runs fully offline (rule-based fallbacks). Add free keys to upgrade:
  GROQ_API_KEY            -> LLM extraction + reasoning (free tier)
  JIRA_URL, JIRA_EMAIL, JIRA_TOKEN, JIRA_PROJECT -> real Jira tickets (free plan)
Optional libs: langgraph (graph runtime), chromadb (vector DB).
"""
import json, math, os, re, sys, base64, urllib.request
from collections import Counter
from typing import TypedDict, List, Dict, Any

HERE = os.path.dirname(os.path.abspath(__file__))

class State(TypedDict, total=False):
    circular_text: str
    circular_ref: str
    obligations: List[Dict[str, Any]]
    impacts: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    tickets: List[Dict[str, Any]]

# ---------- LLM helper (Groq free tier, OpenAI-compatible) ----------
def llm_json(prompt: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        body = json.dumps({
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions", body,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
             "User-Agent": "regdrift/1.0 (+https://github.com/Ajitesh1234/RegDrift)", "Accept": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=60))
        text = out["choices"][0]["message"]["content"]
        m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
        return json.loads(m.group(1)) if m else None
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try: body = " | " + e.read().decode("utf-8", "ignore")[:300]
            except Exception: pass
        print(f"  [llm failed, using fallback: {e}{body}]")
        return None

# ---------- Optional: your LoRA-tuned extractor (set LORA_ADAPTER=path_or_hf_id) ----------
_lora = {}
LORA_PROMPT = "### Instruction:\nExtract measurable obligations as a JSON list.\n### Clause:\n{clause}\n### Output:\n"

def lora_extract(clause_text: str):
    path = os.getenv("LORA_ADAPTER")
    if not path:
        return None
    try:
        if "m" not in _lora:
            import torch
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
            _lora["t"] = AutoTokenizer.from_pretrained(path)
            _lora["m"] = AutoPeftModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None)
        t, m = _lora["t"], _lora["m"]
        ids = t(LORA_PROMPT.format(clause=clause_text), return_tensors="pt").to(m.device)
        out = m.generate(**ids, max_new_tokens=160, do_sample=False)
        txt = t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        mm = re.search(r"\[.*?\]", txt, re.S)
        return json.loads(mm.group(0)) if mm else None
    except Exception as e:
        print(f"  [lora failed, using fallback: {e}]")
        return None

# ---------- Node 1: ingest ----------
def load_text(src: str) -> str:
    """Read a .txt, .pdf or http(s) URL (txt/pdf) into plain text."""
    if src.startswith("http"):
        import tempfile
        data = urllib.request.urlopen(urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
        if data[:4] == b"%PDF":
            f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False); f.write(data); f.close()
            return load_text(f.name)
        return data.decode("utf-8", "ignore")
    if src.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(src).pages)
    return open(src, encoding="utf-8").read()

NUM_START = r"(?:(?:Clause\s+)?\d+\s*[.:)]|\([a-z]\))\s"

CLAUSE_RE = re.compile(r"(?m)^\s*(?:(?:Clause\s+)?(\d+)\s*[.:)]|(\([a-z]\)))\s+(.+)")

def iter_clauses(text):
    for m in CLAUSE_RE.finditer(text):
        yield f"Clause {m.group(1) or m.group(2)}", m.group(3)

def clean(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\r", "")
    text = "\n".join(l for l in text.split("\n") if not l.strip() or sum(ord(c) > 127 for c in l) / len(l) < 0.2)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)               # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n(?!\s*(?:%s|\n))" % NUM_START, " ", text)  # re-join wrapped lines
    return re.sub(r"\n{2,}", "\n", text).strip()

def ingest(state: State) -> State:
    text = state.get("circular_text") or open(os.path.join(HERE, "circular_sample.txt"), encoding="utf-8").read()
    text = clean(text)
    m = re.search(r"\b[A-Z]{3,4}/\d{4}-\d{2,4}/\d+", text)
    ref = m.group(0) if m else ""
    print(f"[1/5] Ingested circular ({len(text)} chars)")
    return {"circular_text": text, "circular_ref": ref}

# ---------- Node 2: obligation extractor ----------
PROMPT_EXTRACT = """Extract machine-checkable obligations from this regulatory text.
Return ONLY a JSON list. Each item: {{"clause": str (e.g. "Clause (a)"), "summary": str (one sentence),
"parameter": snake_case name, "operator": "<=", ">=" or "==", "required_value": number or true/false, "unit": str}}.
Use "==" with true/false for yes/no rules (e.g. a thing must or must not be allowed/captured).
Where the rule concerns one of these existing bot settings, reuse that exact parameter name:
{params}
If the circular changes an old limit to a new one, use the NEW value. Skip clauses with no checkable rule.
TEXT:
{text}"""

# Fallback keyword map (stand-in for the LoRA-tuned model until you train it)
FALLBACK_RULES = [
    (r"re-verif", "kyc_reverification_years_high_risk", "<=", "years"),
    (r"suspicious transaction", "str_filing_days", "<=", "days"),
    (r"sanctions", "sanctions_screening_frequency_days", "<=", "days"),
]

def known_params():
    with open(os.path.join(HERE, "bots.json")) as fh:
        return sorted({p for b in json.load(fh) for p in b["parameters"]})

def extract(state: State) -> State:
    obs = None
    if os.getenv("LORA_ADAPTER"):
        obs = []
        for clause, body in iter_clauses(state["circular_text"]):
            for o in lora_extract(body) or []:
                obs.append({"clause": clause, "summary": body, **o})
    if not obs:
        obs = llm_json(PROMPT_EXTRACT.format(text=state["circular_text"], params=known_params()))
    if not obs:
        obs = []
        for clause, body in iter_clauses(state["circular_text"]):
            for pat, param, op, unit in FALLBACK_RULES:
                if re.search(pat, body, re.I):
                    nums = re.findall(r"(\d+)\s*(?:years?|days?)", body)
                    if nums:
                        obs.append({"clause": clause, "summary": body, "parameter": param,
                                    "operator": op, "required_value": float(nums[-1]), "unit": unit})
                    break
    print(f"[2/5] Extracted {len(obs)} measurable obligations")
    return {"obligations": obs}

# ---------- Node 3: impact analyzer (RAG) ----------
def tok(s): return re.findall(r"[a-z]+", s.lower())

def cosine(a: Counter, b: Counter):
    num = sum(a[t] * b[t] for t in a)
    den = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return num / den if den else 0.0

def bot_doc(b): return f"{b['name']} {b['description']} {' '.join(b['tags'])}"

def retrieve_chroma(bots, query, k):
    import chromadb  # default embedding model downloads once, then runs locally
    col = chromadb.Client().get_or_create_collection("bots", metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        col.add(ids=[b["id"] for b in bots], documents=[bot_doc(b) for b in bots])
    res = col.query(query_texts=[query], n_results=k)
    return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]  # cosine distance -> similarity

def retrieve_bow(bots, query, k):
    q = Counter(tok(query))
    scored = sorted(((b["id"], cosine(q, Counter(tok(bot_doc(b))))) for b in bots),
                    key=lambda x: -x[1])
    return scored[:k]

SIM_THRESHOLD = {"chroma": float(os.getenv("CHROMA_MIN_SIM", "0.35")), "bow": 0.12}

def retrieve(state: State) -> State:
    with open(os.path.join(HERE, "bots.json")) as fh:
        bots = json.load(fh)
    by_id = {b["id"]: b for b in bots}
    impacts = []
    for ob in state["obligations"]:
        hits, kind = {}, "chroma"
        # (a) exact structured match: a bot that exposes the rule's parameter is always affected
        for b in bots:
            if ob.get("parameter") in b["parameters"]:
                hits[b["id"]] = 1.0
        # (b) semantic retrieval for bots without that parameter
        try:
            found = retrieve_chroma(bots, ob["summary"], 3)
        except Exception:
            found, kind = retrieve_bow(bots, ob["summary"], 3), "bow"
        for bid, score in found:
            if score >= SIM_THRESHOLD[kind]:
                hits.setdefault(bid, score)
        for bid, score in hits.items():
            impacts.append({"obligation": ob, "bot": by_id[bid], "score": round(score, 2)})
    print(f"[3/5] Retrieved {len(impacts)} (rule, bot) pairs")
    return {"impacts": impacts}

# ---------- Node 4: gap reasoner ----------
PROMPT_REASON = """Rule: {rule}
Bot: {bot} ({platform}) - {desc}
Bot parameters: {params}
First decide if this rule concerns what the bot actually does. If the bot's job is unrelated to the
rule, answer not_applicable. Otherwise judge compliance. Return ONLY JSON:
{{"status": "not_applicable"|"compliant"|"at_risk"|"violating", "confidence": 0-1, "reason": str}}"""

def judge(imp):
    ob, bot = imp["obligation"], imp["bot"]
    param = ob["parameter"]
    if param in bot["parameters"]:  # deterministic check on structured data first
        cur, req = bot["parameters"][param], ob["required_value"]
        op = ob["operator"]
        ok = cur <= req if op == "<=" else cur >= req if op == ">=" else cur == req
        return {"status": "compliant" if ok else "violating", "confidence": 0.95,
                "reason": " ".join(f"{param} is {cur} {ob.get('unit') or ''}, rule requires {op} {req}.".split()).replace(" ,", ",")}
    r = llm_json(PROMPT_REASON.format(rule=ob["summary"], bot=bot["name"], platform=bot["platform"],
                                      desc=bot["description"], params=bot["parameters"]))
    if isinstance(r, dict) and r.get("status") in ("not_applicable", "compliant", "at_risk", "violating"):
        return r
    return {"status": "at_risk", "confidence": 0.5,
            "reason": f"Bot touches this area but exposes no '{param}' setting; needs manual review."}

def reason(state: State) -> State:
    findings = []
    for imp in state["impacts"]:
        j = judge(imp)
        if j["status"] == "not_applicable":
            print(f"  (skipped {imp['bot']['name']} / {imp['obligation']['clause']}: not applicable)")
            continue
        findings.append({"bot_id": imp["bot"]["id"], "bot": imp["bot"]["name"],
                         "platform": imp["bot"]["platform"], "clause": imp["obligation"]["clause"],
                         "rule": imp["obligation"]["summary"], "obligation": imp["obligation"], "ref": state.get("circular_ref", ""), **j})
    print(f"[4/5] Judged {len(findings)} findings")
    return {"findings": findings}

# ---------- Node 5: action agent (Jira) ----------
SEVERITY = {"violating": "High", "at_risk": "Medium"}

def jira_ready():
    return all(os.getenv(k) for k in ("JIRA_URL", "JIRA_EMAIL", "JIRA_TOKEN", "JIRA_PROJECT"))

def jira_request(path, body=None):
    url, email, tok_ = (os.getenv(k) for k in ("JIRA_URL", "JIRA_EMAIL", "JIRA_TOKEN"))
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode() if body else None, {
        "Content-Type": "application/json", "Accept": "application/json", "User-Agent": "regdrift/1.0",
        "Authorization": "Basic " + base64.b64encode(f"{email}:{tok_}".encode()).decode()})
    return json.load(urllib.request.urlopen(req, timeout=30))

def ticket_summary(f):
    ref = f" ({f['ref']})" if f.get("ref") else ""
    return f"[RegDrift] {f['bot']} {f['status']} - {f['clause']}{ref}"

def jira_existing():
    """Summaries of RegDrift tickets already in the project (so re-runs do not create duplicates)."""
    from urllib.parse import quote
    jql = quote(f'project = "{os.getenv("JIRA_PROJECT")}" AND summary ~ "RegDrift"')
    found, token = set(), ""
    for _ in range(10):
        r = jira_request(f"/rest/api/3/search/jql?jql={jql}&fields=summary&maxResults=100" + (f"&nextPageToken={token}" if token else ""))
        found |= {i["fields"]["summary"] for i in r.get("issues", [])}
        token = r.get("nextPageToken", "")
        if not token or r.get("isLast"): break
    return found

def jira_create(f):
    text = (f"{f['clause']}: {f['rule']}\n\nStatus: {f['status']} (confidence {f['confidence']})\n"
            f"Reason: {f['reason']}\nBot: {f['bot']} [{f['platform']}]\nCircular: {f.get('ref') or 'n/a'}")
    body = {"fields": {"project": {"key": os.getenv("JIRA_PROJECT")}, "issuetype": {"name": "Task"},
            "summary": ticket_summary(f), "labels": ["regdrift"],
            "description": {"type": "doc", "version": 1, "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": text}]}]}}}
    return jira_request("/rest/api/3/issue", body)["key"]

def act(state: State) -> State:
    tickets, live, existing = [], jira_ready(), set()
    if live:
        try:
            existing = jira_existing()
        except Exception as e:
            print(f"  [could not check Jira for existing tickets, duplicates possible: {e}]")
    for f in state["findings"]:
        if f["status"] == "compliant":
            continue
        t = {"severity": SEVERITY[f["status"]], **f, "decision": None}
        if live and ticket_summary(f) in existing:
            t["jira_key"] = "SKIPPED (already in Jira)"
        else:
            try:
                t["jira_key"] = jira_create(f) if live else "DRY-RUN"
            except Exception as e:
                t["jira_key"] = f"FAILED: {e}"
        tickets.append(t)
    with open(os.path.join(HERE, "tickets.json"), "w") as fh:
        json.dump(tickets, fh, indent=2)
    new = sum(1 for t in tickets if t["jira_key"] not in ("DRY-RUN",) and not t["jira_key"].startswith(("SKIPPED", "FAILED")))
    skipped = sum(1 for t in tickets if t["jira_key"].startswith("SKIPPED"))
    print(f"[5/5] {len(tickets)} findings: {new} new Jira tickets, {skipped} already existed" if live
          else f"[5/5] {len(tickets)} findings (dry-run, saved to tickets.json)")
    return {"tickets": tickets}

# ---------- Graph ----------
NODES = [("ingest", ingest), ("extract", extract), ("retrieve", retrieve), ("reason", reason), ("act", act)]

def run(state: State) -> State:
    try:
        from langgraph.graph import StateGraph, END
        g = StateGraph(State)
        for n, fn in NODES: g.add_node(n, fn)
        g.set_entry_point("ingest")
        for (a, _), (b, _) in zip(NODES, NODES[1:]): g.add_edge(a, b)
        g.add_edge("act", END)
        return g.compile().invoke(state)
    except ImportError:
        print("(langgraph not installed, running nodes sequentially)")
        for _, fn in NODES: state = {**state, **fn(state)}
        return state

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    final = run({"circular_text": load_text(path)} if path else {})
    print("\n=== RESULTS ===")
    for t in final["tickets"]:
        print(f"{t['severity']:6} {t['bot']:24} {t['clause']}: {t['reason']}  [{t['jira_key']}]")
