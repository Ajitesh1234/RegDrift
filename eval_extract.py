"""Measure the rule-extraction step against a hand-labelled set.

    python eval_extract.py                      # run the extractor once on every labelled circular
    python eval_extract.py --repeat 5           # run 5 times: shows how steady the AI is, not just how good
    python eval_extract.py --predictions p.json # score saved output, no network (see --save)
    python eval_extract.py --save p.json        # also write what the extractor returned

With GROQ_API_KEY set the AI reads each circular; without it only the built-in keyword rules run, which is
the baseline. Each circular lists rules that must be found, rules that are acceptable, and things that must never
be reported (commencement wording, the OLD value that a circular replaces, legal-powers boilerplate).

Reported per run: required-rule recall, must-not violations, and extra rules nobody labelled (to read, not a score).
Exit status is 1 if any run has a must-not violation, so it can gate a change; recall is printed, not gated, because
the AI's answers vary.
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "eval", "extract_gold.json")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _haystack(ob: dict) -> str:
    parts = [ob.get("summary", ""), ob.get("clause", ""), str(ob.get("parameter", "")).replace("_", " ")]
    return " ".join(str(p) for p in parts).lower().replace("‑", "-")


def _numbers(v):
    if isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(m.replace(",", "")) for m in NUM_RE.findall(str(v)) if m.replace(",", "").replace(".", "", 1).lstrip("-").isdigit()]


def matches(spec: dict, ob: dict) -> bool:
    """Does this extracted obligation satisfy the labelled rule or must-not entry?"""
    text = _haystack(ob)
    if not all(any(w in text for w in group) for group in spec.get("all", [])):
        return False
    want = spec.get("value")
    if want is not None:
        got = _numbers(ob.get("required_value"))
        if not any(abs(g - w) < 1e-9 for g in got for w in want):
            return False
    return True


def score_circular(case: dict, obligations: list) -> dict:
    found, missed, optional_hit = [], [], []
    for r in case["rules"]:
        hit = any(matches(r, o) for o in obligations)
        if r.get("required"):
            (found if hit else missed).append(r["id"])
        elif hit:
            optional_hit.append(r["id"])
    violations = [{"id": m["id"], "rule": o.get("summary", ""), "value": o.get("required_value")}
                  for m in case.get("must_not", []) for o in obligations if matches(m, o)]
    labelled = case["rules"] + case.get("must_not", [])
    extras = [o.get("summary", "") for o in obligations if not any(matches(s, o) for s in labelled)]
    req = len(found) + len(missed)
    return {"id": case["id"], "title": case.get("title", ""), "found": found, "missed": missed,
            "optional_hit": optional_hit, "violations": violations, "extras": extras,
            "recall": (len(found) / req) if req else None, "reported": len(obligations)}


def load_gold(path: str = GOLD) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _text(case: dict, base: str) -> str:
    with open(os.path.join(base, case["text"]), encoding="utf-8") as fh:
        return fh.read()


def run_extractor(case: dict, base: str) -> list:
    import regdrift
    if hasattr(regdrift, "start_job_clock"):
        regdrift.start_job_clock(1e9)  # no time limit for a measurement run
    st = regdrift.extract({"circular_text": _text(case, base)})
    return st["obligations"]


def summarise(runs: list) -> dict:
    """runs: list of {circular_id: score} dicts, one per repeat."""
    total_req = sum(len(s["found"]) + len(s["missed"]) for r in runs for s in r.values())
    total_found = sum(len(s["found"]) for r in runs for s in r.values())
    rule_hits = {}
    for r in runs:
        for cid, s in r.items():
            for k in s["found"]:
                rule_hits.setdefault((cid, k), [0, 0])[0] += 1
                rule_hits[(cid, k)][1] += 1
            for k in s["missed"]:
                rule_hits.setdefault((cid, k), [0, 0])[1] += 1
    return {"runs": len(runs), "recall": (total_found / total_req) if total_req else None,
            "violations": sum(len(s["violations"]) for r in runs for s in r.values()),
            "rule_hit_rate": {f"{c} / {k}": h / n for (c, k), (h, n) in sorted(rule_hits.items())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--predictions", help="score saved extractor output instead of running it")
    ap.add_argument("--save", help="write the extractor output to this file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    gold = load_gold(args.gold)
    base = os.path.dirname(args.gold)
    cases = gold["circulars"]
    runs, saved = [], []
    if args.predictions:
        with open(args.predictions, encoding="utf-8") as fh:
            preds = json.load(fh)
        preds = preds if isinstance(preds, list) else [preds]
        for p in preds:
            runs.append({c["id"]: score_circular(c, p.get(c["id"], [])) for c in cases})
    else:
        for _ in range(max(1, args.repeat)):
            out = {c["id"]: run_extractor(c, base) for c in cases}
            saved.append(out)
            runs.append({c["id"]: score_circular(c, out[c["id"]]) for c in cases})
        if args.save:
            with open(args.save, "w", encoding="utf-8") as fh:
                json.dump(saved, fh, indent=1, default=str)
    summ = summarise(runs)
    if args.json:
        print(json.dumps({"summary": summ, "runs": runs}, indent=1, default=str))
    else:
        mode = "LLM" if os.getenv("GROQ_API_KEY") and not args.predictions else ("saved output" if args.predictions else "keyword rules only")
        print(f"Extraction vs hand labels ({mode}), {summ['runs']} run(s)")
        print(f"Required-rule recall: {summ['recall']:.0%}   Must-not violations: {summ['violations']}\n")
        for cid, s in runs[0].items():
            r = "n/a" if s["recall"] is None else f"{s['recall']:.0%}"
            print(f"{cid}: found {len(s['found'])}, missed {len(s['missed'])} (recall {r}), reported {s['reported']}")
            for m in s["missed"]:
                print(f"    MISSED   {m}")
            for v in s["violations"]:
                print(f"    VIOLATION {v['id']}: {v['rule'][:90]} (value {v['value']})")
            for e in s["extras"][:3]:
                print(f"    extra    {e[:90]}")
        if summ["runs"] > 1:
            print("\nHit rate per required rule across runs:")
            for k, v in summ["rule_hit_rate"].items():
                print(f"  {v:>4.0%}  {k}")
    return 1 if summ["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
