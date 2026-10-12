"""Score the deterministic decision layer against a hand-labelled gold set.

    python eval_judge.py                 # prints accuracy, per-status precision/recall, and every miss
    python eval_judge.py --json          # machine-readable result

No network and no LLM: each case is one (rule, bot setting) pair whose correct verdict was written down
from the rules themselves, before looking at program output. A miss means the logic and the labels
disagree, so either the logic has a bug or the label is wrong. Exit status is 1 if any case misses.
"""
import argparse, json, os, sys, tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "eval", "judge_gold.json")
STATUSES = ["compliant", "violating", "at_risk", "optional_gap", "not_applicable"]


def score(gold_path: str = GOLD) -> dict:
    with open(gold_path, encoding="utf-8") as fh:
        gold = json.load(fh)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(gold["bots"], tmp)
    old = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY", "LORA_ADAPTER")}
    os.environ["REGDRIFT_BOTS"] = tmp.name
    try:
        import regdrift
        bots = {b["name"]: b for b in regdrift.load_bots()}
        rows = []
        for c in gold["cases"]:
            got = regdrift.judge({"obligation": c["obligation"], "bot": bots[c["bot"]]})
            rows.append({"id": c["id"], "note": c["note"], "expect": c["expect"], "got": got["status"],
                         "reason": got["reason"], "ok": got["status"] == c["expect"]})
    finally:
        os.environ.pop("REGDRIFT_BOTS", None)
        os.unlink(tmp.name)
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v
    per = {}
    for s in STATUSES:
        tp = sum(r["expect"] == s and r["got"] == s for r in rows)
        fp = sum(r["expect"] != s and r["got"] == s for r in rows)
        fn = sum(r["expect"] == s and r["got"] != s for r in rows)
        per[s] = {"support": tp + fn, "precision": tp / (tp + fp) if tp + fp else None,
                  "recall": tp / (tp + fn) if tp + fn else None}
    n_ok = sum(r["ok"] for r in rows)
    return {"cases": len(rows), "correct": n_ok, "accuracy": n_ok / len(rows) if rows else 0.0,
            "per_status": per, "confusion": dict(Counter((r["expect"], r["got"]) for r in rows if not r["ok"])),
            "misses": [r for r in rows if not r["ok"]], "rows": rows}


def _fmt(x):
    return "  -  " if x is None else f"{x:.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = score(args.gold)
    if args.json:
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=str))
    else:
        print(f"Decision layer: {res['correct']}/{res['cases']} correct ({res['accuracy']:.1%})\n")
        print(f"{'status':<16}{'cases':>6}{'precision':>11}{'recall':>8}")
        for s, m in res["per_status"].items():
            print(f"{s:<16}{m['support']:>6}{_fmt(m['precision']):>11}{_fmt(m['recall']):>8}")
        for r in res["misses"]:
            print(f"\nMISS {r['id']}: expected {r['expect']}, got {r['got']}\n  {r['note']}\n  {r['reason']}")
    return 0 if not res["misses"] else 1


if __name__ == "__main__":
    sys.exit(main())
