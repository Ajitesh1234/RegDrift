"""Step 2: a human accepts / fixes / rejects each Groq proposal. Only reviewed rows become training data.

    python review_candidates.py            # in Colab use:  %run review_candidates.py
    python review_candidates.py --stats
    python review_candidates.py --auto     # settle the easy ones automatically, then review only the rest

Keys:  a accept   n no checkable rule   e edit (paste a JSON list)   s skip   q save and quit
"""
import argparse, json, os, sys
from collections import Counter

import regdrift
from distill import CANDIDATES, load_rows, save_rows

REVIEWED = {"accepted", "edited", "no_rule"}


def parse_edit(text: str):
    data = json.loads(text)
    data = data if isinstance(data, list) else [data]
    return [regdrift.canon_obligation(o) for o in data]


def apply(row, action, payload=None):
    if action == "a":
        row["status"], row["final"] = "accepted", row["proposed"]
    elif action == "n":
        row["status"], row["final"] = "no_rule", []
    elif action == "e":
        row["status"], row["final"] = "edited", parse_edit(payload)
    elif action == "s":
        row["status"] = "skipped"
    else:
        raise ValueError(f"unknown action {action!r}")
    return row


def stats(rows):
    c = Counter(r["status"] for r in rows)
    print(f"{len(rows)} candidates: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    by = Counter(r["circular"] for r in rows if r["status"] in REVIEWED)
    print("reviewed per circular:", dict(by))


HINT = __import__("re").compile(r"(?i)\b(shall|must|may|within|at least|not (?:exceed|less|more)|upto|up to|"
                                r"permitted|prohibited|not permit)\b|\d")


def _grounded(o, text):
    """Is the proposed value actually written in the clause? (a cheap guard against invented numbers)"""
    v, low = o.get("required_value"), text.lower().replace(",", "")
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return any(float(n) == float(v) for n in __import__("re").findall(r"\d+(?:\.\d+)?", low))
    return str(v).lower() in low


def auto_review(path=CANDIDATES):
    """Settle easy rows; leave doubtful ones pending for a human.
    - no proposal AND the clause has no obligation words or numbers -> no_rule
    - every proposed value is written in the clause text -> accepted
    Anything else stays pending."""
    rows = load_rows(path)
    n_no = n_acc = 0
    for r in rows:
        if r["status"] != "pending":
            continue
        if not r["proposed"]:
            if not HINT.search(r["clause_text"]):
                apply(r, "n"); r["auto"] = True; n_no += 1
        elif all(_grounded(o, r["clause_text"]) for o in r["proposed"]):
            apply(r, "a"); r["auto"] = True; n_acc += 1
    save_rows(path, rows)
    print(f"auto: {n_acc} accepted, {n_no} marked no_rule. Remaining for you: "
          f"{sum(1 for r in rows if r['status'] == 'pending')}")
    return rows


def review(path=CANDIDATES, ask=input):
    rows = load_rows(path)
    todo = [r for r in rows if r["status"] in ("pending", "skipped")]
    print(f"{len(todo)} to review. (a)ccept (n)o rule (e)dit (s)kip (q)uit\n")
    for i, r in enumerate(todo, 1):
        print(f"--- {i}/{len(todo)}  {r['circular']}  {r['clause']}\n{r['clause_text']}\n")
        print("Proposed:", json.dumps(r["proposed"]) if r["proposed"] else "[] (no rule)")
        while True:
            a = ask("[a/n/e/s/q] > ").strip().lower()
            if a == "q":
                save_rows(path, rows)
                stats(rows)
                return
            try:
                apply(r, a, ask("paste JSON list > ") if a == "e" else None)
                break
            except (ValueError, json.JSONDecodeError) as e:
                print("  invalid:", e)
        save_rows(path, rows)
    stats(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=CANDIDATES)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--auto", action="store_true", help="auto-settle easy rows first")
    a = ap.parse_args()
    if a.auto:
        auto_review(a.file)
    if a.stats:
        stats(load_rows(a.file))
    else:
        review(a.file)
