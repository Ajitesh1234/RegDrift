"""Human-in-the-loop review. Approve/reject each ticket; labels feed LoRA training.
Usage: python review.py            (interactive; in Colab use:  %run review.py)
Non-interactive: edit tickets.json, set "decision" to "approve" or "reject"
(optionally "corrected_obligation": {...}), then run: python review.py --collect
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
T, F = os.path.join(HERE, "tickets.json"), os.path.join(HERE, "data", "feedback.jsonl")

def collect(tickets):
    done = {json.loads(l)["key"] for l in open(F)} if os.path.exists(F) else set()
    n = 0
    with open(F, "a") as out:
        for t in tickets:
            key = f"{t['bot_id']}|{t['clause']}|{t['rule']}"
            if t.get("decision") in ("approve", "reject") and key not in done:
                out.write(json.dumps({"key": key, "clause_text": t["rule"], "obligation": t.get("obligation"),
                                      "decision": t["decision"], "corrected": t.get("corrected_obligation")}) + "\n")
                n += 1
    print(f"Saved {n} new labels to data/feedback.jsonl")

if __name__ == "__main__":
    tickets = json.load(open(T))
    if "--collect" not in sys.argv:
        for t in tickets:
            if t.get("decision"): continue
            print(f"\n{t['severity']} | {t['bot']} | {t['clause']}\n  {t['rule']}\n  -> {t['status']}: {t['reason']}")
            a = input("  [a]pprove / [r]eject / [s]kip: ").strip().lower()
            if a == "a": t["decision"] = "approve"
            elif a == "r":
                t["decision"] = "reject"
                c = input("  corrected obligation JSON (or blank): ").strip()
                if c: t["corrected_obligation"] = json.loads(c)
        json.dump(tickets, open(T, "w"), indent=2)
    collect(tickets)
