"""Step 3: build train/test files from seeds + reviewed candidates + reviewer feedback.

    python build_lora_data.py

The test set is held out BY CIRCULAR (never by clause), so the evaluation measures generalisation to
documents the model has not seen. Seeds and feedback labels are train-only.
Outputs: data/train.jsonl, data/test.jsonl, data/split_report.json
"""
import hashlib, json, os, sys

import regdrift
from distill import load_rows

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TRAIN_ONLY = ("seed", "feedback")


def completion(obs) -> str:
    return json.dumps([regdrift.canon_obligation(o) for o in obs])


def collect(data_dir=DATA):
    items = []
    for r in load_rows(os.path.join(data_dir, "seed_examples.jsonl")):
        items.append({"circular": "seed", "clause_text": r["clause_text"], "final": r["obligations"]})
    for r in load_rows(os.path.join(data_dir, "candidates.jsonl")):
        if r["status"] in ("accepted", "edited", "no_rule") and r.get("final") is not None:
            items.append({"circular": r["circular"], "clause_text": r["clause_text"], "final": r["final"]})
    for r in load_rows(os.path.join(data_dir, "feedback.jsonl")):          # written by review.py
        if r["decision"] == "approve" and r.get("obligation"):
            items.append({"circular": "feedback", "clause_text": r["clause_text"], "final": [r["obligation"]]})
        elif r["decision"] == "reject" and r.get("corrected"):
            c = r["corrected"]
            items.append({"circular": "feedback", "clause_text": r["clause_text"], "final": c if isinstance(c, list) else [c]})
    return items


def choose_test_circulars(names, fraction=0.25, min_circulars=4):
    """Deterministic hold-out of whole circulars. With fewer than `min_circulars` there is no honest test set."""
    names = sorted(names, key=lambda n: hashlib.sha1(n.encode()).hexdigest())
    if len(names) < min_circulars:
        return set()
    return set(names[:max(1, round(len(names) * fraction))])


def build(data_dir=DATA):
    items = collect(data_dir)
    circulars = {i["circular"] for i in items if i["circular"] not in TRAIN_ONLY}
    test_c = choose_test_circulars(circulars)
    test = [i for i in items if i["circular"] in test_c]
    test_texts = {i["clause_text"] for i in test}
    seen, train = set(), []
    for i in items:
        if i["circular"] in test_c or i["clause_text"] in test_texts or i["clause_text"] in seen:
            continue                                   # no leakage, no duplicates
        seen.add(i["clause_text"])
        train.append(i)
    with open(os.path.join(data_dir, "train.jsonl"), "w", encoding="utf-8") as fh:
        for i in train:
            p, c = regdrift.LORA_PROMPT.format(clause=i["clause_text"]), completion(i["final"])
            fh.write(json.dumps({"prompt": p, "completion": c, "text": p + c, "circular": i["circular"]}) + "\n")
    with open(os.path.join(data_dir, "test.jsonl"), "w", encoding="utf-8") as fh:
        for i in test:
            fh.write(json.dumps({"clause_text": i["clause_text"], "circular": i["circular"],
                                 "expected": [regdrift.canon_obligation(o) for o in i["final"]]}) + "\n")
    report = {"train_rows": len(train), "test_rows": len(test), "test_circulars": sorted(test_c),
              "train_circulars": sorted(circulars - test_c), "train_positive": sum(1 for i in train if i["final"]),
              "train_negative": sum(1 for i in train if not i["final"])}
    with open(os.path.join(data_dir, "split_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


if __name__ == "__main__":
    r = build()
    print(f"train rows: {r['train_rows']}  (positive {r['train_positive']}, negative {r['train_negative']})")
    print(f"test rows:  {r['test_rows']}  from held-out circulars: {r['test_circulars'] or 'NONE'}")
    if not r["test_circulars"]:
        print("\nNote: fewer than 4 reviewed circulars, so no held-out test set yet. Add more circulars, "
              "re-run distill.py + review_candidates.py, then rebuild.")
