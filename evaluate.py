"""Step 5: compare extractors on the held-out test circulars.

    python evaluate.py                       # keyword baseline (+ Groq if GROQ_API_KEY, + LoRA if LORA_ADAPTER)
    python evaluate.py --backends rules,groq,lora

Metrics (per clause):
  detection P/R/F1   did it notice that the clause contains a checkable rule?
  exact match        predicted obligation set == reviewed set (parameter, operator, value, modality)
  content match      same, ignoring the parameter NAME (rule semantics right, naming free)
Writes docs/eval_results.md. With a small test set, read the numbers as indicative, not definitive.
"""
import argparse, json, os, sys, time

import regdrift
from distill import load_rows

HERE = os.path.dirname(os.path.abspath(__file__))


def _val(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    return str(v).strip().lower()


def sig(o, with_param=True):
    t = (o["operator"], _val(o["required_value"]), o.get("modality", "mandatory"))
    return ((o["parameter"],) + t) if with_param else t


def normalise(preds):
    out = []
    for o in preds or []:
        v = regdrift.validate_obligation(o)
        if v:
            out.append(v)
    return out


def score(pred_lists, gold_lists):
    tp = fp = fn = exact = content = 0
    for p, g in zip(pred_lists, gold_lists):
        p, g = normalise(p), normalise(g)
        tp += bool(p) and bool(g)
        fp += bool(p) and not g
        fn += bool(g) and not p
        exact += {sig(o) for o in p} == {sig(o) for o in g}
        content += {sig(o, False) for o in p} == {sig(o, False) for o in g}
    n = len(gold_lists)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": n, "positives": sum(1 for g in gold_lists if g), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "exact_match": exact / n if n else 0.0, "content_match": content / n if n else 0.0}


def run_backend(name, rows, delay=2.5):
    fn = {"rules": regdrift.rules_extract_clause, "groq": regdrift.llm_extract_clause, "lora": regdrift.lora_extract}[name]
    preds = []
    for r in rows:
        p = fn(r["clause_text"])
        if p is None:
            return None                         # backend unavailable
        preds.append(p)
        if name == "groq":
            time.sleep(delay)
    return preds


def to_markdown(results, n, circulars):
    lines = ["# Extractor evaluation (held-out circulars)", "",
             f"Test clauses: **{n}** from {len(circulars)} circular(s) never seen in training: {', '.join(circulars) or 'none'}.", ""]
    if n < 30:
        lines += ["> Small test set: treat these numbers as indicative, not definitive.", ""]
    lines += ["| Extractor | Detection F1 | Precision | Recall | Exact match | Content match |", "|---|---|---|---|---|---|"]
    label = {"rules": "Keyword baseline", "groq": "Groq LLM (zero-shot)", "lora": "LoRA-tuned small model"}
    for k, m in results.items():
        lines.append(f"| {label[k]} | {m['f1']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['exact_match']:.2f} | {m['content_match']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=os.path.join(HERE, "data", "test.jsonl"))
    ap.add_argument("--backends", default="rules,groq,lora")
    ap.add_argument("--out", default=os.path.join(HERE, "docs", "eval_results.md"))
    ap.add_argument("--delay", type=float, default=2.5)
    a = ap.parse_args()
    rows = load_rows(a.test)
    if not rows:
        sys.exit("No test rows. Run build_lora_data.py with at least 4 reviewed circulars first.")
    gold = [r["expected"] for r in rows]
    results = {}
    for b in a.backends.split(","):
        b = b.strip()
        if b == "groq" and not os.getenv("GROQ_API_KEY"):
            print("skip groq: no GROQ_API_KEY"); continue
        if b == "lora" and not os.getenv("LORA_ADAPTER"):
            print("skip lora: no LORA_ADAPTER"); continue
        preds = run_backend(b, rows, a.delay)
        if preds is None:
            print(f"skip {b}: backend unavailable"); continue
        results[b] = score(preds, gold)
        print(b, {k: round(v, 3) if isinstance(v, float) else v for k, v in results[b].items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    md = to_markdown(results, len(rows), sorted({r["circular"] for r in rows}))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
