"""Step 1 of the LoRA pipeline: let Groq PROPOSE clause -> obligation labels for every circular in circulars/.

    python distill.py                 # resumable; safe to re-run after rate limits
    python distill.py --delay 4       # slower, kinder to the free tier

Output: data/candidates.jsonl (status "pending"). Nothing here is training data until a human reviews it
with review_candidates.py (this is "distillation with a human in the loop").
"""
import argparse, hashlib, json, os, sys, time

import regdrift

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = os.path.join(HERE, "data", "candidates.jsonl")


def unit_id(circular: str, text: str) -> str:
    return hashlib.sha1(f"{circular}|{text}".encode("utf-8")).hexdigest()[:12]


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file


def propose(circulars_dir, out=CANDIDATES, delay=2.5, sleep=time.sleep):
    rows = load_rows(out)
    seen = {r["id"] for r in rows}
    files = sorted(f for f in os.listdir(circulars_dir) if f.lower().endswith((".pdf", ".txt")) and f != "placeholder.txt")
    if not files:
        print(f"No .pdf/.txt circulars found in {circulars_dir}")
        return 1
    added = 0
    for name in files:
        text = regdrift.clean(regdrift.load_text(os.path.join(circulars_dir, name)))
        for label, body in regdrift.split_units(text):
            uid = unit_id(name, body)
            if uid in seen:
                continue
            raw = regdrift.llm_extract_clause(body)
            if raw is None:
                save_rows(out, rows)
                print(f"\nGroq unavailable or rate-limited. Saved {added} new candidates; re-run later to continue.")
                return 1
            props = []
            for o in raw:
                v = regdrift.validate_obligation(o, default_clause=label, default_summary=body)
                if v:
                    props.append(regdrift.canon_obligation(v))
            rows.append({"id": uid, "circular": name, "clause": label, "clause_text": body,
                         "proposed": props, "status": "pending", "final": None})
            seen.add(uid)
            added += 1
            save_rows(out, rows)
            print(f"  {name} {label}: {len(props)} proposed")
            sleep(delay)
    print(f"\nDone: {added} new candidates ({len(rows)} total) in {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(HERE, "circulars"))
    ap.add_argument("--out", default=CANDIDATES)
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between Groq calls (free-tier rate limits)")
    a = ap.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY first (see .env.example).")
    sys.exit(propose(a.dir, a.out, a.delay))
