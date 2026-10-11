"""Builds data/train.jsonl for the LoRA extractor from seed examples + reviewer feedback.
Format matches LORA_PROMPT in regdrift.py. Usage: python build_lora_data.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = "### Instruction:\nExtract measurable obligations as a JSON list.\n### Clause:\n{c}\n### Output:\n"
KEYS = ("parameter", "operator", "required_value", "unit")

def row(clause, obs):
    obs = [{k: o[k] for k in KEYS} for o in obs]
    return {"text": P.format(c=clause) + json.dumps(obs)}

rows = []
for f in ("seed_examples.jsonl", "feedback.jsonl"):
    p = os.path.join(HERE, "data", f)
    if not os.path.exists(p): continue
    for l in open(p):
        d = json.loads(l)
        if f.startswith("seed"):
            rows.append(row(d["clause_text"], d["obligations"]))
        elif d["decision"] == "approve" and d.get("obligation"):
            rows.append(row(d["clause_text"], [d["obligation"]]))
        elif d["decision"] == "reject" and d.get("corrected"):
            c = d["corrected"]; rows.append(row(d["clause_text"], c if isinstance(c, list) else [c]))
with open(os.path.join(HERE, "data", "train.jsonl"), "w") as out:
    for r in rows: out.write(json.dumps(r) + "\n")
print(f"Wrote {len(rows)} training rows to data/train.jsonl")
