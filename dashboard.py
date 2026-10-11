"""Turn reports/*.report.json into one self-contained HTML compliance dashboard (no dependencies).

    python dashboard.py [--reports reports] [--out reports/dashboard.html]
"""
import argparse, glob, html, json, os
SEV = {"violating": ("High", "#b42318"), "at_risk": ("Medium", "#b54708"), "optional_gap": ("Low", "#175cd3"),
       "compliant": ("OK", "#067647")}


def load_reports(folder):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.report.json"))):
        with open(p, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def render(reports) -> str:
    e = html.escape
    rows, totals = [], {k: 0 for k in SEV}
    tickets = {}
    for r in reports:
        for t in r.get("tickets", []):
            tickets[(t.get("bot_id"), t.get("clause"), t.get("ref"))] = t.get("jira_key", "")
    for r in reports:
        for f in r.get("findings", []):
            label, colour = SEV.get(f["status"], (f["status"], "#475467"))
            totals[f["status"]] = totals.get(f["status"], 0) + 1
            key = tickets.get((f["bot_id"], f["clause"], f.get("ref")), "")
            rows.append(f"<tr><td><b style='color:{colour}'>{e(label)}</b></td><td>{e(f['bot'])}</td>"
                        f"<td>{e(str(f['clause']))}</td><td>{e(r.get('circular_ref') or r.get('source', ''))}</td>"
                        f"<td>{e(f['reason'])}</td><td>{e(str(key)) or '&ndash;'}</td></tr>")
    cards = "".join(f"<div class='card'><div class='n' style='color:{c}'>{totals.get(k, 0)}</div>{l}</div>"
                    for k, (l, c) in SEV.items())
    generated = max((r.get("run_at_utc", "") for r in reports), default="")
    body = "".join(rows) or "<tr><td colspan=6>No findings yet. Run regdrift.py first.</td></tr>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RegDrift Dashboard</title><style>
body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#f8fafc;color:#101828}}
h1{{margin:0 0 4px}} .sub{{color:#667085;margin-bottom:20px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:12px 20px;min-width:90px;text-align:center}}
.n{{font-size:28px;font-weight:700}} table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e4e7ec}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #eaecf0;vertical-align:top}} th{{background:#f2f4f7}}
@media(prefers-color-scheme:dark){{body{{background:#0c111d;color:#f2f4f7}}.card,table{{background:#161b26;border-color:#333741}}th{{background:#1f242f}}th,td{{border-color:#333741}}}}
</style></head><body><h1>RegDrift compliance dashboard</h1>
<div class="sub">{len(reports)} circular(s) analysed &middot; last run {e(generated)}</div>
<div class="cards">{cards}</div>
<table><tr><th>Severity</th><th>Bot</th><th>Clause</th><th>Circular</th><th>Finding</th><th>Jira</th></tr>{body}</table>
</body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.reports, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(load_reports(a.reports)))
    print("wrote", out)


if __name__ == "__main__":
    main()
