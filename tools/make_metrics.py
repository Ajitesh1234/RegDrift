"""Draw docs/activity.svg ("repo pulse") from this repository's own git history. Standard library only.

    python tools/make_metrics.py            # writes docs/activity.svg

Numbers shown: commits per week for the last 12 weeks, total commits, number of unit tests, lines of Python.
The scheduled workflow .github/workflows/pulse.yml runs this weekly, so the chart stays current with no outside service.
"""
import datetime as dt, glob, os, re, subprocess, sys
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "activity.svg")
WEEKS = 8


def week_start(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def weekly_counts(dates, today: dt.date, weeks: int = WEEKS):
    """dates: iterable of datetime.date. Returns [(week_start, commits)] for the last `weeks` weeks, oldest first."""
    first = week_start(today) - dt.timedelta(weeks=weeks - 1)
    buckets = {first + dt.timedelta(weeks=i): 0 for i in range(weeks)}
    for d in dates:
        w = week_start(d)
        if w in buckets:
            buckets[w] += 1
    return sorted(buckets.items())


def git_dates():
    out = subprocess.run(["git", "log", "--format=%ad", "--date=short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [dt.date.fromisoformat(x) for x in out.split()]


def count_tests() -> int:
    n = 0
    for f in glob.glob(os.path.join(ROOT, "tests", "test_*.py")):
        with open(f, encoding="utf-8") as fh:
            n += len(re.findall(r"^\s+def test_", fh.read(), re.M))
    return n


def count_python_lines() -> int:
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    n = 0
    for f in files:
        try:
            with open(os.path.join(ROOT, f), encoding="utf-8", errors="ignore") as fh:
                n += sum(1 for _ in fh)
        except OSError:
            pass
    return n


def render(weekly, total_commits: int, tests: int, py_lines: int, updated: dt.date) -> str:
    W, H = 1280, 300
    top = max(1, max(c for _, c in weekly))
    chart_x, chart_y, chart_w, chart_h = 520, 78, 700, 150
    slot = chart_w / len(weekly)
    bw = slot * 0.62
    bars, labels = [], []
    for i, (wk, c) in enumerate(weekly):
        h = 0 if c == 0 else max(5.0, chart_h * c / top)
        x = chart_x + i * slot + (slot - bw) / 2
        y = chart_y + chart_h - h
        last = i == len(weekly) - 1
        fill = "url(#ac)" if c else "#1e2a4a"
        fade = ' opacity=".55"' if last and c else ""   # this week is still in progress
        tip = f'{wk.strftime("%d %b")}: {c} commit{"s" if c != 1 else ""}'
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h, 3):.1f}" rx="5" fill="{fill}"{fade}>'
                    f'<title>{escape(tip)}</title></rect>')
        if c:
            bars.append(f'<text x="{x + bw / 2:.1f}" y="{y - 7:.1f}" font-size="12" font-weight="700" fill="#cbd5e1" text-anchor="middle">{c}</text>')
        if True:
            labels.append(f'<text x="{x + bw / 2:.1f}" y="{chart_y + chart_h + 24}" font-size="11" fill="#64748b" text-anchor="middle">{escape(wk.strftime("%d %b"))}</text>')
    stats = [(f"{total_commits:,}", "commits"), (f"{tests:,}", "unit tests"), (f"{py_lines:,}", "lines of Python")]
    stat_svg = "".join(
        f'<text x="64" y="{118 + i * 66}" font-size="40" font-weight="800" fill="url(#ac)">{escape(v)}</text>'
        f'<text x="250" y="{118 + i * 66}" font-size="16" fill="#94a3b8">{escape(k)}</text>'
        for i, (v, k) in enumerate(stats))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t">
  <title id="t">Repo pulse: commits per week over the last {len(weekly)} weeks</title>
  <defs>
    <linearGradient id="ac" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#2dd4bf"/><stop offset="1" stop-color="#60a5fa"/></linearGradient>
    <clipPath id="r"><rect width="{W}" height="{H}" rx="22"/></clipPath>
  </defs>
  <g clip-path="url(#r)" font-family="ui-sans-serif,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif">
    <rect width="{W}" height="{H}" fill="#0b1224"/>
    <text x="64" y="52" font-size="13" font-weight="800" letter-spacing="2.4" fill="#2dd4bf">REPO PULSE</text>
    {stat_svg}
    <text x="{chart_x}" y="52" font-size="13" font-weight="800" letter-spacing="2.4" fill="#64748b">COMMITS PER WEEK · LAST {len(weekly)} WEEKS</text>
    <line x1="{chart_x}" y1="{chart_y + chart_h + 1}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h + 1}" stroke="#22315a"/>
    {"".join(bars)}
    {"".join(labels)}
    <text x="{W - 40}" y="{H - 22}" font-size="11" fill="#475569" text-anchor="end">drawn from git history by tools/make_metrics.py · updated {updated.isoformat()}</text>
  </g>
</svg>
'''


def main() -> int:
    today = dt.date.today()
    dates = git_dates()
    svg = render(weekly_counts(dates, today), len(dates), count_tests(), count_python_lines(), today)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({len(dates)} commits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
