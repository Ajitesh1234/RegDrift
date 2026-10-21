# Benchmark datasheet

**What it is.** 101 hand-labelled (clause, bot) pairs in `bench_cases.json`, checked against an 18-bot pool in
`bench_bots.json` (the 8 demo bots plus 10 more). `python bench.py` runs four approaches on exactly the same cases and
writes `docs/benchmark.md`, a chart, and a page that lets a reviewer open every case and read each approach's reasoning.

**Version 1.0.** Changes to labels or cases are made by editing the JSON and noting it here, so results stay comparable.

## Each case
| Field | Meaning |
|---|---|
| `clause` | one regulatory clause, written in the style of an RBI direction (`source: real` when the wording comes from one of the five real circulars in `eval/extract_texts/`) |
| `bot` | the bot it is checked against; its description and settings are in `bench_bots.json` |
| `expected` | the correct verdict: `compliant`, `violating`, `at_risk` (a human must check), `optional_gap` (the rule only permits something the bot does not use), or `not_applicable` |
| `category` | the kind of case: threshold, boundary, yes_no, permissive, list, paraphrase, old_new, unit_trap, not_applicable, lookalike, missing_info, contradiction, exception, ambiguous |
| `label_confidence` | `high`, or `review` when a reasonable reviewer could label it differently (19 of 101) |

## How labels were written
From the clause and the bot's settings only, before any system was run. A mandatory rule (`shall`, `must`) that the setting
breaches is `violating`; the same rule met exactly at its limit is `compliant`; a permission (`may`) the bot does not use is
`optional_gap`; ambiguous wording ("promptly"), a clause that contradicts itself, or a bot that does the activity without
exposing a setting is `at_risk`, because a person must decide; a clause about something else is `not_applicable`.

## The four approaches
`rules` (regular expressions, no AI) · `llm` (one prompt with the clause and the bot) · `rag` (retrieve the 3 most similar
bots, then the same prompt) · `full` (RegDrift: LLM extraction, retrieval, deterministic judge). The first arm was given the
obvious fixes (spelled-out numbers, "at least once in N years", ceiling and threshold wording) so it is not a strawman.

## Metrics
The positive class is `violating`, because a wrong High ticket is the costly mistake. Reported: 5-way accuracy, precision,
recall, F1, false-positive rate, and the count of High findings raised on cases that are compliant or not applicable. Intervals
are 95%: Wilson for proportions, a seeded bootstrap for F1. Results are shown for all cases and for `high`-confidence labels only.

## Limits, stated plainly
- **Small.** 101 cases; intervals are wide, and `optional_gap` has only 4 cases. Differences inside the intervals are not evidence.
- **Labels came from an LLM and have not been reviewed by a compliance expert.** Review the 19 `review` cases first.
- **Mostly synthetic clauses and simulated bots.** It shows how the approaches compare on the same inputs, not how they perform on a bank's real automations.
- **Time context** (withdrawn, expired, not yet in force) is not covered here; it has its own unit tests.
- **LLM answers vary and depend on the model Groq serves that day.** Answers are cached in `eval/cache/bench_llm.jsonl`, so a rerun on the cache is exactly reproducible; fresh answers may differ.
