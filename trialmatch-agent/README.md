# TrialMatch Agent

A standalone AI agent that matches a patient to clinical trials. You give it a patient description;
it searches a trial database, reads eligibility criteria, judges each trial it reads, and returns a
ranked list with a short reason for each judgment. It runs in the terminal on **Claude Haiku 5.5**
(the cheapest Claude model), with spending limits built into the code.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # paste your ANTHROPIC_API_KEY into .env
python -m unittest discover tests                                   # free check
python -m trialmatch_agent run --config configs/sample.json --topic 2 --mock   # free demo
python -m trialmatch_agent run --config configs/sample.json --topic 2          # real, < $0.01
```
Run every command from the project folder. Add `--mock` to any `run` or `batch` to test for free
with a fake model.

## How it works

1. **Start:** the patient note is split into numbered sentences (S0, S1, ...). A keyword search
   over the note finds starting candidate trials, and the top ones are shown to the agent.
2. **Agent loop:** Claude chooses one of four tools, the code runs it and returns the result, and
   this repeats until Claude calls `finish` or the tool-call budget is used up.
3. **Ranking:** trials are ordered *eligible → uncertain → excluded → not read → irrelevant*,
   using confidence to break ties. Trials the agent never read keep their search order, so the
   list is long enough to evaluate.
4. **Saving:** every step, judgment, token count, and cost is saved to a trace file.

### The agent's tools

| Tool | What it does |
|---|---|
| `search_trials(query, k)` | Keyword search; returns trial ids, titles, conditions |
| `read_trial(nct_id)` | Summary, age/sex limits, numbered inclusion (I0, I1, ...) and exclusion (E0, ...) criteria |
| `record_judgment(nct_id, verdict, confidence, reason)` | Saves *eligible / uncertain / excluded / irrelevant*, with a reason citing criteria and sentences, e.g. "I2 met (S3); E1 not present" |
| `finish()` | Ends the run |

Only the agent's own turns use the API; the tools themselves run locally and cost nothing.

## Commands

All commands start with `python -m trialmatch_agent` and accept `--config FILE`.

| Command | What it does | Useful options |
|---|---|---|
| `run` | Match one patient, printing each step | `--topic ID` or `--text "..."`, `--budget N`, `--mock`, `--quiet`, `--name` |
| `batch` | Match many patients; asks before spending; picks up where it left off | `--limit N`, `--topics IDs`, `--budget N`, `--name`, `--mock`, `--yes` |
| `evaluate` | Score a batch against the relevance labels | `--name` |
| `prepare` | Convert downloaded TREC files (free, local) | `--trials`, `--topics`, `--qrels`, `--judged-only` |
| `spend` | Show how much credit you've used | |

## Configuration

Settings live in `configs/*.json`; a file only lists what it changes from the defaults in
`trialmatch_agent/config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `spend_limit_usd` | 2.00 | Stops all API calls once your **total** spending (all runs) reaches this |
| `per_run_limit_usd` | 0.10 | Stops one patient's run at this cost |
| `agent.max_tool_calls` | 20 | Tool calls per patient (enforced by the code) |
| `agent.seed_k` | 15 | Starting candidates shown to the agent |
| `agent.max_criteria_shown` | 20 | Criteria per section returned by `read_trial` |
| `agent.compact_above_tokens` | 40000 | Above this prompt size, older tool results are shortened |
| `temperature` | 0.0 | Set to `null` if the API rejects it |
| `paths.trials / topics / qrels` | sample data | Data files |

Provided configs: `sample.json` (10 made-up trials), `trec2021.json`, `trec2022.json`.

## Protecting your budget

- A **total spending limit** is checked before every API call (`outputs/ledger.json`). It counts
  all runs, so raise it deliberately: e.g. $1 for testing, then $8, then higher.
- A **per-patient limit** and a **tool-call budget** stop runaway runs.
- A **response cache**: repeating an identical request costs nothing.
- **Prompt caching** makes the repeated part of the conversation cheaper on each turn.
- **Old tool results are shortened** so prompts stay well under 100k tokens, where Haiku's price
  rises 5×. A warning prints if a prompt gets close.
- `batch` asks before spending and never re-runs finished patients.

Expect roughly $0.02–0.05 per patient at 20 tool calls; check the real figure with a few patients
before running many.

## Output files

| File | Contents |
|---|---|
| `outputs/runs/<name>/traces/<patient>.json` | Everything about one run: each tool call and result, judgments with reasons, ranking, tokens, cost, time, stop reason |
| `outputs/runs/<name>/run.txt` | Ranking in standard TREC format |
| `outputs/runs/<name>/summary.csv` | Per-patient cost, tool calls, trials judged, time (from `batch`) |
| `outputs/ledger.json`, `outputs/usage.csv` | Total spend; one row per API call |

**Stop reasons:** `finished`, `budget_used`, `max_turns`, `spend_limit`, `no_tool_call`
(Claude replied without using a tool; check the trace).

## Evaluation

`evaluate` reports, averaged over patients, using the TREC grades (2 = eligible, 1 = excluded,
0 = not relevant; trials without a grade count as 0):

- **nDCG@10:** quality of the top 10, rewarding eligible trials near the top.
- **P@10 (eligible):** share of the top 10 that are eligible.
- **Recall@100 (eligible):** share of all eligible trials that appear in the top 100.

## Using the real TREC data

Download the trial corpus, topics, and qrels from trec-cds.org (2021 and 2022 track pages) into
`data/raw/`, then:

```bash
python -m trialmatch_agent prepare --trials data/raw/trials \
    --topics data/raw/topics2021.xml data/raw/topics2022.xml \
    --qrels data/raw/qrels2021.txt data/raw/qrels2022.txt --judged-only
python -m trialmatch_agent run --config configs/trec2021.json --topic 1
python -m trialmatch_agent batch --config configs/trec2021.json --limit 5 --name try1
python -m trialmatch_agent evaluate --config configs/trec2021.json --name try1
```
`--judged-only` keeps only trials that have relevance labels (much smaller and faster than the full
corpus). Output names follow input names, so keep the names above or edit the configs.

## Code map

| File | Role |
|---|---|
| `cli.py` | Terminal commands |
| `agent.py` | The agent loop, budget enforcement, saving results |
| `tools.py` | The four tools and the final ranking |
| `prompts.py` | All instructions given to Claude (edit here to change behaviour) |
| `llm.py` | API calls with spending limits, cost tracking, and caching |
| `mock_llm.py` | Fake model for free testing |
| `retrieval.py` | Keyword (BM25) search |
| `data.py` | Loading and converting data files |
| `evaluate.py` | Metrics |
| `config.py` | Default settings |
