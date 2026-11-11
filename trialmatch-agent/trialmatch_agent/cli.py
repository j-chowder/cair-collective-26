"""Terminal commands. Run `python -m trialmatch_agent --help` from the project root."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from .config import load_config
from .data import (load_qrels, load_run, load_topics, load_trials, parse_topics_xml,
                   prepare_trials, write_jsonl)
from .evaluate import evaluate, ndcg_at_10


def _load(args):
    cfg = load_config(args.config)
    from .agent import Agent
    from .llm import LLM
    from .retrieval import load_index
    trials = load_trials(cfg["paths"]["trials"])
    agent = Agent(cfg, LLM(cfg, mock=args.mock), trials, load_index(cfg, trials),
                  verbose=not getattr(args, "quiet", False))
    return cfg, trials, agent


def cmd_run(args):
    from .agent import save_result
    cfg, trials, agent = _load(args)
    if args.text:
        tid, text = "custom", args.text
    else:
        topics = load_topics(cfg["paths"]["topics"])
        if str(args.topic) not in topics:
            sys.exit(f"Topic {args.topic} not found. Some valid ids: {', '.join(list(topics)[:10])}")
        tid, text = str(args.topic), topics[str(args.topic)]

    res = agent.run(tid, text, args.budget)
    run_dir = Path(cfg["paths"]["outputs"]) / "runs" / args.name
    save_result(res, run_dir)

    c = res["cost"]
    print(f"\nStopped: {res['stop_reason']} | {res['tool_calls']} tool calls | {res['seconds']}s")
    print(f"Cost: ${c['cost_usd']:.4f} ({c['calls']} LLM calls, {c['input_tokens']} in / "
          f"{c['output_tokens']} out tokens, largest prompt {c['max_prompt_tokens']})"
          + ("  [mock: nothing billed]" if args.mock else ""))
    if not args.mock:
        print(f"Total spent so far: ${agent.llm.total:.4f} of ${cfg['spend_limit_usd']:.2f} limit")
    grades = load_qrels(cfg["paths"]["qrels"]).get(tid, {})
    print("\nTop 10:")
    for i, n in enumerate(res["ranking"][:10], 1):
        j = res["judgments"].get(n)
        gold = f"  gold={grades.get(n, '-')}" if grades else ""
        print(f" {i:2d}. {n}  {(j['verdict'] if j else 'not read'):10s}{gold}  {trials[n]['title'][:55]}")
        if j:
            print(f"       why: {j['reason'][:110]}")
    if grades:
        print(f"\nnDCG@10 for this patient: {ndcg_at_10(res['ranking'], grades):.3f}")
    print(f"Full trace: {run_dir / 'traces' / (tid + '.json')}")


def cmd_batch(args):
    from .agent import save_result
    cfg, trials, agent = _load(args)
    agent.verbose = False
    topics = load_topics(cfg["paths"]["topics"])
    ids = (args.topics or list(topics))[: args.limit]
    budget = args.budget or cfg["agent"]["max_tool_calls"]
    run_dir = Path(cfg["paths"]["outputs"]) / "runs" / args.name
    todo = [t for t in ids if not (run_dir / "traces" / f"{t}.json").exists()]
    print(f"{len(todo)} patients to run ({len(ids) - len(todo)} already done). Budget {budget} "
          f"tool calls each. Spent so far ${agent.llm.total:.2f} of ${cfg['spend_limit_usd']:.2f}.")
    if not args.mock and todo and not args.yes and input("Continue? [y/N] ").lower() != "y":
        sys.exit("Cancelled.")
    for i, tid in enumerate(todo, 1):
        res = agent.run(tid, topics[tid], budget)
        save_result(res, run_dir)
        print(f" {i}/{len(todo)} patient {tid}: {res['stop_reason']}, {res['tool_calls']} calls, "
              f"{len(res['judgments'])} judged, ${res['cost']['cost_usd']:.4f}, {res['seconds']}s")
        if res["stop_reason"] == "spend_limit":
            sys.exit("Spend limit reached. Raise spend_limit_usd to continue; finished patients are kept.")

    rows = []
    for p in sorted((run_dir / "traces").glob("*.json")):
        r = json.loads(p.read_text())
        rows.append({"patient": r["topic_id"], "stop_reason": r["stop_reason"],
                     "tool_calls": r["tool_calls"], "trials_judged": len(r["judgments"]),
                     "cost_usd": round(r["cost"]["cost_usd"], 5), "seconds": r["seconds"]})
    if rows:
        with open(run_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n = len(rows)
        print(f"\nAverage per patient: ${sum(r['cost_usd'] for r in rows) / n:.4f}, "
              f"{sum(r['tool_calls'] for r in rows) / n:.1f} tool calls, "
              f"{sum(r['seconds'] for r in rows) / n:.1f}s. Saved {run_dir / 'summary.csv'}")


def cmd_evaluate(args):
    cfg = load_config(args.config)
    run_file = Path(cfg["paths"]["outputs"]) / "runs" / args.name / "run.txt"
    per, means = evaluate(load_run(run_file), load_qrels(cfg["paths"]["qrels"]))
    if not per:
        sys.exit("No evaluated patients: check that the run and qrels use the same topic ids.")
    print(f"{run_file}: {len(per)} patients")
    for m, v in means.items():
        print(f"  {m:22s} {v:.4f}")


def cmd_prepare(args):
    out = Path(args.out_dir)
    keep = None
    if args.judged_only:
        keep = {n for q in args.qrels for g in load_qrels(q).values() for n in g}
        print(f"Keeping only the {len(keep)} trials that appear in the qrels.")
    if args.trials:
        ok, bad = prepare_trials(args.trials, out / "trials.jsonl", keep)
        print(f"Wrote {ok} trials to {out / 'trials.jsonl'} ({bad} files could not be parsed)")
    for t in args.topics:
        write_jsonl(out / (Path(t).stem + ".jsonl"), parse_topics_xml(t))
        print(f"Wrote {out / (Path(t).stem + '.jsonl')}")
    for q in args.qrels:
        shutil.copy(q, out / Path(q).name)
        print(f"Copied {out / Path(q).name}")


def cmd_spend(args):
    cfg = load_config(args.config)
    ledger = Path(cfg["paths"]["outputs"]) / "ledger.json"
    total = json.loads(ledger.read_text())["total_usd"] if ledger.exists() else 0.0
    print(f"Spent ${total:.4f} of your ${cfg['spend_limit_usd']:.2f} limit.")


def main():
    p = argparse.ArgumentParser(prog="python -m trialmatch_agent",
                                description="Standalone clinical-trial matching agent")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--config", help="JSON config file (default: built-in sample settings)")
        s.set_defaults(fn=fn)
        return s

    r = add("run", cmd_run, "match one patient, showing every step")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--topic", help="patient id from the topics file")
    g.add_argument("--text", help="type your own patient description")
    r.add_argument("--budget", type=int, help="max tool calls")
    r.add_argument("--name", default="single", help="folder under outputs/runs/")
    r.add_argument("--mock", action="store_true", help="fake model, free")
    r.add_argument("--quiet", action="store_true", help="hide the step-by-step output")

    b = add("batch", cmd_batch, "match many patients (resumes if stopped)")
    b.add_argument("--topics", nargs="*", help="patient ids (default: all)")
    b.add_argument("--limit", type=int, help="only the first N patients")
    b.add_argument("--budget", type=int, help="max tool calls per patient")
    b.add_argument("--name", default="batch", help="folder under outputs/runs/")
    b.add_argument("--mock", action="store_true", help="fake model, free")
    b.add_argument("--yes", action="store_true", help="don't ask before spending")

    e = add("evaluate", cmd_evaluate, "score a run against the relevance labels")
    e.add_argument("--name", default="batch", help="run folder under outputs/runs/")

    pr = add("prepare", cmd_prepare, "convert raw TREC / ClinicalTrials.gov files")
    pr.add_argument("--trials", help="XML file, .zip, or folder of either")
    pr.add_argument("--topics", nargs="*", default=[], help="TREC topics .xml files")
    pr.add_argument("--qrels", nargs="*", default=[], help="TREC qrels files")
    pr.add_argument("--out-dir", default="data/processed")
    pr.add_argument("--judged-only", action="store_true", help="keep only trials in the qrels")

    add("spend", cmd_spend, "show how much credit you have used")

    args = p.parse_args()
    args.fn(args)
