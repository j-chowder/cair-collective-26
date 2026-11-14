"""The agent loop: Claude picks a tool, the code runs it, repeat until Claude
calls finish or the tool-call budget is used up (enforced here, not by the model)."""
import copy
import json
import time
from pathlib import Path

from .data import split_sentences, truncate
from .llm import SpendLimitReached
from .prompts import SYSTEM, first_message
from .tools import TOOL_SCHEMAS, Tools


class Agent:
    def __init__(self, cfg, llm, trials, index, verbose=True):
        self.cfg, self.a = cfg, cfg["agent"]
        self.llm, self.trials, self.index, self.verbose = llm, trials, index, verbose

    def say(self, msg):
        if self.verbose:
            print(msg)

    def run(self, topic_id, patient_text, budget=None):
        budget = int(budget or self.a["max_tool_calls"])
        self.llm.new_run()
        start = time.time()
        sentences = split_sentences(patient_text)
        tools = Tools(self.cfg, self.trials, self.index)

        # Starting candidates: BM25 on the whole note (free). The full list also
        # fills the tail of the final ranking.
        pool = [n for n, _ in self.index.search(patient_text, self.a["pool_depth"])]
        tools.add_candidates(pool)
        shown = [{"nct_id": n, "title": truncate(self.trials[n]["title"], 90),
                  "conditions": truncate(", ".join(self.trials[n]["conditions"]), 70)}
                 for n in pool[: self.a["seed_k"]]]

        system = [{"type": "text", "text": SYSTEM.format(budget=budget)}]
        messages = [{"role": "user", "content": [
            {"type": "text", "text": first_message(sentences, shown, budget)}]}]
        if self.cfg["prompt_caching"]:
            system[0]["cache_control"] = {"type": "ephemeral"}

        calls = turns = 0
        stop, steps = None, []
        self.say(f"\n=== Patient {topic_id} | budget {budget} tool calls ===")
        while stop is None:
            if calls >= budget:
                stop = "budget_used"
                break
            if turns >= self.a["max_turns"]:
                stop = "max_turns"
                break
            try:
                resp = self.llm.create(system=system, tools=TOOL_SCHEMAS,
                                       messages=self._with_cache_mark(messages),
                                       max_tokens=self.a["max_tokens"])
            except SpendLimitReached as e:
                stop = "spend_limit"
                self.say(f"  STOP: {e}")
                break
            turns += 1
            content = [b for b in resp["content"]
                       if b["type"] == "tool_use" or (b["type"] == "text" and b["text"].strip())]
            uses = [b for b in content if b["type"] == "tool_use"]
            if not uses:
                stop = "no_tool_call"
                break
            messages.append({"role": "assistant", "content": content})
            for b in content:
                if b["type"] == "text":
                    self.say(f"  Claude: {truncate(b['text'].strip(), 160)}")

            results = []
            for u in uses:
                if calls < budget:
                    calls += 1
                    out = tools.run(u["name"], u["input"])
                    if u["name"] == "finish":
                        stop = "finished"
                else:
                    out = {"error": "tool-call budget used up"}
                out["budget_left"] = budget - calls
                text = json.dumps(out, ensure_ascii=False)
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": text})
                steps.append({"tool": u["name"], "input": u["input"], "output": out})
                self.say(f"  -> {u['name']}({truncate(json.dumps(u['input']), 100)})\n"
                         f"     {truncate(text, 160)}")
            messages.append({"role": "user", "content": results})
            self.say(f"     [{calls}/{budget} tool calls, ${self.llm.run['cost_usd']:.4f} so far]")

            u = resp["usage"]
            if u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"] \
                    > self.a["compact_above_tokens"]:
                self._compact(messages)

        return {"topic_id": str(topic_id), "budget": budget, "stop_reason": stop,
                "tool_calls": calls, "turns": turns, "seconds": round(time.time() - start, 1),
                "cost": copy.deepcopy(self.llm.run), "searches": tools.searches,
                "judgments": tools.judgments, "ranking": tools.ranking(self.a["pool_depth"]),
                "steps": steps, "model": self.cfg["model"], "mock": self.llm.mock}

    def _with_cache_mark(self, messages):
        """Mark the newest message so the whole conversation so far is cached."""
        if not self.cfg["prompt_caching"]:
            return messages
        msgs = copy.deepcopy(messages)
        msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        return msgs

    def _compact(self, messages):
        """Shorten tool results older than the last 3 turns (keeps prompts small and cheap)."""
        for msg in messages[1:-6]:
            for b in msg["content"]:
                if b.get("type") == "tool_result" and not b["content"].endswith("[shortened]") \
                        and len(b["content"]) > 200:
                    b["content"] = b["content"][:180] + "... [shortened]"


def save_result(result, run_dir):
    """Write the full trace (JSON) and the ranking in TREC run format."""
    run_dir = Path(run_dir)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces" / f"{result['topic_id']}.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    run_file, tid = run_dir / "run.txt", result["topic_id"]
    lines = [l for l in run_file.read_text().splitlines(True) if l.split()[0] != tid] \
        if run_file.exists() else []
    n = len(result["ranking"])
    lines += [f"{tid} Q0 {nct} {r} {n - r + 1} agent\n" for r, nct in enumerate(result["ranking"], 1)]
    run_file.write_text("".join(lines))
