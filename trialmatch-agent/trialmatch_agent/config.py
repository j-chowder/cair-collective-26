"""Settings. A config file (JSON) only needs the keys it changes."""
import copy
import json

DEFAULTS = {
    "model": "claude-haiku-5-5",      # cheapest current Claude model
    "prices": {                        # USD per million tokens
        "input": 0.10, "output": 0.50,
        "input_long": 0.50, "output_long": 2.50, "long_threshold": 100000,
        "cache_write_mult": 1.25, "cache_read_mult": 0.10,
    },
    "spend_limit_usd": 2.00,           # total across ALL runs; raise on purpose
    "per_run_limit_usd": 0.10,         # one patient
    "temperature": 0.0,                # set to null if the API rejects it
    "prompt_caching": True,
    "agent": {
        "max_tool_calls": 20,
        "max_turns": 25,
        "max_tokens": 800,             # per model reply
        "seed_k": 15,                  # starting candidates shown to the agent
        "pool_depth": 1000,            # length of the final ranked list
        "max_criteria_shown": 20,      # per section, when reading a trial
        "compact_above_tokens": 40000, # shrink old tool results above this prompt size
    },
    "paths": {
        "trials": "sample_data/trials.jsonl",
        "topics": "sample_data/topics.jsonl",
        "qrels": "sample_data/qrels.txt",
        "outputs": "outputs",
    },
}


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, json.load(f))
    out = cfg["paths"]["outputs"]
    name = cfg["paths"]["trials"].replace("/", "_").replace(".", "_")
    cfg["paths"].setdefault("index", f"{out}/index/{name}.pkl")
    return cfg
