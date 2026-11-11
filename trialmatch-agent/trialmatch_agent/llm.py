"""Claude API wrapper with spending protection.

Every call: (1) is refused if the total or per-run spend limit is reached,
(2) is priced from its token usage, (3) is logged to outputs/usage.csv and
added to outputs/ledger.json, (4) is saved to a disk cache so repeating an
identical request costs nothing.
"""
import csv
import hashlib
import json
import os
import time
from pathlib import Path


class SpendLimitReached(RuntimeError):
    pass


def _to_dict(resp):
    if isinstance(resp, dict):  # mock client
        return resp
    blocks = []
    for b in resp.content:
        if b.type == "text":
            blocks.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    u = resp.usage
    usage = {k: getattr(u, k, 0) or 0 for k in (
        "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    return {"content": blocks, "usage": usage}


class LLM:
    def __init__(self, cfg, mock=False):
        self.cfg, self.mock = cfg, mock
        out = Path(cfg["paths"]["outputs"])
        self.cache_dir = out / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log, self.ledger_path = out / "usage.csv", out / "ledger.json"
        self.total = json.loads(self.ledger_path.read_text())["total_usd"] if self.ledger_path.exists() else 0.0
        self.new_run()
        if mock:
            from .mock_llm import MockClient
            self.client = MockClient()
        else:
            if Path(".env").exists():  # minimal .env loader
                for line in Path(".env").read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY not set. Put it in .env, or use --mock to test for free.")
            import anthropic
            self.client = anthropic.Anthropic(max_retries=3)

    def new_run(self):
        self.run = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                    "cost_usd": 0.0, "max_prompt_tokens": 0}

    def price(self, u):
        p = self.cfg["prices"]
        prompt = u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]
        long_ = prompt > p["long_threshold"]
        p_in, p_out = (p["input_long"], p["output_long"]) if long_ else (p["input"], p["output"])
        return (u["input_tokens"] * p_in + u["cache_creation_input_tokens"] * p_in * p["cache_write_mult"]
                + u["cache_read_input_tokens"] * p_in * p["cache_read_mult"]
                + u["output_tokens"] * p_out) / 1e6, prompt

    def create(self, **kwargs):
        req = {"model": self.cfg["model"], **kwargs}
        if self.cfg.get("temperature") is not None:
            req["temperature"] = self.cfg["temperature"]
        cache_file = self.cache_dir / (hashlib.sha256(
            json.dumps(req, sort_keys=True).encode()).hexdigest() + ".json")

        if not self.mock and cache_file.exists():
            data, billed = json.loads(cache_file.read_text()), False
        else:
            if not self.mock:
                if self.total >= self.cfg["spend_limit_usd"]:
                    raise SpendLimitReached(f"total spend ${self.total:.2f} reached spend_limit_usd "
                                            f"${self.cfg['spend_limit_usd']:.2f}")
                if self.run["cost_usd"] >= self.cfg["per_run_limit_usd"]:
                    raise SpendLimitReached("per_run_limit_usd reached for this patient")
            data, billed = _to_dict(self.client.messages.create(**req)), not self.mock
            if billed:
                cache_file.write_text(json.dumps(data))

        cost, prompt = self.price(data["usage"])
        r = self.run
        r["calls"] += 1
        r["input_tokens"] += data["usage"]["input_tokens"]
        r["output_tokens"] += data["usage"]["output_tokens"]
        r["cache_read_tokens"] += data["usage"]["cache_read_input_tokens"]
        r["cost_usd"] += cost  # counted even when cached, so reported costs are realistic
        r["max_prompt_tokens"] = max(r["max_prompt_tokens"], prompt)
        if billed:
            self.total += cost
            self.ledger_path.write_text(json.dumps({"total_usd": self.total}))
        if prompt > 0.9 * self.cfg["prices"]["long_threshold"]:
            print(f"  WARNING: {prompt}-token prompt is close to the 100k higher-price tier")
        new = not self.usage_log.exists()
        with open(self.usage_log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "billed", "input", "output", "cache_read", "cost_usd", "total_usd"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), billed, data["usage"]["input_tokens"],
                        data["usage"]["output_tokens"], data["usage"]["cache_read_input_tokens"],
                        f"{cost:.6f}", f"{self.total:.4f}"])
        return data
