"""Fake Claude client for free testing. It follows a fixed script:
search -> read 3 trials -> judge each -> finish. Nothing is billed."""
import json
import re


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text") or str(b.get("content", "")) for b in content)


class MockClient:
    def __init__(self):
        self.messages = self
        self._n = 0

    def create(self, messages, tools=None, system=None, **_):
        first = _text(messages[0]["content"])
        seeds = list(dict.fromkeys(re.findall(r"NCT\d{8}", first)))[:3]
        m = re.search(r"^S0: (.+)$", first, re.M)
        query = " ".join((m.group(1) if m else "patient").split()[:8])

        plan = [("search_trials", {"query": query, "k": 10})]
        for i, n in enumerate(seeds):
            verdict = ["eligible", "uncertain", "excluded"][i % 3]
            plan += [("read_trial", {"nct_id": n}),
                     ("record_judgment", {"nct_id": n, "verdict": verdict, "confidence": 60,
                                          "reason": "mock: I0 met (S0)"})]
        plan.append(("finish", {}))
        done = sum(1 for msg in messages if msg["role"] == "assistant"
                   for b in msg["content"] if b["type"] == "tool_use")
        name, args = plan[min(done, len(plan) - 1)]
        self._n += 1
        content = [{"type": "text", "text": f"(mock) {name}"},
                   {"type": "tool_use", "id": f"toolu_mock_{self._n}", "name": name, "input": args}]
        size = len(json.dumps([system, messages, tools], default=str)) // 4
        return {"content": content, "usage": {"input_tokens": size, "output_tokens": 40,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
