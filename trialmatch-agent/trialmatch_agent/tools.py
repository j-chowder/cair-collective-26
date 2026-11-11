"""The agent's four tools: their descriptions for Claude, and the code that runs them."""
from .data import truncate

VERDICTS = ["eligible", "uncertain", "excluded", "irrelevant"]

TOOL_SCHEMAS = [
    {"name": "search_trials",
     "description": "Keyword search over all trials. Returns ids, titles and conditions.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "k": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query"]}},
    {"name": "read_trial",
     "description": "Read one trial: summary, age/sex limits, and numbered inclusion (I) and exclusion (E) criteria.",
     "input_schema": {"type": "object", "properties": {"nct_id": {"type": "string"}},
                      "required": ["nct_id"]}},
    {"name": "record_judgment",
     "description": "Record your verdict on a trial you read.",
     "input_schema": {"type": "object", "properties": {
         "nct_id": {"type": "string"},
         "verdict": {"type": "string", "enum": VERDICTS},
         "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
         "reason": {"type": "string", "description": "Short; cite I<n>/E<n> and S<n>."}},
         "required": ["nct_id", "verdict", "confidence", "reason"]}},
    {"name": "finish",
     "description": "End the session.",
     "input_schema": {"type": "object", "properties": {}}},
]


class Tools:
    def __init__(self, cfg, trials, index):
        self.cfg, self.trials, self.index = cfg["agent"], trials, index
        self.candidates = {}   # nct_id -> retrieval score (higher = earlier in ranking)
        self.judgments = {}    # nct_id -> {verdict, confidence, reason}
        self.searches = []

    def add_candidates(self, ids):
        for rank, n in enumerate(ids):  # reciprocal-rank fusion across searches
            self.candidates[n] = self.candidates.get(n, 0.0) + 1.0 / (20 + rank + 1)

    def run(self, name, args):
        try:
            if name == "search_trials":
                return self.search_trials(**args)
            if name == "read_trial":
                return self.read_trial(**args)
            if name == "record_judgment":
                return self.record_judgment(**args)
            if name == "finish":
                return {"ok": True}
            return {"error": f"unknown tool {name}"}
        except Exception as e:  # tell the agent what went wrong instead of crashing
            return {"error": f"{type(e).__name__}: {e}"}

    def _get(self, nct_id):
        t = self.trials.get(str(nct_id).strip().upper())
        if t is None:
            raise ValueError(f"{nct_id} is not in the corpus")
        return t

    def search_trials(self, query, k=10):
        ids = [n for n, _ in self.index.search(query, max(1, min(int(k), 20)))]
        self.add_candidates(ids)
        self.searches.append(query)
        return {"hits": [{"nct_id": n, "title": truncate(self.trials[n]["title"], 100),
                          "conditions": truncate(", ".join(self.trials[n]["conditions"]), 80),
                          "judged": n in self.judgments} for n in ids]}

    def read_trial(self, nct_id):
        t, m = self._get(nct_id), self.cfg["max_criteria_shown"]
        crit = lambda items, p: [f"{p}{i}: {truncate(c, 250)}" for i, c in enumerate(items[:m])]
        out = {"nct_id": t["nct_id"], "title": t["title"],
               "conditions": ", ".join(t["conditions"]),
               "summary": truncate(t.get("brief_summary", ""), 400),
               "sex": t.get("gender", ""), "age": f"{t.get('min_age') or 'N/A'} to {t.get('max_age') or 'N/A'}",
               "inclusion": crit(t["inclusion"], "I"), "exclusion": crit(t["exclusion"], "E")}
        hidden = max(0, len(t["inclusion"]) - m) + max(0, len(t["exclusion"]) - m)
        if hidden:
            out["note"] = f"{hidden} more criteria not shown"
        return out

    def record_judgment(self, nct_id, verdict, confidence=50, reason=""):
        t = self._get(nct_id)
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        self.judgments[t["nct_id"]] = {"verdict": verdict, "confidence": int(confidence),
                                       "reason": truncate(str(reason), 400)}
        return {"ok": True, "trials_judged": len(self.judgments)}

    def ranking(self, depth):
        """eligible > uncertain > excluded > unjudged candidates > irrelevant.
        Ties within a verdict break on confidence; unjudged keep search order."""
        order = {"eligible": 0, "uncertain": 1, "excluded": 2}
        j = self.judgments
        judged = sorted((n for n in j if j[n]["verdict"] in order),
                        key=lambda n: (order[j[n]["verdict"]], -j[n]["confidence"]))
        unjudged = sorted((n for n in self.candidates if n not in j), key=lambda n: -self.candidates[n])
        irrelevant = [n for n in j if j[n]["verdict"] == "irrelevant"]
        return (judged + unjudged + irrelevant)[:depth]
