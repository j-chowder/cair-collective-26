"""BM25 keyword search over trials (pure Python, saved to disk after the first build)."""
import heapq
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by for from has have he her his in is it its of on or she "
            "that the their this to was were with year years old patient presents history".split())


def tokenize(text):
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def _trial_text(t):
    title, conds = t.get("title", ""), " ".join(t.get("conditions", []))
    return " ".join([title, title, conds, conds, " ".join(t.get("interventions", [])),
                     t.get("brief_summary", ""), " ".join(t.get("inclusion", []))])


class BM25:
    def __init__(self, trials, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        postings = defaultdict(list)
        self.ids, self.lens = list(trials), []
        for i, nct in enumerate(self.ids):
            toks = tokenize(_trial_text(trials[nct]))
            self.lens.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        n = len(self.ids)
        self.avgdl = sum(self.lens) / n if n else 1.0
        self.postings = dict(postings)
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in postings.items()}

    def search(self, query, k=20):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, []):
                norm = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return [(self.ids[i], s) for i, s in heapq.nlargest(k, scores.items(), key=lambda x: x[1])]


def load_index(cfg, trials):
    path = Path(cfg["paths"]["index"])
    if path.exists():
        with open(path, "rb") as f:
            index = pickle.load(f)
        if len(index.ids) == len(trials):
            return index
    print(f"Building search index over {len(trials)} trials (one time, free)...")
    index = BM25(trials)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return index
