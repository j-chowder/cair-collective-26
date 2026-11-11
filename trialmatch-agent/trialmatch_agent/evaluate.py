"""How good is the ranking? Uses TREC grades: 2 = eligible, 1 = excluded, 0 = not relevant.
Unjudged trials count as not relevant."""
import math


def ndcg_at_10(ranked, grades):
    dcg = lambda gs: sum(g / math.log2(i + 2) for i, g in enumerate(gs))
    ideal = dcg(sorted(grades.values(), reverse=True)[:10])
    return dcg([grades.get(n, 0) for n in ranked[:10]]) / ideal if ideal else 0.0


def precision_at_10(ranked, grades):
    return sum(grades.get(n, 0) == 2 for n in ranked[:10]) / 10


def recall_at_100(ranked, grades):
    eligible = {n for n, g in grades.items() if g == 2}
    return len(eligible & set(ranked[:100])) / len(eligible) if eligible else 0.0


def evaluate(run, qrels):
    """Returns ({topic: {metric: value}}, {metric: mean})."""
    per = {t: {"nDCG@10": ndcg_at_10(r, qrels[t]), "P@10 (eligible)": precision_at_10(r, qrels[t]),
               "Recall@100 (eligible)": recall_at_100(r, qrels[t])}
           for t, r in run.items() if t in qrels}
    means = {m: sum(v[m] for v in per.values()) / len(per) for m in next(iter(per.values()), {})}
    return per, means
