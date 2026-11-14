"""Free tests using the fake model. Run: python -m unittest discover tests"""
import json
import shutil
import tempfile
import unittest

from trialmatch_agent.agent import Agent
from trialmatch_agent.config import load_config
from trialmatch_agent.data import load_topics, load_trials, split_criteria
from trialmatch_agent.evaluate import ndcg_at_10
from trialmatch_agent.llm import LLM
from trialmatch_agent.retrieval import load_index


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = load_config("configs/sample.json")
        self.cfg["paths"]["outputs"] = self.tmp
        self.cfg["paths"]["index"] = f"{self.tmp}/index.pkl"
        trials = load_trials(self.cfg["paths"]["trials"])
        self.llm = LLM(self.cfg, mock=True)
        self.agent = Agent(self.cfg, self.llm, trials, load_index(self.cfg, trials), verbose=False)
        self.topics = load_topics(self.cfg["paths"]["topics"])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_full_run(self):
        res = self.agent.run("1", self.topics["1"], budget=20)
        self.assertEqual(res["stop_reason"], "finished")
        self.assertEqual(len(res["judgments"]), 3)
        self.assertEqual(len(res["ranking"]), len(set(res["ranking"])))
        self.assertEqual(self.llm.total, 0.0)  # mock never spends
        json.dumps(res)

    def test_budget_enforced(self):
        res = self.agent.run("2", self.topics["2"], budget=3)
        self.assertEqual((res["tool_calls"], res["stop_reason"]), (3, "budget_used"))

    def test_criteria_parsing(self):
        inc, exc = split_criteria("Inclusion Criteria:\n\n - Age 18+\n\n - Asthma for\n   1 year\n\n"
                                  "Exclusion Criteria:\n\n - Pregnancy\n")
        self.assertEqual((inc, exc), (["Age 18+", "Asthma for 1 year"], ["Pregnancy"]))

    def test_ndcg(self):
        self.assertAlmostEqual(ndcg_at_10(["A", "B"], {"A": 2, "B": 1}), 1.0)


if __name__ == "__main__":
    unittest.main()
