import unittest
from pathlib import Path
from matching_agent import ProfileMatchingAgent

class TestAgenticProfileMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resumes_dir = str(Path(__file__).parent.parent / "data" / "resumes")
        cls.agent = ProfileMatchingAgent(cls.resumes_dir)

    def test_flow_1_initial_graph_execution(self):
        """Validates LangGraph initial pipeline execution."""
        state = self.agent.state
        self.assertIn("step_history", state)
        self.assertIn("Parse JD", state["step_history"])
        self.assertIn("Extract Requirements", state["step_history"])
        self.assertIn("Search Resumes", state["step_history"])
        self.assertIn("Rank Candidates", state["step_history"])
        self.assertIn("Generate Report", state["step_history"])
        self.assertIn("Human Feedback Loop", state["step_history"])
        self.assertGreaterEqual(len(state["shortlist"]), 5)

    def test_flow_2_natural_language_search(self):
        """Conversational query: Find candidates with React and 3+ years experience."""
        resp = self.agent.chat("Find me candidates with React and 3+ years experience")
        self.assertIn("Requirements Updated & Re-ranked", resp)
        self.assertIn("React", str(self.agent.state["must_have_skills"]).title())
        top_cand = self.agent.state["shortlist"][0]["candidate"]["name"]
        self.assertIn(top_cand, ["Ananya Verma", "Jane Smith", "Lucas Silva"])

    def test_flow_3_head_to_head_comparison(self):
        """Conversational query: Compare the top 3 matches side by side."""
        resp = self.agent.chat("Compare the top 3 matches side by side")
        self.assertIn("Head-to-Head Candidate Comparison", resp)
        self.assertIn("| Candidate Name | Role / Title |", resp)

    def test_flow_4_explainability_ranking_delta(self):
        """Conversational query: Why did John rank higher than Jane?"""
        resp = self.agent.chat("Why did John rank higher than Jane?")
        self.assertIn("Comparative Ranking Analysis", resp)
        self.assertIn("John Doe", resp)
        self.assertIn("Jane Smith", resp)

    def test_flow_5_interview_question_generation(self):
        """Conversational query: Interview questions for John Doe."""
        resp = self.agent.chat("Interview questions for John Doe")
        self.assertIn("Screening Interview Questions", resp)
        self.assertIn("John Doe", resp)

    def test_flow_6_iterative_refinement(self):
        """Iterative refinement: Add strict Kubernetes requirement."""
        resp = self.agent.chat("Find candidates with Kubernetes and 5+ years experience")
        self.assertIn("Requirements Updated & Re-ranked", resp)
        top_names = [x["candidate"]["name"] for x in self.agent.state["shortlist"][:5]]
        # Candidates with Kubernetes and 5+ yrs (Priya Sharma, Carlos Rodriguez, Deepak Verma)
        self.assertTrue(any(name in ["Priya Sharma", "Carlos Rodriguez", "Deepak Verma"] for name in top_names))

if __name__ == "__main__":
    unittest.main()
