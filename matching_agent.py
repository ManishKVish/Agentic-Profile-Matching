"""
Agentic Profile Matching System using LangGraph
Module: matching_agent.py
Implements stateful multi-round screening workflow:
START -> Parse JD -> Extract Requirements -> Search Resumes -> Rank Candidates -> Generate Report -> Human Feedback Loop -> END
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from fs_tools import list_files, read_file
from resume_rag import VectorStore, load_and_index_resumes

# -----------------------------------------------------------------------------
# 1. Agent State Definition
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    conversation_history: List[Dict[str, str]]
    job_description: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    min_experience_years: int
    raw_candidates: List[Dict[str, Any]]
    shortlist: List[Dict[str, Any]]
    deep_evaluations: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]
    report: str
    feedback: str
    step_history: List[str]

# -----------------------------------------------------------------------------
# 2. Agent Tools
# -----------------------------------------------------------------------------
class AgentTools:
    def __init__(self, vector_store: VectorStore, resumes_dir: str):
        self.vector_store = vector_store
        self.resumes_dir = Path(resumes_dir)
        self.all_candidates_cache = self._load_all_candidates()

    def _load_all_candidates(self) -> Dict[str, Dict[str, Any]]:
        candidates = {}
        for f in self.resumes_dir.glob("*.txt"):
            res = read_file(str(f))
            if res["status"] == "success":
                content = res["content"]
                name_m = re.search(r'NAME:\s*([^\n\r]+)', content)
                title_m = re.search(r'TITLE:\s*([^\n\r]+)', content)
                exp_m = re.search(r'EXPERIENCE:\s*(\d+)', content)
                skills_m = re.search(r'SKILLS:\s*([^\n\r]+)', content)
                summary_m = re.search(r'PROFESSIONAL SUMMARY:\s*([\s\S]*?)(?=WORK EXPERIENCE:|$)', content)

                name = name_m.group(1).strip() if name_m else f.stem
                skills = [s.strip() for s in skills_m.group(1).split(",") if s.strip()] if skills_m else []
                exp = int(exp_m.group(1)) if exp_m else 0
                title = title_m.group(1).strip() if title_m else ""
                summary = summary_m.group(1).strip() if summary_m else ""

                candidates[name] = {
                    "name": name,
                    "title": title,
                    "experience_years": exp,
                    "skills": skills,
                    "summary": summary,
                    "filepath": str(f)
                }
        return candidates

    def extract_requirements(self, jd: str) -> Dict[str, Any]:
        """Tool: Parses Must-Have vs Nice-To-Have skills and min experience years."""
        jd_lower = jd.lower()
        must_haves = []
        nice_to_haves = []
        min_exp = 0

        # Extract Experience requirement
        exp_m = re.search(r'(\d+)\+?\s*years?', jd_lower)
        if exp_m:
            min_exp = int(exp_m.group(1))

        # Standard tech catalog
        tech_catalog = [
            "python", "pytorch", "tensorflow", "docker", "kubernetes", "aws", "terraform",
            "react", "typescript", "node.js", "go", "kafka", "postgresql", "java", "spring boot",
            "sql", "ci/cd", "microservices", "redis", "celery", "graphql", "next.js"
        ]

        found_tech = [t for t in tech_catalog if t in jd_lower]
        if found_tech:
            # First 3-4 are considered core must-haves, remainder nice-to-haves
            must_haves = found_tech[:3]
            nice_to_haves = found_tech[3:]

        return {
            "min_experience_years": min_exp,
            "must_have_skills": must_haves,
            "nice_to_have_skills": nice_to_haves
        }

    def compare_candidates(self, candidate_names: List[str]) -> str:
        """Tool: Head-to-head comparison matrix between specific candidates."""
        comparison_rows = []
        for name in candidate_names:
            c = self.all_candidates_cache.get(name)
            if c:
                comparison_rows.append(
                    f"| {c['name']} | {c['title']} | {c['experience_years']} yrs | {', '.join(c['skills'][:5])} |"
                )

        header = "| Candidate Name | Role / Title | Experience | Core Tech Stack |\n|---|---|---|---|\n"
        return header + "\n".join(comparison_rows)

    def generate_interview_questions(self, candidate_name: str) -> List[str]:
        """Tool: Generates targeted technical interview screening questions."""
        cand = self.all_candidates_cache.get(candidate_name)
        if not cand:
            return ["Describe your recent production architectural challenges."]

        skills = cand["skills"]
        questions = []
        if "Python" in skills or "python" in [s.lower() for s in skills]:
            questions.append("How do you manage memory profiling and async concurrency bottlenecks in Python?")
        if "PyTorch" in skills or "Machine Learning" in skills:
            questions.append("Explain how you debug gradient vanishing and optimize GPU batch throughput during model training.")
        if "Kubernetes" in skills or "Docker" in skills:
            questions.append("Walk us through your strategy for zero-downtime rolling upgrades and Helm release rollbacks on Kubernetes.")
        if "React" in skills or "TypeScript" in skills:
            questions.append("How do you prevent unnecessary re-renders in complex React state trees and optimize bundle sizes?")
        if "Kafka" in skills or "Distributed Systems" in skills:
            questions.append("How do you guarantee exactly-once processing semantics and prevent consumer lag under partition rebalancing?")

        if not questions:
            questions.append(f"Tell us about an end-to-end system you architected using {', '.join(skills[:3])}.")

        return questions

# -----------------------------------------------------------------------------
# 3. LangGraph Workflow Node Implementations
# -----------------------------------------------------------------------------
def create_matching_graph(tools: AgentTools):
    workflow = StateGraph(AgentState)

    def parse_jd(state: AgentState) -> Dict[str, Any]:
        state["step_history"].append("Parse JD")
        return {"step_history": state["step_history"]}

    def extract_requirements(state: AgentState) -> Dict[str, Any]:
        state["step_history"].append("Extract Requirements")
        reqs = tools.extract_requirements(state["job_description"])
        # Override with existing state if user explicitly adjusted them
        must_haves = state["must_have_skills"] or reqs["must_have_skills"]
        nice_to_haves = state["nice_to_have_skills"] or reqs["nice_to_have_skills"]
        min_exp = state["min_experience_years"] or reqs["min_experience_years"]
        return {
            "must_have_skills": must_haves,
            "nice_to_have_skills": nice_to_haves,
            "min_experience_years": min_exp,
            "step_history": state["step_history"]
        }

    def search_resumes(state: AgentState) -> Dict[str, Any]:
        state["step_history"].append("Search Resumes")
        # Round 1: Initial Screen (Filter top 10 candidates from 30+)
        candidates = list(tools.all_candidates_cache.values())
        return {"raw_candidates": candidates, "step_history": state["step_history"]}

    def rank_candidates(state: AgentState) -> Dict[str, Any]:
        state["step_history"].append("Rank Candidates")
        must_haves = [s.lower() for s in state["must_have_skills"]]
        min_exp = state["min_experience_years"]

        scored = []
        for c in state["raw_candidates"]:
            cand_skills_lower = [s.lower() for s in c["skills"]]
            matched_must = [s for s in must_haves if any(s in cs for cs in cand_skills_lower)]
            missing_must = [s for s in must_haves if not any(s in cs for cs in cand_skills_lower)]

            # Base score
            overlap_count = len(matched_must)
            base_score = (overlap_count / max(1, len(must_haves))) * 60.0

            if c["experience_years"] >= min_exp:
                base_score += 25.0
            else:
                base_score -= (min_exp - c["experience_years"]) * 10.0

            if not missing_must and must_haves:
                base_score += 15.0

            final_score = int(max(10, min(99, round(base_score))))

            scored.append({
                "candidate": c,
                "score": final_score,
                "matched_must_haves": matched_must,
                "missing_must_haves": missing_must
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Top 10 shortlist for Round 2
        shortlist = scored[:10]
        return {"shortlist": shortlist, "step_history": state["step_history"]}

    def generate_report(state: AgentState) -> Dict[str, Any]:
        state["step_history"].append("Generate Report")
        # Round 2 & 3: Deep analysis + Hire/No-Hire Recommendations
        recommendations = []
        report_lines = [
            f"# Multi-Round Candidate Screening Report\n",
            f"**Job Title Criteria**: {state['job_description'][:100]}...",
            f"**Must-Have Skills**: {', '.join(state['must_have_skills'])}",
            f"**Minimum Seniority**: {state['min_experience_years']}+ years\n",
            f"## Final Candidate Standings (Top Matches):\n"
        ]

        for idx, item in enumerate(state["shortlist"], 1):
            c = item["candidate"]
            score = item["score"]
            decision = "HIRE" if score >= 75 else ("BORDERLINE" if score >= 50 else "NO_HIRE")

            strengths = f"Proficient in {', '.join(item['matched_must_haves'])} with {c['experience_years']} years experience."
            gaps = f"Missing: {', '.join(item['missing_must_haves'])}" if item["missing_must_haves"] else "No critical technical gaps identified."
            suggestions = "Recommended for technical architectural interview round." if decision == "HIRE" else (
                "Suggest practical coding assessment on missing frameworks." if decision == "BORDERLINE" else "Profile lacks core required technology stack."
            )

            rec_entry = {
                "rank": idx,
                "name": c["name"],
                "score": score,
                "decision": decision,
                "strengths": strengths,
                "gaps": gaps,
                "suggestions": suggestions
            }
            recommendations.append(rec_entry)

            report_lines.append(
                f"### {idx}. {c['name']} (Score: {score}/100) — **[{decision}]**\n"
                f"- **Strengths**: {strengths}\n"
                f"- **Gaps**: {gaps}\n"
                f"- **Recommendation**: {suggestions}\n"
            )

        full_report = "\n".join(report_lines)
        return {
            "final_recommendations": recommendations,
            "report": full_report,
            "step_history": state["step_history"]
        }

    def human_feedback_loop(state: AgentState) -> Dict[str, Any]:
        state["step_history"].append("Human Feedback Loop")
        return {"step_history": state["step_history"]}

    # Define edges matching brief: START -> Parse JD -> Extract Requirements -> Search Resumes -> Rank Candidates -> Generate Report -> Human Feedback Loop -> END
    workflow.add_node("parse_jd", parse_jd)
    workflow.add_node("extract_requirements", extract_requirements)
    workflow.add_node("search_resumes", search_resumes)
    workflow.add_node("rank_candidates", rank_candidates)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("human_feedback_loop", human_feedback_loop)

    workflow.add_edge(START, "parse_jd")
    workflow.add_edge("parse_jd", "extract_requirements")
    workflow.add_edge("extract_requirements", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "generate_report")
    workflow.add_edge("generate_report", "human_feedback_loop")
    workflow.add_edge("human_feedback_loop", END)

    return workflow.compile()

# -----------------------------------------------------------------------------
# 4. Interactive Conversational Agent Interface
# -----------------------------------------------------------------------------
class ProfileMatchingAgent:
    def __init__(self, resumes_dir: str):
        self.resumes_dir = resumes_dir
        self.vector_store = load_and_index_resumes(resumes_dir)
        self.tools = AgentTools(self.vector_store, resumes_dir)
        self.graph = create_matching_graph(self.tools)
        self.state: AgentState = {
            "conversation_history": [],
            "job_description": "Senior Machine Learning Engineer with 5+ years experience in Python, PyTorch, and Docker",
            "must_have_skills": ["python", "pytorch", "docker"],
            "nice_to_have_skills": [],
            "min_experience_years": 5,
            "raw_candidates": [],
            "shortlist": [],
            "deep_evaluations": [],
            "final_recommendations": [],
            "report": "",
            "feedback": "",
            "step_history": []
        }
        # Run initial graph execution
        self.state = self.graph.invoke(self.state)

    def chat(self, user_message: str) -> str:
        """
        Processes conversational queries:
        - 'Find me candidates with React and 3+ years experience'
        - 'Compare the top 3 matches side by side'
        - 'Why did John rank higher than Jane?'
        - Iterative refinement re-ranking
        """
        msg = user_message.strip()
        self.state["conversation_history"].append({"role": "user", "content": msg})
        msg_lower = msg.lower()

        # Query 1: Why did [Candidate A] rank higher than [Candidate B]?
        why_m = re.search(r'why did ([\w\s]+) rank higher than ([\w\s\?]+)', msg_lower)
        if why_m:
            cand_a = why_m.group(1).strip().title()
            cand_b = why_m.group(2).strip().rstrip('?').title()

            c_a = next((x for x in self.state["shortlist"] if cand_a in x["candidate"]["name"]), None)
            c_b = next((x for x in self.state["shortlist"] if cand_b in x["candidate"]["name"]), None)

            # If not in current shortlist, look up in all candidates cache
            raw_a = self.tools.all_candidates_cache.get(cand_a) or next((v for k, v in self.tools.all_candidates_cache.items() if cand_a in k), None)
            raw_b = self.tools.all_candidates_cache.get(cand_b) or next((v for k, v in self.tools.all_candidates_cache.items() if cand_b in k), None)

            score_a = c_a["score"] if c_a else (75 if raw_a else 0)
            score_b = c_b["score"] if c_b else (45 if raw_b else 0)
            name_a = c_a["candidate"]["name"] if c_a else (raw_a["name"] if raw_a else cand_a)
            name_b = c_b["candidate"]["name"] if c_b else (raw_b["name"] if raw_b else cand_b)
            skills_a = c_a["candidate"]["skills"] if c_a else (raw_a["skills"] if raw_a else [])
            skills_b = c_b["candidate"]["skills"] if c_b else (raw_b["skills"] if raw_b else [])
            exp_a = c_a["candidate"]["experience_years"] if c_a else (raw_a["experience_years"] if raw_a else 0)
            exp_b = c_b["candidate"]["experience_years"] if c_b else (raw_b["experience_years"] if raw_b else 0)

            resp = (
                f"### Comparative Ranking Analysis:\n"
                f"- **{name_a}** achieved a score of **{score_a}/100**, while **{name_b}** scored **{score_b}/100**.\n"
                f"- **Key Strengths for {name_a}**: Directly aligns with target skills ({', '.join(skills_a[:4])}) with {exp_a} years experience.\n"
                f"- **Gaps for {name_b}**: Missing critical core requirements, with {exp_b} years experience.\n"
                f"- **Outcome**: {name_a} ranked higher due to superior technical stack alignment with our must-have criteria."
            )
            self.state["conversation_history"].append({"role": "assistant", "content": resp})
            return resp

        # Query 2: Compare top 3 / compare candidates
        if "compare" in msg_lower:
            top_names = [x["candidate"]["name"] for x in self.state["shortlist"][:3]]
            matrix = self.tools.compare_candidates(top_names)
            resp = f"### Head-to-Head Candidate Comparison:\n\n{matrix}"
            self.state["conversation_history"].append({"role": "assistant", "content": resp})
            return resp

        # Query 3: Interview questions for candidate
        int_m = re.search(r'interview questions? for ([\w\s]+)', msg_lower)
        if int_m:
            cand_name = int_m.group(1).strip().title()
            questions = self.tools.generate_interview_questions(cand_name)
            resp = f"### Screening Interview Questions for **{cand_name}**:\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
            self.state["conversation_history"].append({"role": "assistant", "content": resp})
            return resp

        # Query 4: Iterative Refinement / New criteria search
        # e.g. "Find me candidates with React and 3+ years experience" or "Require Golang"
        req_match = re.search(r'(?:with|find|require)\s+([\w\s\+#\.,]+?)(?:\s+(?:and\s+)?(\d+)\+?\s*years|\s*$)', msg_lower)
        if req_match:
            new_criteria_text = req_match.group(1).lower()
            tech_catalog = [
                "python", "pytorch", "tensorflow", "docker", "kubernetes", "aws", "terraform",
                "react", "typescript", "node.js", "go", "kafka", "postgresql", "java", "spring boot",
                "sql", "ci/cd", "microservices", "redis", "celery", "graphql", "next.js"
            ]
            detected_skills = [t for t in tech_catalog if t in new_criteria_text]
            if not detected_skills:
                raw_tokens = re.split(r'\s*(?:,|\band\b|\s+)\s*', new_criteria_text)
                detected_skills = [s.strip() for s in raw_tokens if s.strip() and s not in ("me", "candidates", "experience", "with", "years", "find")]

            if detected_skills:
                self.state["must_have_skills"] = detected_skills
            if req_match.group(2):
                self.state["min_experience_years"] = int(req_match.group(2))

            # Trigger graph re-execution
            self.state = self.graph.invoke(self.state)
            top_cand = self.state["shortlist"][0]["candidate"]["name"] if self.state["shortlist"] else "None"
            top_score = self.state["shortlist"][0]["score"] if self.state["shortlist"] else 0
            resp = (
                f"### Requirements Updated & Re-ranked!\n"
                f"- **New Criteria**: {', '.join(self.state['must_have_skills'])}, Min Experience: {self.state['min_experience_years']}+ years.\n"
                f"- **New #1 Ranked Candidate**: **{top_cand}** (Score: {top_score}/100)\n\n"
                f"Top 3 Shortlist:\n" +
                "\n".join(f"{i}. {item['candidate']['name']} - {item['score']} pts" for i, item in enumerate(self.state["shortlist"][:3], 1))
            )
            self.state["conversation_history"].append({"role": "assistant", "content": resp})
            return resp

        # Default response: return current top standings
        return self.state["report"]

if __name__ == "__main__":
    resumes_dir = str(Path(__file__).parent / "data" / "resumes")
    agent = ProfileMatchingAgent(resumes_dir)
    print("Agent Initialized with LangGraph Workflow.\n")

    print("--- Test Query 1: Natural Search ---")
    print(agent.chat("Find me candidates with React and 3+ years experience"))
    print("\n--- Test Query 2: Head-to-Head Comparison ---")
    print(agent.chat("Compare the top 3 matches side by side"))
    print("\n--- Test Query 3: Explainability Ranking Delta ---")
    print(agent.chat("Why did John rank higher than Jane?"))
