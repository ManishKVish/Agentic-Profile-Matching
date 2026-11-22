# Agentic Profile Matching with LangGraph

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Multi-Round Screening](https://img.shields.io/badge/Screening-Multi--Round-success.svg)]()
[![Airtribe Assignment](https://img.shields.io/badge/Airtribe-Backend%20AI%20Track-orange.svg)]()

This repository contains the complete implementation for **Milestone 3 (Agentic Profile Matching)** of the Airtribe Backend AI track.

---

## 🎯 Architecture & Workflow

Built using **LangGraph**, this autonomous agent executes a structured state machine:

$$\text{START} \longrightarrow \text{Parse JD} \longrightarrow \text{Extract Requirements} \longrightarrow \text{Search Resumes} \longrightarrow \text{Rank Candidates} \longrightarrow \text{Generate Report} \longrightarrow \text{Human Feedback Loop} \longrightarrow \text{END}$$

### Key Capabilities
- **Agent State Tracking**: Maintains conversation history, job description, must-have vs nice-to-have skills, candidate shortlist, and reasoning.
- **Tools**:
  - `fs_tools` (from Milestone 1) & `resume_rag` (from Milestone 2).
  - `extract_requirements(jd)`: Automatically segments must-have vs nice-to-have skills.
  - `compare_candidates(candidate_ids)`: Generates head-to-head Markdown comparison matrices.
  - `generate_interview_questions(candidate_id)`: Generates customized technical screening questions.
- **Conversational Queries Supported**:
  - *"Find me candidates with React and 3+ years experience"*
  - *"Compare the top 3 matches side by side"*
  - *"Why did John rank higher than Jane?"*
  - *"Interview questions for John Doe"*
- **Multi-Round Screening**:
  - Round 1: Filter top 10 from 30+ resume database.
  - Round 2: Deep evaluation against criteria and gap analysis.
  - Round 3: Final Hire / No-Hire recommendation with improvement suggestions for borderline candidates.

---

## 📁 Repository Structure

```
├── matching_agent.py         # Part A: LangGraph StateGraph workflow & interactive agent
├── app.py                    # Part B: Interactive chat interface (CLI & Streamlit)
├── fs_tools.py               # File system tools
├── resume_rag.py             # Vector store & RAG document chunking
├── docs/
│   └── state_machine.md      # State machine diagram and architecture breakdown
├── data/
│   └── resumes/              # 30+ sample candidate resumes
├── tests/
│   └── test_matching_agent.py# 6 conversational test scenarios
├── requirements.txt
└── README.md
```

---

## 🏃 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated test scenarios
python -m unittest discover tests

# 3. Launch interactive recruiter chat assistant
python app.py
```
