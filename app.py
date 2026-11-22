"""
Interactive Chat Interface for Agentic Profile Matching (CLI + Streamlit Compatible)
Module: app.py
Run in CLI mode: python app.py
Run in Streamlit mode: streamlit run app.py
"""
import sys
from pathlib import Path
from matching_agent import ProfileMatchingAgent

def run_cli():
    resumes_dir = str(Path(__file__).parent / "data" / "resumes")
    print("=" * 70)
    print("  Agentic Profile Matching - Interactive Assistant")
    print("=" * 70)
    print("Type your questions or criteria updates (e.g.):")
    print("  - 'Find me candidates with React and 3+ years experience'")
    print("  - 'Compare the top 3 matches side by side'")
    print("  - 'Why did John rank higher than Jane?'")
    print("  - 'Interview questions for John Doe'")
    print("Type 'exit' to quit.\n")

    agent = ProfileMatchingAgent(resumes_dir)
    print("\nInitial Baseline Standings Generated:\n")
    print(agent.state["report"])

    while True:
        try:
            user_input = input("\nRecruiter Query > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                print("Session ended.")
                break

            response = agent.chat(user_input)
            print(f"\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    run_cli()
