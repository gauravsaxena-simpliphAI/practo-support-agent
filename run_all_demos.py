"""Master Demonstration & End-to-End Test Runner for Practo Domain Support Agent.

Executes all 16 tasks across Parts 1-4, asserts all acceptance criteria,
and outputs complete formatted transcripts.
"""

import sys
import time

def print_banner(title: str):
    print("\n" + "#" * 84)
    print(f"## {title.center(78)} ##")
    print("#" * 84 + "\n")


def main():
    start_total = time.time()
    print_banner("PRACTO DOMAIN SUPPORT AGENT - FULL CAPSTONE DEMONSTRATION")
    print("Track: Healthcare (Practo)")
    print("Execution Mode: Deterministic MOCK_LLM (Zero Network, Zero API Keys)")

    # ----------------------------------------------------
    # PART 1: DATASET DESIGN & RAG CORE
    # ----------------------------------------------------
    print_banner("PART 1: DATASET DESIGN & RAG CORE (TASKS 1 - 5)")
    
    # Task 1: Dataset Validation
    from dataset import validate_and_report_dataset
    print(">>> Executing Task 1: Dataset Generation & Validation...")
    ds_report = validate_and_report_dataset()
    assert ds_report["total"] >= 40
    assert 10.0 <= ds_report["follow_up_pct"] <= 30.0

    # Task 3: ChromaDB Indexing
    from rag.indexing import build_and_index_collections
    print("\n>>> Executing Task 3: Indexing Collections in ChromaDB...")
    build_and_index_collections()

    # Task 4: Empirical Fallback Calibration & Grounded Generation
    from rag.grounded_generation import calibrate_threshold, demonstrate_grounded_generation
    print("\n>>> Executing Task 4: Empirical Calibration & Grounded Generation...")
    calib = calibrate_threshold()
    print(f"In-Scope Cluster: [{calib['min_in_scope']:.4f}, {calib['max_in_scope']:.4f}]")
    print(f"Out-of-Scope Cluster: [{calib['min_out_scope']:.4f}, {calib['max_out_scope']:.4f}]")
    print(f"Empirical Calibrated Threshold: {calib['recommended_threshold']:.4f}")
    demonstrate_grounded_generation()

    # Task 5: Chunking Strategy Comparative Evaluation
    from rag.evaluation import run_comparative_evaluation
    print("\n>>> Executing Task 5: Chunking Strategy Precision/Recall Evaluation...")
    run_comparative_evaluation()

    # ----------------------------------------------------
    # PART 2: CREWAI MULTI-AGENT ORCHESTRATION & GUARDRAILS
    # ----------------------------------------------------
    print_banner("PART 2: CREWAI MULTI-AGENT ORCHESTRATION (TASKS 6 - 10)")

    # Task 6: Appointment Lookup Tool with Escalation Score
    from tools.appointment_lookup import check_appointment_status, RECOMMENDED_ESCALATION_THRESHOLD
    print(">>> Executing Task 6: Appointment Lookup & Escalation Score...")
    sample_lookup = check_appointment_status("PRAC-1002")
    print(f"Sample Lookup PRAC-1002 -> Status: {sample_lookup['status']}, Fee: INR {sample_lookup['consultation_fee_inr']}, Escalation: {sample_lookup['escalation_score']} (Threshold: {RECOMMENDED_ESCALATION_THRESHOLD})")

    # Task 7 & Task 9: CrewAI Multi-Agent Kickoff & Structured Outputs
    from crew.support_crew import demonstrate_crew
    print("\n>>> Executing Task 7 & Task 9: CrewAI Multi-Agent Kickoff & Structured Validation...")
    demonstrate_crew()

    # Task 8: LangChain Session Memory
    from memory.session_memory import demonstrate_session_memory
    print("\n>>> Executing Task 8: LangChain Multi-Turn Session Memory...")
    demonstrate_session_memory()

    # Task 10: Input & Output Guardrails
    from guardrails.guardrails import demonstrate_guardrails
    print("\n>>> Executing Task 10: Guardrails Demonstration (PII, Injection, Groundedness)...")
    demonstrate_guardrails()

    # ----------------------------------------------------
    # PART 3: EVALUATION, OBSERVABILITY & FASTAPI
    # ----------------------------------------------------
    print_banner("PART 3: EVALUATION, OBSERVABILITY & FASTAPI (TASKS 11 - 13)")

    # Task 11 & Task 12: FastAPI Endpoints, WebSocket, and ELK Logging
    from tests.test_api import (
        test_health_endpoint,
        test_appointment_lookup_endpoint,
        test_ask_policy_endpoint,
        test_ask_pii_masking_and_logging,
        test_ask_prompt_injection_blocked,
        test_add_document_endpoint,
        test_websocket_chat_and_disconnect,
    )
    print(">>> Executing Task 11 & Task 12: FastAPI Endpoints, WebSocket Disconnect & ELK Logging...")
    test_health_endpoint()
    test_appointment_lookup_endpoint()
    test_ask_policy_endpoint()
    test_ask_pii_masking_and_logging()
    test_ask_prompt_injection_blocked()
    test_add_document_endpoint()
    test_websocket_chat_and_disconnect()
    print("FastAPI HTTP, WebSocket disconnect handling, and structured disk logging verified successfully!")

    # Task 13: 15-Query Evaluation at Scale
    from evaluation.eval_suite import run_full_evaluation
    print("\n>>> Executing Task 13: 15-Query Evaluation Suite (LLM-as-Judge)...")
    eval_benchmark = run_full_evaluation()
    print(f"Benchmark Averages -> Accuracy: {eval_benchmark['averages']['accuracy']:.2f}, Grounding: {eval_benchmark['averages']['grounding']:.2f}, Completeness: {eval_benchmark['averages']['completeness']:.2f}, Safety: {eval_benchmark['averages']['safety']:.2f}")

    # ----------------------------------------------------
    # PART 4: RESILIENCE & GOVERNANCE
    # ----------------------------------------------------
    print_banner("PART 4: RESILIENCE & GOVERNANCE (TASKS 14 - 16)")

    # Task 14: AutoGen Review Stage
    from governance.autogen_review import demonstrate_autogen_review
    print(">>> Executing Task 14: AutoGen 2-Agent Review Stage...")
    demonstrate_autogen_review()

    # Task 15: AI Governance Policy
    from governance.policy import demonstrate_governance
    print("\n>>> Executing Task 15: AI Governance Model (Least Autonomy, Risk, Budget Cap)...")
    demonstrate_governance()

    # Task 16: Response Caching
    from cache.response_cache import demonstrate_response_cache
    print("\n>>> Executing Task 16: Response Caching Demonstration...")
    demonstrate_response_cache()

    duration_total = time.time() - start_total
    print_banner(f"ALL 16 CAPSTONE TASKS COMPLETED SUCCESSFULLY ({duration_total:.1f}s)")


if __name__ == "__main__":
    main()
