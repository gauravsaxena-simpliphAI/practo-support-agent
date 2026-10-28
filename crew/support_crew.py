"""CrewAI multi-agent orchestration for Practo Domain Support Agent.

Implements a 3-agent Crew:
1. Retrieval Agent (equipped with rag_kb_search tool)
2. Lookup Agent (equipped with check_appointment_status tool)
3. Response Composer Agent (synthesizes findings without direct database tools - Least Autonomy)

Conforms to Task 7 and Task 9 (Pydantic structured output validation).
"""

import os
import re
import json
from typing import Dict, Any, List

# Ensure telemetry is disabled before any CrewAI imports
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from crew.mock_llm import MockCrewLLM
from tools.appointment_lookup import check_appointment_status as lookup_fn
from rag.grounded_generation import generate_grounded_answer
from schemas.models import AgentResponse

# 1. Define CrewAI Tools
@tool("rag_kb_search")
def rag_kb_search(query: str) -> str:
    """Retrieve official clinic policy documents from the Practo Knowledge Base."""
    res = generate_grounded_answer(query, collection_name="practo_kb_sentence", top_k=2)
    return (
        f"[KB Search Result]\n"
        f"Grounded: {res['grounded']}\n"
        f"Sources: {', '.join(res['sources'])}\n"
        f"Content: {res['answer']}"
    )


@tool("check_appointment_status")
def check_appointment_status_tool(record_id: str) -> str:
    """Look up appointment record by ID (e.g. PRAC-1002) in clinic database."""
    res = lookup_fn(record_id)
    if not res.get("found"):
        return f"[Appointment Record]\nNot Found: {res.get('error')}"
    return (
        f"[Appointment Record {res['record_id']}]\n"
        f"Category: {res['category']}\n"
        f"Status: {res['status']}\n"
        f"Consultation Fee: INR {res['consultation_fee_inr']}\n"
        f"Days Since Created: {res['days_since_created']}\n"
        f"Follow-Up Required: {res['follow_up_required']}\n"
        f"Escalation Score: {res['escalation_score']} (Threshold: {res['escalation_threshold']})\n"
        f"Escalation Recommended: {res['escalation_recommended']}"
    )


# 2. Build Agent Team Factory
def create_practo_crew(query: str):
    llm = MockCrewLLM()

    # Agent 1: Retrieval Agent
    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Retrieve authoritative clinic policy guidelines from the Practo Knowledge Base.",
        backstory="Senior clinic policy officer trained on official Practo healthcare operating standards.",
        tools=[rag_kb_search],
        llm=llm,
        verbose=False,
    )

    # Agent 2: Lookup Agent (Only agent permitted to access patient appointment records)
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Query patient appointment records, verify statuses, and compute calibrated escalation scores.",
        backstory="Practo clinic database administrator with secure access to appointment records.",
        tools=[check_appointment_status_tool],
        llm=llm,
        verbose=False,
    )

    # Agent 3: Response Composer Agent (Least autonomy: no direct tools, synthesizes drafts)
    composer_agent = Agent(
        role="Response Composer",
        goal="Synthesize policy and appointment findings into a grounded, empathetic, patient-facing response.",
        backstory="Lead patient-experience specialist dedicated to providing accurate, verified support.",
        tools=[],  # Strictly no tools attached (Principle of Least Autonomy)
        llm=llm,
        verbose=False,
    )

    return retrieval_agent, lookup_agent, composer_agent


def run_practo_crew(query: str) -> AgentResponse:
    """Run CrewAI multi-agent workflow on a user query and validate against AgentResponse schema."""
    retrieval_agent, lookup_agent, composer_agent = create_practo_crew(query)
    is_appointment_query = bool(re.search(r"PRAC-\d+", query, re.IGNORECASE) or "appointment" in query.lower() and ("status" in query.lower() or "check" in query.lower()))

    tasks = []
    if is_appointment_query:
        # Route to Lookup Agent
        m = re.search(r"PRAC-\d+", query, re.IGNORECASE)
        rec_id = m.group(0).upper() if m else "PRAC-1002"
        task1 = Task(
            description=f"Look up appointment status and escalation score for {rec_id}. Query: {query}",
            expected_output="Detailed appointment status and escalation assessment.",
            agent=lookup_agent,
        )
        task2 = Task(
            description=f"Compose an empathetic patient response summarizing appointment {rec_id}.",
            expected_output="Final patient-facing response.",
            agent=composer_agent,
        )
        tasks = [task1, task2]
        crew = Crew(agents=[lookup_agent, composer_agent], tasks=tasks, verbose=False)
        raw_result = str(crew.kickoff())

        # Perform lookup for structured attributes
        lookup_data = lookup_fn(rec_id)
        structured_response = AgentResponse(
            query=query,
            category="appointment_status",
            answer=raw_result,
            sources=[rec_id],
            escalation_flag=lookup_data.get("escalation_recommended", False),
            escalation_score=lookup_data.get("escalation_score", 0.0),
            confidence=0.98,
        )
    else:
        # Route to Retrieval Agent
        task1 = Task(
            description=f"Retrieve official clinic policy for query: {query}",
            expected_output="Authoritative policy excerpts from knowledge base.",
            agent=retrieval_agent,
        )
        task2 = Task(
            description="Compose a clear, grounded patient response based on the retrieved policy.",
            expected_output="Final patient-facing policy explanation.",
            agent=composer_agent,
        )
        tasks = [task1, task2]
        crew = Crew(agents=[retrieval_agent, composer_agent], tasks=tasks, verbose=False)
        raw_result = str(crew.kickoff())

        # Grounded check on query
        rag_res = generate_grounded_answer(query)
        structured_response = AgentResponse(
            query=query,
            category="policy_inquiry" if rag_res["grounded"] else "out_of_scope",
            answer=rag_res["answer"],
            sources=rag_res["sources"],
            escalation_flag=False,
            escalation_score=None,
            confidence=0.95 if rag_res["grounded"] else 0.20,
        )

    # Validate against declared Pydantic schema
    validated = AgentResponse.model_validate(structured_response.model_dump())
    return validated


def demonstrate_crew():
    """Demonstrate both RAG tool and Lookup tool invocations on different sample queries."""
    print("=" * 80)
    print("CREWAI MULTI-AGENT ORCHESTRATION DEMONSTRATION (TASK 7 & TASK 9)")
    print("=" * 80)

    # Case 1: Policy inquiry -> Demonstrates RAG tool invocation
    policy_query = "What is the cancellation window and refund policy for clinic appointments?"
    print(f"\n[DEMO 1: Policy Inquiry] Invoking Crew on: '{policy_query}'")
    resp_policy = run_practo_crew(policy_query)
    print("Structured Output Validation: SUCCESS (AgentResponse model)")
    print(f"  Category: {resp_policy.category}")
    print(f"  Answer:   {resp_policy.answer}")
    print(f"  Sources:  {resp_policy.sources}")
    print(f"  Escalation Flag: {resp_policy.escalation_flag}")

    # Case 2: Appointment inquiry -> Demonstrates check_appointment_status tool invocation
    appt_query = "Please check the status and fee for appointment PRAC-1002."
    print(f"\n[DEMO 2: Appointment Inquiry] Invoking Crew on: '{appt_query}'")
    resp_appt = run_practo_crew(appt_query)
    print("Structured Output Validation: SUCCESS (AgentResponse model)")
    print(f"  Category: {resp_appt.category}")
    print(f"  Answer:   {resp_appt.answer}")
    print(f"  Sources:  {resp_appt.sources}")
    print(f"  Escalation Score: {resp_appt.escalation_score}")
    print(f"  Escalation Flag:  {resp_appt.escalation_flag}")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_crew()
