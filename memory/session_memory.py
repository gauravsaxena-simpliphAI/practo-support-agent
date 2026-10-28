"""LangChain session-based memory for Practo Domain Support Agent.

Uses InMemoryChatMessageHistory and RunnableWithMessageHistory to maintain
in-process conversation history across multiple turns within one session.
Conforms to Task 8 requirements.
"""

import re
from typing import Dict, Any, Optional
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from tools.appointment_lookup import check_appointment_status
from rag.grounded_generation import generate_grounded_answer

# In-process session history registry
_SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieve or create in-memory chat history for a session."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def clear_session_history(session_id: str) -> None:
    """Clear session history for testing or resetting."""
    if session_id in _SESSION_STORE:
        _SESSION_STORE[session_id].clear()


def _chat_turn_logic(input_dict: Dict[str, Any]) -> str:
    """Core conversational logic that inspects current input and historical context."""
    user_query = input_dict.get("query", "").strip()
    history_messages: list[BaseMessage] = input_dict.get("history", [])

    # Scan history for previously referenced appointment ID
    active_record_id: Optional[str] = None
    historical_text = " ".join([m.content for m in history_messages if isinstance(m, (HumanMessage, AIMessage))])

    # Check for direct record ID in current query
    m_curr = re.search(r"PRAC-\d+", user_query, re.IGNORECASE)
    if m_curr:
        active_record_id = m_curr.group(0).upper()
    else:
        # Check if an appointment was mentioned in prior turns
        m_hist = re.findall(r"PRAC-\d+", historical_text, re.IGNORECASE)
        if m_hist:
            active_record_id = m_hist[-1].upper()

    # If the user is asking an appointment follow-up question
    is_appt_question = any(k in user_query.lower() for k in ["fee", "status", "follow-up", "follow up", "appointment", "doctor", "specialty"])

    if is_appt_question:
        if active_record_id:
            data = check_appointment_status(active_record_id)
            if not data.get("found"):
                return f"I remembered your appointment ID {active_record_id}, but it was not found in the records."

            fu_str = "does require a follow-up consultation" if data["follow_up_required"] else "does not require a follow-up"
            return (
                f"Recalling your previous appointment reference ({active_record_id}): "
                f"The consultation fee was INR {data['consultation_fee_inr']}, "
                f"the status is '{data['status']}', and it {fu_str} (Escalation score: {data['escalation_score']})."
            )
        else:
            # Check if this is a fresh conversation without appointment context
            if any(k in user_query.lower() for k in ["that appointment", "my fee", "what was", "my appointment"]):
                return (
                    "I cannot find any prior appointment mentioned in our current conversation. "
                    "Could you please share your appointment record ID (e.g., PRAC-1002) so I can check for you?"
                )

    # General RAG query
    rag_res = generate_grounded_answer(user_query)
    return rag_res["answer"]


# Construct Runnable with Message History
_base_runnable = RunnableLambda(_chat_turn_logic)
conversational_chain = RunnableWithMessageHistory(
    _base_runnable,
    get_session_history=get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)


def ask_conversational_agent(query: str, session_id: str = "default_session") -> str:
    """Invoke conversational agent maintaining multi-turn memory."""
    config = {"configurable": {"session_id": session_id}}
    result = conversational_chain.invoke({"query": query}, config=config)
    return str(result)


def demonstrate_session_memory():
    """Demonstrate state carry-over across turns vs. fresh conversation absent state."""
    print("=" * 80)
    print("TASK 8: SESSION MEMORY DEMONSTRATION")
    print("=" * 80)

    # Transcript 1: Multi-turn conversation with memory carry-over
    session_alpha = "patient_session_alpha"
    clear_session_history(session_alpha)

    print(f"\n--- TRANSCRIPT 1: Multi-Turn Conversation (Session: {session_alpha}) ---")

    turn1_q = "Hi, my appointment record is PRAC-1002. What is its current status and fee?"
    print(f"\n[Turn 1] User: {turn1_q}")
    turn1_a = ask_conversational_agent(turn1_q, session_id=session_alpha)
    print(f"[Turn 1] Agent: {turn1_a}")

    # Notice: In Turn 2, the user does NOT mention PRAC-1002 or the fee, relying entirely on memory
    turn2_q = "Does that appointment require a follow-up, and what was my fee again?"
    print(f"\n[Turn 2] User: {turn2_q}")
    turn2_a = ask_conversational_agent(turn2_q, session_id=session_alpha)
    print(f"[Turn 2] Agent: {turn2_a}")

    # Verify state carried
    assert "PRAC-1002" in turn2_a, "Memory Failed: Agent did not carry PRAC-1002 into Turn 2!"
    print("\n[Memory Verification]: SUCCESS - PRAC-1002 and fee correctly recalled from Turn 1 context!")

    # Transcript 2: Fresh conversation with no memory state
    session_beta = "patient_session_beta_fresh"
    clear_session_history(session_beta)

    print(f"\n" + "-" * 80)
    print(f"--- TRANSCRIPT 2: Fresh Independent Conversation (Session: {session_beta}) ---")

    fresh_q = "Does that appointment require a follow-up, and what was my fee again?"
    print(f"\n[Turn 1] User: {fresh_q}")
    fresh_a = ask_conversational_agent(fresh_q, session_id=session_beta)
    print(f"[Turn 1] Agent: {fresh_a}")

    # Verify state absent
    assert "cannot find any prior appointment" in fresh_a or "share your appointment record ID" in fresh_a, (
        "Memory Isolation Failed: State leaked into fresh session!"
    )
    print("\n[Fresh Session Verification]: SUCCESS - Prior appointment state correctly absent and reset!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_session_memory()
