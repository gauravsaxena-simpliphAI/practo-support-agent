"""Four-layer AI Governance Model for Practo Domain Support Agent.

Implements:
1. Application Layer: Principle of least autonomy (only Lookup Agent may access appointment records).
2. Risk Classification: High Risk classification with clinical justification.
3. Runtime Layer: Per-request token and character budget cap rejecting oversized requests.

Conforms strictly to Task 15 requirements.
"""

from typing import List, Dict, Any, Optional

# Authorized tool permissions mapping (Principle of Least Autonomy)
AUTHORIZED_AGENT_TOOLS = {
    "Lookup Agent": ["check_appointment_status"],
    "Retrieval Agent": ["rag_kb_search"],
    "Response Composer": [],
}

# Runtime token and character budget caps
MAX_REQUEST_CHARACTERS = 1500  # ~375 tokens
MAX_PERMITTED_TOKENS = 400


class UnauthorizedToolAccessException(PermissionError):
    """Raised when an unauthorized agent attempts to register or invoke a restricted tool."""
    pass


class RequestBudgetExceededException(ValueError):
    """Raised when an incoming request breaches the runtime token/character budget cap."""
    pass


def verify_tool_access_authorization(agent_role: str, tool_name: str) -> bool:
    """Enforce Principle of Least Autonomy at the Application Layer.

    Only Lookup Agent may call check_appointment_status. No other agent may call it.
    """
    allowed_tools = AUTHORIZED_AGENT_TOOLS.get(agent_role, [])
    if tool_name not in allowed_tools:
        raise UnauthorizedToolAccessException(
            f"GOVERNANCE VIOLATION (Application Layer): Agent '{agent_role}' is not authorized "
            f"to access restricted tool '{tool_name}'. Permitted tools for '{agent_role}': {allowed_tools}. "
            f"Strict Principle of Least Autonomy enforced."
        )
    return True


def enforce_runtime_budget(query: str, max_chars: int = MAX_REQUEST_CHARACTERS) -> int:
    """Enforce per-request token/cost budget cap at the Runtime Layer.

    Rejects oversized requests rather than silently exceeding the budget.
    """
    char_count = len(query)
    approx_tokens = (char_count + 3) // 4

    if char_count > max_chars or approx_tokens > MAX_PERMITTED_TOKENS:
        raise RequestBudgetExceededException(
            f"GOVERNANCE VIOLATION (Runtime Layer): Request size ({char_count} chars, ~{approx_tokens} tokens) "
            f"exceeds the permitted per-request budget cap of {max_chars} chars (~{MAX_PERMITTED_TOKENS} tokens). "
            f"Request rejected to prevent resource exhaustion and unbounded inference cost."
        )
    return approx_tokens


def get_risk_classification() -> Dict[str, str]:
    """Return official system risk classification and one-paragraph justification."""
    return {
        "classification": "HIGH RISK",
        "taxonomy": "Low (transcription/summarization), Medium (general code/support tickets), High (medical/financial/hiring)",
        "justification": (
            "Under the standard Low/Medium/High AI governance taxonomy (where Low encompasses summarization/transcription, "
            "Medium encompasses code generation/general support tickets, and High encompasses medical data, clinical decisions, "
            "and financial records), the Practo Domain Support Agent is classified as **High Risk**. The system directly handles "
            "identifiable patient appointment scheduling, clinical follow-up necessity, specialty triage, and financial consultation tariffs. "
            "Misinformation regarding emergency triage protocols, follow-up requirements, or medication refills could delay urgent "
            "clinical care and lead to patient harm. Consequently, strict four-layer governance—including input PII scrubbing, "
            "least-autonomy tool scoping, dual-agent compliance review, and deterministic fallback thresholds—is mandatory."
        ),
        "least_autonomy_explanation": (
            "Under the Principle of Least Autonomy in healthcare AI governance, agents are granted only the minimal permissions "
            "necessary to execute their assigned role. In our Practo architecture, patient appointment records contain sensitive "
            "clinical and transactional details (specialty, status, fees, and follow-up indicators). Wiring check_appointment_status "
            "to the Retrieval Agent or Response Composer would violate separation of concerns and risk unauthorized record exfiltration "
            "during open-ended policy synthesis. Only the Lookup Agent is authorized and audited to query patient records, while the "
            "Composer receives only sanitized, necessary attributes."
        ),
    }


def demonstrate_governance():
    """Demonstrate least autonomy enforcement, risk justification, and runtime budget cap rejection."""
    print("=" * 80)
    print("TASK 15: AI GOVERNANCE MODEL DEMONSTRATION")
    print("=" * 80)

    # 1. Application Layer: Least Autonomy Demonstration
    print("\n[APPLICATION LAYER: Least Autonomy Enforcement]")
    # Permitted call
    try:
        verify_tool_access_authorization("Lookup Agent", "check_appointment_status")
        print("  - Lookup Agent accessing check_appointment_status: AUTHORIZED (PASS)")
    except UnauthorizedToolAccessException as e:
        print("  - Failed:", e)

    # Blocked call: Retrieval Agent attempting to access appointment records
    try:
        verify_tool_access_authorization("Retrieval Agent", "check_appointment_status")
        print("  - Retrieval Agent accessing check_appointment_status: UNEXPECTED ALLOW (FAIL)")
    except UnauthorizedToolAccessException as e:
        print("  - Retrieval Agent accessing check_appointment_status: BLOCKED (PASS)")
        print(f"    Message: {e}")

    # Blocked call: Response Composer attempting to access appointment records
    try:
        verify_tool_access_authorization("Response Composer", "check_appointment_status")
        print("  - Response Composer accessing check_appointment_status: UNEXPECTED ALLOW (FAIL)")
    except UnauthorizedToolAccessException as e:
        print("  - Response Composer accessing check_appointment_status: BLOCKED (PASS)")

    # 2. Risk Classification
    gov_info = get_risk_classification()
    print("\n" + "-" * 80)
    print("[SYSTEM RISK CLASSIFICATION]")
    print(f"  Classification: {gov_info['classification']}")
    print(f"\n  One-Paragraph Justification:\n  {gov_info['justification']}")
    print(f"\n  Least Autonomy Guard Justification:\n  {gov_info['least_autonomy_explanation']}")

    # 3. Runtime Layer: Budget Cap Enforcement
    print("\n" + "-" * 80)
    print("[RUNTIME LAYER: Cost & Token Budget Cap Enforcement]")
    normal_query = "What is the cancellation window for appointments?"
    tokens = enforce_runtime_budget(normal_query)
    print(f"  - Normal Request ({len(normal_query)} chars, ~{tokens} tokens): ACCEPTED (PASS)")

    # Deliberately oversized query (e.g. 2400 chars)
    oversized_query = "Please check my appointment status. " * 80  # ~2880 chars
    print(f"  - Testing Oversized Request ({len(oversized_query)} chars)...")
    try:
        enforce_runtime_budget(oversized_query)
        print("  - Oversized Request: UNEXPECTED ACCEPTANCE (FAIL)")
    except RequestBudgetExceededException as e:
        print("  - Oversized Request: REJECTED AT RUNTIME LAYER (PASS)")
        print(f"    Error: {e}")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_governance()
