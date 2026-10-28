"""MockCrewLLM extending crewai.llms.base_llm.BaseLLM.

Implements zero-network, keyless, deterministic ReAct agent simulation for CrewAI.
Safely avoids both common ReAct pitfalls:
1. Does NOT match "Observation:" in the system-prompt template (filters out system messages).
2. Dispatches tool calls off the tool's declared argument schema (not substring name matching).
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from crewai.llms.base_llm import BaseLLM

# Ensure telemetry is disabled
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


class MockCrewLLM(BaseLLM):
    """Deterministic, keyless BaseLLM implementation for CrewAI."""

    def __init__(self, **kwargs):
        super().__init__(model="practo-mock-llm", **kwargs)

    def supports_stop_words(self) -> bool:
        return True

    def call(
        self,
        messages: str | list,
        tools: Optional[list] = None,
        callbacks: Optional[list] = None,
        available_functions: Optional[dict] = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Optional[type[BaseModel]] = None,
    ) -> str:
        """Execute deterministic ReAct reasoning loop step."""
        # 1. Pitfall 1 Guard: Separate system template from actual execution conversation
        conv_text = ""
        user_query = ""

        if isinstance(messages, list):
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                # Only inspect non-system messages so we never match the ReAct example template
                if role != "system":
                    conv_text += f"\n[{role}]: {content}"
                if role in ("user", "human"):
                    user_query = content
        else:
            conv_text = str(messages)
            user_query = conv_text

        agent_role = getattr(from_agent, "role", "") if from_agent else ""

        # Check if a tool has produced an actual execution result in the conversation
        has_tool_run = (
            "Tool Result:" in conv_text
            or ("Observation:" in conv_text and "the result of the action" not in conv_text.split("Observation:")[-1][:60])
        )

        # 2. Pitfall 2 Guard: Dispatch off declared argument schema, NOT tool name substring
        schema_to_tool: Dict[str, str] = {}
        if tools:
            for t in tools:
                tname = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "")
                schema = getattr(t, "args_schema", None)
                if schema and hasattr(schema, "model_fields"):
                    fields = list(schema.model_fields.keys())
                elif isinstance(t, dict) and "parameters" in t:
                    fields = list(t["parameters"].get("properties", {}).keys())
                else:
                    fields = []
                for f in fields:
                    schema_to_tool[f] = tname

        # Turn 1: If tool not yet executed and tool schema is present, invoke appropriate tool
        if not has_tool_run and schema_to_tool:
            # Check for appointment lookup schema (requires 'record_id')
            if "record_id" in schema_to_tool:
                target_tool = schema_to_tool["record_id"]
                # Extract record ID (e.g. PRAC-1002) from conversation text
                m = re.search(r"PRAC-\d+", conv_text, re.IGNORECASE)
                rec_id = m.group(0).upper() if m else "PRAC-1002"
                return (
                    f"Thought: I need to verify the appointment record details for {rec_id}.\n"
                    f"Action: {target_tool}\n"
                    f"Action Input: {{\"record_id\": \"{rec_id}\"}}"
                )

            # Check for RAG retrieval schema (requires 'query')
            elif "query" in schema_to_tool:
                target_tool = schema_to_tool["query"]
                # Clean query for search
                clean_query = user_query.replace("Current Task:", "").strip()
                if len(clean_query) > 120:
                    clean_query = clean_query[:120]
                return (
                    f"Thought: I need to retrieve official clinic policy documentation from the knowledge base.\n"
                    f"Action: {target_tool}\n"
                    f"Action Input: {{\"query\": \"{clean_query}\"}}"
                )

        # Turn 2: Synthesize findings into final answer
        if "Lookup" in agent_role:
            m = re.search(r"PRAC-\d+", conv_text, re.IGNORECASE)
            rec_id = m.group(0).upper() if m else "PRAC-1002"
            return (
                f"Thought: I have retrieved the appointment status for {rec_id} from the clinic database.\n"
                f"Final Answer: Appointment {rec_id} details verified. Status is Scheduled, consultation fee is INR 750, and escalation score is calibrated at 0.12 (standard routine care)."
            )

        elif "Retrieval" in agent_role:
            return (
                "Thought: I have retrieved the authoritative policy document.\n"
                "Final Answer: Policy retrieved: Patients may cancel or reschedule up to 2 hours before the appointment free of penalty. Late cancellations incur a 20% fee."
            )

        else:
            # Response Composer Agent: combines all findings into a structured final draft
            if "PRAC-" in conv_text.upper():
                m = re.search(r"PRAC-\d+", conv_text, re.IGNORECASE)
                rec_id = m.group(0).upper() if m else "PRAC-1002"
                return (
                    f"Thought: Synthesizing appointment lookup findings.\n"
                    f"Final Answer: Hello! Your appointment {rec_id} is currently confirmed as Scheduled. "
                    f"The consultation fee is INR 750. No urgent escalation is required at this time."
                )
            else:
                return (
                    "Thought: Synthesizing policy retrieval findings.\n"
                    "Final Answer: According to official Practo clinic policy, patients may reschedule or cancel "
                    "their appointments free of charge up to two hours prior to the booked consultation slot. "
                    "Cancellations requested within less than two hours incur a 20% late fee."
                )

    async def acall(self, *args, **kwargs) -> str:
        return self.call(*args, **kwargs)
