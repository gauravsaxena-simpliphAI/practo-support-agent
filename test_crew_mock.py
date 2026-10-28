"""Test script for verifying CrewAI ReAct loop with MockCrewLLM."""
import os
import re
import json

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew
from crewai.tools import tool
from crewai.llms.base_llm import BaseLLM

@tool("check_appointment_status")
def check_status_tool(record_id: str) -> str:
    """Look up appointment status by record_id."""
    return f"Appointment {record_id} is Scheduled in Orthopedics. Fee: INR 750. Escalation score: 0.12 (Normal)."

@tool("rag_kb_search")
def rag_search_tool(query: str) -> str:
    """Retrieve policy documentation from knowledge base."""
    return f"Knowledge Base Result for '{query}': Appointments can be cancelled free of penalty up to 2 hours before consultation."

class MockCrewLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(model="mock-practo-model", **kwargs)

    def supports_stop_words(self) -> bool:
        return True

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None, response_model=None):
        # Extract full text of messages, but distinguish system template from agent dialogue
        conv_text = ""
        last_user_content = ""
        if isinstance(messages, list):
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if role != "system":
                    conv_text += f"\n[{role}]: {content}"
                if role in ("user", "human"):
                    last_user_content = content
        else:
            conv_text = str(messages)
            last_user_content = conv_text

        agent_role = getattr(from_agent, "role", "") if from_agent else ""

        # Check if a tool has already run in THIS dialogue by checking for real tool results in non-system messages
        has_tool_run = ("Tool Result:" in conv_text) or ("\nObservation:" in conv_text and "the result of the action" not in conv_text.split("Observation:")[-1][:50])

        # Dispatch based on tool schemas available to this agent (Pitfall 2: dispatch off tool declared argument schema)
        tool_param_names = []
        tool_name_by_schema = {}
        if tools:
            for t in tools:
                # Each tool in tools has name, description, parameters schema
                name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "")
                params = getattr(t, "args_schema", None)
                if params and hasattr(params, "model_fields"):
                    fields = list(params.model_fields.keys())
                elif isinstance(t, dict) and "parameters" in t:
                    fields = list(t["parameters"].get("properties", {}).keys())
                else:
                    fields = []
                for f in fields:
                    tool_name_by_schema[f] = name

        if not has_tool_run and tool_name_by_schema:
            # First turn: agent should call a tool based on the schema
            if "record_id" in tool_name_by_schema:
                # Match record ID like PRAC-XXXX
                m = re.search(r"PRAC-\d+", conv_text, re.IGNORECASE)
                rec_id = m.group(0).upper() if m else "PRAC-1002"
                tname = tool_name_by_schema["record_id"]
                return f"Thought: I need to check the appointment status for record {rec_id}.\nAction: {tname}\nAction Input: {{\"record_id\": \"{rec_id}\"}}"

            elif "query" in tool_name_by_schema:
                tname = tool_name_by_schema["query"]
                q_text = "cancellation window" if "cancellation" in conv_text.lower() else "clinic policy"
                return f"Thought: I need to retrieve official clinic policy.\nAction: {tname}\nAction Input: {{\"query\": \"{q_text}\"}}"

        # Subsequent turn or Composer agent: return Final Answer
        if "Lookup" in agent_role:
            return "Thought: I have retrieved the appointment status.\nFinal Answer: Appointment record verified: Status is Scheduled with consultation fee INR 750. Escalation score is 0.12 (no escalation required)."
        elif "Retrieval" in agent_role:
            return "Thought: I have retrieved the relevant policy document.\nFinal Answer: Policy verified: Appointments can be cancelled free of penalty up to 2 hours before the scheduled slot."
        else:
            return "Thought: I am composing the complete customer response.\nFinal Answer: Here is the verified information for your inquiry: Appointments may be cancelled free of charge up to 2 hours prior to your consultation slot."

    async def acall(self, *args, **kwargs):
        return self.call(*args, **kwargs)

if __name__ == "__main__":
    llm = MockCrewLLM()
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Look up appointment records",
        backstory="Healthcare database specialist",
        tools=[check_status_tool],
        llm=llm,
        verbose=False
    )
    task1 = Task(
        description="Verify status of appointment PRAC-1002",
        expected_output="Appointment status report",
        agent=lookup_agent
    )
    crew = Crew(agents=[lookup_agent], tasks=[task1], verbose=False)
    res = crew.kickoff()
    print("TEST LOOKUP CREW RESULT:", res)
