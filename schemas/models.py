"""Pydantic data models and schemas for Practo Domain Support Agent.

Includes response schemas, validation models, API request/response contracts,
and governance verdict models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Declared structured output schema that every crew response must conform to."""
    query: str = Field(description="The user query processed by the agent.")
    category: str = Field(description="Classification of query (policy_inquiry, appointment_status, escalation, out_of_scope, etc.).")
    answer: str = Field(description="Grounded final answer provided to the patient.")
    sources: List[str] = Field(default_factory=list, description="Authoritative source documents or record IDs referenced.")
    escalation_flag: bool = Field(default=False, description="Flag indicating if clinic manager escalation is required.")
    escalation_score: Optional[float] = Field(default=None, description="Designed escalation score in [0, 1] if appointment was queried.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model confidence score.")


class ReviewVerdict(BaseModel):
    """Structured output verdict model for Autogen multi-agent review stage."""
    approved: bool = Field(description="True if draft answer is approved without modification; False if revised.")
    final_answer: str = Field(description="The final approved or revised patient-facing answer.")
    reason: str = Field(description="Audit justification for approval or specific corrections made.")


class QueryRequest(BaseModel):
    """FastAPI request body for /ask endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question or appointment inquiry.")
    session_id: Optional[str] = Field(default="default_session", description="Session identifier for multi-turn memory.")


class AddDocumentRequest(BaseModel):
    """FastAPI request body for dynamic KB ingestion via /add-document."""
    doc_id: str = Field(..., description="Unique document ID (e.g., 'KB-DOC-013').")
    topic: str = Field(..., description="Topic of policy document.")
    title: str = Field(..., description="Readable title.")
    content: str = Field(..., min_length=10, description="Body text (2-5 sentences).")


class AppointmentStatusResponse(BaseModel):
    """Structured response for appointment lookup."""
    found: bool
    record_id: str
    category: Optional[str] = None
    status: Optional[str] = None
    consultation_fee_inr: Optional[int] = None
    days_since_created: Optional[int] = None
    follow_up_required: Optional[bool] = None
    escalation_score: Optional[float] = None
    escalation_recommended: Optional[bool] = None
