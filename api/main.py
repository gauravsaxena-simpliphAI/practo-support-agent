"""FastAPI backend application for Practo Domain Support Agent.

Exposes:
- POST /ask (Processes user query with guardrails, memory, and structured AgentResponse)
- POST /add-document (Dynamic knowledge base ingestion with ChromaDB indexing)
- GET /appointments/{record_id} (Direct status lookup)
- GET /health (Health probe)
- WebSocket /ws/chat (Real-time multi-turn chat handling graceful disconnects)

Conforms to Task 11 and Task 12 requirements.
"""

import time
import uuid
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from schemas.models import (
    AgentResponse,
    QueryRequest,
    AddDocumentRequest,
    AppointmentStatusResponse,
)
from guardrails.guardrails import (
    mask_contact_numbers,
    detect_prompt_injection,
    verify_output_groundedness,
)
from memory.session_memory import ask_conversational_agent
from tools.appointment_lookup import check_appointment_status
from rag.indexing import get_chroma_client, get_embedding_model
from rag.chunking import fixed_size_chunking, sentence_based_chunking
from api.logger import log_structured_event

app = FastAPI(
    title="Practo Domain Support Agent API",
    description="Production-grade domain support agent for Practo healthcare clinics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Timing and trace tracking middleware."""
    trace_id = request.headers.get("X-Trace-ID", f"trace-{uuid.uuid4().hex[:12]}")
    request.state.trace_id = trace_id
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000.0
    response.headers["X-Trace-ID"] = trace_id
    return response


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health status and configuration info."""
    return {
        "status": "healthy",
        "service": "Practo Domain Support Agent",
        "mode": "MOCK_LLM",
        "zero_network_access": True,
        "collections": ["practo_kb_fixed", "practo_kb_sentence"],
    }


@app.get("/appointments/{record_id}", response_model=AppointmentStatusResponse)
def get_appointment(record_id: str):
    """Direct appointment status and escalation score lookup."""
    res = check_appointment_status(record_id)
    if not res.get("found"):
        raise HTTPException(status_code=404, detail=res.get("error"))
    return res


@app.post("/ask", response_model=AgentResponse)
async def ask_agent(req: QueryRequest, request: Request) -> AgentResponse:
    """Process customer inquiry with guardrails, memory, and structured response."""
    trace_id = getattr(request.state, "trace_id", f"trace-{uuid.uuid4().hex[:12]}")
    start_time = time.time()
    raw_query = req.query
    session_id = req.session_id or "default_session"

    # 1. Input Guardrail: PII Masking
    masked_query, pii_detected, _ = mask_contact_numbers(raw_query)

    # 2. Input Guardrail: Prompt Injection Detection
    is_injection, reason = detect_prompt_injection(masked_query)
    if is_injection:
        duration_ms = (time.time() - start_time) * 1000.0
        log_structured_event(
            endpoint="/ask",
            method="POST",
            query_text=raw_query,
            duration_ms=duration_ms,
            status_code=400,
            trace_id=trace_id,
            category="injection_blocked",
            guardrail_status="BLOCKED_INJECTION",
        )
        return AgentResponse(
            query=masked_query,
            category="security_alert",
            answer="Security Policy Violation: Prompt injection attempt detected and blocked by Practo AI Safety Guardrail.",
            sources=[],
            escalation_flag=True,
            confidence=0.0,
        )

    # 3. Execution via Conversational Agent (maintains session memory)
    agent_answer = ask_conversational_agent(masked_query, session_id=session_id)

    # 4. Determine category and sources
    category = "policy_inquiry"
    sources = []
    escalation_score = None
    escalation_flag = False

    if "Recalling your previous appointment" in agent_answer or "PRAC-" in masked_query.upper():
        category = "appointment_status"
        # Extract record ID
        import re
        m = re.search(r"PRAC-\d+", masked_query + " " + agent_answer, re.IGNORECASE)
        if m:
            rec_id = m.group(0).upper()
            sources.append(rec_id)
            data = check_appointment_status(rec_id)
            escalation_score = data.get("escalation_score")
            escalation_flag = data.get("escalation_recommended", False)
    elif "cannot find any prior appointment" in agent_answer:
        category = "memory_reset"
    elif "official Practo clinic knowledge base" in agent_answer:
        category = "out_of_scope"

    duration_ms = (time.time() - start_time) * 1000.0

    # Structured logging with pre-disk PII masking
    log_structured_event(
        endpoint="/ask",
        method="POST",
        query_text=raw_query,
        duration_ms=duration_ms,
        status_code=200,
        trace_id=trace_id,
        category=category,
        guardrail_status="PASSED",
    )

    return AgentResponse(
        query=masked_query,
        category=category,
        answer=agent_answer,
        sources=sources,
        escalation_flag=escalation_flag,
        escalation_score=escalation_score,
        confidence=0.96 if category != "out_of_scope" else 0.20,
    )


@app.post("/add-document")
async def add_document(req: AddDocumentRequest, request: Request) -> Dict[str, Any]:
    """Dynamically index a new knowledge base document into ChromaDB."""
    trace_id = getattr(request.state, "trace_id", f"trace-{uuid.uuid4().hex[:12]}")
    start_time = time.time()

    doc_data = {
        "doc_id": req.doc_id,
        "topic": req.topic,
        "title": req.title,
        "content": req.content,
    }

    client = get_chroma_client()
    model = get_embedding_model()

    # Index into fixed-size collection
    fixed_chunks = fixed_size_chunking([doc_data])
    if fixed_chunks:
        fixed_coll = client.get_collection("practo_kb_fixed")
        fixed_embs = model.encode([c["text"] for c in fixed_chunks], show_progress_bar=False).tolist()
        fixed_coll.upsert(
            ids=[c["chunk_id"] for c in fixed_chunks],
            embeddings=fixed_embs,
            documents=[c["text"] for c in fixed_chunks],
            metadatas=[{
                "doc_id": c["doc_id"],
                "topic": c["topic"],
                "title": c["title"],
                "chunk_idx": c["chunk_idx"],
                "strategy": c["strategy"],
            } for c in fixed_chunks]
        )

    # Index into sentence-based collection
    sent_chunks = sentence_based_chunking([doc_data])
    if sent_chunks:
        sent_coll = client.get_collection("practo_kb_sentence")
        sent_embs = model.encode([c["text"] for c in sent_chunks], show_progress_bar=False).tolist()
        sent_coll.upsert(
            ids=[c["chunk_id"] for c in sent_chunks],
            embeddings=sent_embs,
            documents=[c["text"] for c in sent_chunks],
            metadatas=[{
                "doc_id": c["doc_id"],
                "topic": c["topic"],
                "title": c["title"],
                "chunk_idx": c["chunk_idx"],
                "strategy": c["strategy"],
            } for c in sent_chunks]
        )

    duration_ms = (time.time() - start_time) * 1000.0
    log_structured_event(
        endpoint="/add-document",
        method="POST",
        query_text=f"Add Doc: {req.doc_id} - {req.title}",
        duration_ms=duration_ms,
        status_code=200,
        trace_id=trace_id,
        category="kb_ingestion",
        guardrail_status="PASSED",
    )

    return {
        "status": "success",
        "doc_id": req.doc_id,
        "message": f"Successfully indexed '{req.title}' into both fixed and sentence collections.",
        "chunks_indexed": len(fixed_chunks) + len(sent_chunks),
    }


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """Real-time multi-turn WebSocket chat handling client disconnects gracefully."""
    await websocket.accept()
    session_id = f"ws_session_{uuid.uuid4().hex[:8]}"

    try:
        while True:
            # Receive client message
            raw_text = await websocket.receive_text()
            if not raw_text.strip():
                continue

            # Process with guardrails & session memory
            masked_query, _, _ = mask_contact_numbers(raw_text)
            is_injection, _ = detect_prompt_injection(masked_query)

            if is_injection:
                resp_payload = {
                    "query": masked_query,
                    "category": "security_alert",
                    "answer": "Security Policy Violation: Prompt injection attempt detected.",
                    "escalation_flag": True,
                }
            else:
                answer = ask_conversational_agent(masked_query, session_id=session_id)
                resp_payload = {
                    "query": masked_query,
                    "category": "conversational_turn",
                    "answer": answer,
                    "session_id": session_id,
                }

            await websocket.send_json(resp_payload)

    except WebSocketDisconnect:
        # Graceful disconnect handling conforming to Task 11:
        # Server catches WebSocketDisconnect and keeps running cleanly for other clients
        log_structured_event(
            endpoint="/ws/chat",
            method="WEBSOCKET",
            query_text=f"Client disconnected gracefully: {session_id}",
            duration_ms=0.0,
            status_code=1000,
            category="websocket_disconnect",
        )
    except Exception as exc:
        log_structured_event(
            endpoint="/ws/chat",
            method="WEBSOCKET",
            query_text=f"WebSocket error: {str(exc)}",
            duration_ms=0.0,
            status_code=1011,
            category="websocket_error",
        )
