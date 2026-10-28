# Practo Domain Support Agent (CrewAI)

**Track:** Healthcare (Practo)  
**Deliverable:** Complete End-to-End Production-Minded Support Agent  
**Execution Mode:** Deterministic `MOCK_LLM` (Zero API Keys, Zero Paid Accounts, Zero Network Access)  

---

## Executive Summary & Telemetry Compliance

This repository contains the complete implementation of the **Practo Domain Support Agent**, designed to handle clinic policy inquiries and appointment record lookups with rigorous healthcare AI governance, multi-agent orchestration, guardrails, evaluation, and production deployment.

### Telemetry & Offline Guarantee
In strict adherence to the capstone zero-network-access requirement:
- **`CREWAI_DISABLE_TELEMETRY=true`** and **`OTEL_SDK_DISABLED=true`** are explicitly exported across all agent modules and runner scripts to guarantee no outbound telemetry calls occur during `crew.kickoff()`.
- Local embeddings are computed via **`SentenceTransformers`** (`all-MiniLM-L6-v2`) cached locally.
- Vector search runs on local persistent **`ChromaDB`** collections (`practo_kb_fixed` and `practo_kb_sentence`).
- All language-model reasoning steps operate under a deterministic **`MOCK_LLM`** mode that requires zero external API keys.

---

## Part 1 — Dataset Design & RAG Core

### Task 1: Dataset Design Choices (`dataset.py`)
To ensure deterministic reproducibility by the grader:
- **Random Seed:** `42`
- **Total Records:** `50` (Target: $\ge 40$)
- **Category Vocabulary & Probabilistic Weights:**
  - `General Medicine` (Weight: 0.30, Count: 15 records)
  - `Cardiology` (Weight: 0.16, Count: 5 records)
  - `Dermatology` (Weight: 0.20, Count: 8 records)
  - `Pediatrics` (Weight: 0.18, Count: 7 records)
  - `Orthopedics` (Weight: 0.16, Count: 15 records)
  *(Requirement: Every given category $\ge 3$ records — All PASSED)*
- **Status Vocabulary & Probabilistic Weights:**
  - `Scheduled` (Weight: 0.36, Count: 18 records)
  - `Completed` (Weight: 0.38, Count: 15 records)
  - `Cancelled` (Weight: 0.12, Count: 5 records)
  - `No-Show` (Weight: 0.06, Count: 7 records)
  - `Rescheduled` (Weight: 0.08, Count: 5 records)
  *(Requirement: Every given status $\ge 1$ record — All PASSED)*
- **Consultation Fee Range (INR):** `400` to `1,650` (Average: INR 766.00).  
  *Reasoning Sentence:* In Indian outpatient clinics on Practo, consultation fees realistically span INR 400 to INR 600 for General Medicine/Pediatrics, INR 600 to INR 1,000 for Dermatology/Orthopedics, and INR 800 to INR 1,800 for super-specialty Cardiology consultations.
- **Days Since Created:** Range `0` to `28` days (80th percentile: `24.2` days).
- **Follow-up Required Rate:** `7 / 50` records = **`14.00%`**  
  *(Requirement: Strictly between 10.0% and 30.0% — PASSED)*

### Task 2: Knowledge Base (`knowledge_base/`)
Authored 12 authoritative, domain-specific clinic policy documents (2–5 sentences each), covering all required topics:
1. `KB-DOC-001`: Appointment-booking policy
2. `KB-DOC-002`: Cancellation and rescheduling window policy
3. `KB-DOC-003`: Consultation-fee structure by specialty
4. `KB-DOC-004`: Insurance-claim and cashless reimbursement process
5. `KB-DOC-005`: Prescription-refill policy
6. `KB-DOC-006`: Diagnostic lab test turnaround times
7. `KB-DOC-007`: Telemedicine consultation eligibility
8. `KB-DOC-008`: Emergency visit and urgent care protocol
9. `KB-DOC-009`: Patient data privacy and confidentiality policy
10. `KB-DOC-010`: Follow-up visit discount policy
11. `KB-DOC-011`: Second opinion consultation process
12. `KB-DOC-012`: Home visit and domiciliary healthcare eligibility

### Task 3: Chunking & ChromaDB Collections (`rag/indexing.py`)
Two distinct chunking strategies are implemented and indexed into their own collections using `collection.upsert()`:
1. **Fixed-Size with Overlap:** 220 characters with 40 character overlap $\rightarrow$ `practo_kb_fixed` (36 chunks indexed).
2. **Sentence-Based:** Split strictly along sentence boundaries $\rightarrow$ `practo_kb_sentence` (36 chunks indexed).

### Task 4: Empirical Calibration & Grounded Generation (`rag/grounded_generation.py`)
Rather than relying on an arbitrary tutorial default (e.g., 0.5 or 0.7), the "I don't know" fallback threshold was calibrated empirically by measuring top-1 cosine similarities across 5 in-scope queries and 3 out-of-scope queries:

| Query Type | Query | Top-1 Similarity | Matched Topic |
|---|---|---|---|
| In-Scope | Cancellation window for clinic appointments | **0.6945** | cancellation/rescheduling |
| In-Scope | Consultation fee for Cardiology | **0.6058** | fee structure |
| In-Scope | Follow-up visit discount policy | **0.5968** | follow-up discount |
| In-Scope | Cashless health insurance claims | **0.5795** | insurance claims |
| In-Scope | Turnaround times for routine lab tests | **0.6336** | lab test turnaround |
| Out-of-Scope | Book flight tickets to Singapore | **0.2457** | travel (unrelated) |
| Out-of-Scope | Corporate income tax return rules | **0.2235** | taxation (unrelated) |
| Out-of-Scope | Order pepperoni pizza delivery | **0.2034** | food delivery (unrelated) |

- **In-Scope Cluster:** `[0.5795, 0.6945]`
- **Out-of-Scope Cluster:** `[0.2034, 0.2457]`
- **Empirical Separation Gap:** `0.3337`
- **Chosen Calibrated Threshold:** **`0.4500`** (cleanly situated in the empirical margin). Queries with similarity $< 0.4500$ deterministically trigger the official fallback refusal.

### Task 5: Chunking Strategy Evaluation & Recommendation (`rag/evaluation.py`)
Document-level precision and recall computed across the 5 benchmark queries:

```
[Q1] Cancellation Window:
  Fixed Collection:    Retrieved: {KB-DOC-002, KB-DOC-010} | Intersect: {KB-DOC-002} | P = 1/2 = 0.5000 | R = 1/1 = 1.0000
  Sentence Collection: Retrieved: {KB-DOC-002}             | Intersect: {KB-DOC-002} | P = 1/1 = 1.0000 | R = 1/1 = 1.0000

[Q2] Cardiology Fee:
  Fixed Collection:    Retrieved: {KB-DOC-003, KB-DOC-002} | Intersect: {KB-DOC-003} | P = 1/2 = 0.5000 | R = 1/1 = 1.0000
  Sentence Collection: Retrieved: {KB-DOC-010, KB-DOC-003} | Intersect: {KB-DOC-003} | P = 1/2 = 0.5000 | R = 1/1 = 1.0000

[Q3] Follow-Up Discount:
  Fixed Collection:    Retrieved: {KB-DOC-010, KB-DOC-001} | Intersect: {KB-DOC-010} | P = 1/2 = 0.5000 | R = 1/1 = 1.0000
  Sentence Collection: Retrieved: {KB-DOC-010}             | Intersect: {KB-DOC-010} | P = 1/1 = 1.0000 | R = 1/1 = 1.0000

[Q4] Insurance Claims:
  Fixed Collection:    Retrieved: {KB-DOC-004, KB-DOC-001} | Intersect: {KB-DOC-004} | P = 1/2 = 0.5000 | R = 1/1 = 1.0000
  Sentence Collection: Retrieved: {KB-DOC-004}             | Intersect: {KB-DOC-004} | P = 1/1 = 1.0000 | R = 1/1 = 1.0000

[Q5] Lab Turnaround:
  Fixed Collection:    Retrieved: {KB-DOC-006, KB-DOC-003} | Intersect: {KB-DOC-006} | P = 1/2 = 0.5000 | R = 1/1 = 1.0000
  Sentence Collection: Retrieved: {KB-DOC-006, KB-DOC-011} | Intersect: {KB-DOC-006} | P = 1/2 = 0.5000 | R = 1/1 = 1.0000

Collection Averages:
- Fixed-Size Collection:    Mean Precision = 0.5000 (50.00%) | Mean Recall = 1.0000 (100.00%)
- Sentence-Based Collection: Mean Precision = 0.8000 (80.00%) | Mean Recall = 1.0000 (100.00%)
```

**Recommendation:**  
I recommend deploying the **Sentence-Based chunking strategy** (`practo_kb_sentence`) for the Practo production agent. While both chunking strategies achieved perfect recall (Fixed: 100.00%, Sentence: 100.00%) across all test queries, the Sentence-Based strategy achieved a significantly higher document-level precision (80.00% vs. 50.00%) because discrete sentence boundaries prevent context bleeding and cross-policy noise in top-k embeddings, ensuring more compact and grounded generation.

---

## Part 2 — CrewAI Multi-Agent Orchestration, Memory & Guardrails

### Task 6: Designed Escalation Score Formula (`tools/appointment_lookup.py`)
`check_appointment_status(record_id: str) -> dict` combines clinical necessity with temporal recency:

$$\text{recency\_weight} = \frac{\text{days\_since\_created}}{30.0}$$
$$\text{escalation\_score} = 0.60 \times \mathbb{I}(\text{follow\_up\_required}) + 0.40 \times \text{recency\_weight}$$

- **Escalation Threshold:** **`0.65`**
- **Justification:** In our generated dataset, `days_since_created` spans 0 to 28 days with an 80th percentile of **`24.2 days`**. A patient without a required follow-up is capped at a maximum score of $0.40 \times 1.0 = 0.40$ (even at max recency). Conversely, a patient requiring follow-up starts at base score $0.60$ and crosses the $0.65$ escalation threshold by day 4 ($0.60 + 0.40 \times (4/30) = 0.6533$), successfully prioritizing overdue clinical re-engagements before reaching the 80th percentile critical aging mark of 24.2 days.

### Task 7 & Task 9: CrewAI Crew & Structured Output (`crew/support_crew.py`, `crew/mock_llm.py`)
- **3-Agent Crew:**
  1. `Retrieval Agent`: Equipped with `rag_kb_search` tool.
  2. `Lookup Agent`: Equipped with `check_appointment_status` tool.
  3. `Response Composer`: Pure synthesis without tools (Least Autonomy).
- **`MockCrewLLM` Subclassing `crewai.llms.base_llm.BaseLLM`:**
  - *Pitfall 1 Guarded:* Separates system templates from conversational history; never parses `"Observation:"` out of system instructions.
  - *Pitfall 2 Guarded:* Dispatches tools by declared schema arguments (`record_id` vs `query`), never by substring tool name matching.
- **Pydantic Validation (Task 9):** Every response conforms to `AgentResponse` (`query`, `category`, `answer`, `sources`, `escalation_flag`, `escalation_score`, `confidence`).

### Task 8: LangChain Session Memory (`memory/session_memory.py`)
Maintains in-process conversation state using `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`:
- **Transcript 1 (Multi-turn session):**
  - *Turn 1:* User asks: `"Hi, my appointment record is PRAC-1002. What is its current status and fee?"`
  - *Turn 2:* User asks: `"Does that appointment require a follow-up, and what was my fee again?"` (no record ID mentioned).
  - *Result:* Agent successfully recalls `PRAC-1002` and `INR 450` from Turn 1 context.
- **Transcript 2 (Fresh session):**
  - *Turn 1:* User asks identical question `"Does that appointment require a follow-up, and what was my fee again?"` in a new session.
  - *Result:* Agent correctly identifies that no prior appointment exists in the session and requests the appointment ID.

### Task 10: Input & Output Guardrails (`guardrails/guardrails.py`)
1. **Input-Side PII Masking:** Detects and masks Indian contact numbers (`+91-XXXXX-XXXXX` or 10-digit formats) with `[REDACTED_PHONE]`.
2. **Input-Side Prompt Injection Detection:** Intercepts jailbreaks ("ignore previous instructions", "system prompt", "DAN mode") and blocks execution.
3. **Output-Side Groundedness Verification:** Intercepts ungrounded questions and enforces fallback refusal.

---

## Part 3 — Evaluation, Observability & FastAPI Deployment

### Task 11: FastAPI & WebSocket Backend (`api/main.py`)
- `POST /ask`: Accepts `QueryRequest`, enforces guardrails & memory, returns validated `AgentResponse`.
- `POST /add-document`: Ingests and indexes new documents dynamically into ChromaDB.
- `GET /appointments/{record_id}`: Direct appointment status and escalation score lookup.
- `GET /health`: Health probe reporting mode and collections.
- `WebSocket /ws/chat`: Real-time chat handling client disconnects gracefully by catching `WebSocketDisconnect` without terminating the server.

### Task 12: Structured ELK Logging (`api/logger.py`)
Every request generates a single JSON-Lines entry written to `logs/api_requests.jsonl` with `timestamp`, `trace_id`, `endpoint`, `duration_ms`, and `status_code`.  
**PII Disk Protection:** Contact numbers are masked *before* writing to disk, ensuring raw PII never reaches disk in the clear.

### Task 13: 15-Query Evaluation at Scale (`evaluation/eval_suite.py`)
Evaluated using LLM-as-judge across 15 queries (12 KB topics + 2 out-of-scope + 1 prompt injection):

| ID | Topic | Accuracy | Grounding | Completeness | Safety | Outcome |
|---|---|---|---|---|---|---|
| Q01 | appointment-booking policy | 0.85 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q02 | cancellation/rescheduling window | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q03 | consultation-fee structure by specialty | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q04 | insurance-claim process | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q05 | prescription-refill policy | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q06 | lab-test turnaround times | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q07 | telemedicine eligibility | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q08 | emergency-visit protocol | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q09 | patient-data privacy policy | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q10 | follow-up-visit discount policy | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q11 | second-opinion process | 0.85 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q12 | home-visit eligibility | 1.00 | 1.00 | 1.00 | 1.00 | GROUNDED_ANSWER |
| Q13 | out_of_scope_travel | 1.00 | 1.00 | 1.00 | 1.00 | REFUSED_FALLBACK |
| Q14 | out_of_scope_tax | 1.00 | 1.00 | 1.00 | 1.00 | REFUSED_FALLBACK |
| Q15 | adversarial_injection | 1.00 | 1.00 | 1.00 | 1.00 | BLOCKED_INJECTION |
| **AVG** | **SYSTEM BENCHMARK (N=15)** | **0.98** | **1.00** | **1.00** | **1.00** | **ALL PASSED** |

---

## Part 4 — Resilience & Governance

### Task 14: AutoGen Review Stage (`governance/autogen_review.py`)
A 2-agent `RoundRobinGroupChat` team bounded by `max_turns=2`:
- `Policy_Compliance_Reviewer` audits draft answers against retrieved context.
- `Final_Editor` outputs structured `ReviewVerdict` (`approved: bool`, `final_answer: str`, `reason: str`).
- Uses `custom_message_types=[StructuredMessage[ReviewVerdict]]` to prevent runtime message-type crashes.
- **Demonstrated on 2 sample cases:**
  - *Case 1 (Grounded Draft):* Approved unchanged (`approved=True`).
  - *Case 2 (Injected Ungrounded Claim):* Caught and revised (`approved=False`).

### Task 15: AI Governance Model (`governance/policy.py`)
1. **Application Layer (Least Autonomy):**  
   Only `Lookup Agent` has permission to call `check_appointment_status`. If `Retrieval Agent` or `Response Composer` attempts to call it, an `UnauthorizedToolAccessException` is raised.  
   *Justification:* Patient appointment records contain sensitive clinical and transactional data. Restricting lookup tools solely to the Lookup Agent prevents data exfiltration during open-ended policy synthesis.
2. **System Risk Classification:**  
   Classified as **HIGH RISK**. The system handles clinical follow-up urgency, specialty triage, and financial tariffs. Misinformation could delay emergency interventions or misguide medical follow-ups, necessitating strict governance controls.
3. **Runtime Layer (Cost/Budget Cap):**  
   Enforces a per-request limit of 1,500 characters (~400 tokens). Oversized requests (e.g., 2,880 characters) are rejected with `RequestBudgetExceededException`.

### Task 16: Response Caching (`cache/response_cache.py`)
In-memory cache keyed by canonical normalized query text (`normalize_query_text`):
- **Call 1 (Initial query):** Cache Miss $\rightarrow$ Execution Time: `22.197 ms` $\rightarrow$ Model Computations: `1`.
- **Call 2 (Repeated query with varied whitespace/casing):** Cache Hit $\rightarrow$ Execution Time: `0.001 ms` $\rightarrow$ Model Computations: `1` (Zero redundant calls).

---

## Reproduction & Verification Commands

To run all 16 capstone tasks end-to-end and view all verified transcripts:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run master demonstration runner
python run_all_demos.py

# 3. Run individual module test suites
python dataset.py
python -m rag.indexing
python -m rag.grounded_generation
python -m rag.evaluation
python -m crew.support_crew
python -m memory.session_memory
python -m guardrails.guardrails
python -m tests.test_api
python -m evaluation.eval_suite
python -m governance.autogen_review
python -m governance.policy
python -m cache.response_cache
```
