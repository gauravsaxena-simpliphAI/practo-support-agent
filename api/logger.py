"""Structured ELK-style JSON-Lines logging for Practo Domain Support Agent.

Ensures every incoming request is logged with a trace ID, duration, and status.
Guarantees that fixed-format PII (contact numbers) is masked BEFORE reaching disk.
Conforms to Task 12 requirements.
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from guardrails.guardrails import mask_contact_numbers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "api_requests.jsonl")


def init_logger():
    """Ensure log directory exists."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_structured_event(
    endpoint: str,
    method: str,
    query_text: str,
    duration_ms: float,
    status_code: int,
    trace_id: Optional[str] = None,
    category: Optional[str] = None,
    guardrail_status: Optional[str] = None,
    client_ip: Optional[str] = "127.0.0.1",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write an ELK-compliant JSON-Lines log entry to disk with PII scrubbing."""
    init_logger()

    if not trace_id:
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

    # CRITICAL: Scrub all fixed-format PII before logging to disk
    safe_query, pii_found, _ = mask_contact_numbers(query_text)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "method": method,
        "client_ip": client_ip,
        "duration_ms": round(duration_ms, 2),
        "status_code": status_code,
        "masked_query": safe_query,
        "pii_scrubbed": pii_found,
        "category": category or "general",
        "guardrail_status": guardrail_status or "PASSED",
    }

    if extra:
        log_entry.update(extra)

    # Append single line JSON
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry


def read_latest_logs(n: int = 5) -> list[Dict[str, Any]]:
    """Read latest n structured log entries for verification."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [json.loads(line) for line in lines[-n:]]
