"""Appointment lookup tool with designed escalation scoring for Practo Domain Support Agent.

Implements check_appointment_status conforming to Task 6 requirements.
"""

from typing import Dict, Any, Optional
import numpy as np
from dataset import APPOINTMENTS

# Calculate empirical 80th percentile from generated dataset
_DAYS = [r["days_since_created"] for r in APPOINTMENTS]
EMPIRICAL_80TH_PERCENTILE_DAYS = float(np.percentile(_DAYS, 80))

# Recommended escalation threshold justified by dataset distribution
RECOMMENDED_ESCALATION_THRESHOLD = 0.65

# Exact formula documentation:
# recency_norm = days_since_created / 30.0
# follow_up_term = 0.60 if follow_up_required else 0.00
# recency_term = 0.40 * recency_norm
# escalation_score = follow_up_term + recency_term
#
# Threshold justification:
# In our dataset, days_since_created spans 0 to 28 days with an 80th percentile of 24.2 days.
# A patient without follow_up_required caps at max 0.40. A patient with follow_up_required
# starts at 0.60 and crosses the 0.65 threshold by day 4 (0.60 + 0.40*(4/30) = 0.6533),
# appropriately flagging cases requiring clinical re-engagement before reaching
# the 80th percentile critical aging mark of 24.2 days.


def compute_escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    """Compute designed escalation score in [0, 1].

    Args:
        follow_up_required: Boolean flag indicating if clinical follow-up is needed.
        days_since_created: Number of days elapsed since record creation (0-30).

    Returns:
        Escalation score rounded to 4 decimal places.
    """
    recency_norm = min(1.0, max(0.0, days_since_created / 30.0))
    follow_up_term = 0.60 if follow_up_required else 0.00
    recency_term = 0.40 * recency_norm
    return round(follow_up_term + recency_term, 4)


def check_appointment_status(record_id: str) -> Dict[str, Any]:
    """Look up an appointment's status, fee, and compute its designed escalation score.

    Args:
        record_id: Unique record ID, e.g., 'PRAC-1002'.

    Returns:
        Dict with record_id, category, status, consultation_fee_inr,
        days_since_created, follow_up_required, escalation_score,
        and escalation_recommended flag.
    """
    clean_id = record_id.strip().upper()
    matched = None
    for rec in APPOINTMENTS:
        if rec["record_id"].upper() == clean_id:
            matched = rec
            break

    if not matched:
        return {
            "found": False,
            "record_id": record_id,
            "error": f"Appointment record '{record_id}' not found in Practo clinic database.",
            "status": "Unknown",
            "consultation_fee_inr": 0,
            "escalation_score": 0.0,
            "escalation_recommended": False,
        }

    score = compute_escalation_score(
        follow_up_required=matched["follow_up_required"],
        days_since_created=matched["days_since_created"]
    )
    escalate = score >= RECOMMENDED_ESCALATION_THRESHOLD

    return {
        "found": True,
        "record_id": matched["record_id"],
        "category": matched["category"],
        "status": matched["status"],
        "consultation_fee_inr": matched["consultation_fee_inr"],
        "days_since_created": matched["days_since_created"],
        "follow_up_required": matched["follow_up_required"],
        "escalation_score": score,
        "escalation_recommended": escalate,
        "escalation_threshold": RECOMMENDED_ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    print(f"Empirical 80th Percentile Days: {EMPIRICAL_80TH_PERCENTILE_DAYS:.1f}")
    print(f"Recommended Escalation Threshold: {RECOMMENDED_ESCALATION_THRESHOLD}")
    test_ids = ["PRAC-1001", "PRAC-1003", "PRAC-9999"]
    for tid in test_ids:
        print(f"\nLookup for {tid}:", check_appointment_status(tid))
