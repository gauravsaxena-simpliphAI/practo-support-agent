"""Practo Domain Support Agent - Dataset Generator

Track: Healthcare (Practo)
Seeded, deterministic appointment generator.
Generates an APPOINTMENTS dataset conforming to strict distribution constraints:
- >=40 records (50 records generated)
- Category vocabulary: General Medicine, Cardiology, Dermatology, Pediatrics, Orthopedics (each >= 3)
- Status vocabulary: Scheduled, Completed, Cancelled, No-Show, Rescheduled (each >= 1)
- Consultation fee: realistic OPD range (INR 400 - 1800)
- Days since created: 0 - 30 days
- Follow-up required rate: between 10% and 30%
"""

import random
from typing import List, Dict, Any

# Deterministic seed for exact reproducibility
RANDOM_SEED = 42

CATEGORIES = [
    "General Medicine",
    "Cardiology",
    "Dermatology",
    "Pediatrics",
    "Orthopedics",
]

STATUSES = [
    "Scheduled",
    "Completed",
    "Cancelled",
    "No-Show",
    "Rescheduled",
]

# Realistic fee ranges by specialty in Indian clinics on Practo:
# Reasoning: In Indian outpatient clinics on Practo, consultation fees realistically span
# INR 400 to INR 600 for General Medicine/Pediatrics, INR 600 to INR 1,000 for Dermatology/Orthopedics,
# and INR 800 to INR 1,800 for super-specialty Cardiology consultations.
FEE_RANGES = {
    "General Medicine": (400, 600),
    "Pediatrics": (500, 750),
    "Dermatology": (600, 1000),
    "Orthopedics": (700, 1200),
    "Cardiology": (800, 1800),
}

# Probabilistic weights for category distribution ensuring realistic clinic OPD volume
CATEGORY_WEIGHTS = [0.30, 0.16, 0.20, 0.18, 0.16]

# Probabilistic weights for status distribution reflecting typical appointment lifecycles
STATUS_WEIGHTS = [0.36, 0.38, 0.12, 0.06, 0.08]

def generate_appointments(n: int = 50, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Generate deterministic appointment records."""
    rng = random.Random(seed)
    records = []

    # Guarantee at least 3 for every category and 1 for every status first
    for i, cat in enumerate(CATEGORIES):
        for rep in range(3):
            status = STATUSES[(i * 3 + rep) % len(STATUSES)]
            fee_min, fee_max = FEE_RANGES[cat]
            fee = rng.randint(fee_min // 50, fee_max // 50) * 50
            days = rng.randint(0, 30)
            # Follow-up probability calibrated to ~20%
            follow_up = rng.random() < 0.20
            rec_id = f"PRAC-{1001 + len(records)}"
            records.append({
                "record_id": rec_id,
                "category": cat,
                "status": status,
                "consultation_fee_inr": fee,
                "days_since_created": days,
                "follow_up_required": follow_up,
            })

    # Fill the remaining records up to n
    while len(records) < n:
        cat = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        fee_min, fee_max = FEE_RANGES[cat]
        fee = rng.randint(fee_min // 50, fee_max // 50) * 50
        days = rng.randint(0, 30)
        follow_up = rng.random() < 0.20
        rec_id = f"PRAC-{1001 + len(records)}"
        records.append({
            "record_id": rec_id,
            "category": cat,
            "status": status,
            "consultation_fee_inr": fee,
            "days_since_created": days,
            "follow_up_required": follow_up,
        })

    return records


# Generate global immutable dataset
APPOINTMENTS = generate_appointments(n=50, seed=RANDOM_SEED)


def validate_and_report_dataset(records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate dataset constraints and print summary report."""
    if records is None:
        records = APPOINTMENTS

    total = len(records)
    category_counts = {cat: 0 for cat in CATEGORIES}
    status_counts = {st: 0 for st in STATUSES}
    follow_up_count = 0
    total_fee = 0
    days_list = []

    for r in records:
        category_counts[r["category"]] += 1
        status_counts[r["status"]] += 1
        if r["follow_up_required"]:
            follow_up_count += 1
        total_fee += r["consultation_fee_inr"]
        days_list.append(r["days_since_created"])

    follow_up_pct = (follow_up_count / total) * 100.0

    print("=" * 60)
    print("PRACTO APPOINTMENT DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Records Generated: {total} (Target: >= 40)")
    print(f"Random Seed: {RANDOM_SEED}")
    print("\nCount per Category (Requirement: Every category >= 3):")
    for cat, count in category_counts.items():
        assert count >= 3, f"Category {cat} has {count} < 3 records!"
        print(f"  - {cat:20s}: {count} records (PASS)")

    print("\nCount per Status (Requirement: Every status >= 1):")
    for st, count in status_counts.items():
        assert count >= 1, f"Status {st} has {count} < 1 records!"
        print(f"  - {st:20s}: {count} records (PASS)")

    print(f"\nFollow-up Required:")
    print(f"  - Count: {follow_up_count}/{total}")
    print(f"  - Percentage: {follow_up_pct:.2f}% (Requirement: 10.0% - 30.0%)")
    assert 10.0 <= follow_up_pct <= 30.0, f"Follow-up rate {follow_up_pct:.2f}% out of [10%, 30%] band!"
    print("  - Constraint Check: PASS")

    print("\nSummary Statistics:")
    print(f"  - Average Fee: INR {total_fee / total:.2f}")
    print(f"  - Min Fee: INR {min(r['consultation_fee_inr'] for r in records)}")
    print(f"  - Max Fee: INR {max(r['consultation_fee_inr'] for r in records)}")
    print(f"  - Days Range: {min(days_list)} to {max(days_list)} days")
    print("=" * 60)

    return {
        "total": total,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "follow_up_pct": follow_up_pct,
        "avg_fee": total_fee / total,
        "min_fee": min(r["consultation_fee_inr"] for r in records),
        "max_fee": max(r["consultation_fee_inr"] for r in records),
        "min_days": min(days_list),
        "max_days": max(days_list),
    }


if __name__ == "__main__":
    validate_and_report_dataset()
