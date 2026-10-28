"""Knowledge Base Documents for Practo Domain Support Agent

Contains 12 authoritative policy documents covering all required topics.
Each document contains 2 to 5 carefully crafted sentences in authentic Practo domain language.
"""

from typing import List, Dict

DOCUMENTS: List[Dict[str, str]] = [
    {
        "doc_id": "KB-DOC-001",
        "topic": "appointment-booking policy",
        "title": "Appointment Booking Policy",
        "content": (
            "Patients can schedule outpatient consultations across all specialties through the Practo app, web portal, or verified clinic helplines up to 30 days in advance. "
            "Every booking requires a verified patient contact number, valid government identification, and primary reason for consultation before confirmation. "
            "A confirmed booking generates a unique appointment record ID (e.g., PRAC-1001) along with an instant SMS and email confirmation containing clinic location and doctor details."
        )
    },
    {
        "doc_id": "KB-DOC-002",
        "topic": "cancellation/rescheduling window",
        "title": "Cancellation and Rescheduling Window Policy",
        "content": (
            "Patients may cancel or reschedule their scheduled appointments free of penalty up to two hours prior to the booked consultation slot. "
            "Cancellations requested within less than two hours of the appointment time incur a late cancellation fee equal to 20% of the consultation charge. "
            "If an appointment is cancelled by the consulting doctor or clinic due to emergencies, patients receive an immediate 100% refund or priority rebooking at no added cost."
        )
    },
    {
        "doc_id": "KB-DOC-003",
        "topic": "consultation-fee structure by specialty",
        "title": "Consultation Fee Structure by Specialty",
        "content": (
            "Consultation fees are tiered strictly by medical specialty and doctor clinical seniority across all partner clinics. "
            "Standard OPD consultations range from INR 400 to INR 600 for General Medicine and Pediatrics, INR 600 to INR 1,000 for Dermatology and Orthopedics, and INR 800 to INR 1,800 for Cardiology and super-specialty consultations. "
            "All published fees are all-inclusive of routine clinical examinations, vital measurements, and preliminary digital prescriptions."
        )
    },
    {
        "doc_id": "KB-DOC-004",
        "topic": "insurance-claim process",
        "title": "Insurance Claim and Reimbursement Process",
        "content": (
            "Practo partner clinics support cashless claims with major empanelled health insurance providers and Third Party Administrators (TPAs). "
            "Patients must present a valid physical insurance e-card and photo ID at the clinic billing desk at least thirty minutes prior to their consultation for eligibility pre-authorization. "
            "For non-network insurers, the clinic provides an itemized medical bill, doctor consultation summary, and digital payment receipt within 24 hours to facilitate direct patient reimbursement."
        )
    },
    {
        "doc_id": "KB-DOC-005",
        "topic": "prescription-refill policy",
        "title": "Prescription Refill and Renewal Policy",
        "content": (
            "Prescription refills for chronic, maintenance medications can be requested digitally through the Practo patient portal within 90 days of the last clinical visit. "
            "Controlled substances, antibiotics, narcotic analgesics, and specialized cardiac drugs cannot be refilled without an in-person or live teleconsultation review. "
            "The treating physician reviews and approves standard refill requests within 24 business hours, dispatching a digitally signed e-prescription directly to the patient's verified mobile app."
        )
    },
    {
        "doc_id": "KB-DOC-006",
        "topic": "lab-test turnaround times",
        "title": "Diagnostic Lab Test Turnaround Times",
        "content": (
            "Routine diagnostic tests including complete blood counts, urine analysis, and basic metabolic panels are completed with digital reports uploaded within 12 to 24 hours of sample collection. "
            "Specialized diagnostic tests, such as histopathology, hormone panels, microbiology cultures, and advanced cardiac markers, require between 48 to 72 business hours for comprehensive pathologist review. "
            "Critical lab alerts indicating abnormal findings are flagged automatically for urgent physician review and direct telephonic patient notification."
        )
    },
    {
        "doc_id": "KB-DOC-007",
        "topic": "telemedicine eligibility",
        "title": "Telemedicine Consultation Eligibility",
        "content": (
            "Telemedicine consultations are available for non-emergency follow-ups, minor acute ailments, second opinions, and routine chronic disease management across eligible specialties. "
            "Patients experiencing severe chest pain, acute respiratory distress, severe trauma, sudden neurological deficits, or high-risk pediatric conditions are strictly ineligible for teleconsultation and must visit an emergency room. "
            "All virtual visits are conducted over encrypted end-to-end video connections adhering to national Telemedicine Practice Guidelines."
        )
    },
    {
        "doc_id": "KB-DOC-008",
        "topic": "emergency-visit protocol",
        "title": "Emergency Visit and Urgent Care Protocol",
        "content": (
            "Practo outpatient clinics operate scheduled services and do not provide emergency trauma care or intensive life support. "
            "Patients exhibiting critical warning signs—such as severe chest pain radiating to the left arm, acute stroke symptoms, severe breathing difficulty, or heavy bleeding—must immediately dial 112/108 or proceed to the nearest hospital casualty department. "
            "If a patient arrives at a clinic in unstable condition, clinic staff will provide basic first-aid stabilization and coordinate immediate ambulance transfer to a tertiary hospital."
        )
    },
    {
        "doc_id": "KB-DOC-009",
        "topic": "patient-data privacy policy",
        "title": "Patient Data Privacy and Confidentiality Policy",
        "content": (
            "All electronic medical records, clinical notes, prescriptions, and communication logs are stored in encrypted databases compliant with ISO 27001, HIPAA, and Indian Digital Personal Data Protection standards. "
            "Access to patient health information is strictly role-based and granted only to the consulting doctor, assigned clinical staff, and the authenticated patient. "
            "Practo never sells, monetizes, or shares identifiable patient health information with pharmaceutical companies, advertisers, or unassociated commercial entities."
        )
    },
    {
        "doc_id": "KB-DOC-010",
        "topic": "follow-up-visit discount policy",
        "title": "Follow-Up Visit Discount Policy",
        "content": (
            "Patients attending an in-person follow-up consultation with the same doctor within 7 calendar days of an initial completed visit receive a 50% discount on the consultation fee. "
            "Follow-up visits scheduled between day 8 and day 14 post-consultation are eligible for a 25% courtesy discount on standard clinic consultation rates. "
            "Any visit occurring after 14 days or addressing an unrelated new medical complaint is classified as a fresh primary consultation billed at standard tariff."
        )
    },
    {
        "doc_id": "KB-DOC-011",
        "topic": "second-opinion process",
        "title": "Second Opinion Consultation Process",
        "content": (
            "Patients seeking an independent expert second opinion can book dedicated 30-minute review sessions with senior specialty consultants through the Practo Second Opinion Desk. "
            "Prior to the scheduled consultation, patients must securely upload prior diagnostic reports, imaging scans, and clinical case summaries for advance specialist review. "
            "The consultant provides a comprehensive written clinical evaluation within 48 hours, highlighting alternative diagnoses, updated treatment plans, and evidence-based recommendations."
        )
    },
    {
        "doc_id": "KB-DOC-012",
        "topic": "home-visit eligibility",
        "title": "Home Visit and Domiciliary Healthcare Eligibility",
        "content": (
            "Home healthcare visits by qualified general physicians and nursing personnel are reserved for geriatric patients, mobility-impaired individuals, and bedbound post-operative patients. "
            "Home visit requests must be booked at least 24 hours in advance and remain subject to physician serviceability within a 10-kilometer radius of the partner clinic hub. "
            "Emergency medical cases, complex pediatric assessments, and surgical procedures are strictly excluded from home visit services and require clinic or hospital admission."
        )
    }
]

def get_kb_documents() -> List[Dict[str, str]]:
    """Return all 12 knowledge base documents."""
    return DOCUMENTS
