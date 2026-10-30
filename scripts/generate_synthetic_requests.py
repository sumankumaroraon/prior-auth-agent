"""Generate synthetic prior-authorization (PA) submissions with ground-truth labels.

Real PA requests contain PHI, so the requests here are synthetic. The reference data
they point to is real: requesting providers are real NPIs from the CMS NPI Registry,
diagnosis codes come from the NLM ICD-10-CM API, and drugs come from RxNorm.

Each request is written as messy free text in one of three intake channels (provider
portal, faxed form after OCR, or email). The agent has to extract structured fields
from that text. Ground truth (scenario, expected route, true fields) is stored
separately so the agent never sees it.

Usage:  python scripts/generate_synthetic_requests.py --n 240 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NPI = json.load(open(ROOT / "data/reference/npi_registry_snapshot.json"))["records"]
ICD = json.load(open(ROOT / "data/reference/icd10cm_snapshot.json"))["records"]

FIRST = ["Ava", "Liam", "Maya", "Noah", "Zara", "Ethan", "Priya", "Lucas", "Amara", "Owen", "Sofia", "Mateo",
         "Chloe", "Isaac", "Leah", "Andre", "Nina", "Caleb", "Hana", "Marcus", "Elena", "Jamal", "Grace", "Ravi",
         "Talia", "Victor", "Imani", "Felix", "Rosa", "Darius"]
LAST = ["Reyes", "Okafor", "Patel", "Kim", "Nguyen", "Brennan", "Haddad", "Silva", "Kowalski", "Mensah",
        "Duarte", "Lindqvist", "Moreau", "Ibrahim", "Castillo", "Fischer", "Tanaka", "Adeyemi", "Russo", "Walsh"]


def npis_for(*specialties: str) -> list[str]:
    return [n for n, r in NPI.items() if r["primary_specialty"] in specialties]


def corrupt_npi(npi: str) -> str:
    """Change the check digit so the NPI fails the Luhn check (simulates a typo / OCR error)."""
    return npi[:-1] + str((int(npi[-1]) + 3) % 10)


# --------------------------------------------------------------------------- service profiles
# Each profile knows which diagnoses / prescribers fit it and how to write clinical facts
# for "clean", "not_met" and "missing" variants.

def wk(rng, lo, hi):
    return rng.randint(lo, hi)


PROFILES = {
    "73721": dict(kind="service", name="MRI right knee without contrast", dx=["M17.11", "M17.12", "M17.0", "M23.211", "S83.241A"],
                  specs=["Orthopaedic Surgery", "Physical Medicine & Rehabilitation", "Family Medicine"]),
    "72148": dict(kind="service", name="MRI lumbar spine without contrast", dx=["M54.50", "M51.16"],
                  specs=["Physical Medicine & Rehabilitation", "Family Medicine", "Orthopaedic Surgery", "Neurology"]),
    "78815": dict(kind="service", name="PET/CT skull base to mid-thigh", dx=["C34.90", "C50.911"],
                  specs=["Family Medicine", "Hospitalist"]),
    "71250": dict(kind="service", name="CT chest without contrast", dx=["C34.90"], specs=["Family Medicine", "Hospitalist"]),
    "27447": dict(kind="service", name="Total knee arthroplasty", dx=["M17.11", "M17.12", "M17.0"], specs=["Orthopaedic Surgery"]),
    "29881": dict(kind="service", name="Knee arthroscopy with medial meniscectomy", dx=["M23.211", "S83.241A"], specs=["Orthopaedic Surgery"]),
    "E0601": dict(kind="service", name="CPAP device", dx=["G47.33"], specs=["Sleep Medicine"]),
    "K0823": dict(kind="service", name="Power wheelchair, group 2 standard", dx=["G12.21", "Z99.81"],
                  specs=["Physical Medicine & Rehabilitation", "Neurology"]),
    "95810": dict(kind="service", name="Attended in-lab polysomnography", dx=["G47.30", "G47.33"], specs=["Sleep Medicine"]),
    "97110": dict(kind="service", name="Physical therapy - therapeutic exercise", dx=["M54.50", "M17.11"],
                  specs=["Physical Medicine & Rehabilitation", "Orthopaedic Surgery"]),
    "adalimumab": dict(kind="drug", names=["Humira 40 mg/0.4 mL pen SC every 14 days", "adalimumab 40 mg SC q2 weeks",
                                            "adalimumab-adaz 40 mg/0.4 mL SC every other week"],
                       dx_spec={"M05.79": "Rheumatology", "M06.9": "Rheumatology", "L40.50": "Rheumatology",
                                "L40.0": "Dermatology", "K50.90": "Gastroenterology", "K51.90": "Gastroenterology"}),
    "ustekinumab": dict(kind="drug", names=["Stelara 45 mg SC at weeks 0 and 4, then q12 weeks", "ustekinumab 90 mg SC q8 weeks"],
                        dx_spec={"L40.0": "Dermatology", "K50.90": "Gastroenterology"}),
    "infliximab": dict(kind="drug", names=["infliximab 5 mg/kg IV at weeks 0, 2, 6 then q8 weeks"],
                       dx_spec={"K50.90": "Gastroenterology", "K51.90": "Gastroenterology"}),
    "dupilumab": dict(kind="drug", names=["Dupixent 300 mg SC every 2 weeks after 600 mg loading dose", "dupilumab 300 mg SC q2w"],
                      dx_spec={"L20.9": "Dermatology", "J45.50": "Allergy & Immunology"}),
    "semaglutide": dict(kind="drug", names=["Ozempic 0.5 mg SC weekly", "semaglutide 1 mg SC once weekly"],
                        dx_spec={"E11.9": "Endocrinology, Diabetes & Metabolism", "E11.65": "Endocrinology, Diabetes & Metabolism"}),
    "tirzepatide": dict(kind="drug", names=["Mounjaro 5 mg SC weekly", "tirzepatide 2.5 mg SC weekly x4 then 5 mg"],
                        dx_spec={"E11.9": "Endocrinology, Diabetes & Metabolism", "E11.65": "Family Medicine"}),
    "erenumab": dict(kind="drug", names=["Aimovig 70 mg SC monthly", "erenumab 140 mg SC every month"],
                     dx_spec={"G43.709": "Neurology", "G43.909": "Neurology"}),
    "apixaban": dict(kind="drug", names=["Eliquis 5 mg PO BID", "apixaban 5 mg twice daily"],
                     dx_spec={"I48.91": "Cardiovascular Disease"}),
}
NO_PA = ["71250", "97110", "apixaban"]
MD_REVIEW = ["78815", "K0823"]
CLEAN_POOL = ["73721", "72148", "27447", "29881", "E0601", "95810", "adalimumab", "ustekinumab", "infliximab",
              "dupilumab", "semaglutide", "tirzepatide", "erenumab"]


def clean_facts(rng, key, dx):
    """Facts that satisfy policy for this service/drug + diagnosis."""
    f: dict = {}
    if key in ("73721", "29881"):
        if not dx.startswith("S83") or rng.random() < 0.5:
            f["conservative_therapy_weeks"] = wk(rng, 6, 16)
    elif key == "72148":
        if rng.random() < 0.25:
            f["red_flags"] = True
        else:
            f["conservative_therapy_weeks"] = wk(rng, 6, 14)
    elif key == "27447":
        f["conservative_therapy_weeks"] = wk(rng, 12, 40)
        f["bmi"] = round(rng.uniform(23, 38.5), 1)
    elif key == "E0601":
        if rng.random() < 0.6:
            f["ahi"] = round(rng.uniform(15, 45), 1)
        else:
            f["ahi"] = round(rng.uniform(5.5, 14), 1)
            f["osa_symptoms"] = True
    elif key == "95810":
        f["hsat_inconclusive"] = True
    elif key == "adalimumab" and dx.startswith(("M05", "M06")):
        f["prior_therapies"] = [{"drug": "methotrexate", "weeks": wk(rng, 12, 52)}]
    elif key == "ustekinumab":
        f["prior_therapies"] = [{"drug": rng.choice(["methotrexate", "adalimumab", "phototherapy"]), "weeks": wk(rng, 12, 40)}]
    elif key == "dupilumab" and dx.startswith("L20"):
        f["prior_therapies"] = [{"drug": "topical corticosteroid", "weeks": wk(rng, 4, 20)}]
    elif key in ("semaglutide", "tirzepatide"):
        f["prior_therapies"] = [{"drug": "metformin", "weeks": wk(rng, 12, 104)}]
    elif key == "erenumab":
        f["prior_therapies"] = [{"drug": d, "weeks": wk(rng, 8, 26)} for d in rng.sample(["topiramate", "propranolol", "amitriptyline"], 2)]
    return f


def not_met_facts(rng, key, dx):
    """Facts that are documented but do NOT satisfy policy (potential denial -> clinician)."""
    if key in ("73721", "72148", "29881"):
        return {"conservative_therapy_weeks": wk(rng, 1, 4)}
    if key == "27447":
        return rng.choice([{"conservative_therapy_weeks": wk(rng, 14, 30), "bmi": round(rng.uniform(41, 47), 1)},
                           {"conservative_therapy_weeks": wk(rng, 3, 8), "bmi": round(rng.uniform(25, 35), 1)}])
    if key == "E0601":
        return {"ahi": round(rng.uniform(5.5, 12), 1), "osa_symptoms": False}
    if key == "95810":
        return {"hsat_inconclusive": False}
    if key == "adalimumab":
        return {"prior_therapies": [{"drug": "methotrexate", "weeks": wk(rng, 3, 8)}]}
    if key in ("semaglutide", "tirzepatide"):
        return {"prior_therapies": []}
    if key == "erenumab":
        return {"prior_therapies": [{"drug": "topiramate", "weeks": wk(rng, 8, 16)}]}
    if key == "dupilumab":
        return {"prior_therapies": [{"drug": "topical corticosteroid", "weeks": wk(rng, 1, 2)}]}
    raise KeyError(key)


NOT_MET_POOL = ["73721", "72148", "29881", "27447", "E0601", "95810", "adalimumab", "semaglutide", "tirzepatide", "erenumab", "dupilumab"]
MISSING_POOL = ["73721", "72148", "27447", "E0601", "95810", "adalimumab", "semaglutide", "erenumab"]


# --------------------------------------------------------------------------- text rendering
# STYLE="standard" uses the phrasings the rule-based extractor was developed against.
# STYLE="holdout" adds wording and an EHR-export layout it has never seen, to measure how the
# workflow behaves when extraction fails (it should pend or escalate, never auto-approve).
STYLE = "standard"


def holdout_line(rng, f: dict) -> list[str] | None:
    """Unseen phrasings for the same facts (used for ~50% of holdout requests)."""
    out = []
    if "conservative_therapy_weeks" in f:
        w = f["conservative_therapy_weeks"]
        out.append(f"{w // 4} months of supervised physiotherapy, no improvement." if w % 4 == 0 else
                   f"Has completed a {w}-week course of physiotherapy and anti-inflammatories.")
    if f.get("red_flags"):
        out.append("New bowel/bladder dysfunction and foot drop on exam.")
    if "bmi" in f:
        out.append(f"BMI is {f['bmi']}.")
    if "ahi" in f:
        out.append(f"Sleep study showed an apnea-hypopnea index of {f['ahi']}.")
    if f.get("osa_symptoms") is True:
        out.append("Falls asleep while driving; partner reports loud snoring and gasping.")
    if f.get("osa_symptoms") is False:
        out.append("No hypersomnolence reported.")
    if f.get("hsat_inconclusive") is True:
        out.append("Prior at-home study failed due to sensor dislodgement.")
    if f.get("hsat_inconclusive") is False:
        out.append("Has not had any home testing.")
    if "prior_therapies" in f:
        if not f["prior_therapies"]:
            out.append("Has not been on any medication for this.")
        for p in f["prior_therapies"]:
            out.append(f"{p['drug'].capitalize()} was used for {p['weeks']} weeks without benefit.")
    return out


def fact_lines(rng, f: dict, key: str) -> list[str]:
    if STYLE == "holdout" and rng.random() < 0.5:
        return holdout_line(rng, f)
    out = []
    if "conservative_therapy_weeks" in f:
        w = f["conservative_therapy_weeks"]
        out.append(rng.choice([f"Completed {w} weeks of physical therapy and NSAIDs without adequate relief.",
                               f"Conservative tx: PT x {w} wks + NSAIDs, persistent pain.",
                               f"Pt has done {w} weeks of conservative therapy (home exercise, PT, naproxen)."]))
    if f.get("red_flags"):
        out.append(rng.choice(["Red flags: progressive lower-extremity weakness and new saddle anesthesia.",
                               "RED FLAG SYMPTOMS - progressive neurologic deficit noted on exam."]))
    if "bmi" in f:
        out.append(rng.choice([f"BMI {f['bmi']}.", f"Body mass index: {f['bmi']} kg/m2."]))
    if "ahi" in f:
        out.append(rng.choice([f"Home sleep apnea test: AHI {f['ahi']} events/hr.", f"HSAT AHI = {f['ahi']}/hr."]))
    if f.get("osa_symptoms") is True:
        out.append("Symptoms: excessive daytime sleepiness (Epworth 14), witnessed apneas.")
    if f.get("osa_symptoms") is False:
        out.append("Denies daytime sleepiness; Epworth 5. No witnessed apneas.")
    if f.get("hsat_inconclusive") is True:
        out.append(rng.choice(["Home sleep test was technically inadequate / inconclusive.",
                               "HSAT inconclusive (insufficient recording time); in-lab study needed."]))
    if f.get("hsat_inconclusive") is False:
        out.append("No home sleep apnea test has been attempted; requesting in-lab study as first test.")
    if "prior_therapies" in f:
        pts = f["prior_therapies"]
        if not pts:
            out.append(rng.choice(["No prior therapies tried; treatment-naive.", "Prior medications: none."]))
        for p in pts:
            out.append(rng.choice([f"Tried {p['drug']} for {p['weeks']} weeks - inadequate response.",
                                   f"Failed {p['drug']} x {p['weeks']} wks.",
                                   f"Prior therapy: {p['drug']} ({p['weeks']} weeks), discontinued for lack of efficacy."]))
    return out


def render(rng, req: dict) -> str:
    """Write the request as free text in one of three channels."""
    ch = req["channel"]
    prov = NPI.get(req["_true_npi"], {"name": "Unknown", "credential": ""})
    dx_txt = "; ".join(f"{c} {ICD[c]['description'] or ''}".strip() for c in req["dx_codes"])
    svc = req["service_text"]
    urg = req["urgency"]
    notes = " ".join(req["notes"])
    cred = f", {prov['credential']}" if prov.get("credential") else ""
    if ch == "ehr":
        last_first = ", ".join(reversed(req["member_name"].split(" ", 1)))
        return "\n".join([
            "AUTHORIZATION REQUEST (EHR export)",
            f"Priority: {'STAT - patient deteriorating, time-sensitive' if urg == 'expedited' else 'Routine'}",
            f"Patient: {last_first} | Pt ID: {req['member_id']} | Birth date: {req['dob']}",
            f"Prescriber NPI: {req['npi']} ({prov['name']}{cred})",
            f"Order: {svc}",
            f"Dx: {', '.join(req['dx_codes'])}",
            f"Assessment/Plan: {notes}",
        ])
    if ch == "portal":
        return "\n".join([
            "PROVIDER PORTAL - PRIOR AUTHORIZATION REQUEST",
            f"Review type: {'EXPEDITED' if urg == 'expedited' else 'Standard'}",
            f"Member: {req['member_name']}  DOB: {req['dob']}  Member ID: {req['member_id']}",
            f"Requesting provider: {prov['name']}{cred}  NPI: {req['npi']}",
            f"Requested: {svc}",
            f"Diagnosis: {dx_txt}",
            f"Clinical notes: {notes}",
        ])
    if ch == "fax":
        urg_line = rng.choice(["*** URGENT - EXPEDITED REVIEW REQUESTED - delay could seriously jeopardize health ***",
                               "URGENT/EXPEDITED"]) if urg == "expedited" else "ROUTINE"
        return "\n".join([
            "FAX COVER - UTILIZATION MANAGEMENT  pg 1/3",
            urg_line,
            f"PT NAME: {req['member_name'].upper()}   D.O.B. {req['dob']}",
            f"MBR #: {req['member_id']}",
            f"ORDERING MD: {prov['name'].upper()} NPI# {req['npi']}",
            f"SERVICE/RX: {svc.upper()}",
            f"ICD10: {', '.join(req['dx_codes'])}",
            f"HX: {notes.upper()}",
        ])
    # email
    urg_phrase = "This is an urgent request; please expedite." if urg == "expedited" else "Standard review is fine."
    return (f"Subject: PA request for {req['member_name']}\n\n"
            f"Hello UM team, I am requesting prior authorization for {svc} for my patient {req['member_name']} "
            f"(DOB {req['dob']}, member ID {req['member_id']}). Diagnosis codes: {', '.join(req['dx_codes'])}. "
            f"{notes} {urg_phrase}\n\nThank you,\n{prov['name']}{cred}\nNPI {req['npi']}")


# --------------------------------------------------------------------------- scenario builder
SCENARIO_WEIGHTS = [
    ("clean", 0.36), ("no_pa_required", 0.08), ("missing_info", 0.12), ("criteria_not_met", 0.12),
    ("md_review", 0.05), ("invalid_npi", 0.04), ("invalid_icd", 0.04), ("specialty_mismatch", 0.05),
    ("off_label_or_excluded", 0.05), ("unrecognized_drug", 0.02), ("duplicate", 0.04), ("expedited_clean", 0.03),
]
REQUEST_TYPE = {"73721": "imaging", "72148": "imaging", "78815": "imaging", "71250": "imaging", "27447": "surgical",
                "29881": "surgical", "E0601": "dme", "K0823": "dme", "95810": "sleep_study", "97110": "therapy"}


def pick_npi(rng, specialty: str) -> str:
    return rng.choice(npis_for(specialty))


def build_one(rng, i: int, scenario: str, received: datetime) -> dict:
    urgency = "expedited" if scenario == "expedited_clean" or (scenario not in ("duplicate",) and rng.random() < 0.12) else "standard"
    if scenario == "clean" and urgency == "expedited":
        scenario = "expedited_clean"
    facts: dict = {}
    if scenario in ("clean", "expedited_clean"):
        key = rng.choice(CLEAN_POOL)
    elif scenario == "no_pa_required":
        key = rng.choice(NO_PA)
    elif scenario == "missing_info":
        key = rng.choice(MISSING_POOL)
    elif scenario == "criteria_not_met":
        key = rng.choice(NOT_MET_POOL)
    elif scenario == "md_review":
        key = rng.choice(MD_REVIEW)
    elif scenario == "specialty_mismatch":
        key = rng.choice(["adalimumab", "dupilumab"])
    elif scenario == "off_label_or_excluded":
        key = rng.choice(["semaglutide", "erenumab"])
    elif scenario == "unrecognized_drug":
        key = "adalimumab"
    elif scenario == "invalid_icd":
        key = rng.choice(["73721", "semaglutide", "ocrelizumab"])
    else:  # invalid_npi
        key = rng.choice(CLEAN_POOL)

    # diagnosis + prescriber
    if key == "ocrelizumab":
        prof = dict(kind="drug", names=["Ocrevus 600 mg IV every 6 months"])
        dx, spec = "G35", "Neurology"
    else:
        prof = PROFILES[key]
        if prof["kind"] == "drug":
            dx = rng.choice(list(prof["dx_spec"]))
            spec = prof["dx_spec"][dx]
        else:
            dx = rng.choice(prof["dx"])
            spec = rng.choice(prof["specs"])
    if scenario == "invalid_icd" and key == "73721":
        dx = "M17.99"
    if scenario == "invalid_icd" and key == "semaglutide":
        dx = "E11.99"
    if scenario == "specialty_mismatch":
        spec = rng.choice(["Dentist", "Family Medicine"]) if key == "adalimumab" else "Family Medicine"
        dx = "M06.9" if key == "adalimumab" else "L20.9"
    if scenario == "off_label_or_excluded":
        dx = "E66.01" if key == "semaglutide" else "R51.9"
    if scenario == "missing_info" and key == "adalimumab":
        dx = rng.choice(["M05.79", "M06.9"])
    if scenario == "criteria_not_met" and key == "adalimumab":
        dx = rng.choice(["M05.79", "M06.9"])
    if scenario == "criteria_not_met" and key == "dupilumab":
        dx = "L20.9"
    if scenario == "criteria_not_met" and key in ("73721", "29881"):
        dx = rng.choice(["M17.11", "M23.211"]) if key == "73721" else "M23.211"

    npi = pick_npi(rng, spec)
    true_npi = npi
    if scenario == "invalid_npi":
        npi = corrupt_npi(npi)

    # facts
    if scenario in ("clean", "expedited_clean", "invalid_npi", "specialty_mismatch", "unrecognized_drug", "invalid_icd"):
        facts = clean_facts(rng, key, dx) if key in PROFILES else {}
    elif scenario == "criteria_not_met":
        facts = not_met_facts(rng, key, dx)
    elif scenario == "missing_info":
        facts = clean_facts(rng, key, dx)
        if key in ("73721", "72148"):
            dx = "M17.11" if key == "73721" else "M54.50"
            facts = {}
        elif key == "27447":
            facts.pop(rng.choice(["bmi", "conservative_therapy_weeks"]))
        elif key == "E0601":
            facts = {}
        elif key == "95810":
            facts = {}
        else:  # drugs with step therapy -> omit prior therapy section entirely
            facts = {}
    elif scenario == "off_label_or_excluded":
        facts = {"prior_therapies": [{"drug": "metformin", "weeks": 26}]} if key == "semaglutide" else \
            {"prior_therapies": [{"drug": "topiramate", "weeks": 12}, {"drug": "propranolol", "weeks": 10}]}
    elif scenario == "md_review":
        facts = {}

    if prof["kind"] == "drug":
        service_text = rng.choice(prof["names"])
        if scenario == "unrecognized_drug":
            service_text = "Humera 40 mg SC every 2 weeks"
        if scenario == "off_label_or_excluded" and key == "semaglutide":
            service_text = rng.choice(["Wegovy 2.4 mg SC weekly for weight management", "Ozempic 1 mg SC weekly for weight loss"])
        service_code = None
        request_type = "pharmacy"
    else:
        service_text = f"CPT/HCPCS {key} - {prof['name']}"
        service_code = key
        request_type = REQUEST_TYPE[key]

    expected = {
        "clean": "AUTO_APPROVE", "expedited_clean": "AUTO_APPROVE", "no_pa_required": "NO_PA_REQUIRED",
        "missing_info": "PEND_INFO", "invalid_npi": "PEND_INFO", "invalid_icd": "PEND_INFO",
        "criteria_not_met": "HUMAN_REVIEW", "md_review": "HUMAN_REVIEW", "specialty_mismatch": "HUMAN_REVIEW",
        "off_label_or_excluded": "HUMAN_REVIEW", "unrecognized_drug": "HUMAN_REVIEW",
    }[scenario]
    # Expedited requests cannot wait for a request-for-information cycle: a clinician handles them.
    if expected == "PEND_INFO" and urgency == "expedited":
        expected = "HUMAN_REVIEW"

    mname = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    dob = (datetime(1950, 1, 1) + timedelta(days=rng.randint(0, 365 * 55))).strftime("%m/%d/%Y")
    req = {
        "request_id": f"PA-2026-{i:05d}",
        "received_at": received.isoformat(timespec="minutes"),
        "channel": rng.choice(["portal", "portal", "fax", "email", "ehr", "ehr"] if STYLE == "holdout" else ["portal", "portal", "fax", "email"]),
        "member_name": mname, "dob": dob, "member_id": f"DHP{rng.randint(10**7, 10**8 - 1)}",
        "npi": npi, "_true_npi": true_npi, "dx_codes": [dx], "service_text": service_text,
        "urgency": urgency, "notes": [],
    }
    req["notes"] = fact_lines(rng, facts, key)
    req["text"] = render(rng, req)
    truth = {"request_id": req["request_id"], "scenario": scenario, "expected_route": expected,
             "request_type": request_type, "urgency": urgency, "service_code": service_code,
             "drug_key": key if prof["kind"] == "drug" else None, "dx_codes": [dx], "npi": npi,
             "member_id": req["member_id"], "facts": facts}
    return {"request": req, "truth": truth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--style", choices=["standard", "holdout"], default="standard")
    ap.add_argument("--prefix", default="", help="output file prefix, e.g. holdout_")
    a = ap.parse_args()
    global STYLE
    STYLE = a.style
    rng = random.Random(a.seed)
    start = datetime(2026, 10, 1, 8, 0)
    names, weights = zip(*SCENARIO_WEIGHTS)
    items = []
    t = start
    for i in range(1, a.n + 1):
        t += timedelta(minutes=rng.randint(20, 240))
        scen = rng.choices(names, weights)[0]
        originals = [x for x in items if x["truth"]["expected_route"] == "AUTO_APPROVE"]
        if scen == "duplicate" and originals:
            src = rng.choice(originals[-15:])
            r = dict(src["request"])
            r["request_id"] = f"PA-2026-{i:05d}"
            r["received_at"] = t.isoformat(timespec="minutes")
            r["channel"] = rng.choice(["portal", "fax", "email"])
            r["request_id"] = f"PA-2026-{i:05d}" if not a.prefix else f"HO-2026-{i:05d}"
            r["urgency"] = "standard"
            r["text"] = render(rng, r)
            truth = dict(src["truth"], request_id=r["request_id"], scenario="duplicate", expected_route="HUMAN_REVIEW",
                         urgency="standard", duplicate_of=src["request"]["request_id"])
            items.append({"request": r, "truth": truth})
            continue
        if scen == "duplicate":
            scen = "clean"
        it = build_one(rng, i, scen, t)
        if a.prefix:
            it["request"]["request_id"] = it["truth"]["request_id"] = f"HO-2026-{i:05d}"
        items.append(it)

    out = ROOT / "data/synthetic"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.prefix}pa_requests.jsonl", "w") as f:
        for it in items:
            r = it["request"]
            f.write(json.dumps({"request_id": r["request_id"], "received_at": r["received_at"],
                                "channel": r["channel"], "text": r["text"]}) + "\n")
    with open(out / f"{a.prefix}ground_truth.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it["truth"]) + "\n")
    from collections import Counter
    print(len(items), "requests")
    print(Counter(i["truth"]["scenario"] for i in items))
    print(Counter(i["truth"]["expected_route"] for i in items))


if __name__ == "__main__":
    main()
