import json

from pa_agent.extraction import rule_classify, rule_extract
from pa_agent.policy_engine import evaluate
from pa_agent.tools import check_pa_requirement, get_drug_label, normalize_drug, npi_luhn_valid, verify_provider

POLICY = json.load(open("data/policy/medical_policy.json"))


def test_npi_checksum():
    assert npi_luhn_valid("1588608731")          # real NPI from the registry snapshot
    assert not npi_luhn_valid("1588608734")      # check digit altered
    assert not npi_luhn_valid("12345")
    assert verify_provider("1588608734")["checksum_valid"] is False


def test_knee_mri_criteria():
    pol = POLICY["services"]["73721"]
    met = evaluate(pol, ["M17.11"], {"conservative_therapy_weeks": 8}, None)
    assert [r.status for r in met] == ["met", "met"]
    short = evaluate(pol, ["M17.11"], {"conservative_therapy_weeks": 2}, None)
    assert short[1].status == "not_met"
    missing = evaluate(pol, ["M17.11"], {}, None)
    assert missing[1].status == "missing"
    acute = evaluate(pol, ["S83.241A"], {}, None)
    assert acute[1].status == "waived"


def test_step_therapy_and_exclusion():
    ozempic = POLICY["drugs"]["1991302"]
    r = evaluate(ozempic, ["E66.01"], {"prior_therapies": [{"drug": "metformin", "weeks": 26}]}, None)
    assert r[0].status == "not_met" and "EXCLUSION" in r[0].detail
    r = evaluate(ozempic, ["E11.9"], {"prior_therapies": []}, None)
    assert r[2].status == "not_met"
    r = evaluate(ozempic, ["E11.9"], {}, None)
    assert r[2].status == "missing"


def test_specialty_criterion():
    humira = POLICY["drugs"]["327361"]
    r = evaluate(humira, ["M06.9"], {"prior_therapies": [{"drug": "methotrexate", "weeks": 20}]},
                 "Dentist, Oral and Maxillofacial Pathology")
    assert r[1].status == "not_met"


def test_rxnorm_and_label():
    d = normalize_drug("Humira 40 mg")
    assert d["ingredient_rxcui"] == "327361"
    assert normalize_drug("Humera")["found"] is False
    lab = get_drug_label("2045613", "erenumab", None, ["R51.9", "G43.909"])
    assert [c["on_label"] for c in lab["dx_label_checks"]] == [False, True]
    assert check_pa_requirement(service_code="71250")["pa_required"] is False


def test_extract_fax():
    t = ("FAX COVER\nURGENT/EXPEDITED\nPT NAME: AVA KIM   D.O.B. 01/02/1970\nMBR #: DHP12345678\n"
         "ORDERING MD: JOSEPH ABBOUD NPI# 1588608731\nSERVICE/RX: CPT/HCPCS 27447 - TOTAL KNEE ARTHROPLASTY\n"
         "ICD10: M17.11\nHX: CONSERVATIVE TX: PT X 14 WKS + NSAIDS, PERSISTENT PAIN. BMI 33.1.")
    x = rule_extract(t)
    assert (x.member_id, x.requesting_npi, x.service_code, x.dx_codes, x.urgency) == \
        ("DHP12345678", "1588608731", "27447", ["M17.11"], "expedited")
    assert x.clinical.conservative_therapy_weeks == 14 and x.clinical.bmi == 33.1
    assert rule_classify(x).request_type == "surgical"


def test_extract_email_drug():
    t = ("Subject: PA\n\nHello, I am requesting prior authorization for Dupixent 300 mg SC every 2 weeks for my patient "
         "Leah Walsh (DOB 03/04/1990, member ID DHP87654321). Diagnosis codes: L20.9. "
         "Failed topical corticosteroid x 6 wks. Standard review is fine.\n\nThanks\nNPI 1558689216")
    x = rule_extract(t)
    assert x.drug_name == "Dupixent" and x.dx_codes == ["L20.9"]
    assert x.clinical.prior_therapies[0].drug == "topical corticosteroid"
    assert rule_classify(x).request_type == "pharmacy"
