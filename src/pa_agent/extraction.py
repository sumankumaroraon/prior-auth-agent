"""Intake: turn a free-text PA submission into structured fields, then classify it.

Two interchangeable back ends:
* LLM mode       - chat model with structured output (Pydantic schema enforced)
* rule-based mode - regex extractor and CPT-range classifier; deterministic, no API key

Both return the same Pydantic objects, so everything downstream is identical.
"""
from __future__ import annotations

import re

from .schemas import Classification, ClinicalFacts, ExtractedRequest, PriorTherapy

EXTRACTION_PROMPT = """You are an intake specialist at a health plan's utilization-management department.
Extract the prior-authorization request below into the schema. Rules:
- Copy identifiers (NPI, member ID, ICD-10 and CPT/HCPCS codes) EXACTLY as written. Never correct or invent them.
- If a medication is requested, put only the drug name (no dose) in drug_name; otherwise fill service_code.
- urgency is "expedited" only if the submitter marks it urgent/expedited.
- Clinical facts: fill only what is stated. Unknown = null. If the note explicitly says no prior therapies, use an empty list.

REQUEST:
{text}"""

CLASSIFY_PROMPT = """Classify this prior-authorization request.
request_type: pharmacy (any drug), imaging (CPT 70010-79999), surgical (CPT 10004-69990), dme (HCPCS E/K/L codes),
sleep_study (CPT 95800-95811), therapy (CPT 97010-97799), other.
urgency: expedited if the submitter requests expedited/urgent review, else standard.

Extracted request (JSON): {extracted}"""


# ----------------------------------------------------------------------------- rule-based
_NUM = r"(\d+(?:\.\d+)?)"


def _find(pattern: str, text: str, flags: int = re.I) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n", text) if s.strip()]


def _clinical(text: str) -> ClinicalFacts:
    low = text.lower()
    f: dict = {}
    for s in _sentences(low):
        if re.search(r"physical therapy|\bpt\b|conservative|nsaid", s):
            m = re.search(r"(\d+)\s*(?:weeks|wks|wk)\b", s) or re.search(r"\bx\s*(\d+)\s*(?:weeks|wks)", s)
            mo = re.search(r"(\d+)\s*months?\b", s)
            if m:
                f["conservative_therapy_weeks"] = int(m.group(1))
            elif mo:
                f["conservative_therapy_weeks"] = int(mo.group(1)) * 4
    v = _find(r"(?:bmi|body mass index)\s*:?\s*" + _NUM, low)
    if v:
        f["bmi"] = float(v)
    v = _find(r"ahi\s*(?:=|of|:)?\s*" + _NUM, low)
    if v:
        f["ahi"] = float(v)
    if re.search(r"denies daytime sleepiness|no witnessed apneas", low):
        f["osa_symptoms"] = False
    elif re.search(r"daytime sleepiness|witnessed apneas|epworth 1\d", low):
        f["osa_symptoms"] = True
    if re.search(r"red flag|saddle anesthesia|progressive neurologic deficit", low):
        f["red_flags"] = True
    if re.search(r"(?:home sleep|hsat).{0,40}(?:inconclusive|technically inadequate)", low):
        f["hsat_inconclusive"] = True
    elif re.search(r"no home sleep (?:apnea )?test has been attempted|hsat not (?:performed|attempted)", low):
        f["hsat_inconclusive"] = False
    pts: list[PriorTherapy] = []
    explicit_none = bool(re.search(r"no prior therap|treatment-naive|prior medications:\s*none", low))
    for s in _sentences(low):
        m = re.search(r"(?:tried|failed|prior therapy:)\s+([a-z][a-z \-]+?)\s*(?:for|x|\()\s*(\d+)\s*(?:weeks|wks)", s)
        if m:
            pts.append(PriorTherapy(drug=m.group(1).strip(), weeks=int(m.group(2))))
    if pts:
        f["prior_therapies"] = pts
    elif explicit_none:
        f["prior_therapies"] = []
    return ClinicalFacts(**f)


def rule_extract(text: str) -> ExtractedRequest:
    out: dict = {}
    out["member_name"] = _find(r"(?:Member|PT NAME|for my patient)\s*:?\s*([A-Za-z][A-Za-z' \-]+?)\s*(?:\(|D\.?O\.?B|\s{2,}|$)", text, re.M)
    if out["member_name"]:
        out["member_name"] = out["member_name"].title()
    out["dob"] = _find(r"D\.?O\.?B\.?\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    out["member_id"] = _find(r"(?:Member ID|MBR\s*#|member ID)\s*:?\s*([A-Z0-9]{6,})", text)
    out["requesting_npi"] = _find(r"NPI\s*#?\s*:?\s*(\d{6,12})", text)
    out["service_code"] = _find(r"CPT/HCPCS\s+([A-Z]?\d{4,5}[A-Z]?)", text)
    if not out["service_code"]:
        svc = _find(r"(?:Requested|SERVICE/RX|prior authorization for)\s*:?\s*([^\n]+?)(?:\s+for my patient|\n|$)", text)
        if svc:
            out["drug_name"] = re.split(r"\s+\d|\s+(?:IV|SC|PO)\b", svc.strip())[0].strip().title()
    dx_line = _find(r"((?:Diagnosis|ICD10|Diagnosis codes)[^\n]*)", text) or ""
    out["dx_codes"] = re.findall(r"\b([A-TV-Z]\d{2}(?:\.[0-9A-Z]{1,4})?)\b", dx_line.upper().replace("ICD10", ""))
    out["urgency"] = "expedited" if re.search(r"\burgent\b|\bexpedit", text, re.I) else "standard"
    out["clinical"] = _clinical(text)
    return ExtractedRequest(**out)


def rule_classify(x: ExtractedRequest) -> Classification:
    rt = "other"
    code = (x.service_code or "").upper()
    if x.drug_name and not code:
        rt = "pharmacy"
    elif re.fullmatch(r"[EKL]\d{4}", code):
        rt = "dme"
    elif code.isdigit():
        n = int(code)
        rt = ("imaging" if 70010 <= n <= 79999 else "surgical" if 10004 <= n <= 69990 else
              "sleep_study" if 95800 <= n <= 95811 else "therapy" if 97010 <= n <= 97799 else "other")
    why = f"{'drug requested' if rt == 'pharmacy' else f'code {code} falls in the {rt} range'}; urgency marked {x.urgency}"
    return Classification(request_type=rt, urgency=x.urgency, rationale=why)


# ----------------------------------------------------------------------------- LLM
def llm_extract(model, text: str) -> ExtractedRequest:
    return model.with_structured_output(ExtractedRequest).invoke(EXTRACTION_PROMPT.format(text=text))


def llm_classify(model, x: ExtractedRequest) -> Classification:
    return model.with_structured_output(Classification).invoke(CLASSIFY_PROMPT.format(extracted=x.model_dump_json()))
