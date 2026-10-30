"""The approved tool catalog.

These are the ONLY actions the agent can take against outside systems. Each tool has a
Pydantic argument schema, so a malformed call from the LLM is rejected before anything
runs. The same functions are served over MCP by mcp_server.py.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .audit import AuditStore
from .clients import ReferenceData
from .config import get_settings
from .policy_engine import evaluate


@lru_cache(maxsize=1)
def _ref() -> ReferenceData:
    return ReferenceData()


@lru_cache(maxsize=1)
def _policy() -> dict:
    return json.load(open(get_settings().policy_path))


def npi_luhn_valid(npi: str) -> bool:
    """NPI check digit: Luhn over '80840' + first 9 digits (CMS NPI standard)."""
    if not re.fullmatch(r"\d{10}", npi or ""):
        return False
    digits = [int(d) for d in "80840" + npi[:9]]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return (10 - total % 10) % 10 == int(npi[9])


# ICD-10 prefix -> condition wording used on FDA labels (for the on-label check).
DX_TO_LABEL_TERM = {
    "M05": "rheumatoid arthritis", "M06": "rheumatoid arthritis", "L40.5": "psoriatic arthritis", "L40": "plaque psoriasis",
    "K50": "crohn", "K51": "ulcerative colitis", "L20": "atopic dermatitis", "J45": "asthma", "E11": "type 2 diabetes",
    "E66": "obesity", "G43": "migraine", "I48": "atrial fibrillation", "G35": "multiple sclerosis", "M45": "ankylosing spondylitis",
}


# --------------------------------------------------------------------------- argument schemas
class VerifyProviderArgs(BaseModel):
    npi: str = Field(description="10-digit National Provider Identifier of the requesting provider")


class ValidateDiagnosisArgs(BaseModel):
    icd10_code: str = Field(description="One ICD-10-CM code, e.g. M17.11")


class NormalizeDrugArgs(BaseModel):
    drug_name: str = Field(description="Drug name as written on the request (brand or generic), without dose")


class DrugLabelArgs(BaseModel):
    ingredient_rxcui: str = Field(description="RxNorm ingredient RXCUI returned by normalize_drug")
    ingredient: str = Field(description="Ingredient name returned by normalize_drug")
    brand: Optional[str] = Field(None, description="Brand name if the request named a brand (e.g. Wegovy)")
    dx_codes: list[str] = Field(default_factory=list, description="Diagnosis codes to check against the label")


class PARequirementArgs(BaseModel):
    service_code: Optional[str] = Field(None, description="CPT/HCPCS code for procedures, imaging, DME")
    ingredient_rxcui: Optional[str] = Field(None, description="RxNorm ingredient RXCUI for drugs")


class PolicyArgs(BaseModel):
    policy_id: str = Field(description="Policy id returned by check_pa_requirement, e.g. IMG-101")


class EvaluateArgs(BaseModel):
    policy_id: str
    dx_codes: list[str]
    facts: dict[str, Any] = Field(description="ClinicalFacts extracted from the request")
    prescriber_taxonomy: Optional[str] = Field(None, description="Taxonomy description from verify_provider")


class DuplicateArgs(BaseModel):
    request_id: str
    member_id: str
    item_key: str = Field(description="service_code or ingredient_rxcui")
    received_at: str = Field(description="ISO timestamp of this submission")


# --------------------------------------------------------------------------- implementations
def verify_provider(npi: str) -> dict:
    if not npi_luhn_valid(npi):
        return {"npi": npi, "checksum_valid": False, "found": False, "source": "local_rule",
                "message": "NPI fails the CMS Luhn check-digit test (likely typo or OCR error)"}
    r = _ref().lookup_npi(npi)
    if r.get("source") == "cache_miss":
        return {"npi": npi, "checksum_valid": True, "found": None, "source": "cache_miss",
                "message": "Registry unreachable and NPI not in snapshot - cannot verify"}
    return {"checksum_valid": True, **r}


def validate_diagnosis(icd10_code: str) -> dict:
    code = icd10_code.strip().upper()
    if not re.fullmatch(r"[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?", code):
        return {"code": code, "valid": False, "description": None, "source": "local_rule", "message": "not ICD-10-CM format"}
    r = _ref().lookup_icd10(code)
    if r.get("source") == "cache_miss":
        return {"code": code, "valid": None, "source": "cache_miss", "message": "cannot verify code"}
    return r


def normalize_drug(drug_name: str) -> dict:
    name = re.split(r"\s+\d", drug_name.strip())[0].strip()  # drop dose text if passed
    r = _ref().normalize_drug(name)
    if r.get("source") == "cache_miss":
        return {"query": name, "found": False, "source": "cache_miss"}
    return r


def get_drug_label(ingredient_rxcui: str, ingredient: str, brand: str | None = None, dx_codes: list[str] | None = None) -> dict:
    lab = _ref().drug_label(ingredient_rxcui, ingredient, brand)
    if lab.get("source") == "cache_miss":
        return {"ingredient": ingredient, "found": False, "source": "cache_miss"}
    terms = [t.lower() for t in lab.get("indication_terms", [])]
    checks = []
    for dx in dx_codes or []:
        term = next((v for k, v in sorted(DX_TO_LABEL_TERM.items(), key=lambda kv: -len(kv[0])) if dx.upper().startswith(k)), None)
        on = bool(term and any(term in t or t in term for t in terms))
        checks.append({"dx": dx, "condition": term, "on_label": on})
    return {"found": True, **lab, "dx_label_checks": checks}


def check_pa_requirement(service_code: str | None = None, ingredient_rxcui: str | None = None) -> dict:
    pol = _policy()
    entry, key = None, None
    if service_code:
        key = service_code.upper()
        entry = pol["services"].get(key)
    elif ingredient_rxcui:
        key = ingredient_rxcui
        entry = pol["drugs"].get(key)
    if entry is None:
        return {"item_key": key, "recognized": False, "message": "Item not on the plan's PA list or formulary - needs manual lookup"}
    return {"item_key": key, "recognized": True, "pa_required": entry["pa_required"], "policy_id": entry["policy_id"],
            "name": entry["name"], "category": entry["category"], "md_review_required": entry.get("md_review_required", False)}


def _policy_by_id(policy_id: str) -> dict | None:
    pol = _policy()
    for group in ("services", "drugs"):
        for e in pol[group].values():
            if e["policy_id"] == policy_id:
                return e
    return None


def get_medical_policy(policy_id: str) -> dict:
    e = _policy_by_id(policy_id)
    if e is None:
        return {"policy_id": policy_id, "found": False}
    return {"policy_id": policy_id, "found": True, "name": e["name"], "md_review_required": e.get("md_review_required", False),
            "criteria": [{"id": c["id"], "text": c.get("text", "")} for c in e.get("criteria", [])],
            "version": _policy()["_meta"]["version"]}


def evaluate_policy_criteria(policy_id: str, dx_codes: list[str], facts: dict, prescriber_taxonomy: str | None = None) -> dict:
    e = _policy_by_id(policy_id)
    if e is None:
        return {"policy_id": policy_id, "found": False, "results": []}
    res = evaluate(e, dx_codes, facts, prescriber_taxonomy)
    return {"policy_id": policy_id, "found": True, "results": [r.model_dump() for r in res]}


def check_duplicate_submission(request_id: str, member_id: str, item_key: str, received_at: str) -> dict:
    dup = AuditStore(get_settings().audit_db).find_duplicate(request_id, member_id, item_key, received_at, window_days=30)
    return {"is_duplicate": dup is not None, "duplicate_of": dup}


ToolFn = Callable[..., dict]
TOOLS: dict[str, tuple[ToolFn, type[BaseModel], str]] = {
    "verify_provider": (verify_provider, VerifyProviderArgs,
                        "Verify the requesting provider in the CMS NPI Registry (checksum, active status, specialty)."),
    "validate_diagnosis": (validate_diagnosis, ValidateDiagnosisArgs, "Validate one ICD-10-CM code against the NLM ICD-10-CM table."),
    "normalize_drug": (normalize_drug, NormalizeDrugArgs, "Normalize a drug name to its RxNorm ingredient (RXCUI)."),
    "get_drug_label": (get_drug_label, DrugLabelArgs, "Fetch FDA label indications (openFDA) and check each diagnosis is on-label."),
    "check_pa_requirement": (check_pa_requirement, PARequirementArgs,
                             "Check whether the plan requires prior authorization for a service code or drug ingredient."),
    "get_medical_policy": (get_medical_policy, PolicyArgs, "Retrieve the plan's medical-policy criteria for a policy id."),
    "evaluate_policy_criteria": (evaluate_policy_criteria, EvaluateArgs,
                                 "Deterministically evaluate the policy criteria against the documented facts."),
    "check_duplicate_submission": (check_duplicate_submission, DuplicateArgs,
                                   "Check whether the same member/item was already submitted in the last 30 days."),
}
ALLOWED_TOOLS = frozenset(TOOLS)


def run_tool(name: str, args: dict) -> dict:
    """Validate arguments against the tool's schema, then execute. Raises on unknown tool / bad args."""
    if name not in TOOLS:
        raise PermissionError(f"tool '{name}' is not on the approved list")
    fn, schema, _ = TOOLS[name]
    parsed = schema.model_validate(args)
    return fn(**parsed.model_dump())


def openai_tool_specs(names: list[str] | None = None) -> list[dict]:
    """Tool definitions in the function-calling format accepted by LangChain bind_tools()."""
    specs = []
    for n, (_, schema, desc) in TOOLS.items():
        if names and n not in names:
            continue
        specs.append({"type": "function", "function": {"name": n, "description": desc, "parameters": schema.model_json_schema()}})
    return specs
