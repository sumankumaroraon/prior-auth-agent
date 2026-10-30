"""Pydantic schemas: the structured outputs every step must produce.

The LLM (or the rule-based fallback) must return these exact shapes. Anything that
fails validation is treated as an exception and routed to a person, never guessed.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

RequestType = Literal["pharmacy", "imaging", "surgical", "dme", "sleep_study", "therapy", "other"]
Urgency = Literal["expedited", "standard"]
Route = Literal["AUTO_APPROVE", "NO_PA_REQUIRED", "PEND_INFO", "HUMAN_REVIEW"]


class PriorTherapy(BaseModel):
    drug: str = Field(description="Medication or therapy tried, lower case, e.g. 'methotrexate' or 'topical corticosteroid'")
    weeks: Optional[int] = Field(default=None, description="Duration of the trial in weeks")


class ClinicalFacts(BaseModel):
    """Only facts stated in the submission. Leave a field null when it is not mentioned."""
    conservative_therapy_weeks: Optional[int] = Field(None, description="Weeks of PT / NSAIDs / conservative therapy")
    bmi: Optional[float] = None
    ahi: Optional[float] = Field(None, description="Apnea-hypopnea index, events per hour")
    osa_symptoms: Optional[bool] = Field(None, description="True if daytime sleepiness / witnessed apneas documented, False if explicitly denied")
    red_flags: Optional[bool] = Field(None, description="True if red-flag neurologic symptoms are documented")
    hsat_inconclusive: Optional[bool] = Field(None, description="True if home sleep test inconclusive; False if explicitly not attempted")
    prior_therapies: Optional[list[PriorTherapy]] = Field(
        None, description="Prior medication trials. Empty list if the note explicitly says none; null if not mentioned")


class ExtractedRequest(BaseModel):
    member_name: Optional[str] = None
    member_id: Optional[str] = None
    dob: Optional[str] = Field(None, description="MM/DD/YYYY")
    requesting_npi: Optional[str] = Field(None, description="10-digit NPI exactly as written")
    service_code: Optional[str] = Field(None, description="CPT or HCPCS code if a procedure/device is requested")
    drug_name: Optional[str] = Field(None, description="Drug name exactly as written (brand or generic) if a medication is requested")
    dx_codes: list[str] = Field(default_factory=list, description="ICD-10-CM codes exactly as written")
    urgency: Urgency = "standard"
    clinical: ClinicalFacts = Field(default_factory=ClinicalFacts)

    @field_validator("dx_codes")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return [c.strip().upper() for c in v if c and c.strip()]


class Classification(BaseModel):
    request_type: RequestType
    urgency: Urgency
    rationale: str = Field(description="One sentence explaining the classification")


class CriterionResult(BaseModel):
    id: str
    text: str
    status: Literal["met", "not_met", "missing", "waived", "not_applicable"]
    detail: str = ""


class RouteDecision(BaseModel):
    route: Route
    reasons: list[str]
    flags: list[str] = Field(default_factory=list)


class ReviewerDecision(BaseModel):
    """What a human reviewer sends back when a paused case is resumed."""
    decision: Literal["APPROVE", "DENY", "PEND_INFO"]
    reviewer: str
    note: str = ""
