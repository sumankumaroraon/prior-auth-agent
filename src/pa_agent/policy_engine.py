"""Deterministic medical-policy engine.

Policy criteria are data (data/policy/medical_policy.json), not prompts. The LLM never
decides whether a criterion is met: this engine does, so the same facts always give the
same answer and every result can be explained line by line.

Each criterion comes back as met / not_met / missing / waived / not_applicable.
  * missing  -> the provider can supply it  -> request for information (PEND_INFO)
  * not_met  -> potential denial            -> licensed clinician must review
"""
from __future__ import annotations

from typing import Any

from .schemas import CriterionResult


def _dx_match(dx_codes: list[str], prefixes: list[str]) -> bool:
    return any(c.upper().startswith(p.upper()) for c in dx_codes for p in prefixes)


def _eval(c: dict, dx: list[str], facts: dict, taxonomy: str | None) -> tuple[str, str]:
    t = c["type"]
    if t == "dx_in":
        return ("met", f"dx {', '.join(dx)} in {c['prefixes']}") if _dx_match(dx, c["prefixes"]) else \
            ("not_met", f"dx {', '.join(dx) or 'none'} not in covered list {c['prefixes']}")
    if t in ("min_value", "max_value"):
        if c.get("waived_if_dx_prefix") and _dx_match(dx, c["waived_if_dx_prefix"]):
            return "waived", "waived: acute-injury diagnosis"
        if c.get("waived_if_fact") and facts.get(c["waived_if_fact"]) is True:
            return "waived", f"waived: {c['waived_if_fact']} documented"
        v = facts.get(c["fact"])
        if v is None:
            return "missing", f"{c['fact']} not documented"
        ok = v >= c["threshold"] if t == "min_value" else v < c["threshold"]
        op = ">=" if t == "min_value" else "<"
        return ("met" if ok else "not_met"), f"{c['fact']}={v} (requires {op} {c['threshold']})"
    if t == "is_true":
        v = facts.get(c["fact"])
        if v is None:
            return "missing", f"{c['fact']} not documented"
        return ("met", f"{c['fact']}=true") if v else ("not_met", f"{c['fact']}=false")
    if t in ("any_of", "all_of"):
        subs = [_eval(o, dx, facts, taxonomy) for o in c["options"]]
        st = [s for s, _ in subs]
        detail = "; ".join(d for _, d in subs)
        if t == "any_of":
            if "met" in st:
                return "met", detail
            return ("missing" if "missing" in st else "not_met"), detail
        if all(s == "met" for s in st):
            return "met", detail
        return ("not_met" if "not_met" in st else "missing"), detail
    if t == "prescriber_specialty_in":
        if not taxonomy:
            return "missing", "prescriber specialty unknown"
        ok = any(s.lower() in taxonomy.lower() for s in c["specialties"])
        return ("met" if ok else "not_met"), f"prescriber taxonomy '{taxonomy}'"
    if t == "step_therapy":
        if c.get("applies_if_dx_prefix") and not _dx_match(dx, c["applies_if_dx_prefix"]):
            return "not_applicable", "step therapy not required for this diagnosis"
        pts = facts.get("prior_therapies")
        if pts is None:
            return "missing", "prior therapy history not documented"
        req = [r.lower() for r in c["required_any"]]
        tried = [p for p in pts if any(r in (p.get("drug") or "").lower() for r in req)]
        if any((p.get("weeks") or 0) >= c["min_weeks"] for p in tried):
            return "met", f"trial of {tried[0]['drug']} for {max(p.get('weeks') or 0 for p in tried)} weeks"
        if tried:
            return "not_met", f"{tried[0]['drug']} trial {max(p.get('weeks') or 0 for p in tried)} weeks < {c['min_weeks']} required"
        return "not_met", f"no trial of {' / '.join(c['required_any'])} documented"
    if t == "step_count":
        pts = facts.get("prior_therapies")
        if pts is None:
            return "missing", "prior preventive therapies not documented"
        n = sum(1 for p in pts if (p.get("weeks") or 0) >= c["min_weeks"])
        return ("met" if n >= c["min_count"] else "not_met"), f"{n} adequate prior trial(s); {c['min_count']} required"
    if t == "benefit_exclusion":
        if _dx_match(dx, c["dx_prefixes"]):
            return "not_met", f"BENEFIT EXCLUSION: {c['text']}"
        return "met", "no excluded use"
    raise ValueError(f"unknown criterion type {t}")


def evaluate(policy: dict, dx_codes: list[str], facts: dict[str, Any], prescriber_taxonomy: str | None) -> list[CriterionResult]:
    out = []
    for c in policy.get("criteria", []):
        status, detail = _eval(c, dx_codes, facts, prescriber_taxonomy)
        out.append(CriterionResult(id=c["id"], text=c.get("text", c["id"]), status=status, detail=detail))
    return out
