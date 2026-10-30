"""LangGraph workflow for prior-authorization intake.

    intake -> validate -> classify -> gather_evidence -> evaluate_policy -> route
        route -> auto_approve | no_pa_required | request_info | human_review(interrupt) -> finalize

Design rules (responsible-AI controls):
* The LLM extracts, classifies and chooses evidence tools. It never decides coverage:
  criteria are evaluated by the deterministic policy engine, and routing is rule-based.
* The agent can approve, pend for information, or close as no-PA-required. It can never
  deny. Every potential denial pauses at a LangGraph interrupt for a human reviewer.
* Required evidence is enforced: if the LLM skips a required tool, a guardrail runs it.
* Every node, tool call and human action is written to the audit log with timing.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from .audit import AuditStore, member_hash, now_iso
from .config import Settings
from .extraction import llm_classify, llm_extract, rule_classify, rule_extract
from .gateway import ToolGateway
from .schemas import Classification, ExtractedRequest, ReviewerDecision, RouteDecision
from .tools import npi_luhn_valid, openai_tool_specs

POLICY_META = None


class PAState(TypedDict, total=False):
    request_id: str
    received_at: str
    channel: str
    text: str
    extracted: dict
    extraction_source: str
    validation: dict
    classification: dict
    sla_due_at: str
    evidence: dict
    tool_trace: list
    criteria: list
    route: dict
    reviewer_summary: str
    review: dict
    final: dict


TOOL_AGENT_PROMPT = """You are the evidence-gathering step of a health plan's prior-authorization workflow.
Use the tools to collect the evidence a reviewer needs. Typical order:
1. If a drug is requested, normalize_drug first; then check_pa_requirement with the ingredient_rxcui (or service_code).
2. If PA is required: verify_provider, validate_diagnosis for every diagnosis code, get_medical_policy,
   and for drugs get_drug_label with the diagnosis codes.
Call tools only with values present in the request or returned by earlier tools. Do not decide the outcome.
When you have the evidence, reply with the single word DONE.

REQUEST (structured):
{request}"""


def _ser(o: Any) -> Any:
    return o.model_dump() if hasattr(o, "model_dump") else o


def build_graph(settings: Settings, gateway: ToolGateway, audit: AuditStore, model=None, checkpointer=None):
    mode = f"llm:{settings.llm_provider}" if model is not None else "rules"

    def logged(name: str) -> Callable:
        def deco(fn: Callable) -> Callable:
            @wraps(fn)
            async def wrapper(state: PAState) -> dict:
                t0 = time.perf_counter()
                out = await fn(state)
                audit.log(state["request_id"], name, "node_completed",
                          {k: v for k, v in out.items() if k not in ("text", "evidence", "tool_trace")},
                          (time.perf_counter() - t0) * 1000)
                return out
            return wrapper
        return deco

    # ------------------------------------------------------------------ intake
    @logged("intake")
    async def intake(state: PAState) -> dict:
        rid = state["request_id"]
        source = "rules"
        if model is not None:
            try:
                x = llm_extract(model, state["text"])
                source = "llm"
            except Exception as e:  # provider error or schema violation -> deterministic fallback
                audit.log(rid, "intake", "llm_fallback", {"error": f"{type(e).__name__}: {e}"[:500]})
                x = rule_extract(state["text"])
        else:
            x = rule_extract(state["text"])
        audit.upsert_case(rid, thread_id=rid, received_at=state["received_at"], channel=state.get("channel"),
                          member_hash=member_hash(x.member_id), status="IN_PROGRESS", mode=mode)
        return {"extracted": x.model_dump(), "extraction_source": source}

    # ------------------------------------------------------------------ validate
    @logged("validate")
    async def validate(state: PAState) -> dict:
        x = ExtractedRequest.model_validate(state["extracted"])
        defects, checks = [], {}
        required = {"member_id": x.member_id, "requesting_npi": x.requesting_npi, "dx_codes": x.dx_codes,
                    "service_or_drug": x.service_code or x.drug_name}
        missing = [k for k, v in required.items() if not v]
        for k in missing:
            defects.append(f"missing required field: {k}")
        if x.requesting_npi:
            checks["npi_checksum_valid"] = npi_luhn_valid(x.requesting_npi)
            if not checks["npi_checksum_valid"]:
                defects.append(f"NPI {x.requesting_npi} fails the CMS check-digit test")
        if x.dob:
            try:
                d = datetime.strptime(x.dob, "%m/%d/%Y")
                checks["dob_valid"] = d < datetime.now()
            except ValueError:
                checks["dob_valid"] = False
            if not checks["dob_valid"]:
                defects.append("date of birth is not a valid past date")
        conf = round(1 - len(missing) / len(required), 2)
        return {"validation": {"defects": defects, "checks": checks, "extraction_confidence": conf}}

    # ------------------------------------------------------------------ classify
    @logged("classify")
    async def classify(state: PAState) -> dict:
        x = ExtractedRequest.model_validate(state["extracted"])
        c = rule_classify(x)
        if model is not None:
            try:
                llm_c = llm_classify(model, x)
                # safety-first: if either path says expedited, treat as expedited
                urg = "expedited" if "expedited" in (llm_c.urgency, x.urgency) else "standard"
                c = Classification(request_type=llm_c.request_type, urgency=urg, rationale=llm_c.rationale)
            except Exception as e:
                audit.log(state["request_id"], "classify", "llm_fallback", {"error": str(e)[:500]})
        received = datetime.fromisoformat(state["received_at"])
        due = received + (timedelta(hours=72) if c.urgency == "expedited" else timedelta(days=7))
        audit.upsert_case(state["request_id"], request_type=c.request_type, urgency=c.urgency,
                          sla_due_at=due.isoformat(timespec="minutes"))
        return {"classification": c.model_dump(), "sla_due_at": due.isoformat(timespec="minutes")}

    # ------------------------------------------------------------------ gather evidence (tool calling)
    @logged("gather_evidence")
    async def gather_evidence(state: PAState) -> dict:
        rid = state["request_id"]
        x = ExtractedRequest.model_validate(state["extracted"])
        cache: dict[str, dict] = {}
        trace: list[dict] = []

        async def call(name: str, args: dict, origin: str) -> dict:
            key = name + json.dumps(args, sort_keys=True)
            if key not in cache:
                cache[key] = await gateway.call(rid, name, args, origin=origin)
                res = cache[key]
                trace.append({"tool": name, "args": args, "origin": origin, "ok": not res.get("error"),
                              "rejected": bool(res.get("rejected")), "failed": bool(res.get("tool_failed")),
                              "source": res.get("source")})
            return cache[key]

        # 1) LLM chooses tools (tool calling) ------------------------------------------
        if model is not None:
            try:
                from langchain_core.messages import HumanMessage, ToolMessage
                bound = model.bind_tools(openai_tool_specs([
                    "normalize_drug", "check_pa_requirement", "verify_provider", "validate_diagnosis",
                    "get_medical_policy", "get_drug_label"]))
                req = x.model_dump(exclude={"member_name", "dob", "clinical"})  # minimum necessary: no name/DOB to the LLM
                msgs: list = [HumanMessage(TOOL_AGENT_PROMPT.format(request=json.dumps(req)))]
                for _ in range(4):
                    ai = bound.invoke(msgs)
                    msgs.append(ai)
                    if not getattr(ai, "tool_calls", None):
                        break
                    for tc in ai.tool_calls:
                        res = await call(tc["name"], tc.get("args") or {}, origin="llm")
                        msgs.append(ToolMessage(content=json.dumps(res)[:4000], tool_call_id=tc["id"]))
            except Exception as e:
                audit.log(rid, "gather_evidence", "llm_fallback", {"error": f"{type(e).__name__}: {e}"[:500]})

        # 2) Guardrail: required evidence is always collected (cached if the LLM already did it)
        origin = "guardrail" if model is not None else "planner"
        ev: dict[str, Any] = {"drug": None, "pa": None, "provider": None, "dx": [], "policy": None, "label": None, "duplicate": None}
        item_key, brand = x.service_code, None
        if x.drug_name and not x.service_code:
            ev["drug"] = await call("normalize_drug", {"drug_name": x.drug_name}, origin)
            if ev["drug"].get("found"):
                item_key = ev["drug"]["ingredient_rxcui"]
                brand = ev["drug"]["name"] if ev["drug"].get("tty") == "BN" else None
            else:
                item_key = None
        if item_key:
            ev["pa"] = await call("check_pa_requirement",
                                  {"service_code": x.service_code} if x.service_code else {"ingredient_rxcui": item_key}, origin)
        if ev["pa"] and ev["pa"].get("recognized") and ev["pa"].get("pa_required"):
            if x.requesting_npi:
                ev["provider"] = await call("verify_provider", {"npi": x.requesting_npi}, origin)
            ev["dx"] = [await call("validate_diagnosis", {"icd10_code": c}, origin) for c in x.dx_codes]
            ev["policy"] = await call("get_medical_policy", {"policy_id": ev["pa"]["policy_id"]}, origin)
            if ev["drug"] and ev["drug"].get("found"):
                largs = {"ingredient_rxcui": item_key, "ingredient": ev["drug"]["ingredient"], "dx_codes": x.dx_codes}
                if brand:
                    largs["brand"] = brand
                ev["label"] = await call("get_drug_label", largs, origin)
            if x.member_id:
                audit.upsert_case(rid, item_key=item_key, item_name=ev["pa"].get("name"))
                ev["duplicate"] = await call("check_duplicate_submission",
                                             {"request_id": rid, "member_id": x.member_id, "item_key": item_key,
                                              "received_at": state["received_at"]}, origin)
        elif ev["pa"]:
            audit.upsert_case(rid, item_key=item_key, item_name=ev["pa"].get("name"))
        ev["item_key"] = item_key
        return {"evidence": ev, "tool_trace": trace}

    # ------------------------------------------------------------------ evaluate policy
    @logged("evaluate_policy")
    async def evaluate_policy(state: PAState) -> dict:
        ev = state["evidence"]
        if not (ev.get("pa") and ev["pa"].get("recognized") and ev["pa"].get("pa_required")):
            return {"criteria": []}
        x = ExtractedRequest.model_validate(state["extracted"])
        prov = ev.get("provider") or {}
        res = await gateway.call(state["request_id"], "evaluate_policy_criteria", {
            "policy_id": ev["pa"]["policy_id"], "dx_codes": x.dx_codes,
            "facts": x.clinical.model_dump(), "prescriber_taxonomy": prov.get("taxonomy_desc")}, origin="planner")
        if res.get("error"):
            return {"criteria": [{"id": "engine", "text": "policy engine", "status": "error", "detail": res["error"]}]}
        return {"criteria": res["results"]}

    # ------------------------------------------------------------------ route (deterministic)
    @logged("route")
    async def route(state: PAState) -> dict:
        ev, crit = state["evidence"], state.get("criteria", [])
        urg = state["classification"]["urgency"]
        reasons, flags = [], []
        failed = [t["tool"] for t in state.get("tool_trace", []) if t.get("failed")]

        def decide() -> str:
            if ev.get("drug") is not None and not ev["drug"].get("found"):
                reasons.append(f"Drug '{state['extracted'].get('drug_name')}' could not be matched in RxNorm")
                flags.append("unrecognized_item")
                return "HUMAN_REVIEW"
            if not ev.get("pa") or not ev["pa"].get("recognized"):
                reasons.append("Requested item is not on the plan's PA list or formulary")
                flags.append("unrecognized_item")
                return "HUMAN_REVIEW"
            if not ev["pa"]["pa_required"]:
                reasons.append(f"{ev['pa']['name']} does not require prior authorization ({ev['pa']['policy_id']})")
                return "NO_PA_REQUIRED"
            if failed:
                reasons.append(f"Tool failure after retries: {', '.join(sorted(set(failed)))}")
                flags.append("tool_failure")
                return "HUMAN_REVIEW"
            if (ev.get("duplicate") or {}).get("is_duplicate"):
                reasons.append(f"Possible duplicate of {ev['duplicate']['duplicate_of']} (same member and item within 30 days)")
                flags.append("possible_duplicate")
                return "HUMAN_REVIEW"

            # (a) identifier / data defects the submitter can fix
            id_defects = list(state["validation"]["defects"])
            prov = ev.get("provider") or {}
            if prov.get("checksum_valid") and prov.get("found") is False:
                id_defects.append(f"NPI {prov.get('npi')} not found in the NPI Registry")
            for d in ev.get("dx", []):
                if d.get("valid") is False:
                    id_defects.append(f"ICD-10-CM code {d.get('code')} is not a valid billable code")
            if id_defects:
                reasons.extend(id_defects)
                flags.append("data_defect")
                if urg == "expedited":
                    reasons.append("Expedited request cannot wait for a request-for-information cycle")
                    return "HUMAN_REVIEW"
                return "PEND_INFO"

            # (b) clinical exceptions -> only a clinician may decide
            exceptions = []
            if prov.get("found") is None:
                exceptions.append("Provider could not be verified (registry unreachable)")
            if prov.get("found") and prov.get("status") != "Active":
                exceptions.append(f"Provider NPI status is {prov.get('status')}")
            if any(d.get("valid") is None for d in ev.get("dx", [])):
                exceptions.append("Diagnosis code could not be verified")
            if ev["pa"].get("md_review_required"):
                exceptions.append(f"Policy {ev['pa']['policy_id']} requires medical-director review")
                flags.append("md_review_required")
            for chk in (ev.get("label") or {}).get("dx_label_checks", []):
                if not chk["on_label"]:
                    exceptions.append(f"Diagnosis {chk['dx']} is not an FDA-labeled indication (possible off-label use)")
                    flags.append("off_label")
            for c in crit:
                if c["status"] in ("not_met", "error"):
                    exceptions.append(f"Criterion not met - {c['text']}: {c['detail']}")
                    if "BENEFIT EXCLUSION" in c["detail"]:
                        flags.append("benefit_exclusion")
                    elif c["id"] == "prescriber":
                        flags.append("specialty_mismatch")
                    else:
                        flags.append("criteria_not_met")
            if exceptions:
                reasons.extend(exceptions)
                return "HUMAN_REVIEW"

            # (c) missing clinical documentation
            missing = [c for c in crit if c["status"] == "missing"]
            if missing:
                reasons.extend(f"Missing documentation - {c['text']}" for c in missing)
                flags.append("missing_documentation")
                if urg == "expedited":
                    reasons.append("Expedited request cannot wait for a request-for-information cycle")
                    return "HUMAN_REVIEW"
                return "PEND_INFO"

            reasons.append(f"All {len(crit)} criteria of {ev['pa']['policy_id']} met; provider and diagnoses verified")
            return "AUTO_APPROVE"

        r = decide()
        rd = RouteDecision(route=r, reasons=reasons, flags=sorted(set(flags)))
        upd = {"route": r, "reasons": rd.reasons, "flags": rd.flags}
        out: dict = {"route": rd.model_dump()}
        if r == "HUMAN_REVIEW":
            upd["status"] = "PENDING_REVIEW"
            out["reviewer_summary"] = _reviewer_summary(state, rd, model, audit)
        audit.upsert_case(state["request_id"], **upd)
        return out

    # ------------------------------------------------------------------ outcomes
    @logged("auto_approve")
    async def auto_approve(state: PAState) -> dict:
        return {"final": _final(state, "APPROVED", "agent:auto", "Approved: all policy criteria met.")}

    @logged("no_pa_required")
    async def no_pa_required(state: PAState) -> dict:
        return {"final": _final(state, "NO_PA_REQUIRED", "agent:auto", state["route"]["reasons"][0])}

    @logged("request_info")
    async def request_info(state: PAState) -> dict:
        items = "\n".join(f"  - {r}" for r in state["route"]["reasons"])
        letter = (f"Request for additional information - {state['request_id']}\n"
                  f"We need the following to complete review. This is not a denial.\n{items}\n"
                  f"Please reply by the review due date {state['sla_due_at']}.")
        return {"final": _final(state, "PENDED_RFI", "agent:auto", letter)}

    def human_review(state: PAState, config: RunnableConfig) -> dict:
        if sys.version_info < (3, 11):
            # Python 3.10: async graphs don't propagate the run context, which interrupt() reads.
            var_child_runnable_config.set(config)
        packet = {
            "request_id": state["request_id"], "urgency": state["classification"]["urgency"],
            "sla_due_at": state["sla_due_at"], "summary": state.get("reviewer_summary"),
            "reasons": state["route"]["reasons"], "flags": state["route"]["flags"],
            "criteria": state.get("criteria", []),
        }
        raw = interrupt(packet)  # graph pauses here until a reviewer resumes it
        t0 = time.perf_counter()
        try:
            d = ReviewerDecision.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"invalid reviewer decision: {e}") from e
        if d.decision == "DENY" and len(d.note.strip()) < 10:
            raise ValueError("A denial requires a clinical rationale note (>= 10 characters)")
        status = {"APPROVE": "APPROVED", "DENY": "DENIED", "PEND_INFO": "PENDED_RFI"}[d.decision]
        audit.log(state["request_id"], "human_review", "reviewer_decision", d.model_dump(), (time.perf_counter() - t0) * 1000)
        return {"review": d.model_dump(),
                "final": _final(state, status, f"human:{d.reviewer}", d.note or f"Reviewer decision: {d.decision}")}

    @logged("finalize")
    async def finalize(state: PAState) -> dict:
        f = state["final"]
        audit.upsert_case(state["request_id"], status=f["status"], decision=f["status"], decided_by=f["decided_by"],
                          decided_at=f["decided_at"])
        return {}

    def pick(state: PAState) -> str:
        return {"AUTO_APPROVE": "auto_approve", "NO_PA_REQUIRED": "no_pa_required",
                "PEND_INFO": "request_info", "HUMAN_REVIEW": "human_review"}[state["route"]["route"]]

    g = StateGraph(PAState)
    for name, fn in [("intake", intake), ("validate", validate), ("classify", classify),
                     ("gather_evidence", gather_evidence), ("evaluate_policy", evaluate_policy), ("route", route),
                     ("auto_approve", auto_approve), ("no_pa_required", no_pa_required),
                     ("request_info", request_info), ("human_review", human_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "validate")
    g.add_edge("validate", "classify")
    g.add_edge("classify", "gather_evidence")
    g.add_edge("gather_evidence", "evaluate_policy")
    g.add_edge("evaluate_policy", "route")
    g.add_conditional_edges("route", pick, ["auto_approve", "no_pa_required", "request_info", "human_review"])
    for n in ("auto_approve", "no_pa_required", "request_info", "human_review"):
        g.add_edge(n, "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def _final(state: PAState, status: str, by: str, text: str) -> dict:
    return {"status": status, "decided_by": by, "decided_at": now_iso(), "letter": text,
            "sla_due_at": state.get("sla_due_at")}


def _reviewer_summary(state: PAState, rd: RouteDecision, model, audit: AuditStore) -> str:
    x, ev = state["extracted"], state["evidence"]
    prov = ev.get("provider") or {}
    item = (ev.get("pa") or {}).get("name") or x.get("service_code") or x.get("drug_name")
    base = (f"{state['classification']['urgency'].upper()} {state['classification']['request_type']} request for {item}; "
            f"dx {', '.join(x.get('dx_codes') or []) or 'none'}; prescriber {prov.get('name', 'unverified')} "
            f"({prov.get('taxonomy_desc', 'specialty unknown')}). Escalated because: {'; '.join(rd.reasons[:3])}.")
    if model is None:
        return base
    try:
        msg = model.invoke("Rewrite this prior-authorization escalation note for a nurse reviewer in at most 3 sentences. "
                           "Do not add facts or recommend a decision.\n\n" + base)
        return str(msg.content)[:800]
    except Exception as e:
        audit.log(state["request_id"], "route", "llm_fallback", {"error": str(e)[:300]})
        return base
