import asyncio
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pa_agent.audit import AuditStore
from pa_agent.config import get_settings
from pa_agent.gateway import ToolGateway
from pa_agent.graph import build_graph
from pa_agent.runner import process_batch, resume_case
from pa_agent.schemas import ReviewerDecision

CLEAN = ("PROVIDER PORTAL - PRIOR AUTHORIZATION REQUEST\nReview type: Standard\n"
         "Member: Ava Kim  DOB: 01/02/1970  Member ID: DHP11112222\nRequesting provider: Joseph Abboud, M.D.  NPI: 1588608731\n"
         "Requested: CPT/HCPCS 73721 - MRI knee\nDiagnosis: M17.11 Unilateral primary osteoarthritis, right knee\n"
         "Clinical notes: Completed 9 weeks of physical therapy and NSAIDs without adequate relief.")
NOT_MET = CLEAN.replace("9 weeks", "2 weeks").replace("DHP11112222", "DHP33334444")
EXPEDITED_MISSING = CLEAN.replace("Standard", "EXPEDITED").replace(" Completed 9 weeks of physical therapy and NSAIDs without adequate relief.", "").replace("DHP11112222", "DHP55556666")


def reqs(*texts):
    return [{"request_id": f"T-{i}", "received_at": f"2026-10-0{i + 1}T09:00", "channel": "portal", "text": t}
            for i, t in enumerate(texts)]


def test_batch_routes_and_human_in_the_loop():
    out = asyncio.run(process_batch(reqs(CLEAN, NOT_MET, EXPEDITED_MISSING, CLEAN), progress=False))
    routes = [r["route"] for r in out["results"]]
    # clean -> approve; criteria not met -> human; expedited + missing docs -> human; resubmission -> duplicate -> human
    assert routes == ["AUTO_APPROVE", "HUMAN_REVIEW", "HUMAN_REVIEW", "HUMAN_REVIEW"]
    assert "possible_duplicate" in out["results"][3]["flags"]
    audit = AuditStore(get_settings().audit_db)
    assert audit.case("T-1")["status"] == "PENDING_REVIEW"
    assert audit.case("T-0")["sla_due_at"] == "2026-10-08T09:00"     # standard = 7 days
    assert audit.case("T-2")["sla_due_at"] == "2026-10-06T09:00"     # expedited = 72 hours

    # A denial without clinical rationale is refused; with rationale it is recorded against the reviewer.
    with pytest.raises(Exception):
        asyncio.run(resume_case("T-1", ReviewerDecision(decision="DENY", reviewer="rn.lee", note="")))
    final = asyncio.run(resume_case("T-1", ReviewerDecision(decision="DENY", reviewer="rn.lee",
                                                            note="Only 2 weeks of PT; policy IMG-101 requires 6.")))
    assert final["status"] == "DENIED" and final["decided_by"] == "human:rn.lee"
    assert audit.case("T-1")["status"] == "DENIED"
    kinds = {e["event_type"] for e in audit.events("T-1")}
    assert {"node_completed", "tool_call", "reviewer_decision"} <= kinds


class ScriptedToolModel(BaseChatModel):
    """Fake LLM: first turn asks for one approved and one unapproved tool, then says DONE."""
    turns: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kw: Any):
        return self

    def with_structured_output(self, schema: Any, **kw: Any):
        raise NotImplementedError("structured output not supported by fake")  # exercises the fallback path

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        self.turns += 1
        if self.turns == 1:
            msg = AIMessage(content="", tool_calls=[
                {"name": "check_pa_requirement", "args": {"service_code": "73721"}, "id": "c1"},
                {"name": "delete_member_records", "args": {"member_id": "x"}, "id": "c2"},
                {"name": "verify_provider", "args": {"npi": 12}, "id": "c3"}])
        else:
            msg = AIMessage(content="DONE")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_llm_tool_calls_are_policed():
    s = get_settings()
    audit = AuditStore(s.audit_db)

    async def go():
        async with ToolGateway(s, audit) as gw:
            g = build_graph(s, gw, audit, model=ScriptedToolModel())
            st = await g.ainvoke({"request_id": "L-1", "received_at": "2026-10-20T09:00", "channel": "portal",
                                  "text": CLEAN.replace("DHP11112222", "DHP99990000")})
            return st, gw.stats
    st, stats = asyncio.run(go())
    ev = audit.events("L-1")
    rejected = [e for e in ev if e["event_type"] == "tool_rejected"]
    assert len(rejected) == 2                                   # unapproved tool + malformed args never ran
    origins = {t["origin"] for t in st["tool_trace"]}
    assert {"llm", "guardrail"} <= origins                     # guardrail completed the required evidence
    assert any(e["event_type"] == "llm_fallback" for e in ev)  # structured-output failure fell back to rules
    assert st["route"]["route"] == "AUTO_APPROVE"


def test_fault_injection_recovers():
    from dataclasses import replace
    s = replace(get_settings(), fault_rate=1.0)  # every first attempt fails
    audit = AuditStore(s.audit_db)

    async def go():
        async with ToolGateway(s, audit) as gw:
            r = await gw.call("F-1", "validate_diagnosis", {"icd10_code": "M17.11"})
            return r, gw.stats
    r, stats = asyncio.run(go())
    assert r["valid"] is True and stats["retries"] == 1 and stats["failures"] == 0
