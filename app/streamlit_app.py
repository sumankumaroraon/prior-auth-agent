"""Reviewer console: work queue for cases the agent escalated, plus metrics and the audit trail.

    streamlit run app/streamlit_app.py
(Run scripts/run_batch.py first so there are cases to review.)
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_agent.audit import AuditStore  # noqa: E402
from pa_agent.config import OUTPUTS, get_settings  # noqa: E402
from pa_agent.runner import get_paused_state, resume_case  # noqa: E402
from pa_agent.schemas import ReviewerDecision  # noqa: E402

st.set_page_config(page_title="Prior Auth Agent - Reviewer Console", layout="wide")
S = get_settings()
audit = AuditStore(S.audit_db)


def load_metrics(name: str) -> dict | None:
    p = OUTPUTS / name
    return json.loads(p.read_text()) if p.exists() else None


cases = pd.DataFrame(audit.cases())
st.title("Prior Authorization Agent: Reviewer Console")
st.caption("Synthetic requests · real CMS NPI, NLM ICD-10/RxNorm and openFDA reference data · "
           "the agent can approve or pend, never deny")

if cases.empty:
    st.warning("No cases yet. Run `python scripts/run_batch.py` first.")
    st.stop()

tab_q, tab_over, tab_audit, tab_eval = st.tabs(["Review queue", "Overview", "Audit trail", "Evaluation"])

# ------------------------------------------------------------------------------- review queue
with tab_q:
    pend = cases[cases.status == "PENDING_REVIEW"].copy()
    if pend.empty:
        st.success("Queue is empty: every escalated case has a decision.")
    else:
        pend["priority"] = (pend.urgency != "expedited").astype(int)
        pend = pend.sort_values(["priority", "sla_due_at"])
        as_of = datetime.fromisoformat(cases.received_at.max())
        pend["hours_left"] = pend.sla_due_at.apply(lambda d: round((datetime.fromisoformat(d) - as_of).total_seconds() / 3600, 1))
        pend["flag_list"] = pend["flags"].apply(lambda f: ", ".join(json.loads(f or "[]")))
        st.markdown(f"**{len(pend)} cases waiting** · expedited first, then by SLA due time · "
                    f"clock as of last intake {as_of:%b %d %H:%M}")
        st.dataframe(pend[["request_id", "urgency", "request_type", "item_name", "flag_list", "sla_due_at", "hours_left"]],
                     hide_index=True, width="stretch")
        rid = st.selectbox("Open case", pend.request_id.tolist())
        snap = asyncio.run(get_paused_state(rid, S))
        if snap and snap["interrupts"]:
            pkt, vals = snap["interrupts"][0], snap["values"]
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader(f"{rid} · {pkt['urgency'].upper()} · due {pkt['sla_due_at']}")
                st.info(pkt.get("summary") or "")
                st.markdown("**Why the agent escalated**")
                for r in pkt["reasons"]:
                    st.markdown(f"- {r}")
                if pkt["criteria"]:
                    st.markdown("**Policy criteria (deterministic engine)**")
                    st.dataframe(pd.DataFrame(pkt["criteria"])[["status", "text", "detail"]], hide_index=True,
                                 width="stretch")
                with st.expander("Original submission"):
                    st.code(vals["text"], language=None)
                with st.expander("Evidence gathered by tools"):
                    st.json(vals.get("evidence", {}), expanded=False)
            with c2:
                st.markdown("**Reviewer decision**")
                with st.form(f"decide-{rid}"):
                    who = st.text_input("Reviewer ID", value="rn.reviewer")
                    dec = st.radio("Decision", ["APPROVE", "DENY", "PEND_INFO"], horizontal=True)
                    note = st.text_area("Clinical rationale (required to deny)")
                    if st.form_submit_button("Submit decision", type="primary"):
                        try:
                            final = asyncio.run(resume_case(rid, ReviewerDecision(decision=dec, reviewer=who, note=note), S))
                            st.success(f"{rid}: {final['status']} by {final['decided_by']}")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

# ------------------------------------------------------------------------------- overview
with tab_over:
    n = len(cases)
    by_status = cases.status.value_counts()
    k = st.columns(5)
    k[0].metric("Requests processed", n)
    k[1].metric("Auto-approved", int((cases.route == "AUTO_APPROVE").sum()))
    k[2].metric("Pended for info (RFI)", int((cases.route == "PEND_INFO").sum()))
    k[3].metric("Escalated to clinician", int((cases.route == "HUMAN_REVIEW").sum()))
    k[4].metric("Median agent time / request", f"{cases.processing_ms.median():.0f} ms")
    c1, c2 = st.columns(2)
    c1.markdown("**Route by request type**")
    c1.bar_chart(cases.pivot_table(index="request_type", columns="route", values="request_id", aggfunc="count").fillna(0))
    c2.markdown("**Current status**")
    c2.bar_chart(by_status)
    st.dataframe(cases[["request_id", "received_at", "channel", "request_type", "urgency", "item_name", "route", "status",
                        "sla_due_at", "decided_by", "processing_ms"]], hide_index=True, width="stretch")

# ------------------------------------------------------------------------------- audit trail
with tab_audit:
    rid = st.selectbox("Request", cases.request_id.tolist(), key="audit_rid")
    ev = pd.DataFrame(audit.events(rid))
    if not ev.empty:
        ev["detail"] = ev.payload.apply(lambda p: (lambda d: d.get("tool") or ", ".join(list(d)[:4]) if isinstance(d, dict) else "")(json.loads(p)))
        ev["source"] = ev.payload.apply(lambda p: (json.loads(p) or {}).get("source") if p.startswith("{") else None)
        st.dataframe(ev[["ts", "node", "event_type", "detail", "source", "duration_ms"]], hide_index=True, width="stretch")
        with st.expander("Raw events (JSON)"):
            st.json([{**r, "payload": json.loads(r["payload"])} for r in audit.events(rid)], expanded=False)

# ------------------------------------------------------------------------------- evaluation
with tab_eval:
    dev, ho = load_metrics("metrics.json"), load_metrics("metrics_holdout.json")
    cols = st.columns(2)
    for col, m, title in [(cols[0], dev, "Development set (240 requests)"), (cols[1], ho, "Holdout set (unseen formats)")]:
        with col:
            st.subheader(title)
            if not m:
                st.write("not run yet")
                continue
            st.metric("Routing accuracy", f"{m['routing_accuracy_pct']}%")
            st.metric("Unsafe auto-approvals", m["unsafe_auto_approvals"])
            st.metric("Field extraction accuracy", f"{m['field_extraction_overall_pct']}%")
            st.markdown("Confusion matrix (rows = expected, columns = agent)")
            st.dataframe(pd.DataFrame(m["confusion_matrix_expected_x_predicted"]).T)
            st.json({"tool_calls": m["tool_calls"], "latency_ms": m["latency_ms"], "run": m["run"]}, expanded=False)
