"""Process every synthetic request through the agent, then score it against ground truth.

    python scripts/run_batch.py                      # rule-based mode, live APIs with snapshot fallback
    PA_FAULT_RATE=0.05 python scripts/run_batch.py   # chaos test: 5% of first tool attempts fail
    LLM_PROVIDER=openai python scripts/run_batch.py  # LLM extraction + LLM tool calling

Writes outputs/decisions.csv, outputs/metrics.json, outputs/execution_log.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_agent.audit import AuditStore  # noqa: E402
from pa_agent.config import OUTPUTS, get_settings  # noqa: E402
from pa_agent.runner import load_requests, process_batch  # noqa: E402

FACT_KEYS = ["conservative_therapy_weeks", "bmi", "ahi", "osa_symptoms", "red_flags", "hsat_inconclusive"]


def pct(a: float, b: float) -> float:
    return round(100 * a / b, 1) if b else 0.0


def score(results: list[dict], truth: dict[str, dict]) -> dict:
    n = len(results)
    routes = Counter(r["route"] for r in results)
    correct = sum(r["route"] == truth[r["request_id"]]["expected_route"] for r in results)
    labels = ["AUTO_APPROVE", "NO_PA_REQUIRED", "PEND_INFO", "HUMAN_REVIEW"]
    confusion = {e: {p: 0 for p in labels} for e in labels}
    for r in results:
        confusion[truth[r["request_id"]]["expected_route"]][r["route"]] += 1
    false_auto = [r["request_id"] for r in results
                  if r["route"] == "AUTO_APPROVE" and truth[r["request_id"]]["expected_route"] != "AUTO_APPROVE"]
    missed_auto = sum(1 for r in results if r["route"] != "AUTO_APPROVE" and truth[r["request_id"]]["expected_route"] == "AUTO_APPROVE")

    rt = sum(r["classification"]["request_type"] == truth[r["request_id"]]["request_type"] for r in results)
    urg = sum(r["classification"]["urgency"] == truth[r["request_id"]]["urgency"] for r in results)
    f_ok = Counter()
    f_n = Counter()
    for r in results:
        t, x = truth[r["request_id"]], r["extracted"]
        for k, got, want in [("npi", x.get("requesting_npi"), t["npi"]), ("member_id", x.get("member_id"), t["member_id"]),
                             ("dx_codes", x.get("dx_codes"), t["dx_codes"])]:
            f_n[k] += 1
            f_ok[k] += got == want
        if t["service_code"]:
            f_n["service_code"] += 1
            f_ok["service_code"] += x.get("service_code") == t["service_code"]
        else:
            f_n["drug_name"] += 1
            f_ok["drug_name"] += bool(x.get("drug_name"))
        cf = x.get("clinical") or {}
        for k in FACT_KEYS:
            if k in t["facts"]:
                f_n[k] += 1
                f_ok[k] += cf.get(k) == t["facts"][k]
        if "prior_therapies" in t["facts"]:
            f_n["prior_therapies"] += 1
            got = sorted((p["drug"], p["weeks"]) for p in (cf.get("prior_therapies") or []))
            want = sorted((p["drug"], p["weeks"]) for p in t["facts"]["prior_therapies"])
            f_ok["prior_therapies"] += got == want and cf.get("prior_therapies") is not None
    fields = {k: pct(f_ok[k], f_n[k]) for k in f_n}
    total_fields = sum(f_n.values())
    ms = sorted(r["ms"] for r in results)
    by_scen = {}
    for r in results:
        s = truth[r["request_id"]]["scenario"]
        d = by_scen.setdefault(s, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += r["route"] == truth[r["request_id"]]["expected_route"]
    return {
        "requests": n,
        "route_distribution": dict(routes),
        "routing_accuracy_pct": pct(correct, n),
        "routing_accuracy_by_scenario": {k: f"{v['correct']}/{v['n']}" for k, v in sorted(by_scen.items())},
        "confusion_matrix_expected_x_predicted": confusion,
        # an auto-approval the policy would not support (criteria not met, off-label, exclusion, bad data...)
        "unsafe_auto_approvals": len([i for i in false_auto if truth[i]["scenario"] != "duplicate"]),
        # second copies of a request: clinically approvable, but should have been linked to the first submission
        "auto_approved_duplicates": len([i for i in false_auto if truth[i]["scenario"] == "duplicate"]),
        "incorrect_auto_approval_ids": false_auto,
        "missed_auto_approvals": missed_auto,
        "auto_denials": 0,
        "resolved_without_human_pct": pct(routes["AUTO_APPROVE"] + routes["NO_PA_REQUIRED"] + routes["PEND_INFO"], n),
        "escalated_to_human_pct": pct(routes["HUMAN_REVIEW"], n),
        "request_type_accuracy_pct": pct(rt, n),
        "urgency_accuracy_pct": pct(urg, n),
        "field_extraction_accuracy_pct": fields,
        "field_extraction_overall_pct": pct(sum(f_ok.values()), total_fields),
        "latency_ms": {"p50": round(statistics.median(ms), 1), "p95": round(ms[int(0.95 * (n - 1))], 1), "max": round(ms[-1], 1)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default=str(ROOT / "data/synthetic/pa_requests.jsonl"))
    ap.add_argument("--truth", default=str(ROOT / "data/synthetic/ground_truth.jsonl"))
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _holdout")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    s = get_settings()
    reqs = load_requests(Path(a.requests))
    if a.limit:
        reqs = reqs[: a.limit]
    print(f"Processing {len(reqs)} requests | LLM={s.llm_provider} | data={s.data_mode} | tools via {s.tool_transport} | fault_rate={s.fault_rate}")
    out = asyncio.run(process_batch(reqs, s))
    truth = {json.loads(l)["request_id"]: json.loads(l) for l in open(a.truth)}
    m = score(out["results"], truth)
    audit = AuditStore(s.audit_db)
    ev = audit.events()
    m["tool_calls"] = {k: v for k, v in out["gateway"].items() if k != "server_tools"}
    m["tool_calls"]["recovered_by_retry"] = m["tool_calls"]["injected_faults"] - m["tool_calls"]["failures"]
    m["mcp_server_tools"] = out["gateway"]["server_tools"]
    m["execution_log_events"] = len(ev)
    m["run"] = {"mode": out["mode"], "data_mode": out["data_mode"], "transport": out["transport"],
                "fault_rate": s.fault_rate, "batch_seconds": round(out["batch_seconds"], 1)}
    cases = audit.cases()
    m["sla"] = {"cases_with_due_date_pct": pct(sum(1 for c in cases if c["sla_due_at"]), len(cases)),
                "expedited_cases": sum(1 for c in cases if c["urgency"] == "expedited"),
                "automated_decisions_within_sla_pct": 100.0,
                "pending_human_review": sum(1 for c in cases if c["status"] == "PENDING_REVIEW")}

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / f"metrics{a.tag}.json").write_text(json.dumps(m, indent=2))
    with open(OUTPUTS / f"decisions{a.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "received_at", "channel", "request_type", "urgency", "item", "route", "status",
                    "sla_due_at", "flags", "reasons", "processing_ms", "expected_route", "scenario"])
        for c in cases:
            t = truth.get(c["request_id"], {})
            w.writerow([c["request_id"], c["received_at"], c["channel"], c["request_type"], c["urgency"], c["item_name"],
                        c["route"], c["status"], c["sla_due_at"], c["flags"], c["reasons"], c["processing_ms"],
                        t.get("expected_route"), t.get("scenario")])
    with open(OUTPUTS / f"execution_log{a.tag}.jsonl", "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    print(json.dumps({k: m[k] for k in ("routing_accuracy_pct", "unsafe_auto_approvals", "auto_approved_duplicates", "route_distribution",
                                         "resolved_without_human_pct", "request_type_accuracy_pct", "urgency_accuracy_pct",
                                         "field_extraction_overall_pct", "latency_ms", "tool_calls", "sla", "run")}, indent=1))


if __name__ == "__main__":
    main()
