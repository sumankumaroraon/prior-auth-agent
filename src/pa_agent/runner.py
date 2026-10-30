"""Run the workflow over a batch of requests, and resume paused cases after human review."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from .audit import AuditStore
from .config import Settings, get_settings
from .gateway import ToolGateway
from .graph import build_graph
from .llm import get_chat_model
from .schemas import ReviewerDecision


def load_requests(path: Path) -> list[dict]:
    reqs = [json.loads(l) for l in open(path) if l.strip()]
    return sorted(reqs, key=lambda r: r["received_at"])


async def process_batch(requests: list[dict], settings: Settings | None = None, reset: bool = True,
                        progress: bool = True) -> dict:
    s = settings or get_settings()
    audit = AuditStore(s.audit_db)
    if reset:
        audit.reset()
        for p in (s.checkpoint_db, Path(str(s.checkpoint_db) + "-wal"), Path(str(s.checkpoint_db) + "-shm")):
            if p.exists():
                p.unlink()
    model = get_chat_model(s)
    results = []
    t_batch = time.perf_counter()
    async with ToolGateway(s, audit) as gw, AsyncSqliteSaver.from_conn_string(str(s.checkpoint_db)) as cp:
        graph = build_graph(s, gw, audit, model=model, checkpointer=cp)
        for i, r in enumerate(requests, 1):
            t0 = time.perf_counter()
            cfg = {"configurable": {"thread_id": r["request_id"]}}
            state = await graph.ainvoke({"request_id": r["request_id"], "received_at": r["received_at"],
                                         "channel": r.get("channel"), "text": r["text"]}, cfg)
            ms = (time.perf_counter() - t0) * 1000
            paused = bool(state.get("__interrupt__"))
            audit.upsert_case(r["request_id"], processing_ms=round(ms, 1),
                              processed_at=__import__("datetime").datetime.now().isoformat(timespec="seconds"))
            results.append({"request_id": r["request_id"], "route": state["route"]["route"], "paused_for_review": paused,
                            "extracted": state["extracted"], "classification": state["classification"],
                            "flags": state["route"]["flags"], "reasons": state["route"]["reasons"], "ms": ms})
            if progress and i % 25 == 0:
                print(f"  processed {i}/{len(requests)}")
        stats = dict(gw.stats, server_tools=gw.server_tools)
    return {"results": results, "gateway": stats, "batch_seconds": time.perf_counter() - t_batch,
            "mode": f"llm:{s.llm_provider}" if model else "rules", "data_mode": s.data_mode,
            "transport": s.tool_transport}


async def get_paused_state(request_id: str, settings: Settings | None = None) -> dict | None:
    s = settings or get_settings()
    async with AsyncSqliteSaver.from_conn_string(str(s.checkpoint_db)) as cp:
        graph = build_graph(s, None, AuditStore(s.audit_db), checkpointer=cp)  # type: ignore[arg-type]
        snap = await graph.aget_state({"configurable": {"thread_id": request_id}})
        if not snap or not snap.values:
            return None
        return {"values": snap.values, "next": list(snap.next),
                "interrupts": [i.value for t in snap.tasks for i in (t.interrupts or [])]}


async def resume_case(request_id: str, decision: ReviewerDecision, settings: Settings | None = None) -> dict:
    """Resume a case paused at human_review with the reviewer's decision."""
    if decision.decision == "DENY" and len(decision.note.strip()) < 10:
        raise ValueError("A denial requires a clinical rationale note (>= 10 characters)")
    s = replace(settings or get_settings(), tool_transport="inprocess")  # no tools are needed after review
    audit = AuditStore(s.audit_db)
    async with ToolGateway(s, audit) as gw, AsyncSqliteSaver.from_conn_string(str(s.checkpoint_db)) as cp:
        graph = build_graph(s, gw, audit, model=None, checkpointer=cp)
        cfg = {"configurable": {"thread_id": request_id}}
        snap = await graph.aget_state(cfg)
        if "human_review" not in snap.next:
            raise ValueError(f"{request_id} is not waiting for review")
        state = await graph.ainvoke(Command(resume=decision.model_dump()), cfg)
        return state["final"]


def run(coro):
    return asyncio.run(coro)
