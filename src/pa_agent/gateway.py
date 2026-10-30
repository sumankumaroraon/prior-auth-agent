"""Tool gateway: the single choke point between the agent and outside systems.

Every tool call, whether the LLM chose it or the deterministic planner did, goes through
here and gets:
  1. allowlist check        - tools not on the approved list are refused and logged
  2. argument validation    - Pydantic schema per tool; malformed calls never execute
  3. timeout + retries       - exponential backoff on transient errors
  4. execution logging       - tool, args, attempts, latency, data source (live/cache)
The tools run in a separate MCP server process (stdio). An in-process mode exists for
unit tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any

from pydantic import ValidationError

from .audit import AuditStore, member_hash
from .config import ROOT, Settings
from .tools import ALLOWED_TOOLS, TOOLS, run_tool


class TransientToolError(Exception):
    pass


class ToolGateway:
    def __init__(self, settings: Settings, audit: AuditStore):
        self.s, self.audit = settings, audit
        self._stack: AsyncExitStack | None = None
        self.session = None
        self.server_tools: list[str] = []
        self.stats = {"calls": 0, "retries": 0, "failures": 0, "rejected": 0, "injected_faults": 0, "live": 0, "cache": 0}

    async def __aenter__(self) -> "ToolGateway":
        if self.s.tool_transport == "mcp":
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["PA_AUDIT_DB"] = str(self.s.audit_db)
            env["PA_DATA_MODE"] = self.s.data_mode
            params = StdioServerParameters(command=sys.executable, args=["-m", "pa_agent.mcp_server"], env=env)
            self._stack = AsyncExitStack()
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            listed = await self.session.list_tools()
            self.server_tools = [t.name for t in listed.tools]
            unexpected = set(self.server_tools) - ALLOWED_TOOLS
            if unexpected:  # a server advertising tools we never approved is a configuration error
                raise RuntimeError(f"MCP server exposes unapproved tools: {unexpected}")
        else:
            self.server_tools = sorted(ALLOWED_TOOLS)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack:
            await self._stack.aclose()

    # ------------------------------------------------------------------
    def _inject_fault(self, request_id: str, name: str, args: dict, attempt: int) -> bool:
        if self.s.fault_rate <= 0 or attempt > 1:
            return False
        h = hashlib.md5(f"{request_id}:{name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF < self.s.fault_rate

    async def _execute(self, name: str, args: dict) -> dict:
        if self.session is None:
            return await asyncio.to_thread(run_tool, name, args)
        res = await asyncio.wait_for(self.session.call_tool(name, args), timeout=self.s.http_timeout_s * 4)
        if res.isError:
            raise TransientToolError(res.content[0].text if res.content else "tool error")
        if getattr(res, "structuredContent", None):
            sc = res.structuredContent
            return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
        return json.loads(res.content[0].text)

    async def call(self, request_id: str, name: str, args: dict, origin: str = "planner") -> dict:
        self.stats["calls"] += 1
        if name not in ALLOWED_TOOLS:
            self.stats["rejected"] += 1
            self.audit.log(request_id, "tool_gateway", "tool_rejected", {"tool": name, "args": args, "reason": "not on allowlist", "origin": origin})
            return {"error": f"tool '{name}' is not approved", "rejected": True}
        try:
            args = TOOLS[name][1].model_validate(args).model_dump(exclude_none=True)
        except ValidationError as e:
            self.stats["rejected"] += 1
            self.audit.log(request_id, "tool_gateway", "tool_rejected",
                           {"tool": name, "args": args, "reason": "schema validation failed", "errors": e.errors(include_url=False), "origin": origin})
            return {"error": "invalid arguments", "rejected": True}

        logged_args = dict(args, member_id=member_hash(args["member_id"])) if "member_id" in args else args
        t0 = time.perf_counter()
        last_err = ""
        for attempt in range(1, self.s.tool_max_attempts + 1):
            try:
                if self._inject_fault(request_id, name, args, attempt):
                    self.stats["injected_faults"] += 1
                    raise TransientToolError("injected fault: simulated 503 Service Unavailable")
                out = await self._execute(name, args)
                ms = (time.perf_counter() - t0) * 1000
                src = out.get("source") if isinstance(out, dict) else None
                if src in ("live", "cache"):
                    self.stats[src] += 1
                self.audit.log(request_id, "tool_gateway", "tool_call",
                               {"tool": name, "args": logged_args, "attempts": attempt, "origin": origin, "source": src,
                                "result": out}, ms)
                return out
            except Exception as e:  # transient: retry with backoff
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.s.tool_max_attempts:
                    self.stats["retries"] += 1
                    self.audit.log(request_id, "tool_gateway", "tool_retry", {"tool": name, "attempt": attempt, "error": last_err})
                    await asyncio.sleep(0.05 * 2 ** (attempt - 1))
        self.stats["failures"] += 1
        self.audit.log(request_id, "tool_gateway", "tool_failed", {"tool": name, "args": logged_args, "error": last_err},
                       (time.perf_counter() - t0) * 1000)
        return {"error": last_err, "tool_failed": True}
