"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Reference data: "auto" = try the live public API, fall back to the snapshot;
    # "live" = live only (errors surface); "cache" = snapshot only (offline, reproducible).
    data_mode: str = field(default_factory=lambda: os.getenv("PA_DATA_MODE", "auto"))
    http_timeout_s: float = field(default_factory=lambda: float(os.getenv("PA_HTTP_TIMEOUT", "4")))
    # LLM: "none" = deterministic rule-based mode (no API key needed); "openai" or "anthropic".
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "none").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    # Tool transport: "mcp" = tools run in a separate MCP server process over stdio; "inprocess" for unit tests.
    tool_transport: str = field(default_factory=lambda: os.getenv("PA_TOOL_TRANSPORT", "mcp"))
    tool_max_attempts: int = field(default_factory=lambda: int(os.getenv("PA_TOOL_MAX_ATTEMPTS", "3")))
    # Chaos testing: probability that a tool call's first attempt fails with a transient error.
    fault_rate: float = field(default_factory=lambda: float(os.getenv("PA_FAULT_RATE", "0")))
    audit_db: Path = field(default_factory=lambda: Path(os.getenv("PA_AUDIT_DB", str(OUTPUTS / "pa_audit.db"))))
    checkpoint_db: Path = field(default_factory=lambda: Path(os.getenv("PA_CHECKPOINT_DB", str(OUTPUTS / "checkpoints.db"))))
    policy_path: Path = DATA / "policy" / "medical_policy.json"
    reference_dir: Path = DATA / "reference"

    def default_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5"}.get(self.llm_provider, "")


def get_settings() -> Settings:
    return Settings()
