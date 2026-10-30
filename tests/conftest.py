import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_tmp = tempfile.mkdtemp()
os.environ.update({"PA_DATA_MODE": "cache", "PA_TOOL_TRANSPORT": "inprocess", "LLM_PROVIDER": "none",
                   "PA_FAULT_RATE": "0", "PA_AUDIT_DB": f"{_tmp}/audit.db", "PA_CHECKPOINT_DB": f"{_tmp}/cp.db"})
