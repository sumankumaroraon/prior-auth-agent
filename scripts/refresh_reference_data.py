"""Re-pull every reference record from the live public APIs and rewrite data/reference/*.json.

Run on a machine with normal internet access:  python scripts/refresh_reference_data.py
It reuses the agent's own API clients in live mode, so it also works as a smoke test of
the live integrations (NPI Registry, NLM ICD-10-CM, RxNorm, openFDA).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_agent.clients import ReferenceData  # noqa: E402
from pa_agent.config import get_settings  # noqa: E402

REF = ROOT / "data/reference"


def main() -> None:
    ref = ReferenceData(replace(get_settings(), data_mode="live"))
    today = date.today().isoformat()
    changed = 0
    for fname, fn in [("npi_registry_snapshot.json", ref.lookup_npi), ("icd10cm_snapshot.json", ref.lookup_icd10),
                      ("rxnorm_snapshot.json", None), ("openfda_labels_snapshot.json", None)]:
        doc = json.load(open(REF / fname))
        for key, old in list(doc["records"].items()):
            if fname.startswith("rxnorm"):
                new = ref.normalize_drug(old["query"])
            elif fname.startswith("openfda"):
                new = ref.drug_label(key, old["ingredient"], old.get("brand"))
                new.setdefault("related_labels", old.get("related_labels"))
                if not new["related_labels"]:
                    new.pop("related_labels")
            else:
                new = fn(key)
            new.pop("source", None)
            if fname.startswith("npi") and old.get("primary_specialty") and new.get("found"):
                new["primary_specialty"] = old["primary_specialty"]  # keep the search label used by the generator
            if new != old:
                changed += 1
                print(f"  {fname}: {key} changed")
            doc["records"][key] = new
        doc["_meta"]["fetched"] = today
        json.dump(doc, open(REF / fname, "w"), indent=1)
        print(f"refreshed {fname} ({len(doc['records'])} records)")
    print(f"done; {changed} records differ from the previous snapshot")


if __name__ == "__main__":
    main()
