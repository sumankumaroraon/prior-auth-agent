"""Clients for the four public healthcare APIs the agent is allowed to use.

* CMS NPPES NPI Registry   - is the requesting provider real, active, and what specialty?
* NLM Clinical Tables      - is the ICD-10-CM diagnosis code valid?
* NLM RxNav (RxNorm)       - normalize a drug name (brand/generic/misspelling) to an ingredient
* openFDA drug label       - what is the drug FDA-labeled for? (on-label check)

Each client tries the live API first (PA_DATA_MODE=auto) and falls back to a dated
snapshot in data/reference/ when the API is unreachable. A simple circuit breaker
stops hammering an API that is down. Every result says where it came from
("live" or "cache") so the audit log shows the provenance of every fact.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings, get_settings

NPI_URL = "https://npiregistry.cms.hhs.gov/api/"
ICD_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
RX_URL = "https://rxnav.nlm.nih.gov/REST"
FDA_URL = "https://api.fda.gov/drug/label.json"

# Condition vocabulary used to turn label text into indication terms.
CONDITION_TERMS = [
    "rheumatoid arthritis", "juvenile idiopathic arthritis", "psoriatic arthritis", "ankylosing spondylitis", "crohn",
    "ulcerative colitis", "plaque psoriasis", "hidradenitis suppurativa", "uveitis", "atopic dermatitis", "asthma",
    "chronic rhinosinusitis with nasal polyps", "eosinophilic esophagitis", "prurigo nodularis",
    "chronic obstructive pulmonary disease", "type 2 diabetes", "obesity", "overweight", "migraine",
    "atrial fibrillation", "deep vein thrombosis", "pulmonary embolism", "multiple sclerosis", "cardiovascular disease",
]


class ApiUnavailable(Exception):
    pass


class _Breaker:
    """Open after `threshold` consecutive failures; stay open for `cooldown_s`."""

    def __init__(self, threshold: int = 2, cooldown_s: float = 120):
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self.failures, self.opened_at = 0, 0.0

    def allow(self) -> bool:
        return self.failures < self.threshold or (time.time() - self.opened_at) > self.cooldown_s

    def ok(self) -> None:
        self.failures = 0

    def fail(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.time()


class ReferenceData:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        ref: Path = self.s.reference_dir
        self._npi = json.load(open(ref / "npi_registry_snapshot.json"))["records"]
        self._icd = json.load(open(ref / "icd10cm_snapshot.json"))["records"]
        self._rx = json.load(open(ref / "rxnorm_snapshot.json"))["records"]
        self._fda = json.load(open(ref / "openfda_labels_snapshot.json"))["records"]
        self._breakers = {k: _Breaker() for k in ("npi", "icd", "rx", "fda")}
        self._http = httpx.Client(timeout=self.s.http_timeout_s, headers={"User-Agent": "pa-agent-portfolio/1.0"})

    # ------------------------------------------------------------------ plumbing
    def _get(self, url: str, params: dict | None = None) -> Any:
        r = self._http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def _with_fallback(self, api: str, live: Callable[[], dict], cached: Callable[[], dict | None]) -> dict:
        mode = self.s.data_mode
        br = self._breakers[api]
        if mode in ("auto", "live") and br.allow():
            try:
                out = live()
                br.ok()
                out["source"] = "live"
                return out
            except Exception as e:  # network error, timeout, 5xx, bad JSON
                br.fail()
                if mode == "live":
                    raise ApiUnavailable(f"{api} API unavailable: {type(e).__name__}") from e
        hit = cached()
        if hit is None:
            return {"source": "cache_miss"}
        return dict(hit, source="cache")

    # ------------------------------------------------------------------ NPI Registry
    def lookup_npi(self, npi: str) -> dict:
        def live() -> dict:
            j = self._get(NPI_URL, {"version": "2.1", "number": npi})
            if not j.get("result_count"):
                return {"npi": npi, "found": False}
            r = j["results"][0]
            b = r.get("basic", {})
            taxes = r.get("taxonomies", [])
            prim = next((t for t in taxes if t.get("primary")), taxes[0] if taxes else {})
            name = b.get("organization_name") or f"{b.get('first_name', '')} {b.get('last_name', '')}".strip()
            addr = (r.get("addresses") or [{}])[0]
            return {"npi": npi, "found": True, "status": "Active" if b.get("status") == "A" else "Inactive",
                    "name": name.title(), "credential": b.get("credential", ""),
                    "taxonomy_desc": prim.get("desc", ""), "primary_specialty": prim.get("desc", ""),
                    "city": addr.get("city"), "state": addr.get("state"), "enumeration_type": r.get("enumeration_type")}

        return self._with_fallback("npi", live, lambda: self._npi.get(npi))

    # ------------------------------------------------------------------ ICD-10-CM
    def lookup_icd10(self, code: str) -> dict:
        code = code.upper().strip()

        def live() -> dict:
            j = self._get(ICD_URL, {"sf": "code", "terms": code, "maxList": 10})
            hit = next((x for x in (j[3] or []) if x[0] == code), None)
            return {"code": code, "valid": hit is not None, "description": hit[1] if hit else None}

        return self._with_fallback("icd", live, lambda: self._icd.get(code))

    # ------------------------------------------------------------------ RxNorm
    def normalize_drug(self, name: str) -> dict:
        q = name.strip()

        def live() -> dict:
            ids = (self._get(f"{RX_URL}/rxcui.json", {"name": q, "search": 2}).get("idGroup") or {}).get("rxnormId") or []
            if not ids:
                approx = self._get(f"{RX_URL}/approximateTerm.json", {"term": q, "maxEntries": 1})
                cands = (approx.get("approximateGroup") or {}).get("candidate") or []
                return {"query": q, "found": False, "suggestion_rxcui": cands[0]["rxcui"] if cands else None}
            cui = ids[0]
            p = self._get(f"{RX_URL}/rxcui/{cui}/properties.json")["properties"]
            ing_cui, ing = cui, p["name"]
            if p["tty"] != "IN":
                rel = self._get(f"{RX_URL}/rxcui/{cui}/related.json", {"tty": "IN"})
                for g in (rel.get("relatedGroup") or {}).get("conceptGroup") or []:
                    if g.get("conceptProperties"):
                        ing_cui, ing = g["conceptProperties"][0]["rxcui"], g["conceptProperties"][0]["name"]
                        break
            return {"query": q, "found": True, "rxcui": cui, "tty": p["tty"], "name": p["name"],
                    "ingredient_rxcui": ing_cui, "ingredient": ing}

        return self._with_fallback("rx", live, lambda: self._rx.get(q.lower()))

    # ------------------------------------------------------------------ openFDA
    def drug_label(self, ingredient_rxcui: str, ingredient: str, brand: str | None = None) -> dict:
        snap = self._fda.get(ingredient_rxcui)

        def cached() -> dict | None:
            if snap is None:
                return None
            if brand and snap.get("related_labels", {}).get(brand.upper()):
                rel = snap["related_labels"][brand.upper()]
                return {"ingredient": ingredient, "brand": brand.upper(), **rel}
            return {k: v for k, v in snap.items() if k != "related_labels"}

        def live() -> dict:
            field = f'openfda.brand_name:"{brand}"' if brand else f'openfda.generic_name:"{ingredient}"'
            j = self._get(FDA_URL, {"search": field, "limit": 1})
            r = j["results"][0]
            text = re.sub(r"\s+", " ", (r.get("indications_and_usage") or [""])[0])
            low = text.lower()
            return {"ingredient": ingredient, "brand": (brand or "").upper() or None, "set_id": r.get("set_id"),
                    "effective_time": r.get("effective_time"), "indications_excerpt": text[:900],
                    "indication_terms": [t for t in CONDITION_TERMS if t in low]}

        return self._with_fallback("fda", live, cached)
