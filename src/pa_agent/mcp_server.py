"""MCP server that exposes the approved prior-auth tools.

Run it on its own (any MCP client, e.g. Claude Desktop or the MCP Inspector, can use it):
    python -m pa_agent.mcp_server
The LangGraph agent starts it as a subprocess and talks to it over stdio, so the agent
process itself never calls an outside API directly.
"""
from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from . import tools as T

mcp = FastMCP("prior-auth-tools", log_level="WARNING")


@mcp.tool()
def verify_provider(npi: str) -> dict:
    """Verify the requesting provider in the CMS NPI Registry (checksum, active status, specialty)."""
    return T.verify_provider(npi)


@mcp.tool()
def validate_diagnosis(icd10_code: str) -> dict:
    """Validate one ICD-10-CM code against the NLM ICD-10-CM table."""
    return T.validate_diagnosis(icd10_code)


@mcp.tool()
def normalize_drug(drug_name: str) -> dict:
    """Normalize a drug name (brand or generic) to its RxNorm ingredient RXCUI."""
    return T.normalize_drug(drug_name)


@mcp.tool()
def get_drug_label(ingredient_rxcui: str, ingredient: str, brand: Optional[str] = None,
                   dx_codes: Optional[list[str]] = None) -> dict:
    """Fetch FDA label indications from openFDA and check whether each diagnosis is on-label."""
    return T.get_drug_label(ingredient_rxcui, ingredient, brand, dx_codes or [])


@mcp.tool()
def check_pa_requirement(service_code: Optional[str] = None, ingredient_rxcui: Optional[str] = None) -> dict:
    """Check whether the plan requires prior authorization for a CPT/HCPCS code or a drug ingredient."""
    return T.check_pa_requirement(service_code, ingredient_rxcui)


@mcp.tool()
def get_medical_policy(policy_id: str) -> dict:
    """Retrieve the plan's medical-policy criteria for a policy id."""
    return T.get_medical_policy(policy_id)


@mcp.tool()
def evaluate_policy_criteria(policy_id: str, dx_codes: list[str], facts: dict[str, Any],
                             prescriber_taxonomy: Optional[str] = None) -> dict:
    """Deterministically evaluate policy criteria against documented clinical facts."""
    return T.evaluate_policy_criteria(policy_id, dx_codes, facts, prescriber_taxonomy)


@mcp.tool()
def check_duplicate_submission(request_id: str, member_id: str, item_key: str, received_at: str) -> dict:
    """Check whether the same member and item were already submitted within 30 days."""
    return T.check_duplicate_submission(request_id, member_id, item_key, received_at)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
