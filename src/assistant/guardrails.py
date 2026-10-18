from __future__ import annotations

BLOCKED_TERMS = (
    "ignore this rule",
    "jailbreak",
    "bypass",
    "override the system",
    "disregard policy",
    "ignore system",
)
SUPPLY_CHAIN_TERMS = (
    "shipment",
    "shipments",
    "route",
    "routes",
    "port",
    "ports",
    "delay",
    "on-time",
    "on time",
    "booking",
    "cargo",
    "vessel",
    "eta",
    "container",
    "status",
    "logistics",
)


def validate_prompt(prompt: str) -> bool:
    if not isinstance(prompt, str):
        return False
    lowered = prompt.strip().lower()
    if not lowered:
        return False
    return not any(term in lowered for term in BLOCKED_TERMS)


def should_reject_prompt(prompt: str) -> bool:
    if not validate_prompt(prompt):
        return True
    lowered = prompt.strip().lower()
    return not any(term in lowered for term in SUPPLY_CHAIN_TERMS)


def requires_tool_citation(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    citation_markers = (
        "source:",
        "source tool:",
        "query_shipments",
        "get_route_stats",
        "predict_delay",
        "tool=",
    )
    return any(marker in lowered for marker in citation_markers)
