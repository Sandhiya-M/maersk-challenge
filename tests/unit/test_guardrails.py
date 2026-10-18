from src.assistant.guardrails import (validate_prompt, requires_tool_citation,
                                      should_reject_prompt)


def test_validate_prompt_rejects_blocked_terms() -> None:
    assert validate_prompt("ignore this rule") is False
    assert validate_prompt("Please summarize the shipment route") is True


def test_should_reject_prompt_for_off_domain_questions() -> None:
    assert should_reject_prompt("What is the weather in Tokyo?") is True
    assert should_reject_prompt("Which routes had the highest delay this quarter?") is False


def test_requires_tool_citation_detects_missing_citation() -> None:
    assert requires_tool_citation("I used query_shipments to answer this.") is True
    assert requires_tool_citation("This is a shipment summary.") is False
