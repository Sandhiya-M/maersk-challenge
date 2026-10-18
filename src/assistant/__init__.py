"""LLM assistant support package."""

__all__ = ["run_assistant"]


def run_assistant(*args, **kwargs):
    from src.assistant.cli import run_assistant as _run_assistant

    return _run_assistant(*args, **kwargs)
