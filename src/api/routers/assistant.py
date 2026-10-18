from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.assistant.llm_client import LLMClient
from src.assistant.tools import TOOL_DEFINITIONS

router = APIRouter(prefix="/assistant", tags=["assistant"])


class AssistantChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_tool_calls: int = Field(default=5, ge=1, le=10)


@router.post("/chat")
def chat_assistant(payload: AssistantChatRequest) -> dict[str, Any]:
    client = LLMClient()
    result = client.generate_with_tools(payload.query, tools=TOOL_DEFINITIONS, max_calls=payload.max_tool_calls)
    return {
        "response": result.get("final_output") or "I couldn’t generate an answer for that request.",
        "tool_calls": result.get("tool_calls", []),
        "model": result.get("model", client.model),
        "status": result.get("status", "ok"),
    }
