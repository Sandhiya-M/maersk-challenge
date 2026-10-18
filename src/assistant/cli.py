from __future__ import annotations

import argparse

from src.assistant.guardrails import should_reject_prompt
from src.assistant.llm_client import LLMClient
from src.assistant.tools import TOOL_DEFINITIONS


def run_assistant(query: str, *, max_tool_calls: int = 5) -> str:
    if should_reject_prompt(query):
        return "I can only answer supply-chain questions about shipments, routes, ports, delay risk, and logistics performance."

    client = LLMClient()
    result = client.generate_with_tools(query, tools=TOOL_DEFINITIONS, max_calls=max_tool_calls)
    return result["final_output"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Supply chain assistant CLI")
    parser.add_argument("--query", help="Natural-language question to answer")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive terminal loop")
    parser.add_argument("--max-tool-calls", type=int, default=5, help="Maximum tool calls allowed in a single session")
    args = parser.parse_args()

    if args.interactive:
        while True:
            try:
                question = input("assistant> ")
            except EOFError:
                break
            if not question:
                continue
            print(run_assistant(question, max_tool_calls=args.max_tool_calls))
        return

    if not args.query:
        parser.error("Either --query or --interactive is required.")

    print(run_assistant(args.query, max_tool_calls=args.max_tool_calls))


if __name__ == "__main__":
    main()
