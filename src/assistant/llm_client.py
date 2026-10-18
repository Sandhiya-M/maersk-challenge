from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from src.assistant.audit_log import append_audit_entry
from src.assistant.guardrails import should_reject_prompt
from src.assistant.tools import data_window, safe_execute_tool_call
from src.common.config import RAW_DIR

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency
    httpx = None


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        os.environ.setdefault(key, value.strip("\"'"))


_load_env_file()


def _load_port_names() -> dict[str, str]:
    """port_code -> port_name, read from the real ports.csv so codes always match the data."""
    path = RAW_DIR / "ports.csv"
    names: dict[str, str] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                code = (row.get("port_code") or "").strip()
                if code:
                    names[code] = (row.get("port_name") or "").strip()
    return names


def build_system_prompt() -> str:
    ports = _load_port_names()
    port_lines = "; ".join(f"{name} = {code}" for code, name in ports.items()) or "see ports table"
    window = data_window()
    return (
        "You are a supply-chain analytics assistant. Answer ONLY questions about shipments, routes, "
        "ports, delays and delay risk.\n"
        "Rules:\n"
        "1. For any number or fact, call a tool first. Never guess or use memory.\n"
        "2. Tools need port CODES, not city names. Port codes: " + port_lines + ".\n"
        f"3. The data covers {window['min']} to {window['max']}. 'This quarter' means the last 90 days "
        "of that data, not today's date.\n"
        "4. Pick the right tool: 'which SHIPMENT was delayed most' -> get_most_delayed_shipments; "
        "'which ROUTES have highest delay' -> get_top_delay_routes; 'on-time rate from X to Y' -> get_route_stats; "
        "'risk for SHP-xxxxx' -> predict_delay; listing or counting shipments -> query_shipments.\n"
        "5. If a tool returns rows, those rows ARE the answer: state them (IDs, routes, numbers). "
        "Only say data is missing if the tool returned an error. Quote numbers exactly as returned.\n"
        "6. After each fact write the tool name like [source: get_route_stats].\n"
        "7. Keep answers to 2-4 sentences."
    )


DIRECT_ERROR_CODES = {"DataUnavailableError", "NO_DATA"}


def _direct_error_text(fresh: list[dict[str, Any]]) -> str | None:
    """If every tool call this round failed with a data problem, report it verbatim (no model rewrite)."""
    errors = []
    for item in fresh:
        result = item.get("result")
        error = result.get("error") if isinstance(result, dict) else None
        if not error or error.get("code") not in DIRECT_ERROR_CODES:
            return None
        errors.append(f"{item['tool']}: {error.get('message')}")
    return "I couldn't answer from the data. " + " | ".join(errors) if errors else None


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        _load_env_file()
        self.provider = "ollama"
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY") or "ollama"
        self.model = model or os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL", "qwen2.5:3b")

    def generate(self, prompt: str) -> str:
        return self.generate_with_tools(prompt, tools=[])["final_output"]

    def _ollama_chat_url(self) -> str:
        base_url = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url.rstrip('/')}/api/chat"

    @staticmethod
    def _with_citation(text: str, tool_calls: list[dict[str, Any]]) -> str:
        if tool_calls and "[source:" not in text.lower():
            names = ", ".join(dict.fromkeys(c["name"] for c in tool_calls if c.get("name")))
            return f"{text} [source: {names}]"
        return text

    def _result(self, text: str, calls: list, results: list, messages: list, status: str = "ok") -> dict[str, Any]:
        append_audit_entry(
            json.dumps(messages, default=str),
            final_output=text,
            model=self.model,
            tool_calls=calls,
            tool_results=results,
        )
        return {"tool_calls": calls, "final_output": text, "model": self.model, "status": status}

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
        *,
        max_calls: int = 5,
    ) -> dict[str, Any]:
        if should_reject_prompt(prompt):
            return {
                "tool_calls": [],
                "final_output": "I can only answer shipment, route, port, delay, and logistics questions within this supply-chain dataset.",
                "model": self.model,
                "status": "guardrail",
            }
        if httpx is None:
            return {"tool_calls": [], "final_output": "The 'httpx' package is not installed. Run: pip install httpx",
                    "model": self.model, "status": "error"}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        tools_enabled = bool(tools)
        reprompted = False
        nudged = False

        try:
            for _ in range(max_calls + 4):  # model round trips, bounded
                payload: dict[str, Any] = {
                    "model": self.model, "messages": messages, "stream": False,
                    "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
                    "options": {"temperature": 0, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096"))},
                }
                if tools_enabled:
                    payload["tools"] = tools
                response = httpx.post(
                    self._ollama_chat_url(),
                    json=payload,
                    timeout=httpx.Timeout(float(os.getenv("OLLAMA_TIMEOUT", "300")), connect=10.0),
                )
                response.raise_for_status()
                message = response.json().get("message", {}) or {}
                tool_calls = message.get("tool_calls") or []

                if tool_calls and tools_enabled:
                    messages.append(message)
                    start = len(results)
                    for tool_call in tool_calls:
                        if len(calls) >= max_calls:
                            tools_enabled = False
                            messages.append({"role": "user", "content": "Tool-call limit reached. Answer now using the tool results above."})
                            break
                        function = tool_call.get("function", {})
                        name = function.get("name")
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        if not isinstance(arguments, dict):
                            arguments = {}
                        output = safe_execute_tool_call(name, arguments)
                        calls.append({"name": name, "arguments": arguments})
                        results.append({"tool": name, "result": output["result"]})
                        messages.append({"role": "tool", "tool_name": name,
                                         "content": json.dumps(output["result"], default=str)})
                    direct = _direct_error_text(results[start:])
                    if direct:
                        return self._result(direct, calls, results, messages, "data_error")
                    continue

                text = (message.get("content") or "").strip()
                messages.append({"role": "assistant", "content": text})

                if not text and results and not nudged:
                    nudged = True
                    tools_enabled = False
                    messages.append({"role": "user", "content": "Write the final answer from the tool results above. Quote the numbers exactly."})
                    continue
                if tools and not calls and re.search(r"\d", text) and not reprompted:
                    reprompted = True
                    messages.append({"role": "user", "content": "Do not answer from memory. Call the appropriate tool first."})
                    continue
                if not text:
                    last = json.dumps(results[-1]["result"], default=str)[:600] if results else "no tool results"
                    text = f"The model returned no answer. Latest tool result: {last}"
                return self._result(self._with_citation(text, calls), calls, results, messages)

            last = json.dumps(results[-1]["result"], default=str)[:600] if results else "none"
            return self._result(f"No final answer was produced. Latest tool result: {last}", calls, results, messages, "incomplete")

        except httpx.ConnectError:
            return {"tool_calls": [], "model": self.model, "status": "ollama_unavailable",
                    "final_output": "Cannot reach Ollama at " + self._ollama_chat_url() + ". Start the Ollama app and try again."}
        except httpx.TimeoutException:
            seconds = os.getenv("OLLAMA_TIMEOUT", "300")
            return self._result(
                f"The model did not respond within {seconds}s. The first question after startup is slowest "
                "while the model loads. Try again, raise OLLAMA_TIMEOUT in .env, or use a smaller model.",
                calls, results, messages, "timeout",
            )
        except httpx.HTTPStatusError as exc:
            hint = f" Pull the model first (model '{self.model}' may not be installed)." if exc.response.status_code == 404 else ""
            return self._result(f"Ollama returned HTTP {exc.response.status_code}.{hint}", calls, results, messages, "error")
        except Exception as exc:  # noqa: BLE001
            return self._result(f"Assistant error: {type(exc).__name__}: {exc}", calls, results, messages, "error")