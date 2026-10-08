"""A tiny OpenAI-compatible server backed by the simulated model, served over real HTTP on
localhost. Lets tests exercise OpenAICompatibleProvider end to end (requests, status codes,
retries, tool-call wire format) without any external network."""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pma.providers.base import ChatMessage, LLMRequest, ToolCall, ToolSpec
from pma.providers.simulated import SimulatedMarketingLLM


def infer_purpose(messages: list[dict]) -> str:
    """Real OpenAI requests carry no 'purpose'; recover it from the prompt text."""
    text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    first = next((m["content"] for m in messages if m["role"] == "user"), "")
    probe = first if "Task: plan" in first else text
    for marker, purpose in [
        ("Task: plan", "plan"),
        ("Task: revise", "revise"),
        ("Task: write the marketing content for the product below in one pass", "baseline"),
        ("Task: write", "draft"),
        ("strict compliance reviewer", "judge"),
        ("could not be used", "repair"),
    ]:
        if marker in probe:
            return purpose
    return "draft"


@dataclass
class FakeServerState:
    flaw_scripts: dict[str, list[str]] = field(default_factory=dict)
    fail_next: list[int] = field(default_factory=list)  # HTTP statuses to return before succeeding
    delay_next_s: float = 0.0
    requests: list[dict] = field(default_factory=list)


def build_fake_app(state: FakeServerState) -> FastAPI:
    app = FastAPI()
    sim = SimulatedMarketingLLM(state.flaw_scripts, "clean")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        state.requests.append(payload)
        if state.delay_next_s:
            delay, state.delay_next_s = state.delay_next_s, 0.0
            time.sleep(delay)
        if state.fail_next:
            status = state.fail_next.pop(0)
            return JSONResponse({"error": "boom"}, status_code=status, headers={"retry-after": "0"})
        messages = []
        for m in payload["messages"]:
            calls = [
                ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in m.get("tool_calls", [])
            ]
            messages.append(
                ChatMessage(
                    m["role"], m.get("content") or "", calls, m.get("tool_call_id"), m.get("name")
                )
            )
        tools = [
            ToolSpec(
                t["function"]["name"], t["function"]["description"], t["function"]["parameters"]
            )
            for t in payload.get("tools", [])
        ]
        req = LLMRequest(messages=messages, tools=tools, purpose=infer_purpose(payload["messages"]))
        resp = sim(req)
        message: dict = {"role": "assistant", "content": resp.content or None}
        if resp.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in resp.tool_calls
            ]
        return {
            "model": payload["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
        }

    return app


class FakeServer:
    def __init__(self) -> None:
        self.state = FakeServerState()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(
            build_fake_app(self.state), host="127.0.0.1", port=self.port, log_level="error"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> FakeServer:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("fake server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def reset(self) -> None:
        self.state.flaw_scripts.clear()
        self.state.fail_next.clear()
        self.state.delay_next_s = 0.0
        self.state.requests.clear()
