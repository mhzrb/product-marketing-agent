"""Tool-calling runtime with two interchangeable transports.

* **Native** - the provider advertises ``supports_tools``; tools are passed through its tool
  calling API and results come back as ``tool`` messages.
* **JSON protocol** - for endpoints without tool calling (many local models): the model must
  answer with ``{"action": "tool", ...}`` or ``{"action": "final", "output": ...}``. Every reply
  is validated with Pydantic; an invalid reply gets one repair attempt, then the run errors out.

Both transports end in the same place: the final answer as a JSON string for the caller to
validate against its own schema.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pma.errors import OutputValidationError, ToolProtocolError
from pma.llm_client import LLMClient
from pma.prompts import Prompt
from pma.providers.base import ChatMessage
from pma.tools import ToolRegistry


class ToolProtocolMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "final"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None

    @model_validator(mode="after")
    def _consistent(self) -> ToolProtocolMessage:
        if self.action == "tool" and not self.tool:
            raise ValueError("action 'tool' requires a 'tool' name")
        if self.action == "final" and self.output is None:
            raise ValueError("action 'final' requires an 'output'")
        return self


def tool_list_text(registry: ToolRegistry, allowed: list[str]) -> str:
    lines = []
    for spec in registry.specs(allowed):
        lines.append(f"- {spec.name}: {spec.description} arguments: {json.dumps(spec.parameters)}")
    return "\n".join(lines)


def run_tool_loop(
    client: LLMClient,
    registry: ToolRegistry,
    messages: list[ChatMessage],
    *,
    purpose: str,
    prompt: Prompt,
    allowed: list[str],
    max_rounds: int = 6,
) -> tuple[str, str]:
    """Run until the model gives its final answer. Returns ``(final_text, transport)``."""
    if client.provider.supports_tools:
        return _native(client, registry, messages, purpose, prompt, allowed, max_rounds), "native"
    return _protocol(client, registry, messages, purpose, prompt, max_rounds), "json_protocol"


def _native(
    client: LLMClient,
    registry: ToolRegistry,
    messages: list[ChatMessage],
    purpose: str,
    prompt: Prompt,
    allowed: list[str],
    max_rounds: int,
) -> str:
    specs = registry.specs(allowed)
    for _ in range(max_rounds):
        response = client.call(purpose, messages, prompt=prompt, tools=specs)
        if not response.tool_calls:
            return response.content
        messages.append(
            ChatMessage("assistant", response.content, tool_calls=list(response.tool_calls))
        )
        for call in response.tool_calls:
            execution = registry.execute(call.name, call.arguments, invoked_by="model")
            client.trace.add(
                "tool_call",
                tool=call.name,
                arguments=call.arguments,
                ok=execution.ok,
                transport="native",
                result=client.trace.io(execution.to_message_text()),
            )
            messages.append(
                ChatMessage(
                    "tool", execution.to_message_text(), tool_call_id=call.id, name=call.name
                )
            )
    raise ToolProtocolError(f"model did not finish within {max_rounds} tool rounds")


def _protocol(
    client: LLMClient,
    registry: ToolRegistry,
    messages: list[ChatMessage],
    purpose: str,
    prompt: Prompt,
    max_rounds: int,
) -> str:
    for _ in range(max_rounds):
        response = client.call(purpose, messages, prompt=prompt, json_mode=True)
        try:
            msg, _ = client.validate(response.content, ToolProtocolMessage, purpose=purpose)
        except OutputValidationError as exc:
            raise ToolProtocolError(f"invalid tool protocol message: {exc}") from exc
        if msg.action == "final":
            output = msg.output
            return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        assert msg.tool is not None
        execution = registry.execute(msg.tool, msg.arguments, invoked_by="model")
        client.trace.add(
            "tool_call",
            tool=msg.tool,
            arguments=msg.arguments,
            ok=execution.ok,
            transport="json_protocol",
            result=client.trace.io(execution.to_message_text()),
        )
        messages.append(ChatMessage("assistant", response.content))
        messages.append(
            ChatMessage(
                "user",
                f'<tool_result name="{msg.tool}">{execution.to_message_text()}</tool_result>',
            )
        )
    raise ToolProtocolError(f"model did not finish within {max_rounds} tool rounds")
