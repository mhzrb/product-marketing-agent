---
name: tool_protocol
version: 1.0.0
---
Tool calling protocol (this model endpoint has no native tool calling).
To call a tool, reply with exactly one JSON object and nothing else:
{"action": "tool", "tool": "<tool name>", "arguments": { ... }}
When you have everything you need, reply with exactly one JSON object and nothing else:
{"action": "final", "output": <your final JSON answer>}
Tool results are returned to you in messages that start with <tool_result>. Call at most one tool
per reply.

Available tools:
${tool_list}
