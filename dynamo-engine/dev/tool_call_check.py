"""A directive with tool calls must come back as OpenAI tool_calls through each
model's own tool-call parser. Usage: tool_call_check.py <port> <model>..."""

import json
import sys

import requests

port, models = int(sys.argv[1]), sys.argv[2:]
directive = {"text": "", "tool_calls": [{"name": "Read", "arguments": {"file_path": "/tmp/step1.md", "limit": 40}}]}
tools = [{"type": "function", "function": {"name": "Read", "parameters": {
    "type": "object", "properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"}}}}}]
for model in models:
    for stream in (False, True):
        body = {"model": model, "tools": tools, "max_tokens": 64, "stream": stream,
                "messages": [{"role": "user", "content": f"<<respond:{json.dumps(directive)}>>"}]}
        response = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, stream=stream, timeout=60)
        if stream:
            calls, finish = {}, None
            for line in response.iter_lines():
                if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                    continue
                for choice in json.loads(line[5:]).get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    for call in (choice.get("delta") or {}).get("tool_calls") or []:
                        entry = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                        entry["name"] += (call.get("function") or {}).get("name") or ""
                        entry["arguments"] += (call.get("function") or {}).get("arguments") or ""
            result = [(c["name"], json.loads(c["arguments"] or "{}")) for c in calls.values()]
        else:
            message = response.json()["choices"][0]
            finish = message["finish_reason"]
            result = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                      for c in message["message"].get("tool_calls") or []]
        ok = result == [("Read", {"file_path": "/tmp/step1.md", "limit": 40})] and finish == "tool_calls"
        print(f"{'PASS' if ok else 'FAIL'} {model:36s} stream={stream!s:5s} finish={finish} calls={result}")
