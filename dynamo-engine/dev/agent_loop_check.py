"""A chained agent loop must advance through the real chat template and tool
parser: each tool result carries the directive for the next turn, and the
newest one wins. Usage: agent_loop_check.py <port> <model>..."""

import json
import sys

import requests

port, models = int(sys.argv[1]), sys.argv[2:]
tools = [{"type": "function", "function": {"name": "Read", "parameters": {
    "type": "object", "properties": {"file_path": {"type": "string"}}}}}]


def directive(payload):
    return f"<<respond:{json.dumps(payload)}>>"


def read(path):
    return directive({"tool_calls": [{"name": "Read", "arguments": {"file_path": path}}]})


for model in models:
    messages = [{"role": "user", "content": f"Read step1 then follow each file. {read('/w/step1.txt')}"}]
    results = [f"step1 body\n{read('/w/step2.txt')}", f"step2 body\n{directive({'text': 'SESSION COMPLETE'})}"]
    expected = [("tool_calls", "/w/step1.txt"), ("tool_calls", "/w/step2.txt"), ("stop", "SESSION COMPLETE")]
    for turn, (want_finish, want) in enumerate(expected):
        body = {"model": model, "tools": tools, "max_tokens": 64, "messages": messages}
        choice = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=60).json()["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls") or []
        got = json.loads(calls[0]["function"]["arguments"])["file_path"] if calls else (message.get("content") or "").strip()
        ok = choice["finish_reason"] == want_finish and got == want
        print(f"{'PASS' if ok else 'FAIL'} {model:28s} turn={turn} finish={choice['finish_reason']} got={got!r}")
        if not calls:
            break
        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": results[turn]})
